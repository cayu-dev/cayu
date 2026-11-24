"""Agent health: the public readers through Cloud's `agent-read` route, `agent_health`, and
`cayu cloud service health`."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from cayu.cli import main
from cayu.storage import SQLiteTaskStore
from cayu.tasks.creation import TaskCreate

API_KEY = "read-key-canary-material"
OPERATOR = {"username": "operator", "password": "operator-password-canary"}
NOW = datetime.now(UTC)
_AGENT_READ = "/v1/applications/agent-1/agent-read/"
_ALLOWED = ("api/tasks", "api/tasks/latest")


def _at(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def _task(task_id: str, task_type: str, status: str, hours_ago: float, **extra: Any) -> dict:
    return {
        "id": task_id,
        "type": task_type,
        "status": status,
        "status_reason": None,
        "error_code": None,
        "error_summary": None,
        "updated_at": _at(hours_ago),
        "completed_at": _at(hours_ago) if status in {"completed", "failed"} else None,
        **extra,
    }


class FakeCloud:
    """Cayu Cloud's API, its `agent-read` route and the Agent behind it, on one server."""

    def __init__(self) -> None:
        # Whether Cloud offers `agent-read`; without it Cloud answers FastAPI's plain 404.
        self.offers_agent_read = True
        self.tasks: list[dict[str, Any]] = []
        self.service: dict[str, Any] = {}
        self.counts: tuple[int, dict[str, Any]] = (
            200,
            {"counts": {"approval": 0, "question": 0}, "total_count": 0, "complete": True},
        )
        self.inbox_items: list[dict[str, Any]] = []
        # Answers `agent-read` gives before it reads the Agent, such as a wake's 503.
        self.agent_read_answers: list[tuple[int, Any]] = []
        self.requests: list[tuple[str, str | None]] = []
        # The Cloud keys and login tokens Cloud accepts now.
        self.accepted_keys = {API_KEY}
        self.inbox_page: dict[str, Any] = {}


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[FakeCloud]:
    state = FakeCloud()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parts = urlsplit(self.path)
            authorization = self.headers.get("Authorization")
            state.requests.append((self.path, authorization))
            status, body = self._answer(parts.path, parse_qs(parts.query), authorization)
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _answer(
            self, path: str, query: dict[str, list[str]], authorization: str | None
        ) -> tuple[int, Any]:
            if path.startswith("/v1/"):
                if authorization not in {f"Bearer {key}" for key in state.accepted_keys}:
                    return 401, {"detail": "Not authenticated"}
                return self._cloud(path, query)
            if path.startswith("/cayu/api/"):
                expected = (
                    "Basic "
                    + base64.b64encode(
                        f"{OPERATOR['username']}:{OPERATOR['password']}".encode()
                    ).decode()
                )
                if authorization != expected:
                    return 401, {"detail": "Not authenticated"}
                return self._agent(path.removeprefix("/cayu/"), query)
            return 404, {"detail": "Not Found"}

        def _cloud(self, path: str, query: dict[str, list[str]]) -> tuple[int, Any]:
            if path == "/v1/applications":
                return 200, {"items": [{"id": "agent-1", "name": "Agent One"}]}
            if path == "/v1/applications/agent-1/service":
                return 200, state.service
            if path == "/v1/applications/agent-1/operator-credentials":
                return 200, {**OPERATOR, "env": {"username": "U", "password": "P"}}
            if path == "/v1/applications/agent-1/inbox/counts":
                return state.counts
            if path == "/v1/applications/agent-1/inbox":
                assert query.get("limit") == ["50"]
                return 200, {
                    "items": state.inbox_items,
                    "agents": [{"application_id": "agent-1", "state": "read"}],
                    "next_cursor": None,
                    "complete": True,
                    **state.inbox_page,
                }
            if path.startswith(_AGENT_READ) and state.offers_agent_read:
                if state.agent_read_answers:
                    return state.agent_read_answers.pop(0)
                forwarded = unquote(path.removeprefix(_AGENT_READ))
                one_task = forwarded.startswith("api/tasks/") and forwarded.count("/") == 2
                if forwarded not in _ALLOWED and not one_task:
                    return 404, {
                        "detail": {
                            "code": "agent_read_path_not_allowed",
                            "message": "Cloud doesn't forward this read.",
                        }
                    }
                return self._agent(forwarded, query)
            return 404, {"detail": "Not Found"}

        def _agent(self, route: str, query: dict[str, list[str]]) -> tuple[int, Any]:
            if route == "api/tasks":
                rows = [
                    task
                    for task in state.tasks
                    if ("status" not in query or task["status"] == query["status"][0])
                    and ("type" not in query or task["type"] == query["type"][0])
                ]
                rows.sort(key=lambda task: task["updated_at"], reverse=True)
                offset = int(query.get("offset", ["0"])[0])
                limit = int(query.get("limit", ["50"])[0])
                return 200, rows[offset : offset + limit]
            if route.startswith("api/tasks/"):
                task_id = route.removeprefix("api/tasks/")
                for task in state.tasks:
                    if task["id"] == task_id:
                        return 200, {**task, "result": {"ok": True}}
                return 404, {"detail": f"Task not found: {task_id}"}
            return 404, {"detail": "Not Found"}

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    state.service = {
        "status": "running",
        "cayu_url": f"{origin}/cayu/",
        "deployment_id": "rel_0123456789abcdef",
        "paused": False,
        "last_request_at": None,
        "web_service": "internal-service-name",
    }
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setenv("CAYU_CLOUD_API_KEY", API_KEY)
    monkeypatch.setenv("CAYU_CLOUD_API_URL", origin)
    try:
        yield state
    finally:
        server.shutdown()
        thread.join()


def _operator_login_requested(cloud: FakeCloud) -> bool:
    return any(path.endswith("/operator-credentials") for path, _ in cloud.requests)


def test_tasks_are_read_through_cloud_with_only_the_callers_key(cloud: FakeCloud) -> None:
    from cayu.storage import TaskListRequest, open_cloud_agent_readers

    cloud.tasks = [_task("t-1", "digest", "completed", 1)]
    # Cloud is waking the Agent: a 503 `starting` without Retry-After is read again.
    cloud.agent_read_answers = [
        (503, {"detail": {"code": "starting", "message": "The Agent is starting."}})
    ]

    async def read() -> tuple[list[dict[str, Any]], dict[str, Any] | None, Any]:
        async with open_cloud_agent_readers("Agent One") as agent:
            rows = await agent.tasks.list_tasks(TaskListRequest(type="digest"))
            task = await agent.tasks.load_task("t-1")
            missing = await agent.tasks.load_task("t-2")
            assert missing is None
            return rows, task, agent.source

    rows, task, source = asyncio.run(read())

    assert [row["id"] for row in rows] == ["t-1"]
    assert task is not None and task["result"] == {"ok": True}
    assert source == {"kind": "cayu_cloud", "application": "Agent One"}
    assert not _operator_login_requested(cloud)
    assert not any(path.startswith("/cayu/") for path, _ in cloud.requests)


def test_a_cloud_without_agent_read_falls_back_to_the_operator_login(cloud: FakeCloud) -> None:
    from cayu.storage import TaskListRequest, open_cloud_agent_readers

    cloud.offers_agent_read = False
    cloud.tasks = [_task("t-1", "digest", "completed", 1)]

    async def read() -> list[dict[str, Any]]:
        async with open_cloud_agent_readers("agent-1") as agent:
            first = await agent.tasks.list_tasks(TaskListRequest())
            second = await agent.tasks.list_tasks(TaskListRequest(type="digest"))
            return first + second

    rows = asyncio.run(read())

    assert [row["id"] for row in rows] == ["t-1", "t-1"]
    agent_read = [path for path, _ in cloud.requests if path.startswith(_AGENT_READ)]
    # Cloud's answer that it has no such route is taken once, and never tried again.
    assert len(agent_read) == 1
    assert _operator_login_requested(cloud)
    assert any(path.startswith("/cayu/api/tasks") for path, _ in cloud.requests)


def test_a_route_refusal_is_not_taken_for_a_missing_route(cloud: FakeCloud) -> None:
    from cayu.storage import AgentReadError, TaskListRequest, open_cloud_agent_readers

    cloud.agent_read_answers = [
        (403, {"detail": {"code": "scope_required", "message": "The key can't read Agents."}})
    ]

    async def read() -> None:
        async with open_cloud_agent_readers("agent-1") as agent:
            await agent.tasks.list_tasks(TaskListRequest())

    with pytest.raises(AgentReadError) as raised:
        asyncio.run(read())
    assert raised.value.category == "forbidden"
    assert raised.value.code == "scope_required"
    assert not _operator_login_requested(cloud)


def test_agent_health_sums_up_tasks_inbox_and_service(cloud: FakeCloud) -> None:
    from cayu.storage import AgentHealth, agent_health, open_cloud_agent_readers

    cloud.tasks = [
        _task("d-old", "digest", "completed", 30),
        _task("d-1", "digest", "failed", 20, error_summary="timeout"),
        _task("d-2", "digest", "completed", 10),
        _task("d-3", "digest", "failed", 3, error_summary="first"),
        _task(
            "d-4",
            "digest",
            "failed",
            2,
            error_summary="GitHub said:\n  token=ghp_abcdefghijklmnopqrstuvwxyz0123 is bad",
        ),
        _task("d-5", "digest", "running", 1),
        # Updated before the window: its type isn't reported.
        _task("s-1", "sweep", "completed", 48),
        _task("p-1", "pass", "completed", 5),
    ]
    cloud.counts = (
        200,
        {
            "counts": {"approval": 2, "question": 1, "recovery": 0, "knowledge": 0},
            "total_count": 3,
            "complete": True,
        },
    )
    cloud.inbox_items = [
        {"id": "i-2", "kind": "approval", "created_at": _at(1)},
        {"id": "i-1", "kind": "question", "created_at": _at(7)},
    ]

    async def read() -> AgentHealth:
        async with open_cloud_agent_readers("agent-1") as agent:
            return await agent_health(agent, since=NOW - timedelta(hours=24))

    health = asyncio.run(read())
    shown = health.to_dict()

    assert set(shown["tasks"]) == {"digest", "pass"}
    digest = shown["tasks"]["digest"]
    assert digest["last_completed_at"] == _at(10)
    assert digest["last_failed_at"] == _at(2)
    assert digest["failure_streak"] == 2
    reason = digest["last_failure_reason"]
    assert reason.startswith("GitHub said: token=")
    assert "ghp_abcdefghijklmnop" not in reason and "\n" not in reason
    assert shown["tasks"]["pass"]["failure_streak"] == 0
    assert shown["tasks"]["pass"]["last_failed_at"] is None
    assert shown["inbox"] == {"approval": 2, "question": 1, "recovery": 0, "knowledge": 0}
    assert shown["oldest_inbox_item_at"] == _at(7)
    assert shown["service"] == {
        "status": "running",
        "deployment_id": "rel_0123456789abcdef",
        "cayu_version": None,
        "paused": False,
        "last_request_at": None,
    }
    assert shown["unavailable"] == []
    assert json.loads(json.dumps(shown)) == shown
    assert not _operator_login_requested(cloud)


def test_agent_health_names_each_part_it_couldnt_read(cloud: FakeCloud) -> None:
    from cayu.storage import agent_health, open_cloud_agent_readers

    cloud.tasks = [_task("d-1", "digest", "completed", 1)]
    cloud.counts = (
        200,
        {"counts": None, "complete": False, "unavailable_reason": "access_limited"},
    )

    async def read() -> dict[str, Any]:
        async with open_cloud_agent_readers("agent-1") as agent:
            health = await agent_health(
                agent, since=NOW - timedelta(hours=24), task_types=["digest", "digest"]
            )
            cloud.service = {}
            cloud.agent_read_answers = [
                (403, {"detail": {"code": "scope_required", "message": "No."}})
            ]
            later = await agent_health(agent, since=NOW - timedelta(hours=24))
            return {"first": health.to_dict(), "later": later.to_dict()}

    shown = asyncio.run(read())

    first = shown["first"]
    assert list(first["tasks"]) == ["digest"]
    assert first["inbox"] is None and first["oldest_inbox_item_at"] is None
    assert first["unavailable"] == [
        {
            "part": "inbox",
            "category": "access_limited",
            "message": "The Agent's inbox couldn't be counted (access_limited).",
        }
    ]
    later = shown["later"]
    assert later["tasks"] == {}
    assert [(part["part"], part["category"]) for part in later["unavailable"]] == [
        ("tasks", "forbidden"),
        ("inbox", "access_limited"),
    ]
    assert later["service"]["status"] is None


def test_agent_health_reads_a_local_store_the_same_way(tmp_path: Path) -> None:
    from cayu.storage import agent_health, open_store_agent_readers
    from cayu.storage.targets import SessionStoreBackend, SessionStoreTarget

    database = tmp_path / "cayu.db"

    async def seed() -> None:
        store = SQLiteTaskStore(database)
        try:
            await store.create_task(TaskCreate(task_id="done", type="scan"))
            claimed = await store.claim_task("worker-a")
            assert claimed is not None
            await store.complete_task(
                "done",
                {"ok": True},
                worker_id="worker-a",
                lease_expires_at=claimed.lease_expires_at,
            )
        finally:
            await store.close()

    async def read() -> dict[str, Any]:
        target = SessionStoreTarget(backend=SessionStoreBackend.SQLITE, sqlite_path=database)
        async with open_store_agent_readers(target) as agent:
            assert await agent.service() is None
            health = await agent_health(agent, since=datetime.now(UTC) - timedelta(hours=1))
        return health.to_dict()

    asyncio.run(seed())
    shown = asyncio.run(read())

    assert shown["source"] == {"kind": "sqlite", "path": str(database)}
    assert shown["tasks"]["scan"]["last_completed_at"] is not None
    assert shown["tasks"]["scan"]["failure_streak"] == 0
    assert shown["inbox"] == {"approval": 0, "question": 0, "recovery": 0}
    assert shown["service"] is None
    assert shown["unavailable"] == []


def test_agent_health_needs_an_aware_since(cloud: FakeCloud) -> None:
    from cayu.storage import agent_health, open_cloud_agent_readers

    async def read() -> None:
        async with open_cloud_agent_readers("agent-1") as agent:
            await agent_health(agent, since=datetime(2026, 1, 1))

    with pytest.raises(ValueError, match="timezone-aware"):
        asyncio.run(read())


def test_service_health_command_prints_a_summary_or_json(
    cloud: FakeCloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud.tasks = [
        _task("d-1", "digest", "completed", 5),
        _task("d-2", "digest", "failed", 1, error_summary="model refused"),
    ]

    assert main(["cloud", "service", "health", "--application", "agent-1", "--json"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["ok"] is True
    assert envelope["operation"] == "service.health"
    result = envelope["result"]
    assert result["tasks"]["digest"]["failure_streak"] == 1
    assert result["source"] == {"kind": "cayu_cloud", "application": "agent-1"}
    since = datetime.fromisoformat(result["since"])
    assert timedelta(hours=23) < datetime.now(UTC) - since < timedelta(hours=25)

    assert main(["cloud", "service", "health", "--application", "agent-1", "--since", "2h"]) == 0
    out = capsys.readouterr().out
    assert "Service: running" in out
    assert "Inbox: empty" in out
    assert "FAILED IN A ROW" in out
    assert "model refused" in out
    assert API_KEY not in out and OPERATOR["password"] not in out
    assert not _operator_login_requested(cloud)


@pytest.mark.parametrize("span", ["24", "0h", "1w", "400d"])
def test_service_health_refuses_a_bad_since(span: str, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["cloud", "service", "health", "--application", "agent-1", "--since", span])

    assert code != 0
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["category"] == "invalid_input"
    assert "--since" in error["message"]


@pytest.mark.parametrize(
    ("status", "code", "category"),
    [
        (404, "agent_read_not_served", "not_served"),
        (404, "agent_read_path_not_allowed", "unsupported"),
        (502, "credentials_rejected", "credentials_rejected"),
        (502, "identity_refused", "credentials_rejected"),
        (409, "agent_start_failed", "not_serving"),
        (502, "too_large", "too_large"),
        (502, "agent_answer_invalid", "not_cayu"),
    ],
)
def test_clouds_own_refusals_are_reported_and_never_fall_back(
    cloud: FakeCloud, status: int, code: str, category: str
) -> None:
    from cayu.storage import AgentReadError, agent_health, open_cloud_agent_readers

    refusal = (status, {"detail": {"code": code, "message": "Cloud couldn't read the Agent."}})
    cloud.agent_read_answers = [refusal, refusal]

    async def read() -> dict[str, Any]:
        async with open_cloud_agent_readers("agent-1") as agent:
            with pytest.raises(AgentReadError) as raised:
                # Not "no such task": Cloud said why it couldn't read.
                await agent.tasks.load_task("t-1")
            assert (raised.value.category, raised.value.code) == (category, code)
            assert not raised.value.transient
            health = await agent_health(agent, since=NOW - timedelta(hours=1))
            return health.to_dict()

    shown = asyncio.run(read())

    assert shown["unavailable"][0]["part"] == "tasks"
    assert shown["unavailable"][0]["category"] == category
    assert shown["service"]["status"] == "running"
    assert not _operator_login_requested(cloud)
    assert not any(path.startswith("/cayu/") for path, _ in cloud.requests)


def test_an_expiring_cloud_login_is_refreshed_for_forwarded_reads(cloud: FakeCloud) -> None:
    import os

    from cayu.cli._cloud_api import CloudApiClient
    from cayu.storage import TaskListRequest
    from cayu.storage.agent_readers import cloud_agent_readers

    cloud.tasks = [_task("t-1", "digest", "completed", 1)]
    current = {"token": "token-1"}
    replaced: list[str] = []

    def provider() -> str:
        return current["token"]

    def rejected(token: str) -> str:
        replaced.append(token)
        current["token"] = "token-3"
        return "token-3"

    client = CloudApiClient(
        api_url=os.environ["CAYU_CLOUD_API_URL"],
        api_key="unused-key-value",
        api_key_provider=provider,
        rejected_api_key_provider=rejected,
    )

    async def read() -> list[str]:
        readers = cloud_agent_readers(client, "agent-1", source={"kind": "cayu_cloud"})
        ids: list[str] = []
        async with readers:
            for token in ("token-1", "token-2"):
                # The provider's login changes between reads; each read sends the current one.
                cloud.accepted_keys = {token}
                current["token"] = token
                ids += [row["id"] for row in await readers.tasks.list_tasks(TaskListRequest())]
            # Cloud rejects the login: it is replaced once and the read sent again.
            cloud.accepted_keys = {"token-3"}
            ids += [row["id"] for row in await readers.tasks.list_tasks(TaskListRequest())]
        return ids

    assert asyncio.run(read()) == ["t-1", "t-1", "t-1"]
    assert replaced == ["token-2"]
    sent = [auth for path, auth in cloud.requests if path.startswith(_AGENT_READ)]
    assert sent == ["Bearer token-1", "Bearer token-2", "Bearer token-2", "Bearer token-3"]


@pytest.mark.parametrize(
    "page",
    [
        {"complete": False},
        {"agents": [{"state": "read", "issues": [{"code": "inbox_truncated"}]}]},
        {"agents": [{"state": "read", "earlier_release_items": ["i-0"]}]},
    ],
)
def test_the_oldest_inbox_item_is_only_read_from_a_whole_inbox(
    cloud: FakeCloud, page: dict[str, Any]
) -> None:
    from cayu.storage import open_cloud_agent_readers

    cloud.counts = (
        200,
        {"counts": {"approval": 1}, "total_count": 1, "complete": True},
    )
    cloud.inbox_items = [{"id": "i-1", "kind": "approval", "created_at": _at(1)}]
    cloud.inbox_page = page

    async def read() -> datetime | None:
        async with open_cloud_agent_readers("agent-1") as agent:
            return await agent.oldest_inbox_item_at()

    assert asyncio.run(read()) is None
