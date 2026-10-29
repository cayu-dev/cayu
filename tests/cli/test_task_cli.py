"""`cayu task list/show`: one reader seam over a local store and a Cloud Agent's task API."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from cayu.cli import main
from cayu.storage import SQLiteTaskStore
from cayu.storage import migrations as schema
from cayu.tasks.creation import TaskCreate
from cayu.tasks.records import TaskRetryPolicy
from cayu.tasks.retry import TaskRetryAttemptDisposition, TaskRetrySettlementRequest

OPERATOR = {"username": "operator", "password": "operator-password-canary"}


def _seed(path: Path) -> None:
    async def seed() -> None:
        store = SQLiteTaskStore(path)
        try:
            await store.create_task(
                TaskCreate(task_id="done", type="scan", assigned_agent_name="hunter")
            )
            claimed = await store.claim_task("worker-a")
            assert claimed is not None and claimed.id == "done"
            await store.complete_task(
                "done",
                {"findings": 3},
                worker_id="worker-a",
                lease_expires_at=claimed.lease_expires_at,
            )
            await store.create_task(
                TaskCreate(
                    task_id="retried",
                    type="review",
                    retry_policy=TaskRetryPolicy(max_attempts=3, initial_backoff_seconds=0),
                )
            )
            first = await store.claim_task("worker-b")
            assert first is not None and first.retry_series is not None
            await store.settle_task_retry_attempt(
                TaskRetrySettlementRequest(
                    task_id=first.id,
                    worker_id="worker-b",
                    lease_expires_at=first.lease_expires_at,
                    idempotency_key="retried-1",
                    causal_budget_id=first.retry_series.causal_budget_id,
                    disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                    error={"code": "flaky", "message": "try again"},
                )
            )
            await store.create_task(TaskCreate(task_id="stopped", type="scan"))
            await store.cancel_task("stopped", error={"code": "operator", "message": "not needed"})
        finally:
            await store.close()

    asyncio.run(seed())


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "cayu.db"
    _seed(path)
    return path


def _run(capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, Any]:
    code = main(["task", *arguments])
    out = capsys.readouterr().out
    return code, json.loads(out) if out.startswith("{") else out


def test_list_and_show_read_status_result_cancellation_and_retry_series(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, listed = _run(capsys, "list", "--sqlite", str(database), "--json")

    assert code == 0
    assert listed["source"] == {"kind": "sqlite", "path": str(database)}
    rows = {row["id"]: row for row in listed["tasks"]}
    assert rows["done"]["status"] == "completed"
    # The list leaves results out, as the server's does; `show` has them.
    assert "result" not in rows["done"]
    assert rows["stopped"]["cancellation"]["state"] == "cancelled"
    retried = [row for row in listed["tasks"] if row["type"] == "review"]
    assert {row["retry_series"]["attempt"] for row in retried} == {1, 2}

    code, shown = _run(capsys, "show", "done", "--sqlite", str(database), "--json")
    assert code == 0
    assert shown["task"]["result"] == {"findings": 3}
    assert shown["cancellation"] is None
    assert shown["retry_series"] is None

    code, shown = _run(capsys, "show", "stopped", "--sqlite", str(database), "--json")
    assert code == 0
    assert shown["cancellation"]["state"] == "cancelled"
    assert shown["cancellation"]["error_message"] == "not needed"

    successor = next(row["id"] for row in retried if row["retry_series"]["attempt"] == 2)
    code, shown = _run(capsys, "show", successor, "--sqlite", str(database), "--json")
    assert code == 0
    attempts = shown["retry_series"]["attempts"]
    assert [(row["task_id"], row["attempt"]) for row in attempts] == [
        ("retried", 1),
        (successor, 2),
    ]
    assert attempts[0]["status"] == "failed"
    assert shown["retry_series"]["attempts_complete"] is True

    code, out = _run(capsys, "show", successor, "--sqlite", str(database))
    assert code == 0
    assert "\nRetry series " in out and "\\x0a" not in out


def test_table_output_and_filters(database: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = _run(capsys, "list", "--sqlite", str(database), "--status", "completed")

    assert code == 0
    lines = out.splitlines()
    assert lines[0].split()[:3] == ["id", "type", "status"]
    assert [line.split()[0] for line in lines[1:]] == ["done"]

    code, out = _run(capsys, "list", "--sqlite", str(database), "--limit", "1", "--type", "scan")
    assert code == 0
    assert "add --offset 1" in out

    code, out = _run(capsys, "show", "stopped", "--sqlite", str(database))
    assert code == 0
    assert "Cancellation" in out and "not needed" in out


def test_missing_task_and_missing_database_fail_clearly(
    database: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = _run(capsys, "show", "absent", "--sqlite", str(database), "--json")
    assert code == 1
    assert out["error"] == {"code": "TASK_INSPECTION_FAILED", "message": "Task not found: absent"}

    missing = tmp_path / "missing.db"
    code, out = _run(capsys, "list", "--sqlite", str(missing), "--json")
    assert code == 1
    assert "No SQLite database exists" in out["error"]["message"]
    assert not missing.exists()


def test_read_only_sqlite_task_store_cannot_write(database: Path) -> None:
    with pytest.raises(ValueError, match="schema_mode=validate"):
        SQLiteTaskStore(database, read_only=True)

    async def write() -> None:
        store = SQLiteTaskStore(database, schema_mode=schema.SchemaMode.VALIDATE, read_only=True)
        try:
            assert await store.load_task("done") is not None
            await store.create_task(TaskCreate(task_id="new", type="scan"))
        finally:
            await store.close()

    with pytest.raises(Exception, match="readonly|read-only|read only"):
        asyncio.run(write())


class FakeCloud:
    """Cayu Cloud's API and the Agent's own Cayu API behind it, on one loopback server."""

    def __init__(self) -> None:
        self.api_base = "/cayu/api"
        self.tasks: list[dict[str, Any]] = []
        self.service: dict[str, Any] = {}
        self.agent_status: int | None = None
        self.requests: list[tuple[str, str | None]] = []


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[FakeCloud]:
    state = FakeCloud()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parts = urlsplit(self.path)
            state.requests.append((self.path, self.headers.get("Authorization")))
            status, body = self._answer(parts.path, parse_qs(parts.query))
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _answer(self, path: str, query: dict[str, list[str]]) -> tuple[int, Any]:
            if path == "/v1/applications":
                return 200, {"items": [{"id": "bug-hunter", "name": "Bug Hunter"}]}
            if path == "/v1/applications/bug-hunter/service":
                return 200, state.service
            if path == "/v1/applications/bug-hunter/operator-credentials":
                return 200, {**OPERATOR, "env": {"username": "U", "password": "P"}}
            if not path.startswith(state.api_base + "/"):
                return 404, {"detail": "Not Found"}
            if state.agent_status is not None:
                return state.agent_status, {"detail": "Not authenticated"}
            route = path[len(state.api_base) :]
            if route == "/tasks":
                rows = [
                    {key: value for key, value in task.items() if key != "result"}
                    for task in state.tasks
                    if "status" not in query or task["status"] == query["status"][0]
                ]
                return 200, rows
            if route.startswith("/tasks/"):
                task_id = route.removeprefix("/tasks/")
                for task in state.tasks:
                    if task["id"] == task_id:
                        return 200, task
                return 404, {"detail": f"Task not found: {task_id}"}
            return 404, {"detail": "Not Found"}

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    state.service = {"status": "running", "cayu_url": f"{origin}/cayu/"}
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setenv("CAYU_CLOUD_API_KEY", "customer-secret-material")
    monkeypatch.setenv("CAYU_CLOUD_API_URL", origin)
    try:
        yield state
    finally:
        server.shutdown()
        thread.join()


def _api_task(task_id: str, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": task_id,
        "type": "scan",
        "status": status,
        "status_reason": None,
        "status_payload": None,
        "assigned_agent_name": "hunter",
        "worker_id": None,
        "updated_at": "2026-10-28T16:19:00+00:00",
        "retry_series": None,
        "result": None,
        **extra,
    }


def test_application_reads_the_agents_task_api_with_its_operator_login(
    cloud: FakeCloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud.tasks = [
        _api_task("scan-1", "completed", result={"findings": 2}),
        _api_task(
            "scan-2",
            "cancelled",
            status_payload={
                "cancellation_reconciliation": {
                    "cancellation_requested_at": "2026-10-28T16:21:00+00:00",
                    "evidence": {"outcome": "quiescent"},
                }
            },
        ),
    ]

    code, listed = _run(capsys, "list", "--application", "Bug Hunter", "--json")

    assert code == 0
    assert listed["source"] == {"kind": "cayu_cloud", "application": "Bug Hunter"}
    assert [row["id"] for row in listed["tasks"]] == ["scan-1", "scan-2"]
    assert listed["tasks"][1]["cancellation"]["outcome"] == "quiescent"
    expected = "Basic " + base64.b64encode(b"operator:operator-password-canary").decode()
    agent_requests = [item for item in cloud.requests if item[0].startswith("/cayu/api/")]
    assert agent_requests and all(auth == expected for _, auth in agent_requests)
    assert "operator-password-canary" not in json.dumps(listed)

    code, shown = _run(capsys, "show", "scan-1", "--application", "bug-hunter", "--json")
    assert code == 0
    assert shown["task"]["result"] == {"findings": 2}

    code, shown = _run(capsys, "show", "absent", "--application", "bug-hunter", "--json")
    assert code == 1
    assert shown["error"]["message"] == "Task not found: absent"


def test_application_falls_back_to_cayu_serve_api_path(
    cloud: FakeCloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud.api_base = "/api"
    cloud.tasks = [_api_task("scan-1", "running", worker_id="worker-1")]

    code, listed = _run(
        capsys, "list", "--application", "bug-hunter", "--status", "running", "--json"
    )

    assert code == 0
    assert [row["id"] for row in listed["tasks"]] == ["scan-1"]
    assert any(path.startswith("/api/tasks?") for path, _ in cloud.requests)


@pytest.mark.parametrize(
    ("setup", "category", "message"),
    [
        (
            lambda cloud: cloud.service.update(status="stopped"),
            "not_serving",
            "service is stopped, so it can't be read",
        ),
        (
            lambda cloud: cloud.service.update(cayu_url=None),
            "no_address",
            "serves no Cayu address",
        ),
        (
            lambda cloud: setattr(cloud, "agent_status", 401),
            "credentials_rejected",
            "did not accept the credentials",
        ),
        (
            lambda cloud: setattr(cloud, "api_base", "/elsewhere"),
            "not_served",
            "does not serve Cayu's API at /cayu/api/ or /api/",
        ),
    ],
)
def test_application_failures_name_the_cause(
    cloud: FakeCloud,
    capsys: pytest.CaptureFixture[str],
    setup: Any,
    category: str,
    message: str,
) -> None:
    setup(cloud)

    code, out = _run(capsys, "list", "--application", "bug-hunter", "--json")

    assert code == 1
    assert out["error"]["category"] == category
    assert message in out["error"]["message"]


def test_context_requires_application(database: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = _run(capsys, "list", "--sqlite", str(database), "--context", "ctx.json", "--json")

    assert code == 1
    assert out["error"]["message"] == "--context applies only with --application."


def test_store_and_server_readers_render_the_same_task(database: Path) -> None:
    """The local reader projects tasks in the server's own task API shape."""

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from tests.server.server_test_support import _LOCAL_SERVER_CONFIG

    from cayu.applications import CayuApp
    from cayu.cli._cloud_agent_api import CayuApi
    from cayu.cli._task_readers import ApiTaskReader, StoreTaskReader, TaskListRequest
    from cayu.server import create_server

    async def read() -> tuple[Any, Any, Any, Any]:
        local = StoreTaskReader(
            SQLiteTaskStore(database, schema_mode=schema.SchemaMode.VALIDATE, read_only=True),
            source={},
        )
        store = SQLiteTaskStore(database, schema_mode=schema.SchemaMode.VALIDATE)
        app = CayuApp(task_store=store, enable_logging=False)
        with TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)) as client:
            served = ApiTaskReader(CayuApi("http://testserver/", http=client), source={})
            try:
                return (
                    await local.list_tasks(TaskListRequest()),
                    await served.list_tasks(TaskListRequest()),
                    await local.load_task("retried"),
                    await served.load_task("retried"),
                )
            finally:
                await local.close()
                await store.close()

    local_rows, served_rows, local_task, served_task = asyncio.run(read())

    assert served_rows == local_rows
    assert {key: served_task[key] for key in local_task if key != "invocation"} == {
        key: value for key, value in local_task.items() if key != "invocation"
    }


def test_pending_cancellation_shows_the_requests_reason_and_time(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "cayu.db"

    async def seed() -> None:
        store = SQLiteTaskStore(path)
        try:
            await store.create_task(TaskCreate(task_id="draining", type="scan"))
            assert await store.claim_task("worker-a") is not None
            await store.cancel_task("draining", error={"code": "operator", "message": "deploy"})
        finally:
            await store.close()

    asyncio.run(seed())

    code, shown = _run(capsys, "show", "draining", "--sqlite", str(path), "--json")

    assert code == 0
    cancellation = shown["cancellation"]
    assert cancellation["state"] == "requested"
    assert cancellation["reason"] == "cancellation_requested"
    assert (cancellation["error_code"], cancellation["error_message"]) == ("operator", "deploy")
    assert isinstance(cancellation["requested_at"], str)


def test_reconciled_cancellation_names_its_outcome_and_actor() -> None:
    from cayu.cli.task import cancellation_summary

    summary = cancellation_summary(
        {
            "status": "cancelled",
            "status_reason": None,
            "status_payload": {
                "cancellation_reconciliation": {
                    "cancellation_requested_at": "2026-10-28T16:21:00+00:00",
                    "evidence": {"outcome": "quiescent"},
                    "reconciled_by": {"subject": "operator@example.com", "source": "request"},
                }
            },
            "error": None,
        }
    )

    assert summary is not None
    assert summary["outcome"] == "quiescent"
    assert summary["reconciled_by"] == "operator@example.com"
    assert summary["requested_at"] == "2026-10-28T16:21:00+00:00"


def test_a_full_hundred_attempt_series_reads_as_complete() -> None:
    from cayu.cli.task import _retry_series

    tasks = {
        f"t{number}": {
            "id": f"t{number}",
            "status": "failed",
            "retry_series": {
                "series_id": "s",
                "attempt": number,
                "predecessor_task_id": f"t{number - 1}" if number > 1 else None,
                "successor_task_id": f"t{number + 1}" if number < 100 else None,
            },
        }
        for number in range(1, 101)
    }

    class Reader:
        async def load_task(self, task_id: str) -> dict[str, Any] | None:
            return tasks.get(task_id)

    series = asyncio.run(_retry_series(Reader(), tasks["t50"]))  # type: ignore[arg-type]

    assert series is not None
    assert [row["attempt"] for row in series["attempts"]] == list(range(1, 101))
    assert series["attempts_complete"] is True


def test_table_output_escapes_terminal_control_sequences(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "cayu.db"

    async def seed() -> None:
        store = SQLiteTaskStore(path)
        try:
            await store.create_task(
                TaskCreate(task_id="forged", type="scan", title="Review \x1b]52;c;Zm9v\x07 \x1b[2J")
            )
        finally:
            await store.close()

    asyncio.run(seed())

    code, out = _run(capsys, "show", "forged", "--sqlite", str(path))

    assert code == 0
    assert "\x1b" not in out and "\x07" not in out
    assert "Review \\x1b]52;c;Zm9v\\x07 \\x1b[2J" in out
