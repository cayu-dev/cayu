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
        # Answers to the next reads of one task, before it is read from `tasks`.
        self.task_answers: list[tuple[int, Any]] = []


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
                if state.task_answers:
                    return state.task_answers.pop(0)
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


@pytest.mark.parametrize("reason", ["cancellation_requested", "retry_cancellation_requested"])
def test_held_task_with_fence_text_reason_shows_no_cancellation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], reason: str
) -> None:
    # A hold's reason is the caller's own text: nobody asked to cancel this task.
    path = tmp_path / "cayu.db"

    async def seed() -> None:
        store = SQLiteTaskStore(path)
        try:
            await store.create_task(TaskCreate(task_id="held", type="scan"))
            await store.pause_task("held", reason=reason)
        finally:
            await store.close()

    asyncio.run(seed())

    code, shown = _run(capsys, "show", "held", "--sqlite", str(path), "--json")

    assert code == 0
    assert (shown["task"]["status"], shown["task"]["status_reason"]) == ("paused", reason)
    assert shown["cancellation"] is None


def test_cancel_action_is_not_settled_by_a_hold_reason_with_fence_text() -> None:
    from cayu.cli._control_plane_actions import _task_action_settled

    fenced = {"status": "running", "status_reason": "cancellation_requested"}
    held = {"status": "paused", "status_reason": "cancellation_requested"}
    assert _task_action_settled("cancel", fenced) is True
    assert _task_action_settled("cancel", {"status": "cancelled", "status_reason": None}) is True
    assert _task_action_settled("cancel", held) is False


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


def _wait(capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, Any, str]:
    code = main(["task", "wait", *arguments, "--poll-seconds", "0.01", "--json"])
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


def test_wait_exits_with_how_the_task_ended(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fail() -> None:
        store = SQLiteTaskStore(database)
        try:
            await store.create_task(TaskCreate(task_id="broken", type="scan"))
            await store.fail_task("broken", {"code": "crash", "message": "boom"})
        finally:
            await store.close()

    asyncio.run(fail())

    for task_id, exit_code in (("done", 0), ("broken", 3), ("stopped", 4)):
        code, shown, err = _wait(capsys, task_id, "--sqlite", str(database))
        status = shown["task"]["status"]
        assert code == exit_code, task_id
        assert shown["wait"] == {"task_id": task_id, "outcome": status}
        assert f"Task {task_id} is {status}." in err

    code, shown, _ = _wait(capsys, "absent", "--sqlite", str(database))
    assert code == 1
    assert shown["error"]["message"] == "Task not found: absent"


def test_wait_follows_a_retry_series_and_stops_at_the_wait_limit(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, shown, err = _wait(capsys, "retried", "--sqlite", str(database), "--wait-seconds", "0.1")

    # The first attempt failed with a retry scheduled; its successor hasn't run yet.
    successor = shown["task"]["id"]
    assert code == 124
    assert successor != "retried"
    assert shown["task"]["status"] == "pending"
    assert shown["wait"] == {"task_id": "retried", "outcome": "timed_out"}
    assert [row["attempt"] for row in shown["retry_series"]["attempts"]] == [1, 2]
    assert f"its retry series goes on as task {successor}" in err
    assert f"Task {successor} is still pending after waiting 0.1s" in err


def test_wait_sees_a_worker_finish_the_task_in_the_local_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker completes the task through its own connection right after the wait reads
    it claimed, so the wait's next read must see that write. Tying the completion to the
    read, not to a clock, keeps a slow CLI start from letting the worker finish first."""

    from cayu.cli._task_readers import StoreTaskReader

    path = tmp_path / "cayu.db"

    async def claim() -> Any:
        store = SQLiteTaskStore(path)
        try:
            await store.create_task(TaskCreate(task_id="live", type="scan"))
            return await store.claim_task("worker-a")
        finally:
            await store.close()

    claimed = asyncio.run(claim())

    async def complete() -> None:
        store = SQLiteTaskStore(path)
        try:
            await store.complete_task(
                "live",
                {"findings": 1},
                worker_id="worker-a",
                lease_expires_at=claimed.lease_expires_at,
            )
        finally:
            await store.close()

    read_task = StoreTaskReader.load_task

    async def read_then_finish(reader: StoreTaskReader, task_id: str) -> dict[str, Any] | None:
        task = await read_task(reader, task_id)
        if task is not None and task["status"] == "claimed":
            await complete()
        return task

    monkeypatch.setattr(StoreTaskReader, "load_task", read_then_finish)

    code, shown, err = _wait(capsys, "live", "--sqlite", str(path), "--wait-seconds", "30")

    assert code == 0
    assert shown["task"]["result"] == {"findings": 1}
    assert shown["wait"] == {"task_id": "live", "outcome": "completed"}
    assert err.splitlines() == ["Task live is claimed.", "Task live is completed."]


def test_application_wait_reads_through_until_the_task_ends(
    cloud: FakeCloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud.tasks = [_api_task("scan-1", "completed", result={"findings": 2})]
    cloud.task_answers = [
        (200, _api_task("scan-1", "running", worker_id="worker-1")),
        (502, {"detail": "Bad Gateway"}),
        (200, _api_task("scan-1", "running", worker_id="worker-1")),
    ]

    code, shown, err = _wait(capsys, "scan-1", "--application", "bug-hunter")

    assert code == 0
    assert shown["task"]["result"] == {"findings": 2}
    assert shown["wait"] == {"task_id": "scan-1", "outcome": "completed"}
    assert err.count("Task scan-1 is running.") == 1
    assert "answered HTTP 502" in err and "Reading task scan-1 again." in err
    assert "Task scan-1 is completed." in err
    reads = [path for path, _ in cloud.requests if path.startswith("/cayu/api/tasks/scan-1")]
    assert len(reads) == 4


def test_application_wait_names_a_task_still_running_and_a_failure_that_lasts(
    cloud: FakeCloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud.tasks = [_api_task("scan-1", "running", worker_id="worker-1")]

    code = main(["task", "wait", "scan-1", "--application", "bug-hunter", "--wait-seconds", "0.05"])
    captured = capsys.readouterr()

    assert code == 124
    assert "worker-1" in captured.out
    assert "Task scan-1 is still running after waiting 0.05s; it goes on." in captured.err

    cloud.agent_status = 401
    code, shown, _ = _wait(capsys, "scan-1", "--application", "bug-hunter")
    assert code == 1
    assert shown["error"]["category"] == "credentials_rejected"


def test_wait_never_reports_an_earlier_attempts_end_for_an_unread_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed attempt handed the wait to its successor; reads of the successor failed
    until the wait ran out, and the Agent answered again right after. The earlier attempt's
    failure is not how the work ended."""

    import argparse

    import cayu.cli.task as task_cli
    from cayu.cli._cloud_agent_api import CayuApiError

    first = _api_task("a", "failed", retry_series={"attempt": 1, "successor_task_id": "b"})
    second = _api_task("b", "running", retry_series={"attempt": 2, "predecessor_task_id": "a"})

    class Reader:
        source: dict[str, Any] = {}
        reachable = False

        async def load_task(self, task_id: str) -> dict[str, Any] | None:
            if task_id == "a":
                return first
            if not self.reachable:
                raise CayuApiError("unreachable", "Could not connect to the Agent.")
            return second

        async def task_session_ids(self, task: dict[str, Any]) -> None:
            return None

        async def close(self) -> None:
            return None

    reader = Reader()
    read_series = task_cli._retry_series

    async def after_the_wait(reader_: Any, task: dict[str, Any]) -> Any:
        reader.reachable = True
        return await read_series(reader_, task)

    monkeypatch.setattr(task_cli, "_retry_series", after_the_wait)
    args = argparse.Namespace(
        task_id="a",
        wait_seconds=0.05,
        poll_seconds=0.01,
        output_format="json",
        wake_timeout_seconds=180.0,
    )

    with pytest.raises(CayuApiError, match="Could not connect"):
        asyncio.run(task_cli._wait_for_task(args, reader))  # type: ignore[arg-type]


def test_wait_keeps_the_task_it_read_when_its_series_history_cant_be_read(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The awaited attempt completed; reading its predecessor for the retry-series history
    failed the way a waking Agent does. The completion stands, with the history marked
    incomplete."""

    import argparse

    from cayu.cli._cloud_agent_api import CayuApiError
    from cayu.cli.task import _wait_for_task

    retry = _api_task(
        "retry-2",
        "completed",
        result={"findings": 1},
        retry_series={"series_id": "s", "attempt": 2, "predecessor_task_id": "retry-1"},
    )

    class Reader:
        source: dict[str, Any] = {}

        async def load_task(self, task_id: str) -> dict[str, Any] | None:
            if task_id == "retry-2":
                return retry
            raise CayuApiError("unreachable", "Could not connect to the Agent.")

        async def task_session_ids(self, task: dict[str, Any]) -> None:
            return None

        async def close(self) -> None:
            return None

    args = argparse.Namespace(
        task_id="retry-2",
        wait_seconds=1.0,
        poll_seconds=0.01,
        output_format="json",
        wake_timeout_seconds=180.0,
    )

    code = asyncio.run(_wait_for_task(args, Reader()))  # type: ignore[arg-type]

    shown = json.loads(capsys.readouterr().out)
    assert code == 0
    assert shown["wait"] == {"task_id": "retry-2", "outcome": "completed"}
    assert shown["retry_series"]["attempts_complete"] is False
    assert [row["task_id"] for row in shown["retry_series"]["attempts"]] == ["retry-2"]


def test_series_history_still_fails_on_a_refusal_that_wont_pass() -> None:
    from cayu.cli._cloud_agent_api import CayuApiError
    from cayu.cli.task import _retry_series

    class Reader:
        async def load_task(self, task_id: str) -> dict[str, Any] | None:
            raise CayuApiError("credentials_rejected", "The Agent did not accept the login.")

    task = {"id": "b", "status": "failed", "retry_series": {"predecessor_task_id": "a"}}
    with pytest.raises(CayuApiError, match="did not accept"):
        asyncio.run(_retry_series(Reader(), task))  # type: ignore[arg-type]


def test_wait_with_no_wait_fails_at_once_on_a_sleeping_agent() -> None:
    """``--no-wait`` makes the first read of a sleeping Agent fail, so the wait does too,
    rather than reading through the wake as it reads through other transient failures."""

    import argparse

    from cayu.cli._cloud_agent_api import CayuApiError
    from cayu.cli.task import _wait_for_task

    reads: list[str] = []

    class Reader:
        source: dict[str, Any] = {}

        async def load_task(self, task_id: str) -> dict[str, Any] | None:
            reads.append(task_id)
            raise CayuApiError("starting", "The Agent a is starting.", status_code=503)

        async def task_session_ids(self, task: dict[str, Any]) -> None:
            return None

        async def close(self) -> None:
            return None

    args = argparse.Namespace(
        task_id="t",
        wait_seconds=5.0,
        poll_seconds=0.01,
        output_format="json",
        wake_timeout_seconds=0.0,
    )

    with pytest.raises(CayuApiError, match="is starting"):
        asyncio.run(_wait_for_task(args, Reader()))  # type: ignore[arg-type]
    assert reads == ["t"]


def test_field_prints_one_value_and_raw_prints_text_as_is(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = _run(capsys, "show", "done", "--sqlite", str(database), "--field", "result")
    assert code == 0
    assert out == {"findings": 3}

    code, out = _run(
        capsys, "show", "done", "--sqlite", str(database), "--field", "result.findings", "--raw"
    )
    assert code == 0
    # Only text is printed bare; a number stays JSON.
    assert out == "3\n"

    assert (
        main(["task", "show", "done", "--sqlite", str(database), "--field", "result.digest"]) == 1
    )
    assert "Task done has no field result.digest: result has no 'digest'." in (
        capsys.readouterr().err
    )


def test_field_errors_are_json_with_json_output(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = _run(
        capsys, "show", "stopped", "--sqlite", str(database), "--field", "result.x", "--json"
    )

    assert code == 1
    assert out["error"]["message"] == (
        "Task stopped has no field result.x: result has no 'x' (result is null)."
    )


def test_wait_field_keeps_the_outcome_exit_code(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A cancelled task has no result: the field error is a note, and the exit says how it ended.
    code = main(["task", "wait", "stopped", "--sqlite", str(database), "--field", "result.x"])
    err = capsys.readouterr().err
    assert code == 4
    assert "Task stopped has no field result.x" in err
    # A completed task missing the field is still an error.
    assert (
        main(["task", "wait", "done", "--sqlite", str(database), "--field", "result.digest"]) == 1
    )
    capsys.readouterr()


def test_raw_needs_a_field(database: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["task", "show", "done", "--sqlite", str(database), "--raw"]) == 2
    assert "--raw prints one field; add --field PATH." in capsys.readouterr().err


def test_application_field_raw_prints_a_deployed_tasks_markdown(
    cloud: FakeCloud, capsys: pytest.CaptureFixture[str]
) -> None:
    brief = '# Brief\n\n- "quoted" item\n'
    cloud.tasks = [_api_task("digest-1", "completed", result={"markdown": brief, "items": [1]})]

    code, out = _run(
        capsys,
        "show",
        "digest-1",
        "--application",
        "bug-hunter",
        "--field",
        "result.markdown",
        "--raw",
    )

    assert code == 0
    assert out == brief + "\n"
    code, out = _run(
        capsys, "show", "digest-1", "--application", "bug-hunter", "--field", "result.items.0"
    )
    assert (code, out) == (0, "1\n")


def test_a_project_command_reads_a_deployed_agents_tasks_through_the_public_readers(
    cloud: FakeCloud,
) -> None:
    from cayu.storage import TaskListRequest, open_cloud_agent_readers
    from cayu.tasks.records import TaskStatus

    cloud.tasks = [
        _api_task("digest-2", "running"),
        _api_task("digest-1", "completed", result={"markdown": "# Brief"}),
    ]

    async def last_digest() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        async with open_cloud_agent_readers("bug-hunter") as agent:
            assert agent.source == {"kind": "cayu_cloud", "application": "bug-hunter"}
            rows = await agent.tasks.list_tasks(TaskListRequest(status=TaskStatus.COMPLETED))
            task = await agent.tasks.load_task(rows[0]["id"])
            assert task is not None
            return task, rows

    task, rows = asyncio.run(last_digest())

    assert [row["id"] for row in rows] == ["digest-1"]
    assert task["result"] == {"markdown": "# Brief"}
    expected = "Basic " + base64.b64encode(b"operator:operator-password-canary").decode()
    assert all(auth == expected for path, auth in cloud.requests if path.startswith("/cayu/api/"))


def test_the_public_readers_fail_with_a_category_when_the_agent_cant_be_read(
    cloud: FakeCloud,
) -> None:
    from cayu.storage import AgentReadError, open_cloud_agent_readers

    cloud.service.update(status="stopped")

    with pytest.raises(AgentReadError) as raised:
        open_cloud_agent_readers("bug-hunter")
    assert raised.value.category == "not_serving"


def test_the_public_readers_read_a_local_store_the_same_way(database: Path) -> None:
    from cayu.storage import TaskListRequest, open_store_agent_readers
    from cayu.storage.targets import SessionStoreBackend, SessionStoreTarget

    async def read() -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        target = SessionStoreTarget(backend=SessionStoreBackend.SQLITE, sqlite_path=database)
        async with open_store_agent_readers(target) as agent:
            return (
                await agent.tasks.list_tasks(TaskListRequest(type="scan")),
                await agent.tasks.load_task("done"),
            )

    rows, task = asyncio.run(read())

    assert {row["id"] for row in rows} == {"done", "stopped"}
    assert task is not None and task["result"] == {"findings": 3}
