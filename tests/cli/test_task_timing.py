"""The sessions run under a task, and `cayu task show`'s time breakdown and lag over them."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from cayu.cli import main
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.runtime._task_lease_authority import TaskLeaseAuthority, bind_task_lease_authority
from cayu.session_metadata import TASK_METADATA_KEY
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks.creation import TaskCreate

_IDENTITY = SessionIdentity(provider_name="fake", model="m")


def _attempt(n: int) -> dict[str, str]:
    return {"model_step_id": f"mstep_{n:032d}", "model_attempt_id": f"matt_{n:032d}"}


def _in_task(task_id: str):
    return bind_task_lease_authority(
        TaskLeaseAuthority(
            datetime.now(UTC) + timedelta(minutes=5),
            claim_generation=1,
            task_id=task_id,
            worker_id="w",
        )
    )


async def _create(store, session_id: str, **request: Any) -> None:
    await store.create(
        RunRequest(
            agent_name="hunter",
            session_id=session_id,
            messages=[Message.text("user", "x")],
            **request,
        ),
        identity=_IDENTITY,
    )


async def _create_in_task(store, task_id: str, session_id: str) -> None:
    # A handler's code runs in the asyncio task the worker created under its binding.
    with _in_task(task_id):
        running = asyncio.ensure_future(_create(store, session_id))
    await running


async def _append(store, session_id: str, at: datetime, event_type, **fields: Any) -> None:
    await store.append_event(
        session_id,
        Event(type=event_type, session_id=session_id, timestamp=at, **fields),
    )


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_sessions_created_in_a_task_handler_are_listed_under_the_task(
    backend: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    if backend == "postgres":
        from cayu.storage import PostgresSessionStore
        from cayu.storage.migrations import SchemaMode

        dsn = request.getfixturevalue("postgres_dsn")

    def open_store():
        if backend == "memory":
            return InMemorySessionStore()
        if backend == "sqlite":
            return SQLiteSessionStore(tmp_path / "s.db")
        return PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)

    async def scenario() -> list[list[str]]:
        store = open_store()
        try:
            await _create_in_task(store, "scan", "hunt")
            # A child created later, outside the handler, runs under its parent's task.
            await _create(store, "hunt-child", parent_session_id="hunt")
            await _create_in_task(store, "other-task", "elsewhere")
            await _create(store, "unrelated")
            hunt = await store.load("hunt")
            assert hunt is not None and hunt.metadata[TASK_METADATA_KEY] == "scan"
            unrelated = await store.load("unrelated")
            assert unrelated is not None and TASK_METADATA_KEY not in unrelated.metadata
            return [
                sorted(
                    session.id
                    for session in (
                        await store.list_sessions(SessionQuery(task_id=task_id))
                    ).sessions
                )
                for task_id in ("scan", "other-task", "missing")
            ]
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                await close()

    assert asyncio.run(scenario()) == [["hunt", "hunt-child"], ["elsewhere"], []]


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_forks_run_under_the_task_of_their_handler_or_source(
    backend: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    from cayu.sessions import Session, SessionStatus
    from cayu.sessions.forks import fork_session_invocation

    if backend == "postgres":
        from cayu.storage import PostgresSessionStore
        from cayu.storage.migrations import SchemaMode

        dsn = request.getfixturevalue("postgres_dsn")

    def open_store():
        if backend == "memory":
            return InMemorySessionStore()
        if backend == "sqlite":
            return SQLiteSessionStore(tmp_path / "s.db")
        return PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)

    async def fork(store, source_id: str, fork_id: str) -> None:
        source = await store.load(source_id)
        assert source is not None
        await store.create_fork(
            source_session_id=source_id,
            fork=Session(
                id=fork_id,
                agent_name="hunter",
                provider_name="fake",
                model="m",
                parent_session_id=source_id,
                invocation=fork_session_invocation(source),
                status=SessionStatus.COMPLETED,
            ),
            source_statuses={SessionStatus.COMPLETED},
            expected_source_run_epoch=source.run_epoch,
            transcript_cursor=None,
            checkpoint_transform=None,
        )

    async def fork_in_task(store, task_id: str, source_id: str, fork_id: str) -> None:
        with _in_task(task_id):
            running = asyncio.ensure_future(fork(store, source_id, fork_id))
        await running

    async def scenario() -> dict[str, str | None]:
        store = open_store()
        try:
            await _create(store, "f-unmarked")
            await _create_in_task(store, "scan", "f-hunt")
            for source_id in ("f-unmarked", "f-hunt"):
                await store.update_status(source_id, SessionStatus.COMPLETED)
            # A handler's fork runs under its task; a later fork, under its source's.
            await fork_in_task(store, "scan", "f-unmarked", "f-handler-fork")
            await fork(store, "f-hunt", "f-hunt-fork")
            await fork(store, "f-unmarked", "f-plain-fork")
            marks: dict[str, str | None] = {}
            for session_id in ("f-handler-fork", "f-hunt-fork", "f-plain-fork"):
                session = await store.load(session_id)
                assert session is not None
                marks[session_id] = session.metadata.get(TASK_METADATA_KEY)
            return marks
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                await close()

    assert asyncio.run(scenario()) == {
        "f-handler-fork": "scan",
        "f-hunt-fork": "scan",
        "f-plain-fork": None,
    }


def test_callers_cannot_claim_a_session_runs_under_a_task() -> None:
    with pytest.raises(ValueError, match="runtime-owned task authority"):
        RunRequest(
            agent_name="hunter",
            messages=[Message.text("user", "x")],
            metadata={TASK_METADATA_KEY: "scan"},
        )


T0 = datetime(2026, 11, 3, 13, 0, tzinfo=UTC)


def _at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


async def _seed(path: Path) -> None:
    """Task ``scan`` ran two sessions from its handler; ``old`` ran none that are linked."""

    tasks = SQLiteTaskStore(path)
    sessions = SQLiteSessionStore(path)
    try:
        await tasks.create_task(TaskCreate(task_id="scan", type="scan"))
        await tasks.create_task(TaskCreate(task_id="old", type="scan"))
        await _create_in_task(sessions, "scan", "hunt")
        await _create_in_task(sessions, "scan", "verify")
        # A session the old task ran before the link was recorded.
        await _create(sessions, "untracked")
        # hunt: model 10-30, tools 30-40 (read_file) and 32-38 (search_text) in parallel,
        # model 45-50; elapsed 0-60. verify: model 5-15, tool 15-25 (read_file); 0-30.
        for session_id, events in {
            "hunt": [
                (0, EventType.SESSION_STARTED, {}),
                (10, EventType.MODEL_STARTED, {"payload": _attempt(1)}),
                (30, EventType.MODEL_COMPLETED, {"payload": _attempt(1)}),
                (30, EventType.TOOL_CALL_STARTED, _tool("read_file", "a")),
                (32, EventType.TOOL_CALL_STARTED, _tool("search_text", "b")),
                (38, EventType.TOOL_CALL_COMPLETED, _tool("search_text", "b")),
                (40, EventType.TOOL_CALL_COMPLETED, _tool("read_file", "a")),
                (45, EventType.MODEL_STARTED, {"payload": _attempt(2)}),
                (50, EventType.MODEL_COMPLETED, {"payload": _attempt(2)}),
                (55, EventType.RUNTIME_EVENT_LOOP_LAGGED, _lag(4000, 900)),
                (60, EventType.SESSION_COMPLETED, {}),
            ],
            "verify": [
                (0, EventType.SESSION_STARTED, {}),
                (5, EventType.MODEL_STARTED, {"payload": _attempt(3)}),
                (15, EventType.MODEL_COMPLETED, {"payload": _attempt(3)}),
                (15, EventType.TOOL_CALL_STARTED, _tool("read_file", "c")),
                (25, EventType.TOOL_CALL_COMPLETED, _tool("read_file", "c")),
                (30, EventType.SESSION_COMPLETED, {}),
            ],
            "untracked": [(0, EventType.RUNTIME_EVENT_LOOP_LAGGED, _lag(9000, 3000))],
        }.items():
            for seconds, event_type, fields in events:
                await _append(sessions, session_id, _at(seconds), event_type, **fields)
    finally:
        await sessions.close()
        await tasks.close()


def _tool(name: str, call_id: str) -> dict[str, Any]:
    return {"tool_name": name, "payload": {"tool_round_id": "r", "tool_call_id": call_id}}


def _lag(lagging_ms: int, max_lag_ms: int) -> dict[str, Any]:
    return {"payload": {"lagging_ms": lagging_ms, "max_lag_ms": max_lag_ms, "pid": 7}}


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "cayu.db"
    asyncio.run(_seed(path))
    return path


def _show(capsys: pytest.CaptureFixture[str], *argv: str) -> Any:
    code = main(["task", "show", *argv])
    out = capsys.readouterr().out
    assert code == 0, out
    return json.loads(out) if out.startswith("{") else out


def test_task_show_adds_up_the_time_and_lag_of_its_sessions(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shown = _show(capsys, "scan", "--sqlite", str(database), "--json")

    timing = shown["timing"]
    assert timing["session_ids"] == ["hunt", "verify"]
    assert (timing["sessions"], timing["sessions_truncated"], timing["sessions_listed"]) == (
        2,
        False,
        True,
    )
    assert timing["elapsed_ms"] == 90_000
    # Before the first model call: 10 s + 5 s. Parallel tool calls count once.
    assert timing["first_model_call_ms"] == 15_000
    assert (timing["model_ms"], timing["tool_ms"]) == (35_000, 20_000)
    # hunt: 40-45 and 50-60; verify: 25-30.
    assert timing["gap_ms"] == 20_000
    assert (timing["model_calls"], timing["tool_calls"]) == (3, 3)
    assert timing["per_model_call"] == {
        "calls": 3,
        "total_ms": 35_000,
        "average_ms": 11_667,
        "max_ms": 20_000,
    }
    assert timing["per_tool"] == [
        {
            "tool": "read_file",
            "calls": 2,
            "total_ms": 20_000,
            "average_ms": 10_000,
            "max_ms": 10_000,
        },
        {
            "tool": "search_text",
            "calls": 1,
            "total_ms": 6_000,
            "average_ms": 6_000,
            "max_ms": 6_000,
        },
    ]
    lag = shown["event_loop_lag"]
    assert (lag["sessions"], lag["sessions_lagged"], lag["reports"]) == (2, 1, 1)
    assert (lag["lagging_ms"], lag["max_lag_ms"]) == (4000, 900)

    table = _show(capsys, "scan", "--sqlite", str(database))
    assert "Time over 2 sessions (1m30s elapsed, added up)" in table
    assert "Before each session's first model call: 15.0s. After it, 20.0s went to" in table
    parts = ("setup", "model", "tool", "teardown", "waiting", "other", "idle")
    assert sum(timing[f"{part}_ms"] for part in parts) == timing["elapsed_ms"]
    assert all(f"\n{part} " in table for part in parts)
    assert "read_file    2      20.0s  10.0s    10.0s" in table
    assert "while 1 of this task's 2 sessions ran" in table
    assert "cayu session list --task scan" in table


def test_a_task_without_linked_sessions_reports_none(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Sessions created before Cayu recorded the link aren't found.
    shown = _show(capsys, "old", "--sqlite", str(database), "--json")

    assert shown["event_loop_lag"] is None
    assert shown["timing"]["session_ids"] == []
    assert shown["timing"]["sessions"] == 0
    assert "none run under this task" in _show(capsys, "old", "--sqlite", str(database))


def test_session_list_names_the_sessions_run_under_a_task(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["session", "list", "--task", "scan", "--sqlite", str(database), "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert sorted(row["id"] for row in listed["sessions"]) == ["hunt", "verify"]


def test_a_cloud_agent_reports_the_same_task_breakdown(
    database: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from tests.cli.test_session_cli_cloud import _FakeCloud

    from cayu.applications import CayuApp
    from cayu.server import ServerConfig, create_server
    from cayu.server.auth import BasicAuth

    local = _show(capsys, "scan", "--sqlite", str(database), "--json")
    monkeypatch.setattr("cayu.cli.cloud.cloud_client", lambda **_: _FakeCloud())
    sessions = SQLiteSessionStore(database)
    tasks = SQLiteTaskStore(database)
    server = create_server(
        CayuApp(session_store=sessions, task_store=tasks, enable_logging=False),
        config=ServerConfig.protected(BasicAuth(username="operator", password="s3cret-password")),
    )
    with TestClient(server, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(client, "close", lambda: None)
        monkeypatch.setattr("cayu.cli._cloud_agent_api.httpx.Client", lambda **_: client)
        remote = _show(capsys, "scan", "--application", "bug-hunter-agent", "--json")

    assert remote["timing"] == local["timing"]
    assert remote["event_loop_lag"] == local["event_loop_lag"]


def test_an_agent_that_ignores_the_task_filter_lists_no_sessions() -> None:
    from cayu.cli._task_readers import ApiTaskReader

    class OlderAgent:
        description = "the Agent"

        def get(self, path: str, *, query: Any = None, resource: str = "") -> Any:
            if path == "/sessions/unrelated":
                return {"id": "unrelated", "metadata": {}}
            assert path == "/sessions" and query["task_id"] == "scan"
            # Its Cayu doesn't know the filter, so every session comes back.
            return {"sessions": [{"id": "unrelated"}], "next_cursor": None}

    reader = ApiTaskReader(OlderAgent(), source={})  # type: ignore[arg-type]
    task = {"id": "scan", "created_at": T0.isoformat()}

    assert asyncio.run(reader.task_session_ids(task)) is None


def test_session_rollup_keeps_the_sessions_run_under_a_task(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["session", "rollup", "--task", "scan", "--sqlite", str(database), "--json"]
    assert main(argv) == 0
    (group,) = json.loads(capsys.readouterr().out)["groups"]
    assert (group["sessions"], group["model_calls"], group["tool_calls"]) == (2, 3, 3)


def test_a_task_handlers_own_runs_are_marked_with_its_task() -> None:
    from cayu import AgentSpec, CayuApp
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.providers import ModelStreamEvent
    from cayu.tasks.memory import InMemoryTaskStore
    from cayu.tasks.records import Task
    from cayu.tasks.worker import run_task_worker

    async def scenario() -> list[str]:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        reply = [
            ModelStreamEvent.text_delta("Done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
        app.register_provider(ScriptedModelProvider([reply, reply], name="fixture"), default=True)
        app.register_agent(AgentSpec(name="hunter", model="scripted-model"))
        await tasks.create_task(TaskCreate(task_id="scan", type="scan"))

        async def handler(app_: CayuApp, task: Task, worker_id: str) -> None:
            # The handler runs sessions of its own, not attached to the task.
            for name in ("rank", "hunt"):
                async for _ in app_.run(
                    RunRequest(
                        agent_name="hunter",
                        session_id=f"{name}-{task.id}",
                        messages=[Message.text("user", "go")],
                    )
                ):
                    pass
            await tasks.complete_task(
                task.id, {}, worker_id=worker_id, lease_expires_at=task.lease_expires_at
            )

        await run_task_worker(app, tasks, handler, worker_id="w", poll_interval_s=0.01, max_tasks=1)
        page = await sessions.list_sessions(SessionQuery(task_id="scan"))
        return sorted(session.id for session in page.sessions)

    assert asyncio.run(scenario()) == ["hunt-scan", "rank-scan"]
