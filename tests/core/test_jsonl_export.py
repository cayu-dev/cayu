from __future__ import annotations

import asyncio
import io
import json

from cayu.core import Event, EventType, Message
from cayu.runtime import (
    InMemorySessionStore,
    InMemoryTaskStore,
    RunRequest,
    SessionIdentity,
    TaskCreate,
)
from cayu.storage.jsonl_export import (
    ImportedSession,
    export_sessions,
    export_tasks,
    import_sessions,
    import_tasks,
)


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


def _lines(stream: io.StringIO) -> list[dict]:
    text = stream.getvalue()
    assert text.endswith("\n")
    return [json.loads(line) for line in text.splitlines()]


def test_export_sessions_writes_one_line_per_session_with_nested_state():
    async def run() -> None:
        store = InMemorySessionStore()
        # A rich session with events, transcript, and a checkpoint.
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_rich",
                messages=[Message.text("user", "build")],
            ),
            identity=_identity(),
        )
        await store.append_events(
            "sess_rich",
            [Event(type=EventType.SESSION_STARTED, session_id="sess_rich")],
        )
        await store.append_transcript_messages(
            "sess_rich",
            [Message.text("assistant", "building")],
        )
        await store.checkpoint("sess_rich", {"step": 3})
        # A bare session with no events/transcript/checkpoint.
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_bare",
                messages=[Message.text("user", "review")],
            ),
            identity=_identity(),
        )

        stream = io.StringIO()
        count = await export_sessions(store, stream=stream)

        assert count == 2
        lines = _lines(stream)
        assert len(lines) == 2
        assert all(line["type"] == "session" for line in lines)
        assert all(
            {"session", "events", "transcript", "checkpoint"} <= line.keys() for line in lines
        )

        by_id = {line["session"]["id"]: line for line in lines}
        assert set(by_id) == {"sess_rich", "sess_bare"}

        rich = by_id["sess_rich"]
        assert rich["session"]["agent_name"] == "builder"
        assert len(rich["events"]) == 1
        assert rich["events"][0]["type"] == EventType.SESSION_STARTED.value
        assert len(rich["transcript"]) == 1  # the one appended assistant message
        assert rich["transcript"][0]["role"] == "assistant"
        assert rich["checkpoint"] == {"step": 3}

        bare = by_id["sess_bare"]
        assert bare["events"] == []
        assert bare["transcript"] == []
        assert bare["checkpoint"] is None

    asyncio.run(run())


def test_export_sessions_empty_store_returns_zero():
    async def run() -> None:
        store = InMemorySessionStore()
        stream = io.StringIO()
        count = await export_sessions(store, stream=stream)
        assert count == 0
        assert stream.getvalue() == ""

    asyncio.run(run())


def test_export_sessions_pages_past_default_list_limit():
    async def run() -> None:
        store = InMemorySessionStore()
        for index in range(1001):
            await store.create(
                RunRequest(
                    agent_name="builder",
                    session_id=f"sess_{index:03}",
                    messages=[Message.text("user", "build")],
                ),
                identity=_identity(),
            )

        stream = io.StringIO()
        count = await export_sessions(store, stream=stream)

        assert count == 1001
        lines = _lines(stream)
        assert len(lines) == 1001
        assert {line["session"]["id"] for line in lines} == {
            f"sess_{index:03}" for index in range(1001)
        }

    asyncio.run(run())


def test_export_tasks_writes_one_line_per_task():
    async def run() -> None:
        store = InMemoryTaskStore()
        await store.create_task(
            TaskCreate(
                task_id="task_a",
                type="process",
                title="Process A",
                input={"value": 1},
            )
        )
        await store.create_task(
            TaskCreate(
                task_id="task_b",
                type="process",
                input={"value": 2},
            )
        )

        stream = io.StringIO()
        count = await export_tasks(store, stream=stream)

        assert count == 2
        lines = _lines(stream)
        assert len(lines) == 2
        assert all(line["type"] == "task" for line in lines)
        assert all("task" in line for line in lines)

        by_id = {line["task"]["id"]: line["task"] for line in lines}
        assert set(by_id) == {"task_a", "task_b"}
        assert by_id["task_a"]["input"] == {"value": 1}
        assert by_id["task_a"]["title"] == "Process A"
        assert by_id["task_b"]["status"] == "pending"

    asyncio.run(run())


def test_export_tasks_pages_past_default_list_limit():
    async def run() -> None:
        store = InMemoryTaskStore()
        for index in range(1001):
            await store.create_task(
                TaskCreate(
                    task_id=f"task_{index:03}",
                    type="process",
                    input={"index": index},
                )
            )

        stream = io.StringIO()
        count = await export_tasks(store, stream=stream)

        assert count == 1001
        lines = _lines(stream)
        assert len(lines) == 1001
        assert {line["task"]["id"] for line in lines} == {
            f"task_{index:03}" for index in range(1001)
        }

    asyncio.run(run())


def test_export_tasks_empty_store_returns_zero():
    async def run() -> None:
        store = InMemoryTaskStore()
        stream = io.StringIO()
        count = await export_tasks(store, stream=stream)
        assert count == 0
        assert stream.getvalue() == ""

    asyncio.run(run())


def test_export_sessions_keyset_paging_survives_concurrent_delete():
    # An offset walk skips a live session when one ahead of the cursor is
    # deleted mid-export (the window shifts down by one). Keyset paging anchors
    # each page to the last emitted (created_at, id), so it stays correct.
    async def run() -> None:
        store = InMemorySessionStore()
        for index in range(1002):
            await store.create(
                RunRequest(
                    agent_name="builder",
                    session_id=f"sess_{index:04}",
                    messages=[Message.text("user", "build")],
                ),
                identity=_identity(),
            )

        # A store wrapper that deletes a not-yet-reached session on the second
        # page, mid-export, to model a concurrent writer.
        class _DeletingStore:
            def __init__(self, inner):
                self._inner = inner
                self._calls = 0

            async def list_sessions(self, query):
                self._calls += 1
                if self._calls == 2:
                    del self._inner._sessions["sess_1001"]
                return await self._inner.list_sessions(query)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        stream = io.StringIO()
        count = await export_sessions(_DeletingStore(store), stream=stream)

        ids = {line["session"]["id"] for line in _lines(stream)}
        # Exactly the sessions that survived are emitted, none skipped, none dup.
        assert len(ids) == count
        assert "sess_1001" not in ids
        assert ids == {f"sess_{index:04}" for index in range(1001)}

    asyncio.run(run())


def test_import_sessions_round_trips_export():
    async def run() -> None:
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_rich",
                messages=[Message.text("user", "build")],
            ),
            identity=_identity(),
        )
        await store.append_events(
            "sess_rich",
            [Event(type=EventType.SESSION_STARTED, session_id="sess_rich")],
        )
        await store.append_transcript_messages(
            "sess_rich",
            [Message.text("assistant", "building")],
        )
        await store.checkpoint("sess_rich", {"step": 3})
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_bare",
                messages=[Message.text("user", "review")],
            ),
            identity=_identity(),
        )

        stream = io.StringIO()
        await export_sessions(store, stream=stream)

        imported = list(import_sessions(io.StringIO(stream.getvalue())))
        assert all(isinstance(record, ImportedSession) for record in imported)
        by_id = {record.session.id: record for record in imported}
        assert set(by_id) == {"sess_rich", "sess_bare"}

        rich = by_id["sess_rich"]
        assert rich.session == await store.load("sess_rich")
        assert rich.events == await store.load_events("sess_rich")
        assert rich.transcript == await store.load_transcript("sess_rich")
        assert rich.checkpoint == {"step": 3}

        bare = by_id["sess_bare"]
        assert bare.events == []
        assert bare.transcript == []
        assert bare.checkpoint is None

    asyncio.run(run())


def test_import_tasks_round_trips_export():
    async def run() -> None:
        store = InMemoryTaskStore()
        await store.create_task(
            TaskCreate(task_id="task_a", type="process", title="Process A", input={"value": 1})
        )
        await store.create_task(TaskCreate(task_id="task_b", type="process", input={"value": 2}))

        stream = io.StringIO()
        await export_tasks(store, stream=stream)

        imported = list(import_tasks(io.StringIO(stream.getvalue())))
        by_id = {task.id: task for task in imported}
        assert set(by_id) == {"task_a", "task_b"}
        assert by_id["task_a"] == await store.load_task("task_a")
        assert by_id["task_b"] == await store.load_task("task_b")

    asyncio.run(run())


def test_import_skips_blank_lines_and_rejects_wrong_type():
    imported = list(import_sessions(["", "  \n"]))
    assert imported == []

    task_lines = io.StringIO('{"type": "task", "task": {}}\n')
    try:
        list(import_sessions(task_lines))
    except ValueError as exc:
        assert "session record" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError on wrong record type")

    session_lines = io.StringIO('{"type": "session", "session": {}}\n')
    try:
        list(import_tasks(session_lines))
    except ValueError as exc:
        assert "task record" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError on wrong record type")
