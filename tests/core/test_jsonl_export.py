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
from cayu.storage.jsonl_export import export_sessions, export_tasks


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
