from __future__ import annotations

import asyncio
import sqlite3

import pytest

from cayu import SQLiteSessionStore
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
)
from cayu.runtime import CayuApp, ResumeRequest, RunRequest, SessionStatus


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(
        self,
        events: list[ModelStreamEvent] | list[list[ModelStreamEvent]],
    ) -> None:
        if events and isinstance(events[0], list):
            self.event_batches = events  # type: ignore[assignment]
        else:
            self.event_batches = [events]  # type: ignore[list-item]
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        batch_index = len(self.requests) - 1
        if batch_index >= len(self.event_batches):
            raise AssertionError(f"No fake provider event batch for request {batch_index}")
        for event in self.event_batches[batch_index]:
            yield event


async def _close(store: SQLiteSessionStore) -> None:
    await store.close()


async def _collect_app_events(events) -> list[Event]:
    return [event async for event in events]


def test_sqlite_session_store_persists_sessions_events_and_checkpoints(tmp_path):
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)

    async def run_store_operations() -> None:
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_sqlite",
                environment_name="local-dev",
                messages=[Message.text("user", "hi")],
                metadata={"project_id": 123},
            )
        )
        assert session.status == SessionStatus.PENDING

        await store.update_status("sess_sqlite", SessionStatus.RUNNING)
        await store.append_event(
            "sess_sqlite",
            Event(
                type=EventType.SESSION_STARTED,
                session_id="sess_sqlite",
                agent_name="assistant",
                environment_name="local-dev",
                payload={"step": 1},
            ),
        )
        await store.append_event(
            "sess_sqlite",
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="sess_sqlite",
                agent_name="assistant",
                environment_name="local-dev",
                payload={"finish_reason": "stop"},
            ),
        )
        await store.append_transcript_messages(
            "sess_sqlite",
            [
                Message.text("user", "hi"),
                Message.text("assistant", "hello"),
            ],
        )
        await store.checkpoint(
            "sess_sqlite",
            {"messages": [{"role": "user", "content": "hi"}], "step": 1},
        )
        await _close(store)

    asyncio.run(run_store_operations())

    reopened = SQLiteSessionStore(db_path)

    async def assert_reopened_state() -> None:
        session = await reopened.load("sess_sqlite")
        events = await reopened.load_events("sess_sqlite")
        transcript = await reopened.load_transcript("sess_sqlite")
        checkpoint = await reopened.load_checkpoint("sess_sqlite")

        assert session is not None
        assert session.agent_name == "assistant"
        assert session.environment_name == "local-dev"
        assert session.status == SessionStatus.RUNNING
        assert session.metadata == {"project_id": 123}
        assert [event.type for event in events] == [
            EventType.SESSION_STARTED,
            EventType.MODEL_COMPLETED,
        ]
        assert [event.payload for event in events] == [
            {"step": 1},
            {"finish_reason": "stop"},
        ]
        assert [message.role for message in transcript] == ["user", "assistant"]
        assert [message.content[0].text for message in transcript] == ["hi", "hello"]
        assert checkpoint == {
            "messages": [{"role": "user", "content": "hi"}],
            "step": 1,
        }
        await _close(reopened)

    asyncio.run(assert_reopened_state())


def test_sqlite_session_store_exposes_queryable_event_identity_columns(tmp_path):
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)

    async def run_store_operations() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_query_columns",
                messages=[Message.text("user", "hi")],
            )
        )
        await store.append_event(
            "sess_query_columns",
            Event(
                type=EventType.TOOL_CALL_COMPLETED,
                session_id="sess_query_columns",
                agent_name="assistant",
                environment_name="local-dev",
                tool_name="read_file",
                payload={"path": "README.md"},
            ),
        )
        await _close(store)

    asyncio.run(run_store_operations())

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """
            SELECT session_id, event_type, agent_name, environment_name,
                   tool_name, payload_json
            FROM events
            WHERE tool_name = ?
            """,
            ("read_file",),
        ).fetchone()
    finally:
        connection.close()

    assert dict(row) == {
        "session_id": "sess_query_columns",
        "event_type": EventType.TOOL_CALL_COMPLETED,
        "agent_name": "assistant",
        "environment_name": "local-dev",
        "tool_name": "read_file",
        "payload_json": '{"path":"README.md"}',
    }


def test_sqlite_session_store_rejects_duplicate_sessions_and_mismatched_events(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite")

    async def run_store_operations() -> None:
        request = RunRequest(
            agent_name="assistant",
            session_id="sess_duplicate",
            messages=[Message.text("user", "hi")],
        )
        await store.create(request)

        with pytest.raises(ValueError, match="Session already exists"):
            await store.create(request)

        with pytest.raises(ValueError, match="Event session_id"):
            await store.append_event(
                "sess_duplicate",
                Event(
                    type=EventType.SESSION_STARTED,
                    session_id="other_session",
                ),
            )

        event = Event(
            id="event_duplicate",
            type=EventType.SESSION_STARTED,
            session_id="sess_duplicate",
        )
        await store.append_event("sess_duplicate", event)
        with pytest.raises(ValueError, match="Event already exists"):
            await store.append_event("sess_duplicate", event)

        with pytest.raises(KeyError, match="Session not found"):
            await store.load_events("missing_session")

        await _close(store)

    asyncio.run(run_store_operations())


def test_cayu_app_can_use_sqlite_session_store(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    app = CayuApp(session_store=store)
    app.register_provider(
        FakeProvider(
            [
                ModelStreamEvent.text_delta("hello"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run_app() -> None:
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_runtime_sqlite",
                    messages=[Message.text("user", "hi")],
                )
            )
        ]
        persisted_events = await store.load_events("sess_runtime_sqlite")
        session = await store.load("sess_runtime_sqlite")

        assert [event.type for event in events] == [
            EventType.SESSION_STARTED,
            EventType.MODEL_STARTED,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
            EventType.SESSION_COMPLETED,
        ]
        assert persisted_events == events
        assert session is not None
        assert session.status == SessionStatus.COMPLETED
        await _close(store)

    asyncio.run(run_app())


def test_cayu_app_can_resume_with_sqlite_session_store(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run_app() -> None:
        await _collect_app_events(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_resume_sqlite",
                    messages=[Message.text("user", "first request")],
                )
            )
        )
        resume_events = await _collect_app_events(
            app.resume(
                ResumeRequest(
                    session_id="sess_resume_sqlite",
                    messages=[Message.text("user", "second request")],
                )
            )
        )
        transcript = await store.load_transcript("sess_resume_sqlite")
        persisted_events = await store.load_events("sess_resume_sqlite")
        session = await store.load("sess_resume_sqlite")

        assert [event.type for event in resume_events] == [
            EventType.SESSION_RESUMED,
            EventType.MODEL_STARTED,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
            EventType.SESSION_COMPLETED,
        ]
        assert [message.content[0].text for message in provider.requests[1].messages] == [
            "first request",
            "first answer",
            "second request",
        ]
        assert [message.content[0].text for message in transcript] == [
            "first request",
            "first answer",
            "second request",
            "second answer",
        ]
        assert [event.type for event in persisted_events] == [
            EventType.SESSION_STARTED,
            EventType.MODEL_STARTED,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
            EventType.SESSION_COMPLETED,
            EventType.SESSION_RESUMED,
            EventType.MODEL_STARTED,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
            EventType.SESSION_COMPLETED,
        ]
        assert session is not None
        assert session.status == SessionStatus.COMPLETED
        await _close(store)

    asyncio.run(run_app())


def test_sqlite_session_store_rejects_newer_schema_version(tmp_path):
    db_path = tmp_path / "newer.sqlite"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA user_version = 999")
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="newer Cayu schema"):
        SQLiteSessionStore(db_path)
