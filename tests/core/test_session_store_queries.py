from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from cayu import EventQuery, SessionOrder, SessionQuery, SQLiteSessionStore
from cayu.core import Event, EventType, Message
from cayu.runtime import InMemorySessionStore, RunRequest, SessionStatus, SessionStore

StoreFactory = Callable[[object], SessionStore]


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_list_sessions_with_filters_and_pagination(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_builder_1",
                environment_name="local",
                messages=[Message.text("user", "build")],
            )
        )
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_builder_2",
                environment_name="hosted",
                messages=[Message.text("user", "build again")],
            )
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_reviewer",
                environment_name="hosted",
                messages=[Message.text("user", "review")],
            )
        )
        await store.update_status("sess_builder_1", SessionStatus.RUNNING)
        await store.update_status("sess_builder_2", SessionStatus.COMPLETED)

        builder_sessions = await store.list_sessions(
            SessionQuery(
                agent_name="builder",
                order_by=SessionOrder.CREATED_AT_ASC,
            )
        )
        hosted_sessions = await store.list_sessions(
            SessionQuery(environment_name="hosted", order_by=SessionOrder.CREATED_AT_ASC)
        )
        completed_sessions = await store.list_sessions(SessionQuery(status=SessionStatus.COMPLETED))
        paged_sessions = await store.list_sessions(
            SessionQuery(limit=1, offset=1, order_by=SessionOrder.CREATED_AT_ASC)
        )

        assert [session.id for session in builder_sessions] == [
            "sess_builder_1",
            "sess_builder_2",
        ]
        assert [session.id for session in hosted_sessions] == [
            "sess_builder_2",
            "sess_reviewer",
        ]
        assert [session.id for session in completed_sessions] == ["sess_builder_2"]
        assert [session.id for session in paged_sessions] == ["sess_builder_2"]
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_query_events_with_filters_cursors_and_batching(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_builder",
                environment_name="local",
                messages=[Message.text("user", "build")],
            )
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_reviewer",
                environment_name="hosted",
                messages=[Message.text("user", "review")],
            )
        )

        await store.append_events(
            "sess_builder",
            [
                Event(
                    id="event_1",
                    type=EventType.SESSION_STARTED,
                    session_id="sess_builder",
                    agent_name="builder",
                    environment_name="local",
                ),
                Event(
                    id="event_2",
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id="sess_builder",
                    agent_name="builder",
                    environment_name="local",
                    tool_name="read_file",
                ),
                Event(
                    id="event_3",
                    type=EventType.MODEL_COMPLETED,
                    session_id="sess_builder",
                    agent_name="builder",
                    environment_name="local",
                    payload={"finish_reason": "stop"},
                ),
            ],
        )
        await store.append_event(
            "sess_reviewer",
            Event(
                id="event_4",
                type=EventType.SESSION_STARTED,
                session_id="sess_reviewer",
                agent_name="reviewer",
                environment_name="hosted",
            ),
        )

        all_records = await store.query_events(EventQuery(limit=10))
        builder_records = await store.query_events(EventQuery(session_id="sess_builder"))
        read_file_records = await store.query_events(EventQuery(tool_name="read_file"))
        started_records = await store.query_events(EventQuery(event_type=EventType.SESSION_STARTED))
        cursor_records = await store.query_events(
            EventQuery(after_sequence=all_records[1].sequence, limit=10)
        )

        assert [record.sequence for record in all_records] == [1, 2, 3, 4]
        assert [record.event.id for record in builder_records] == [
            "event_1",
            "event_2",
            "event_3",
        ]
        assert [record.event.id for record in read_file_records] == ["event_2"]
        assert [record.event.id for record in started_records] == [
            "event_1",
            "event_4",
        ]
        assert [record.event.id for record in cursor_records] == [
            "event_3",
            "event_4",
        ]

        with pytest.raises(ValueError, match="Event already exists"):
            await store.append_events(
                "sess_builder",
                [
                    Event(
                        id="event_new_rolled_back",
                        type=EventType.MODEL_STARTED,
                        session_id="sess_builder",
                    ),
                    Event(
                        id="event_2",
                        type=EventType.MODEL_COMPLETED,
                        session_id="sess_builder",
                    ),
                ],
            )

        records_after_failed_batch = await store.query_events(EventQuery(limit=10))
        assert [record.event.id for record in records_after_failed_batch] == [
            "event_1",
            "event_2",
            "event_3",
            "event_4",
        ]
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_append_and_load_transcript_messages(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_transcript",
                messages=[Message.text("user", "build")],
            )
        )

        user_message = Message.text("user", "build")
        assistant_message = Message.tool_call(
            tool_call_id="call_1",
            tool_name="read_file",
            arguments={"path": "README.md"},
        )
        tool_message = Message.tool_result(
            tool_call_id="call_1",
            tool_name="read_file",
            content="contents",
            structured={"bytes": 8},
        )

        await store.append_transcript_messages(
            "sess_transcript",
            [user_message, assistant_message],
        )
        user_message.content[0].text = "mutated"
        await store.append_transcript_messages("sess_transcript", [tool_message])

        transcript = await store.load_transcript("sess_transcript")
        assert [message.role for message in transcript] == [
            "user",
            "assistant",
            "tool",
        ]
        assert transcript[0].content[0].text == "build"
        assert transcript[1].content[0].tool_name == "read_file"
        assert transcript[2].content[0].structured == {"bytes": 8}

        transcript[0].content[0].text = "changed after load"
        loaded_again = await store.load_transcript("sess_transcript")
        assert loaded_again[0].content[0].text == "build"

        with pytest.raises(KeyError, match="Session not found"):
            await store.append_transcript_messages(
                "missing_session",
                [Message.text("user", "hi")],
            )
        with pytest.raises(KeyError, match="Session not found"):
            await store.load_transcript("missing_session")

        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_transition_status_atomically(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_transition",
                messages=[Message.text("user", "build")],
            )
        )
        await store.update_status("sess_transition", SessionStatus.COMPLETED)

        transitioned = await store.transition_status(
            "sess_transition",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
        )
        assert transitioned.status == SessionStatus.RUNNING

        with pytest.raises(ValueError, match="transition not allowed"):
            await store.transition_status(
                "sess_transition",
                from_statuses={SessionStatus.COMPLETED},
                to_status=SessionStatus.RUNNING,
            )

        loaded = await store.load("sess_transition")
        assert loaded is not None
        assert loaded.status == SessionStatus.RUNNING

        await _close_store(store)

    asyncio.run(run_store_operations())


def _make_store(store_factory: StoreFactory, tmp_path) -> SessionStore:
    if store_factory is SQLiteSessionStore:
        return SQLiteSessionStore(tmp_path / "sessions.sqlite")
    return store_factory()


async def _close_store(store: SessionStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()
