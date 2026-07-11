from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from cayu import SQLiteSessionStore
from cayu.core import Event, EventType, Message
from cayu.runtime import (
    EventQuery,
    InMemorySessionStore,
    RunRequest,
    Session,
    SessionIdentity,
    SessionOrder,
    SessionQuery,
    SessionStatus,
    SessionStore,
)

_POSTGRES_TABLES = (
    "cayu_knowledge_labels",
    "cayu_knowledge_aspects",
    "cayu_knowledge_impact_targets",
    "cayu_knowledge_chunks",
    "cayu_knowledge_entries",
    "cayu_event_watcher_state",
    "cayu_events",
    "cayu_session_labels",
    "cayu_transcript_messages",
    "cayu_checkpoints",
    "cayu_tasks",
    "cayu_sessions",
    "cayu_schema_migrations",
)


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


async def _truncate_postgres(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            for table in _POSTGRES_TABLES:
                await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await conn.commit()


def _new_postgres_store(dsn: str) -> SessionStore:
    from cayu import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    return PostgresSessionStore(dsn, min_size=1, max_size=4, schema_mode=SchemaMode.CREATE)


async def _close_store(store: SessionStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def session_store_case(request, tmp_path):
    if request.param == "memory":
        return request.param, tmp_path, None
    if request.param == "sqlite":
        return request.param, tmp_path, None
    return request.param, tmp_path, request.getfixturevalue("postgres_dsn")


async def _open_store(case) -> SessionStore:
    store_kind, tmp_path, postgres_dsn = case
    if store_kind == "memory":
        return InMemorySessionStore()
    if store_kind == "sqlite":
        return SQLiteSessionStore(tmp_path / "sessions.sqlite")
    await _truncate_postgres(postgres_dsn)
    return _new_postgres_store(postgres_dsn)


def test_session_store_conformance_atomically_transforms_checkpoint(session_store_case) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_atomic_checkpoint_transform",
                    messages=[Message.text("user", "hello")],
                ),
                identity=_identity(),
            )
            await store.checkpoint("sess_atomic_checkpoint_transform", {"original": True})

            def add_key(key: str):
                def transform(_session: Session, checkpoint: dict[str, Any] | None):
                    updated = {} if checkpoint is None else dict(checkpoint)
                    updated[key] = True
                    return updated

                return transform

            await asyncio.gather(
                store.transform_checkpoint(
                    "sess_atomic_checkpoint_transform",
                    add_key("first"),
                ),
                store.transform_checkpoint(
                    "sess_atomic_checkpoint_transform",
                    add_key("second"),
                ),
            )
            await asyncio.gather(
                store.transform_checkpoint(
                    "sess_atomic_checkpoint_transform",
                    add_key("third"),
                ),
                store.append_transcript_messages_and_transform_checkpoint(
                    "sess_atomic_checkpoint_transform",
                    [Message.text("assistant", "done")],
                    add_key("fourth"),
                ),
            )

            assert await store.load_checkpoint("sess_atomic_checkpoint_transform") == {
                "original": True,
                "first": True,
                "second": True,
                "third": True,
                "fourth": True,
            }
            assert [
                message.content[0].text
                for message in await store.load_transcript("sess_atomic_checkpoint_transform")
            ] == ["done"]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_lists_pending_interruption_cascades(session_store_case) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            for session_id in (
                "sess_cascade_index_a",
                "sess_cascade_index_b",
                "sess_cascade_index_none",
                "sess_cascade_index_running",
            ):
                await store.create(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", session_id)],
                    ),
                    identity=_identity(),
                )
            for session_id in (
                "sess_cascade_index_a",
                "sess_cascade_index_b",
                "sess_cascade_index_none",
            ):
                await store.update_status(session_id, SessionStatus.INTERRUPTED)
            await store.update_status(
                "sess_cascade_index_running",
                SessionStatus.RUNNING,
            )
            for session_id in (
                "sess_cascade_index_a",
                "sess_cascade_index_b",
                "sess_cascade_index_running",
            ):
                await store.checkpoint(
                    session_id,
                    {
                        "pending_interruption_cascade": {
                            "attempt_id": session_id,
                            "interrupt_payload": {"interruption_type": "operator_requested"},
                        }
                    },
                )
            await store.checkpoint(
                "sess_cascade_index_none",
                {"unrelated_checkpoint": True},
            )

            first = await store.list_sessions_with_pending_interruption_cascade(
                SessionQuery(
                    status=SessionStatus.INTERRUPTED,
                    order_by=SessionOrder.CREATED_AT_ASC,
                    limit=1,
                    include_total_count=True,
                )
            )
            second = await store.list_sessions_with_pending_interruption_cascade(
                SessionQuery(
                    status=SessionStatus.INTERRUPTED,
                    order_by=SessionOrder.CREATED_AT_ASC,
                    limit=1,
                    cursor=first.next_cursor,
                )
            )
            running = await store.list_sessions_with_pending_interruption_cascade(
                SessionQuery(status=SessionStatus.RUNNING)
            )

            assert first.total_count == 2
            assert first.next_cursor is not None
            assert [session.id for session in first.sessions + second.sessions] == [
                "sess_cascade_index_a",
                "sess_cascade_index_b",
            ]
            assert second.next_cursor is None
            assert [session.id for session in running.sessions] == ["sess_cascade_index_running"]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_applies_query_filters(session_store_case) -> None:
    async def run() -> None:
        session_store = await _open_store(session_store_case)
        try:
            await session_store.create(
                RunRequest(
                    agent_name="alpha",
                    session_id="sess_query_alpha",
                    causal_budget_id="budget_runtime",
                    environment_name="local",
                    labels={"team": "runtime"},
                    messages=[Message.text("user", "alpha")],
                ),
                identity=_identity(),
            )
            await session_store.create(
                RunRequest(
                    agent_name="beta",
                    session_id="sess_query_beta",
                    causal_budget_id="budget_runtime",
                    environment_name="remote",
                    labels={"team": "review"},
                    messages=[Message.text("user", "beta")],
                ),
                identity=_identity(),
            )
            await session_store.append_events(
                "sess_query_alpha",
                [
                    Event(
                        id="evt_query_alpha",
                        type=EventType.TOOL_CALL_COMPLETED,
                        session_id="sess_query_alpha",
                        agent_name="alpha",
                        environment_name="local",
                        tool_name="read_file",
                        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                    )
                ],
            )
            await session_store.append_events(
                "sess_query_beta",
                [
                    Event(
                        id="evt_query_beta",
                        type=EventType.TOOL_CALL_FAILED,
                        session_id="sess_query_beta",
                        agent_name="beta",
                        environment_name="remote",
                        tool_name="edit_file",
                        timestamp=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
                    )
                ],
            )

            sessions = await session_store.list_sessions(
                SessionQuery(q="ALPHA", labels={"team": "runtime"}, include_total_count=True)
            )
            assert [session.id for session in sessions.sessions] == ["sess_query_alpha"]
            assert sessions.total_count == 1

            records = await session_store.query_events(
                EventQuery(
                    causal_budget_id="budget_runtime",
                    event_types=(EventType.TOOL_CALL_COMPLETED,),
                    agent_name="alpha",
                    tool_name="read_file",
                )
            )
            assert [record.event.id for record in records] == ["evt_query_alpha"]
        finally:
            await _close_store(session_store)

    asyncio.run(run())


def test_session_store_conformance_validates_event_batch_preamble(
    session_store_case,
) -> None:
    async def run() -> None:
        session_store = await _open_store(session_store_case)
        try:
            await session_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_event_preamble",
                    messages=[Message.text("user", "events")],
                ),
                identity=_identity(),
            )
            append_events: Any = session_store.append_events

            with pytest.raises(TypeError, match="Session events must be a list."):
                await append_events("sess_event_preamble", ())
            with pytest.raises(TypeError, match="Session events must be Event instances."):
                await append_events("sess_event_preamble", ["not-an-event"])
            with pytest.raises(ValueError, match="Event session_id does not match target session."):
                await session_store.append_events(
                    "sess_event_preamble",
                    [
                        Event(
                            id="evt_wrong_session",
                            type=EventType.SESSION_STARTED,
                            session_id="sess_other",
                        )
                    ],
                )
            with pytest.raises(ValueError, match="Event already exists for session"):
                await session_store.append_events(
                    "sess_event_preamble",
                    [
                        Event(
                            id="evt_duplicate",
                            type=EventType.SESSION_STARTED,
                            session_id="sess_event_preamble",
                        ),
                        Event(
                            id="evt_duplicate",
                            type=EventType.SESSION_STARTED,
                            session_id="sess_event_preamble",
                        ),
                    ],
                )
        finally:
            await _close_store(session_store)

    asyncio.run(run())


def test_session_store_conformance_validates_fork_request_preamble(
    session_store_case,
) -> None:
    async def run() -> None:
        session_store = await _open_store(session_store_case)
        try:
            source = await session_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_fork_source",
                    messages=[Message.text("user", "fork")],
                ),
                identity=_identity(),
            )

            with pytest.raises(ValueError, match="Fork parent_session_id must match"):
                await session_store.create_fork(
                    source_session_id=source.id,
                    fork=Session(
                        id="sess_bad_parent",
                        agent_name="assistant",
                        provider_name="fake",
                        model="fake-model",
                        parent_session_id="sess_other",
                        causal_budget_id=source.causal_budget_id,
                        status=SessionStatus.PENDING,
                    ),
                    source_statuses={SessionStatus.PENDING},
                    transcript_cursor=None,
                    checkpoint_transform=None,
                )
            with pytest.raises(ValueError, match="transcript_cursor must be greater"):
                await session_store.create_fork(
                    source_session_id=source.id,
                    fork=Session(
                        id="sess_bad_cursor",
                        agent_name="assistant",
                        provider_name="fake",
                        model="fake-model",
                        parent_session_id=source.id,
                        causal_budget_id=source.causal_budget_id,
                        status=SessionStatus.PENDING,
                    ),
                    source_statuses={SessionStatus.PENDING},
                    transcript_cursor=-1,
                    checkpoint_transform=None,
                )
            with pytest.raises(ValueError, match="Source session status is not forkable"):
                await session_store.create_fork(
                    source_session_id=source.id,
                    fork=Session(
                        id="sess_bad_status_source",
                        agent_name="assistant",
                        provider_name="fake",
                        model="fake-model",
                        parent_session_id=source.id,
                        causal_budget_id=source.causal_budget_id,
                        status=SessionStatus.PENDING,
                    ),
                    source_statuses={SessionStatus.COMPLETED},
                    transcript_cursor=None,
                    checkpoint_transform=None,
                )
            with pytest.raises(ValueError, match="Fork status must match source session status"):
                await session_store.create_fork(
                    source_session_id=source.id,
                    fork=Session(
                        id="sess_bad_status_fork",
                        agent_name="assistant",
                        provider_name="fake",
                        model="fake-model",
                        parent_session_id=source.id,
                        causal_budget_id=source.causal_budget_id,
                        status=SessionStatus.COMPLETED,
                    ),
                    source_statuses={SessionStatus.PENDING},
                    transcript_cursor=None,
                    checkpoint_transform=None,
                )
            with pytest.raises(ValueError, match="Fork provider_name must match"):
                await session_store.create_fork(
                    source_session_id=source.id,
                    fork=Session(
                        id="sess_bad_provider",
                        agent_name="assistant",
                        provider_name="other",
                        model="fake-model",
                        parent_session_id=source.id,
                        causal_budget_id=source.causal_budget_id,
                        status=SessionStatus.PENDING,
                    ),
                    source_statuses={SessionStatus.PENDING},
                    transcript_cursor=None,
                    checkpoint_transform=None,
                )
        finally:
            await _close_store(session_store)

    asyncio.run(run())
