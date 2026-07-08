"""Postgres SessionStore parity + concurrency tests.

These mirror the SQLite/InMemory conformance assertions in
``test_sqlite_session_store.py`` and ``test_session_store_queries.py`` so the
identical behavioral contract is proven against a real Dockerized Postgres.
They skip automatically when Docker is unavailable (see ``conftest.py``).
"""

from __future__ import annotations

import asyncio
import base64
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cayu.core import Event, EventType, Message
from cayu.runtime import (
    EventOrder,
    EventQuery,
    RunRequest,
    Session,
    SessionIdentity,
    SessionOrder,
    SessionQuery,
    SessionStatus,
    TranscriptQuery,
)

pytestmark = pytest.mark.usefixtures("postgres_dsn")

_TABLES = (
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


async def _truncate(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            for table in _TABLES:
                await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await conn.commit()


def _new_store(dsn: str):
    from cayu import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    # Tests own a throwaway database and (re)create the schema each run.
    return PostgresSessionStore(dsn, min_size=1, max_size=4, schema_mode=SchemaMode.CREATE)


def _run(dsn: str, coro_factory) -> object:
    async def runner():
        await _truncate(dsn)
        store = _new_store(dsn)
        try:
            return await coro_factory(store)
        finally:
            await store.close()

    return asyncio.run(runner())


def test_postgres_session_store_persists_sessions_events_and_checkpoints(postgres_dsn):
    async def ops(store):
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_pg",
                environment_name="local-dev",
                messages=[Message.text("user", "hi")],
                metadata={"project_id": 123},
            ),
            identity=SessionIdentity(
                provider_name="anthropic",
                model="claude-test",
                runtime_name="cayu",
                runtime_version="test-version",
            ),
        )
        assert session.status == SessionStatus.PENDING
        assert session.provider_name == "anthropic"
        assert session.model == "claude-test"
        assert session.runtime_version == "test-version"

        await store.update_status("sess_pg", SessionStatus.RUNNING)
        await store.append_event(
            "sess_pg",
            Event(
                type=EventType.SESSION_STARTED,
                session_id="sess_pg",
                agent_name="assistant",
                environment_name="local-dev",
                payload={"step": 1},
            ),
        )
        await store.append_event(
            "sess_pg",
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="sess_pg",
                agent_name="assistant",
                environment_name="local-dev",
                payload={"finish_reason": "stop"},
            ),
        )
        await store.append_transcript_messages(
            "sess_pg",
            [Message.text("user", "hi"), Message.text("assistant", "hello")],
        )
        await store.checkpoint(
            "sess_pg",
            {"messages": [{"role": "user", "content": "hi"}], "step": 1},
        )

        loaded = await store.load("sess_pg")
        events = await store.load_events("sess_pg")
        transcript = await store.load_transcript("sess_pg")
        checkpoint = await store.load_checkpoint("sess_pg")

        assert loaded is not None
        assert loaded.agent_name == "assistant"
        assert loaded.environment_name == "local-dev"
        assert loaded.status == SessionStatus.RUNNING
        assert loaded.metadata == {"project_id": 123}
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
        assert checkpoint == {"messages": [{"role": "user", "content": "hi"}], "step": 1}

    _run(postgres_dsn, ops)


def test_postgres_session_store_atomically_appends_transcript_and_checkpoint(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_atomic",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.checkpoint("sess_atomic", {"pending_tool_approval": {"approval_id": "a1"}})
        await store.append_transcript_messages_and_checkpoint(
            "sess_atomic",
            [Message.text("assistant", "done")],
            {"closed": True},
        )
        transcript = await store.load_transcript("sess_atomic")
        checkpoint = await store.load_checkpoint("sess_atomic")
        assert [message.role for message in transcript] == ["assistant"]
        assert transcript[0].content[0].text == "done"
        assert checkpoint == {"closed": True}

    _run(postgres_dsn, ops)


def test_postgres_session_store_atomically_transitions_status_and_checkpoint(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_status_ck",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        session = await store.transition_status_and_checkpoint(
            "sess_status_ck",
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.INTERRUPTING,
            checkpoint_transform=lambda _s, ck: {
                **({} if ck is None else ck),
                "pending_session_interrupt": {"reason": "operator stop"},
            },
        )
        checkpoint = await store.load_checkpoint("sess_status_ck")
        assert session.status == SessionStatus.INTERRUPTING
        assert checkpoint == {"pending_session_interrupt": {"reason": "operator stop"}}

    _run(postgres_dsn, ops)


def test_postgres_session_store_rejects_stale_atomic_status_checkpoint_transition(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_stale",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        # Move it out of PENDING so the guarded transition can no longer match.
        await store.update_status("sess_stale", SessionStatus.RUNNING)

        with pytest.raises(ValueError, match="Session status transition not allowed"):
            await store.transition_status_and_checkpoint(
                "sess_stale",
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.INTERRUPTING,
                checkpoint_transform=lambda _s, ck: {
                    **({} if ck is None else ck),
                    "pending_session_interrupt": {"reason": "operator stop"},
                },
            )

        session = await store.load("sess_stale")
        checkpoint = await store.load_checkpoint("sess_stale")
        assert session is not None
        assert session.status == SessionStatus.RUNNING
        # Failed transition must NOT have written a checkpoint (atomic rollback).
        assert checkpoint is None

    _run(postgres_dsn, ops)


def test_postgres_session_store_transforms_current_checkpoint_during_fork(postgres_dsn):
    async def ops(store):
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_fork_ck_source",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        await store.checkpoint(source.id, {"version": 2})
        await store.append_transcript_messages(
            source.id,
            [Message.text("user", "first request"), Message.text("assistant", "first answer")],
        )

        fork = await store.create_fork(
            source_session_id=source.id,
            fork=Session(
                id="sess_fork_ck_child",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
                parent_session_id=source.id,
                status=SessionStatus.COMPLETED,
            ),
            source_statuses={SessionStatus.COMPLETED},
            transcript_cursor=None,
            checkpoint_transform=lambda _s, ck: {"copied_version": ck["version"] if ck else None},
        )

        assert fork.parent_session_id == source.id
        assert fork.status == SessionStatus.COMPLETED
        assert await store.load_checkpoint("sess_fork_ck_child") == {"copied_version": 2}
        transcript = await store.load_transcript("sess_fork_ck_child")
        assert [m.content[0].text for m in transcript] == ["first request", "first answer"]
        children = (await store.list_sessions(SessionQuery(parent_session_id=source.id))).sessions
        assert [s.id for s in children] == ["sess_fork_ck_child"]

    _run(postgres_dsn, ops)


def test_postgres_session_store_persists_run_request_parent_session_id(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_pg_run_parent",
                messages=[Message.text("user", "parent")],
            ),
            identity=_identity(),
        )
        child = await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_pg_run_child",
                parent_session_id="sess_pg_run_parent",
                causal_budget_id="job_pg_run_parent",
                messages=[Message.text("user", "child")],
            ),
            identity=_identity(),
        )

        assert child.parent_session_id == "sess_pg_run_parent"
        loaded = await store.load("sess_pg_run_child")
        assert loaded is not None
        assert loaded.parent_session_id == "sess_pg_run_parent"
        assert loaded.causal_budget_id == "job_pg_run_parent"
        children = (
            await store.list_sessions(SessionQuery(parent_session_id="sess_pg_run_parent"))
        ).sessions
        assert [session.id for session in children] == ["sess_pg_run_child"]

    _run(postgres_dsn, ops)


def test_postgres_session_store_rejects_missing_parent_session_id(postgres_dsn):
    async def ops(store):
        with pytest.raises(ValueError, match="Parent session not found"):
            await store.create(
                RunRequest(
                    agent_name="reviewer",
                    session_id="sess_pg_missing_parent_child",
                    parent_session_id="sess_pg_missing_parent",
                    messages=[Message.text("user", "child")],
                ),
                identity=_identity(),
            )
        assert await store.load("sess_pg_missing_parent_child") is None

    _run(postgres_dsn, ops)


def test_postgres_session_store_rejects_self_parent_session_id(postgres_dsn):
    async def ops(store):
        with pytest.raises(ValueError, match="own parent"):
            await store.create(
                RunRequest(
                    agent_name="reviewer",
                    session_id="sess_pg_self_parent",
                    parent_session_id="sess_pg_self_parent",
                    messages=[Message.text("user", "child")],
                ),
                identity=_identity(),
            )
        assert await store.load("sess_pg_self_parent") is None

    _run(postgres_dsn, ops)


def test_postgres_session_store_fork_honors_transcript_cursor(postgres_dsn):
    async def ops(store):
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_fork_cursor_source",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        await store.append_transcript_messages(
            source.id,
            [
                Message.text("user", "m1"),
                Message.text("assistant", "m2"),
                Message.text("user", "m3"),
            ],
        )
        await store.create_fork(
            source_session_id=source.id,
            fork=Session(
                id="sess_fork_cursor_child",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
                parent_session_id=source.id,
                status=SessionStatus.COMPLETED,
            ),
            source_statuses={SessionStatus.COMPLETED},
            transcript_cursor=2,
            checkpoint_transform=None,
        )
        transcript = await store.load_transcript("sess_fork_cursor_child")
        assert [m.content[0].text for m in transcript] == ["m1", "m2"]

        with pytest.raises(ValueError, match="transcript_cursor is greater"):
            await store.create_fork(
                source_session_id=source.id,
                fork=Session(
                    id="sess_fork_cursor_overflow",
                    agent_name="assistant",
                    provider_name="fake",
                    model="fake-model",
                    parent_session_id=source.id,
                    status=SessionStatus.COMPLETED,
                ),
                source_statuses={SessionStatus.COMPLETED},
                transcript_cursor=99,
                checkpoint_transform=None,
            )

    _run(postgres_dsn, ops)


def test_postgres_session_store_rejects_fork_status_and_provider_mismatch(postgres_dsn):
    async def ops(store):
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_fork_mismatch_source",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)

        with pytest.raises(ValueError, match="Fork status must match"):
            await store.create_fork(
                source_session_id=source.id,
                fork=Session(
                    id="sess_fork_status_child",
                    agent_name="assistant",
                    provider_name="fake",
                    model="fake-model",
                    parent_session_id=source.id,
                    status=SessionStatus.RUNNING,
                ),
                source_statuses={SessionStatus.COMPLETED},
                transcript_cursor=None,
                checkpoint_transform=None,
            )

        with pytest.raises(ValueError, match="Fork provider_name must match"):
            await store.create_fork(
                source_session_id=source.id,
                fork=Session(
                    id="sess_fork_provider_child",
                    agent_name="assistant",
                    provider_name="other",
                    model="fake-model",
                    parent_session_id=source.id,
                    status=SessionStatus.COMPLETED,
                ),
                source_statuses={SessionStatus.COMPLETED},
                transcript_cursor=None,
                checkpoint_transform=None,
            )

    _run(postgres_dsn, ops)


def test_postgres_session_store_rejects_duplicate_sessions_and_mismatched_events(postgres_dsn):
    async def ops(store):
        request = RunRequest(
            agent_name="assistant",
            session_id="sess_duplicate",
            messages=[Message.text("user", "hi")],
        )
        await store.create(request, identity=_identity())

        with pytest.raises(ValueError, match="Session already exists"):
            await store.create(request, identity=_identity())

        with pytest.raises(ValueError, match="Event session_id"):
            await store.append_event(
                "sess_duplicate",
                Event(type=EventType.SESSION_STARTED, session_id="other_session"),
            )

        event = Event(id="event_dup", type=EventType.SESSION_STARTED, session_id="sess_duplicate")
        await store.append_event("sess_duplicate", event)
        with pytest.raises(ValueError, match="Event already exists"):
            await store.append_event("sess_duplicate", event)

        with pytest.raises(KeyError, match="Session not found"):
            await store.load_events("missing_session")

    _run(postgres_dsn, ops)


def test_postgres_session_store_lists_sessions_with_filters_and_pagination(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_builder_1",
                environment_name="local",
                messages=[Message.text("user", "build")],
            ),
            identity=_identity(),
        )
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_builder_2",
                environment_name="hosted",
                messages=[Message.text("user", "build again")],
            ),
            identity=_identity(),
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_reviewer",
                environment_name="hosted",
                messages=[Message.text("user", "review")],
            ),
            identity=_identity(),
        )
        await store.create(
            RunRequest(
                agent_name="operator",
                session_id="sess_openai_operator",
                environment_name="sandbox",
                labels={"marker": "literal%token", "workflow": "pr-review"},
                messages=[Message.text("user", "review PR")],
            ),
            identity=SessionIdentity(provider_name="openai", model="gpt-5.5"),
        )
        await store.update_status("sess_builder_1", SessionStatus.RUNNING)
        await store.update_status("sess_builder_2", SessionStatus.COMPLETED)

        builder_sessions = (
            await store.list_sessions(
                SessionQuery(agent_name="builder", order_by=SessionOrder.CREATED_AT_ASC)
            )
        ).sessions
        hosted_sessions = (
            await store.list_sessions(
                SessionQuery(environment_name="hosted", order_by=SessionOrder.CREATED_AT_ASC)
            )
        ).sessions
        completed_sessions = (
            await store.list_sessions(SessionQuery(status=SessionStatus.COMPLETED))
        ).sessions
        paged_sessions = (
            await store.list_sessions(
                SessionQuery(limit=1, offset=1, order_by=SessionOrder.CREATED_AT_ASC)
            )
        ).sessions
        openai_sessions = (
            await store.list_sessions(
                SessionQuery(provider_name="openai", order_by=SessionOrder.CREATED_AT_ASC)
            )
        ).sessions
        model_sessions = (
            await store.list_sessions(
                SessionQuery(model="gpt-5.5", order_by=SessionOrder.CREATED_AT_ASC)
            )
        ).sessions
        query_by_agent_sessions = (
            await store.list_sessions(SessionQuery(q="OPER", order_by=SessionOrder.CREATED_AT_ASC))
        ).sessions
        query_by_model_sessions = (
            await store.list_sessions(SessionQuery(q="gpt-5", order_by=SessionOrder.CREATED_AT_ASC))
        ).sessions
        query_by_label_sessions = (
            await store.list_sessions(
                SessionQuery(q="pr-review", order_by=SessionOrder.CREATED_AT_ASC)
            )
        ).sessions
        query_by_literal_percent_sessions = (
            await store.list_sessions(SessionQuery(q="%", order_by=SessionOrder.CREATED_AT_ASC))
        ).sessions

        assert [s.id for s in builder_sessions] == ["sess_builder_1", "sess_builder_2"]
        assert [s.id for s in hosted_sessions] == ["sess_builder_2", "sess_reviewer"]
        assert [s.id for s in completed_sessions] == ["sess_builder_2"]
        assert [s.id for s in paged_sessions] == ["sess_builder_2"]
        assert [s.id for s in openai_sessions] == ["sess_openai_operator"]
        assert [s.id for s in model_sessions] == ["sess_openai_operator"]
        assert [s.id for s in query_by_agent_sessions] == ["sess_openai_operator"]
        assert [s.id for s in query_by_model_sessions] == ["sess_openai_operator"]
        assert [s.id for s in query_by_label_sessions] == ["sess_openai_operator"]
        assert [s.id for s in query_by_literal_percent_sessions] == ["sess_openai_operator"]

    _run(postgres_dsn, ops)


def test_postgres_session_store_preserves_and_filters_session_labels(postgres_dsn):
    async def ops(store):
        created = await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_pg_labels_invoice",
                labels={
                    "owner": "org_123",
                    "project": "ap_q2",
                    "workflow": "invoice-ingestion",
                },
                messages=[Message.text("user", "ingest invoice")],
            ),
            identity=_identity(),
        )
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_pg_labels_research",
                labels={"owner": "org_123", "project": "research"},
                messages=[Message.text("user", "research")],
            ),
            identity=_identity(),
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_pg_labels_other_owner",
                labels={"owner": "org_999", "project": "ap_q2"},
                messages=[Message.text("user", "review")],
            ),
            identity=_identity(),
        )

        loaded = await store.load(created.id)
        owner_sessions = (
            await store.list_sessions(
                SessionQuery(labels={"owner": "org_123"}, order_by=SessionOrder.CREATED_AT_ASC)
            )
        ).sessions
        exact_sessions = (
            await store.list_sessions(
                SessionQuery(
                    labels={"owner": "org_123", "project": "ap_q2"},
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
            )
        ).sessions
        missing_sessions = (
            await store.list_sessions(SessionQuery(labels={"owner": "missing"}))
        ).sessions
        exists_sessions = (
            await store.list_sessions(
                SessionQuery(
                    label_selectors=[{"key": "workflow", "operator": "exists"}],
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
            )
        ).sessions
        in_sessions = (
            await store.list_sessions(
                SessionQuery(
                    label_selectors=[
                        {"key": "project", "operator": "in", "values": ["ap_q2", "research"]}
                    ],
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
            )
        ).sessions
        not_in_sessions = (
            await store.list_sessions(
                SessionQuery(
                    labels={"owner": "org_123"},
                    label_selectors=[
                        {"key": "project", "operator": "not_in", "values": ["research"]}
                    ],
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
            )
        ).sessions
        not_exists_sessions = (
            await store.list_sessions(
                SessionQuery(
                    label_selectors=[{"key": "owner", "operator": "not_exists"}],
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
            )
        ).sessions

        assert loaded is not None
        assert loaded.labels == {
            "owner": "org_123",
            "project": "ap_q2",
            "workflow": "invoice-ingestion",
        }
        assert [session.id for session in owner_sessions] == [
            "sess_pg_labels_invoice",
            "sess_pg_labels_research",
        ]
        assert [session.id for session in exact_sessions] == ["sess_pg_labels_invoice"]
        assert missing_sessions == []
        assert [session.id for session in exists_sessions] == ["sess_pg_labels_invoice"]
        assert [session.id for session in in_sessions] == [
            "sess_pg_labels_invoice",
            "sess_pg_labels_research",
            "sess_pg_labels_other_owner",
        ]
        assert [session.id for session in not_in_sessions] == ["sess_pg_labels_invoice"]
        assert [session.id for session in not_exists_sessions] == []

    _run(postgres_dsn, ops)


def test_postgres_session_store_query_events_with_filters_cursors_and_batching(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_builder",
                environment_name="local",
                messages=[Message.text("user", "build")],
            ),
            identity=_identity(),
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_reviewer",
                environment_name="hosted",
                messages=[Message.text("user", "review")],
            ),
            identity=_identity(),
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
                    timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                ),
                Event(
                    id="event_2",
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id="sess_builder",
                    agent_name="builder",
                    environment_name="local",
                    tool_name="read_file",
                    timestamp=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
                ),
                Event(
                    id="event_3",
                    type=EventType.MODEL_COMPLETED,
                    session_id="sess_builder",
                    agent_name="builder",
                    environment_name="local",
                    timestamp=datetime(2026, 1, 1, 12, 10, tzinfo=UTC),
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
                timestamp=datetime(2026, 1, 1, 12, 15, tzinfo=UTC),
            ),
        )

        all_records = await store.query_events(EventQuery(limit=10))
        desc_records = await store.query_events(
            EventQuery(order_by=EventOrder.SEQUENCE_DESC, limit=2)
        )
        since_records = await store.query_events(
            EventQuery(since=datetime(2026, 1, 1, 12, 5, tzinfo=UTC), limit=10)
        )
        until_records = await store.query_events(
            EventQuery(until=datetime(2026, 1, 1, 12, 10, tzinfo=UTC), limit=10)
        )
        window_records = await store.query_events(
            EventQuery(
                since=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
                until=datetime(2026, 1, 1, 12, 15, tzinfo=UTC),
                limit=10,
            )
        )
        builder_records = await store.query_events(EventQuery(session_id="sess_builder"))
        session_ids_records = await store.query_events(
            EventQuery(session_ids=("sess_reviewer", "sess_builder"), limit=10)
        )
        read_file_records = await store.query_events(EventQuery(tool_name="read_file"))
        started_records = await store.query_events(EventQuery(event_type=EventType.SESSION_STARTED))
        cursor_records = await store.query_events(
            EventQuery(after_sequence=all_records[1].sequence, limit=10)
        )

        assert [r.sequence for r in all_records] == [1, 2, 3, 4]
        assert [r.sequence for r in desc_records] == [4, 3]
        assert [r.event.id for r in since_records] == ["event_2", "event_3", "event_4"]
        assert [r.event.id for r in until_records] == ["event_1", "event_2"]
        assert [r.event.id for r in window_records] == ["event_2", "event_3"]
        assert [r.event.id for r in builder_records] == ["event_1", "event_2", "event_3"]
        assert [r.event.id for r in session_ids_records] == [
            "event_1",
            "event_2",
            "event_3",
            "event_4",
        ]
        assert [r.event.id for r in read_file_records] == ["event_2"]
        assert [r.event.id for r in started_records] == ["event_1", "event_4"]
        assert [r.event.id for r in cursor_records] == ["event_3", "event_4"]

        # A batch containing a duplicate event id must roll back atomically.
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

        records_after = await store.query_events(EventQuery(limit=10))
        assert [r.event.id for r in records_after] == [
            "event_1",
            "event_2",
            "event_3",
            "event_4",
        ]

    _run(postgres_dsn, ops)


def test_postgres_session_store_append_and_load_transcript_messages(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_transcript",
                messages=[Message.text("user", "build")],
            ),
            identity=_identity(),
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

        await store.append_transcript_messages("sess_transcript", [user_message, assistant_message])
        # Messages are frozen: stored transcripts cannot be corrupted through
        # references the caller still holds.
        with pytest.raises(ValidationError):
            user_message.content[0].text = "mutated"  # type: ignore[misc]
        await store.append_transcript_messages("sess_transcript", [tool_message])

        transcript = await store.load_transcript("sess_transcript")
        assert [m.role for m in transcript] == ["user", "assistant", "tool"]
        assert transcript[0].content[0].text == "build"
        assert transcript[1].content[0].tool_name == "read_file"
        assert transcript[2].content[0].structured == {"bytes": 8}

        with pytest.raises(KeyError, match="Session not found"):
            await store.append_transcript_messages("missing_session", [Message.text("user", "hi")])
        with pytest.raises(KeyError, match="Session not found"):
            await store.load_transcript("missing_session")

    _run(postgres_dsn, ops)


def test_postgres_session_store_update_session_active_model(postgres_dsn):
    async def ops(store):
        session = await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_model",
                messages=[Message.text("user", "build")],
            ),
            identity=SessionIdentity(provider_name="fake", model="initial-model"),
        )
        assert session.model == "initial-model"

        updated = await store.update_model("sess_model", "upgraded-model")
        assert updated.model == "upgraded-model"
        assert updated.updated_at >= session.updated_at

        loaded = await store.load("sess_model")
        assert loaded is not None
        assert loaded.model == "upgraded-model"

        with pytest.raises(ValueError, match="model"):
            await store.update_model("sess_model", " ")
        with pytest.raises(KeyError, match="Session not found"):
            await store.update_model("missing_session", "other-model")

    _run(postgres_dsn, ops)


def test_postgres_session_store_transition_status_atomically(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_transition",
                messages=[Message.text("user", "build")],
            ),
            identity=_identity(),
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

    _run(postgres_dsn, ops)


def test_postgres_session_store_concurrent_appends_keep_contiguous_order(postgres_dsn):
    """Concurrent append batches must produce a contiguous per-session order."""

    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_concurrent",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )

        async def append_batch(prefix: str) -> None:
            await store.append_events(
                "sess_concurrent",
                [
                    Event(
                        id=f"{prefix}_{i}",
                        type=EventType.MODEL_TEXT_DELTA,
                        session_id="sess_concurrent",
                        payload={"i": i},
                    )
                    for i in range(10)
                ],
            )

        await asyncio.gather(*(append_batch(f"w{w}") for w in range(5)))

        events = await store.load_events("sess_concurrent")
        assert len(events) == 50
        assert len({e.id for e in events}) == 50

        # The global query cursor (sequence) must be a contiguous 1..50 range.
        records = await store.query_events(EventQuery(session_id="sess_concurrent", limit=1000))
        assert [r.sequence for r in records] == list(range(1, 51))

        # Per-session order is dense and unique under concurrency.
        import psycopg

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT session_order FROM cayu_events WHERE session_id = %s "
                "ORDER BY session_order ASC",
                ("sess_concurrent",),
            )
            orders = [row[0] for row in await cur.fetchall()]
        assert orders == list(range(1, 51))

    _run(postgres_dsn, ops)


def test_postgres_session_store_append_advances_event_seq_counter(postgres_dsn):
    """append_events must advance cayu_sessions.event_seq to the last order."""

    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_counter",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )

        import psycopg

        async def read_counter() -> int:
            async with (
                await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
                conn.cursor() as cur,
            ):
                await cur.execute(
                    "SELECT event_seq FROM cayu_sessions WHERE id = %s",
                    ("sess_counter",),
                )
                row = await cur.fetchone()
            assert row is not None
            return row[0]

        # A freshly created session starts at 0 (no events reserved yet).
        assert await read_counter() == 0

        await store.append_events(
            "sess_counter",
            [
                Event(
                    id=f"a_{i}",
                    type=EventType.MODEL_TEXT_DELTA,
                    session_id="sess_counter",
                    payload={"i": i},
                )
                for i in range(3)
            ],
        )
        assert await read_counter() == 3

        await store.append_event(
            "sess_counter",
            Event(
                id="a_last",
                type=EventType.MODEL_COMPLETED,
                session_id="sess_counter",
                payload={"finish_reason": "stop"},
            ),
        )
        assert await read_counter() == 4

        # An empty batch neither advances the counter nor raises for a live session.
        await store.append_events("sess_counter", [])
        assert await read_counter() == 4

        # The counter tracks the highest stored session_order exactly.
        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT MAX(session_order) FROM cayu_events WHERE session_id = %s",
                ("sess_counter",),
            )
            max_order = (await cur.fetchone())[0]
        assert max_order == 4

        # A missing session still raises, even for an empty batch.
        with pytest.raises(KeyError):
            await store.append_events("missing_counter", [])

    _run(postgres_dsn, ops)


def test_postgres_session_store_failed_checkpoint_transition_is_transactional(postgres_dsn):
    """If the transform raises, neither status nor checkpoint may change."""

    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_tx",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.checkpoint("sess_tx", {"existing": True})

        def boom(_s, _ck):
            raise RuntimeError("transform failure")

        with pytest.raises(RuntimeError, match="transform failure"):
            await store.transition_status_and_checkpoint(
                "sess_tx",
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.INTERRUPTING,
                checkpoint_transform=boom,
            )

        loaded = await store.load("sess_tx")
        assert loaded is not None
        assert loaded.status == SessionStatus.PENDING
        assert await store.load_checkpoint("sess_tx") == {"existing": True}

    _run(postgres_dsn, ops)


def test_postgres_session_store_summarize_events(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_builder",
                environment_name="local",
                messages=[Message.text("user", "build")],
            ),
            identity=_identity(),
        )
        await store.append_events(
            "sess_builder",
            [
                Event(
                    id="event_1",
                    type=EventType.SESSION_STARTED,
                    session_id="sess_builder",
                ),
                Event(
                    id="event_2",
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id="sess_builder",
                    tool_name="read_file",
                ),
                Event(
                    id="event_3",
                    type=EventType.MODEL_COMPLETED,
                    session_id="sess_builder",
                    payload={"finish_reason": "stop"},
                ),
            ],
        )

        summary = await store.summarize_events("sess_builder")
        assert summary.session_id == "sess_builder"
        assert summary.total_events == 3
        assert summary.counts_by_type == {
            "model.completed": 1,
            "session.started": 1,
            "tool.call.completed": 1,
        }
        assert summary.latest_event is not None
        assert summary.latest_event.event.id == "event_3"

        with pytest.raises(KeyError, match="Session not found"):
            await store.summarize_events("missing_session")

    _run(postgres_dsn, ops)


def test_postgres_session_store_batches_large_event_session_id_queries(postgres_dsn, monkeypatch):
    import cayu.storage.postgres as postgres_module

    monkeypatch.setattr(postgres_module, "_EVENT_QUERY_SESSION_IDS_BATCH_SIZE", 2)

    async def ops(store):
        for index in range(5):
            session_id = f"sess_batch_{index}"
            await store.create(
                RunRequest(
                    agent_name="builder",
                    session_id=session_id,
                    environment_name="local",
                    messages=[Message.text("user", f"batch {index}")],
                ),
                identity=_identity(),
            )
            await store.append_event(
                session_id,
                Event(
                    id=f"event_batch_{index}",
                    type=EventType.SESSION_STARTED,
                    session_id=session_id,
                    agent_name="builder",
                    environment_name="local",
                    timestamp=datetime(2026, 1, 1, 12, index, tzinfo=UTC),
                ),
            )

        session_ids = (
            "sess_batch_4",
            "sess_batch_0",
            "sess_batch_2",
            "sess_batch_1",
            "sess_batch_3",
        )
        records = await store.query_events(EventQuery(session_ids=session_ids, limit=10))
        limited_records = await store.query_events(EventQuery(session_ids=session_ids, limit=3))
        cursor_records = await store.query_events(
            EventQuery(
                session_ids=session_ids,
                after_sequence=records[1].sequence,
                limit=10,
            )
        )

        assert [record.event.id for record in records] == [
            "event_batch_0",
            "event_batch_1",
            "event_batch_2",
            "event_batch_3",
            "event_batch_4",
        ]
        assert [record.event.id for record in limited_records] == [
            "event_batch_0",
            "event_batch_1",
            "event_batch_2",
        ]
        assert [record.event.id for record in cursor_records] == [
            "event_batch_2",
            "event_batch_3",
            "event_batch_4",
        ]

    _run(postgres_dsn, ops)


def test_postgres_session_store_summarize_outcome_from_terminal_and_retry_events(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_outcome",
                messages=[Message.text("user", "retry then stop")],
            ),
            identity=_identity(),
        )
        await store.update_status("sess_outcome", SessionStatus.INTERRUPTED)
        await store.append_events(
            "sess_outcome",
            [
                Event(
                    id="event_retry",
                    type=EventType.MODEL_RETRY,
                    session_id="sess_outcome",
                    payload={
                        "provider": "fake",
                        "model": "fake-model",
                        "step": 1,
                        "attempt": 1,
                        "next_attempt": 2,
                        "max_attempts": 2,
                        "reason": "http_status",
                        "status_code": 429,
                        "delay_seconds": 0.0,
                        "error": "rate limited",
                    },
                ),
                Event(
                    id="event_interrupted",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id="sess_outcome",
                    payload={
                        "interruption_type": "limit_reached",
                        "limit": "total_tokens",
                        "actual": 12,
                        "maximum": 10,
                        "message": "Run limit reached.",
                    },
                ),
                Event(
                    id="event_hook_after_terminal",
                    type=EventType.HOOK_COMPLETED,
                    session_id="sess_outcome",
                    payload={"hook": "after_session_interrupted"},
                ),
            ],
        )

        outcome = await store.summarize_outcome("sess_outcome")
        assert outcome.status == SessionStatus.INTERRUPTED
        assert outcome.reason == "limit_reached"
        assert outcome.details == {
            "interruption_type": "limit_reached",
            "limit": "total_tokens",
            "maximum": 10,
            "actual": 12,
            "message": "Run limit reached.",
        }
        assert outcome.retry == {
            "provider": "fake",
            "model": "fake-model",
            "step": 1,
            "attempt": 1,
            "next_attempt": 2,
            "max_attempts": 2,
            "delay_seconds": 0.0,
            "reason": "http_status",
            "status_code": 429,
        }
        assert outcome.terminal_event is not None
        assert outcome.terminal_event.event.id == "event_interrupted"
        assert outcome.latest_retry_event is not None
        assert outcome.latest_retry_event.event.id == "event_retry"

        with pytest.raises(KeyError, match="Session not found"):
            await store.summarize_outcome("missing_session")

    _run(postgres_dsn, ops)


def test_postgres_session_store_summarize_outcome_scopes_to_latest_lifecycle(postgres_dsn):
    """Terminal and retry events before the latest start/resume must be ignored.

    This exercises the COALESCE(MAX(sequence)) lifecycle subquery directly: a clean
    resume after an earlier completion + retry should surface only the post-resume
    terminal event and no stale retry.
    """

    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_resume",
                messages=[Message.text("user", "first")],
            ),
            identity=_identity(),
        )
        await store.update_status("sess_resume", SessionStatus.COMPLETED)
        await store.append_events(
            "sess_resume",
            [
                Event(id="event_started", type=EventType.SESSION_STARTED, session_id="sess_resume"),
                Event(
                    id="event_retry_old",
                    type=EventType.MODEL_RETRY,
                    session_id="sess_resume",
                    payload={
                        "provider": "fake",
                        "model": "fake-model",
                        "step": 1,
                        "attempt": 1,
                        "next_attempt": 2,
                        "max_attempts": 2,
                        "reason": "timeout",
                        "delay_seconds": 0.0,
                    },
                ),
                Event(
                    id="event_completed_first",
                    type=EventType.SESSION_COMPLETED,
                    session_id="sess_resume",
                ),
                Event(id="event_resumed", type=EventType.SESSION_RESUMED, session_id="sess_resume"),
                Event(
                    id="event_completed_after_resume",
                    type=EventType.SESSION_COMPLETED,
                    session_id="sess_resume",
                ),
            ],
        )

        outcome = await store.summarize_outcome("sess_resume")
        assert outcome.status == SessionStatus.COMPLETED
        assert outcome.reason == "completed"
        assert outcome.retry is None
        assert outcome.latest_retry_event is None
        assert outcome.terminal_event is not None
        assert outcome.terminal_event.event.id == "event_completed_after_resume"

    _run(postgres_dsn, ops)


def test_postgres_session_store_query_transcript_pagination_and_role_filter(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_transcript",
                messages=[Message.text("user", "build")],
            ),
            identity=_identity(),
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
            [user_message, assistant_message, tool_message],
        )

        # Stable, gap-free index across the full transcript; offset/limit paginate it.
        page = await store.query_transcript(
            TranscriptQuery(session_id="sess_transcript", offset=1, limit=1)
        )
        assert page.total_records == 3
        assert [record.index for record in page.records] == [1]
        assert page.records[0].message.content[0].tool_name == "read_file"

        # Role filter keeps the original full-transcript index (0), not a re-counted one.
        user_page = await store.query_transcript(
            TranscriptQuery(session_id="sess_transcript", role="user", limit=10)
        )
        assert user_page.total_records == 1
        assert [record.index for record in user_page.records] == [0]
        assert user_page.records[0].message.content[0].text == "build"

        with pytest.raises(KeyError, match="Session not found"):
            await store.query_transcript(TranscriptQuery(session_id="missing_session"))

    _run(postgres_dsn, ops)


def _lifecycle_request(
    session_id: str,
    *,
    parent: str | None = None,
    labels: dict[str, str] | None = None,
    metadata: dict[str, object] | None = None,
) -> RunRequest:
    return RunRequest(
        agent_name="assistant",
        session_id=session_id,
        parent_session_id=parent,
        labels=labels or {},
        metadata=metadata or {},
        messages=[Message.text("user", "hi")],
    )


def test_postgres_session_store_delete_session_cascades_and_is_idempotent(postgres_dsn):
    async def ops(store):
        await store.create(_lifecycle_request("sess_keep"), identity=_identity())
        await store.create(
            _lifecycle_request("sess_drop", labels={"team": "drop"}), identity=_identity()
        )
        await store.append_events(
            "sess_drop",
            [Event(type=EventType.SESSION_STARTED, session_id="sess_drop", agent_name="assistant")],
        )
        await store.append_transcript_messages("sess_drop", [Message.text("assistant", "bye")])
        await store.checkpoint("sess_drop", {"cursor": 1})

        await store.delete_session("sess_drop")

        assert await store.load("sess_drop") is None
        assert await store.query_events(EventQuery(session_id="sess_drop")) == []
        assert await store.load_checkpoint("sess_drop") is None
        assert (await store.list_sessions(SessionQuery(labels={"team": "drop"}))).sessions == []
        assert await store.load("sess_keep") is not None
        await store.create(_lifecycle_request("sess_drop"), identity=_identity())
        assert await store.load("sess_drop") is not None
        await store.delete_session("sess_never_existed")

    _run(postgres_dsn, ops)


def test_postgres_session_store_delete_rejects_in_flight_sessions(postgres_dsn):
    async def ops(store):
        for index, status in enumerate((SessionStatus.RUNNING, SessionStatus.INTERRUPTING)):
            session_id = f"sess_inflight_{index}"
            await store.create(_lifecycle_request(session_id), identity=_identity())
            await store.update_status(session_id, status)
            with pytest.raises(ValueError, match="interrupt it first"):
                await store.delete_session(session_id)
            assert await store.load(session_id) is not None

    _run(postgres_dsn, ops)


def test_postgres_session_store_delete_rechecks_status_after_waiting_for_row_lock(postgres_dsn):
    async def ops(store):
        import psycopg

        session_id = "sess_delete_race"
        await store.create(_lifecycle_request(session_id), identity=_identity())
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE cayu_sessions SET status = %s WHERE id = %s",
                    (str(SessionStatus.RUNNING), session_id),
                )
                delete_task = asyncio.create_task(store.delete_session(session_id))
                await asyncio.sleep(0.1)
                assert delete_task.done() is False
            await conn.commit()

        with pytest.raises(ValueError, match="interrupt it first"):
            await asyncio.wait_for(delete_task, timeout=2.0)
        loaded = await store.load(session_id)
        assert loaded is not None
        assert loaded.status == SessionStatus.RUNNING

    _run(postgres_dsn, ops)


def test_postgres_session_store_delete_parent_nulls_child_parent(postgres_dsn):
    async def ops(store):
        await store.create(_lifecycle_request("sess_parent"), identity=_identity())
        await store.create(
            _lifecycle_request("sess_child", parent="sess_parent"), identity=_identity()
        )
        await store.delete_session("sess_parent")
        child = await store.load("sess_child")
        assert child is not None
        assert child.parent_session_id is None

    _run(postgres_dsn, ops)


def test_postgres_session_store_update_labels_replaces_and_filters(postgres_dsn):
    async def ops(store):
        created = await store.create(
            _lifecycle_request("sess_labeled", labels={"team": "research", "stage": "draft"}),
            identity=_identity(),
        )
        await store.update_status("sess_labeled", SessionStatus.COMPLETED)
        updated = await store.update_labels("sess_labeled", {"stage": "review"})
        assert updated.labels == {"stage": "review"}
        assert updated.updated_at >= created.updated_at
        assert updated.status == SessionStatus.COMPLETED
        reloaded = await store.load("sess_labeled")
        assert reloaded is not None
        assert reloaded.labels == {"stage": "review"}
        matched = (await store.list_sessions(SessionQuery(labels={"stage": "review"}))).sessions
        assert [session.id for session in matched] == ["sess_labeled"]
        stale = (await store.list_sessions(SessionQuery(labels={"team": "research"}))).sessions
        assert stale == []
        cleared = await store.update_labels("sess_labeled", {})
        assert cleared.labels == {}
        assert (await store.list_sessions(SessionQuery(labels={"stage": "review"}))).sessions == []
        with pytest.raises(ValueError, match="reserved"):
            await store.update_labels("sess_labeled", {"cayu:internal": "x"})
        with pytest.raises(KeyError, match="Session not found"):
            await store.update_labels("sess_missing", {"k": "v"})

    _run(postgres_dsn, ops)


def test_postgres_session_store_update_metadata_replaces(postgres_dsn):
    async def ops(store):
        await store.create(
            _lifecycle_request("sess_meta", metadata={"a": 1, "keep": False}),
            identity=_identity(),
        )
        await store.update_status("sess_meta", SessionStatus.COMPLETED)
        updated = await store.update_metadata("sess_meta", {"b": [1, 2]})
        assert updated.metadata == {"b": [1, 2]}
        assert updated.status == SessionStatus.COMPLETED
        reloaded = await store.load("sess_meta")
        assert reloaded is not None
        assert reloaded.metadata == {"b": [1, 2]}
        with pytest.raises(KeyError, match="Session not found"):
            await store.update_metadata("sess_missing", {"k": "v"})

    _run(postgres_dsn, ops)


def test_postgres_session_store_cursor_pagination_is_stable_across_orders(postgres_dsn):
    async def ops(store):
        for index in range(5):
            await store.create(_lifecycle_request(f"sess_{index}"), identity=_identity())
        for order in SessionOrder:
            full = (await store.list_sessions(SessionQuery(order_by=order, limit=100))).sessions
            expected_ids = [session.id for session in full]
            collected: list[str] = []
            cursor: str | None = None
            while True:
                page = await store.list_sessions(
                    SessionQuery(order_by=order, limit=2, cursor=cursor, include_total_count=True)
                )
                assert page.total_count == len(expected_ids)
                collected.extend(session.id for session in page.sessions)
                if page.next_cursor is None:
                    break
                cursor = page.next_cursor
            assert collected == expected_ids, order

    _run(postgres_dsn, ops)


def test_postgres_session_store_cursor_survives_concurrent_insert(postgres_dsn):
    async def ops(store):
        for index in range(4):
            await store.create(_lifecycle_request(f"sess_{index}"), identity=_identity())
        order = SessionOrder.CREATED_AT_ASC
        first = await store.list_sessions(SessionQuery(order_by=order, limit=2))
        seen = [session.id for session in first.sessions]
        await store.create(_lifecycle_request("sess_inserted"), identity=_identity())
        cursor = first.next_cursor
        while cursor is not None:
            page = await store.list_sessions(SessionQuery(order_by=order, limit=2, cursor=cursor))
            seen.extend(session.id for session in page.sessions)
            cursor = page.next_cursor
        assert len(seen) == len(set(seen)), seen
        assert {"sess_0", "sess_1", "sess_2", "sess_3"} <= set(seen)

    _run(postgres_dsn, ops)


def test_postgres_session_store_rejects_invalid_cursor(postgres_dsn):
    async def ops(store):
        await store.create(_lifecycle_request("sess_only"), identity=_identity())
        with pytest.raises(ValueError, match="[Cc]ursor"):
            await store.list_sessions(SessionQuery(cursor="!!!not-a-cursor"))
        forged = base64.urlsafe_b64encode(b'["not-a-timestamp","sess_only"]').decode("ascii")
        with pytest.raises(ValueError, match="[Cc]ursor"):
            await store.list_sessions(SessionQuery(cursor=forged))

    _run(postgres_dsn, ops)


def test_postgres_session_store_cursor_pagination_empty_result(postgres_dsn):
    async def ops(store):
        await store.create(_lifecycle_request("sess_only"), identity=_identity())
        page = await store.list_sessions(
            SessionQuery(labels={"absent": "1"}, limit=2, include_total_count=True)
        )
        assert page.sessions == []
        assert page.next_cursor is None
        assert page.total_count == 0

    _run(postgres_dsn, ops)
