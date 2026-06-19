from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cayu import EventQuery, SessionOrder, SessionQuery, SQLiteSessionStore, TranscriptQuery
from cayu.core import Event, EventType, Message
from cayu.runtime import (
    InMemorySessionStore,
    RunRequest,
    SessionIdentity,
    SessionStatus,
    SessionStore,
)
from cayu.runtime.sessions import event_summary_from_records, session_outcome_from_records

StoreFactory = Callable[[object], SessionStore]


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_default_causal_budget_id_from_task_or_session(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        explicit = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_explicit_causal",
                causal_budget_id="job_explicit",
                task_id="task_explicit",
                messages=[Message.text("user", "explicit")],
            ),
            identity=_identity(),
        )
        task_scoped = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_task_causal",
                task_id="task_shared",
                messages=[Message.text("user", "task")],
            ),
            identity=_identity(),
        )
        session_scoped = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_default_causal",
                messages=[Message.text("user", "default")],
            ),
            identity=_identity(),
        )

        assert explicit.causal_budget_id == "job_explicit"
        assert task_scoped.causal_budget_id == "task_shared"
        assert session_scoped.causal_budget_id == "sess_default_causal"
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_reject_invalid_parent_session_id(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        with pytest.raises(ValueError, match="Parent session not found"):
            await store.create(
                RunRequest(
                    agent_name="reviewer",
                    session_id="sess_missing_parent_child",
                    parent_session_id="sess_missing_parent",
                    messages=[Message.text("user", "child")],
                ),
                identity=_identity(),
            )
        assert await store.load("sess_missing_parent_child") is None
        with pytest.raises(ValueError, match="own parent"):
            await store.create(
                RunRequest(
                    agent_name="reviewer",
                    session_id="sess_self_parent",
                    parent_session_id="sess_self_parent",
                    messages=[Message.text("user", "child")],
                ),
                identity=_identity(),
            )
        assert await store.load("sess_self_parent") is None
        await _close_store(store)

    asyncio.run(run_store_operations())


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
                causal_budget_id="job_build",
                environment_name="local",
                messages=[Message.text("user", "build")],
            ),
            identity=_identity(),
        )
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_builder_2",
                causal_budget_id="job_build",
                environment_name="hosted",
                messages=[Message.text("user", "build again")],
            ),
            identity=_identity(),
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_reviewer",
                causal_budget_id="job_review",
                environment_name="hosted",
                messages=[Message.text("user", "review")],
            ),
            identity=_identity(),
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
        causal_sessions = await store.list_sessions(
            SessionQuery(causal_budget_id="job_build", order_by=SessionOrder.CREATED_AT_ASC)
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
        assert [session.id for session in causal_sessions] == [
            "sess_builder_1",
            "sess_builder_2",
        ]
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_preserve_and_filter_session_labels(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        owner_labels = {
            "owner": "org_123",
            "project": "ap_q2",
            "workflow": "invoice-ingestion",
        }
        created = await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_labels_invoice",
                labels=owner_labels,
                messages=[Message.text("user", "ingest invoice")],
            ),
            identity=_identity(),
        )
        owner_labels["owner"] = "mutated"
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_labels_research",
                labels={"owner": "org_123", "project": "research"},
                messages=[Message.text("user", "research")],
            ),
            identity=_identity(),
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_labels_other_owner",
                labels={"owner": "org_999", "project": "ap_q2"},
                messages=[Message.text("user", "review")],
            ),
            identity=_identity(),
        )

        assert created.labels == {
            "owner": "org_123",
            "project": "ap_q2",
            "workflow": "invoice-ingestion",
        }
        loaded = await store.load("sess_labels_invoice")
        assert loaded is not None
        assert loaded.labels == created.labels

        owner_sessions = await store.list_sessions(
            SessionQuery(labels={"owner": "org_123"}, order_by=SessionOrder.CREATED_AT_ASC)
        )
        project_sessions = await store.list_sessions(
            SessionQuery(labels={"project": "ap_q2"}, order_by=SessionOrder.CREATED_AT_ASC)
        )
        exact_sessions = await store.list_sessions(
            SessionQuery(
                labels={"owner": "org_123", "project": "ap_q2"},
                order_by=SessionOrder.CREATED_AT_ASC,
            )
        )
        missing_sessions = await store.list_sessions(SessionQuery(labels={"owner": "missing"}))

        assert [session.id for session in owner_sessions] == [
            "sess_labels_invoice",
            "sess_labels_research",
        ]
        assert [session.id for session in project_sessions] == [
            "sess_labels_invoice",
            "sess_labels_other_owner",
        ]
        assert [session.id for session in exact_sessions] == ["sess_labels_invoice"]
        assert missing_sessions == []
        await _close_store(store)

    asyncio.run(run_store_operations())


def test_session_labels_are_strict_clean_string_maps():
    with pytest.raises(ValidationError, match="labels.bad"):
        RunRequest(
            agent_name="assistant",
            session_id="sess_bad_label_value",
            labels={"bad": 123},
            messages=[Message.text("user", "hi")],
        )
    with pytest.raises(ValidationError, match="must not start or end with whitespace"):
        SessionQuery(labels={" owner": "org_123"})
    with pytest.raises(ValidationError, match="cannot be blank"):
        RunRequest(
            agent_name="assistant",
            session_id="sess_blank_label_value",
            labels={"owner": " "},
            messages=[Message.text("user", "hi")],
        )
    with pytest.raises(ValidationError, match="reserved for Cayu"):
        RunRequest(
            agent_name="assistant",
            session_id="sess_reserved_label",
            labels={"cayu:causal-budget": "budget_123"},
            messages=[Message.text("user", "hi")],
        )


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
                causal_budget_id="job_build",
                environment_name="local",
                messages=[Message.text("user", "build")],
            ),
            identity=_identity(),
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_reviewer",
                causal_budget_id="job_review",
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
        causal_records = await store.query_events(EventQuery(causal_budget_id="job_build"))
        cursor_records = await store.query_events(
            EventQuery(after_sequence=all_records[1].sequence, limit=10)
        )
        event_summary = await store.summarize_events("sess_builder")

        assert [record.sequence for record in all_records] == [1, 2, 3, 4]
        assert [record.event.id for record in since_records] == [
            "event_2",
            "event_3",
            "event_4",
        ]
        assert [record.event.id for record in until_records] == ["event_1", "event_2"]
        assert [record.event.id for record in window_records] == ["event_2", "event_3"]
        assert [record.event.id for record in builder_records] == [
            "event_1",
            "event_2",
            "event_3",
        ]
        assert [record.event.id for record in session_ids_records] == [
            "event_1",
            "event_2",
            "event_3",
            "event_4",
        ]
        assert [record.event.id for record in read_file_records] == ["event_2"]
        assert [record.event.id for record in started_records] == [
            "event_1",
            "event_4",
        ]
        assert [record.event.id for record in causal_records] == [
            "event_1",
            "event_2",
            "event_3",
        ]
        assert [record.event.id for record in cursor_records] == [
            "event_3",
            "event_4",
        ]
        assert event_summary.total_events == 3
        assert event_summary.counts_by_type == {
            "model.completed": 1,
            "session.started": 1,
            "tool.call.completed": 1,
        }
        assert event_summary.latest_event is not None
        assert event_summary.latest_event.event.id == "event_3"

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
        with pytest.raises(KeyError, match="Session not found"):
            await store.summarize_events("missing_session")
        await _close_store(store)

    asyncio.run(run_store_operations())


def test_sqlite_session_store_batches_large_event_session_id_queries(tmp_path, monkeypatch):
    import cayu.storage.sqlite as sqlite_module

    monkeypatch.setattr(sqlite_module, "_EVENT_QUERY_SESSION_IDS_BATCH_SIZE", 2)
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite")

    async def run_store_operations() -> None:
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
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_summarize_outcome_from_terminal_and_retry_events(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
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
        records = await store.query_events(EventQuery(session_id="sess_outcome", limit=100))
        session = await store.load("sess_outcome")
        assert session is not None
        helper_summary = event_summary_from_records("sess_outcome", records)
        helper_outcome = session_outcome_from_records(session, records)
        assert helper_summary.total_events == 3
        assert helper_summary.counts_by_type["model.retry"] == 1
        assert helper_summary.counts_by_type["session.interrupted"] == 1
        assert helper_summary.latest_event is not None
        assert helper_summary.latest_event.event.id == "event_hook_after_terminal"
        assert helper_outcome == outcome

        with pytest.raises(KeyError, match="Session not found"):
            await store.summarize_outcome("missing_session")
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_summarize_outcome_prefers_current_active_status(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_resumed_running",
                messages=[Message.text("user", "first")],
            ),
            identity=_identity(),
        )
        await store.update_status("sess_resumed_running", SessionStatus.COMPLETED)
        await store.append_event(
            "sess_resumed_running",
            Event(
                id="event_completed_before_resume",
                type=EventType.SESSION_COMPLETED,
                session_id="sess_resumed_running",
            ),
        )
        await store.update_status("sess_resumed_running", SessionStatus.RUNNING)

        outcome = await store.summarize_outcome("sess_resumed_running")

        assert outcome.status == SessionStatus.RUNNING
        assert outcome.reason == "running"
        assert outcome.details == {}
        assert outcome.terminal_event is None

        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_summarize_outcome_ignores_mismatched_terminal_event(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_terminal_mismatch",
                messages=[Message.text("user", "first")],
            ),
            identity=_identity(),
        )
        await store.update_status("sess_terminal_mismatch", SessionStatus.COMPLETED)
        await store.append_event(
            "sess_terminal_mismatch",
            Event(
                id="event_completed_before_failed_status",
                type=EventType.SESSION_COMPLETED,
                session_id="sess_terminal_mismatch",
            ),
        )
        await store.update_status("sess_terminal_mismatch", SessionStatus.FAILED)

        outcome = await store.summarize_outcome("sess_terminal_mismatch")

        assert outcome.status == SessionStatus.FAILED
        assert outcome.reason == "failed"
        assert outcome.details == {}
        assert outcome.terminal_event is None

        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_summarize_outcome_scopes_retry_to_latest_lifecycle(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_retry_then_clean_resume",
                messages=[Message.text("user", "first")],
            ),
            identity=_identity(),
        )
        await store.update_status("sess_retry_then_clean_resume", SessionStatus.COMPLETED)
        await store.append_events(
            "sess_retry_then_clean_resume",
            [
                Event(
                    id="event_started",
                    type=EventType.SESSION_STARTED,
                    session_id="sess_retry_then_clean_resume",
                ),
                Event(
                    id="event_retry_old",
                    type=EventType.MODEL_RETRY,
                    session_id="sess_retry_then_clean_resume",
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
                    session_id="sess_retry_then_clean_resume",
                ),
                Event(
                    id="event_resumed",
                    type=EventType.SESSION_RESUMED,
                    session_id="sess_retry_then_clean_resume",
                ),
                Event(
                    id="event_completed_after_clean_resume",
                    type=EventType.SESSION_COMPLETED,
                    session_id="sess_retry_then_clean_resume",
                ),
            ],
        )

        outcome = await store.summarize_outcome("sess_retry_then_clean_resume")

        assert outcome.status == SessionStatus.COMPLETED
        assert outcome.reason == "completed"
        assert outcome.retry is None
        assert outcome.latest_retry_event is None
        assert outcome.terminal_event is not None
        assert outcome.terminal_event.event.id == "event_completed_after_clean_resume"

        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_summarize_outcome_scopes_terminal_to_latest_lifecycle(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_terminal_then_resume",
                messages=[Message.text("user", "first")],
            ),
            identity=_identity(),
        )
        await store.update_status("sess_terminal_then_resume", SessionStatus.COMPLETED)
        await store.append_events(
            "sess_terminal_then_resume",
            [
                Event(
                    id="event_started",
                    type=EventType.SESSION_STARTED,
                    session_id="sess_terminal_then_resume",
                ),
                Event(
                    id="event_completed_before_resume",
                    type=EventType.SESSION_COMPLETED,
                    session_id="sess_terminal_then_resume",
                ),
                Event(
                    id="event_resumed",
                    type=EventType.SESSION_RESUMED,
                    session_id="sess_terminal_then_resume",
                ),
            ],
        )

        outcome = await store.summarize_outcome("sess_terminal_then_resume")

        assert outcome.status == SessionStatus.COMPLETED
        assert outcome.reason == "completed"
        assert outcome.details == {}
        assert outcome.terminal_event is None

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

        transcript_page = await store.query_transcript(
            TranscriptQuery(
                session_id="sess_transcript",
                offset=1,
                limit=1,
            )
        )
        assert transcript_page.total_records == 3
        assert [record.index for record in transcript_page.records] == [1]
        assert transcript_page.records[0].message.content[0].tool_name == "read_file"

        user_page = await store.query_transcript(
            TranscriptQuery(
                session_id="sess_transcript",
                role="user",
                limit=10,
            )
        )
        assert user_page.total_records == 1
        assert [record.index for record in user_page.records] == [0]
        assert user_page.records[0].message.content[0].text == "build"

        user_page.records[0].message.content[0].text = "changed after query"
        user_page_again = await store.query_transcript(
            TranscriptQuery(session_id="sess_transcript", role="user")
        )
        assert user_page_again.records[0].message.content[0].text == "build"

        with pytest.raises(KeyError, match="Session not found"):
            await store.append_transcript_messages(
                "missing_session",
                [Message.text("user", "hi")],
            )
        with pytest.raises(KeyError, match="Session not found"):
            await store.load_transcript("missing_session")
        with pytest.raises(KeyError, match="Session not found"):
            await store.query_transcript(TranscriptQuery(session_id="missing_session"))

        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_update_session_active_model(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
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
