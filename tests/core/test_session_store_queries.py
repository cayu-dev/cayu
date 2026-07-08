from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cayu import (
    EventOrder,
    EventQuery,
    SessionOrder,
    SessionQuery,
    SQLiteSessionStore,
    TranscriptQuery,
)
from cayu.core import Event, EventType, Message, ThinkingPart, ToolCallPart
from cayu.runtime import (
    InMemorySessionStore,
    RunRequest,
    Session,
    SessionDebugState,
    SessionIdentity,
    SessionStatus,
    SessionStore,
)
from cayu.runtime.sessions import event_summary_from_records, session_outcome_from_records

StoreFactory = Callable[[object], SessionStore]


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


def test_session_store_lifecycle_methods_are_not_abstract() -> None:
    # delete/update_labels/update_metadata are concrete (NotImplementedError) defaults,
    # not @abstractmethod, so existing out-of-tree SessionStore subclasses still instantiate.
    assert {"delete_session", "update_labels", "update_metadata"}.isdisjoint(
        SessionStore.__abstractmethods__
    )


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
        await store.create(
            RunRequest(
                agent_name="operator",
                session_id="sess_openai_operator",
                causal_budget_id="job_ops",
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
                SessionQuery(
                    agent_name="builder",
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
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
        causal_sessions = (
            await store.list_sessions(
                SessionQuery(causal_budget_id="job_build", order_by=SessionOrder.CREATED_AT_ASC)
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
        assert [session.id for session in openai_sessions] == ["sess_openai_operator"]
        assert [session.id for session in model_sessions] == ["sess_openai_operator"]
        assert [session.id for session in query_by_agent_sessions] == ["sess_openai_operator"]
        assert [session.id for session in query_by_model_sessions] == ["sess_openai_operator"]
        assert [session.id for session in query_by_label_sessions] == ["sess_openai_operator"]
        assert [session.id for session in query_by_literal_percent_sessions] == [
            "sess_openai_operator"
        ]
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_list_sessions_with_debug_state_filter(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def create(session_id: str, status: SessionStatus, events: list[Event]) -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "debug")],
            ),
            identity=_identity(),
        )
        await store.update_status(session_id, status)
        await store.append_events(session_id, events)

    async def run_store_operations() -> None:
        await create(
            "debug_query_normal",
            SessionStatus.COMPLETED,
            [Event(type=EventType.SESSION_COMPLETED, session_id="debug_query_normal")],
        )
        await create(
            "debug_query_tool_failure",
            SessionStatus.COMPLETED,
            [
                Event(type=EventType.TOOL_CALL_FAILED, session_id="debug_query_tool_failure"),
                Event(type=EventType.SESSION_COMPLETED, session_id="debug_query_tool_failure"),
            ],
        )
        await create(
            "debug_query_tool_blocked",
            SessionStatus.COMPLETED,
            [
                Event(type=EventType.TOOL_CALL_BLOCKED, session_id="debug_query_tool_blocked"),
                Event(type=EventType.SESSION_COMPLETED, session_id="debug_query_tool_blocked"),
            ],
        )
        await create(
            "debug_query_session_failure",
            SessionStatus.FAILED,
            [Event(type=EventType.SESSION_FAILED, session_id="debug_query_session_failure")],
        )
        await create(
            "debug_query_interruption",
            SessionStatus.INTERRUPTED,
            [Event(type=EventType.SESSION_INTERRUPTED, session_id="debug_query_interruption")],
        )

        tool_issues = await store.list_sessions(
            SessionQuery(
                debug_state=SessionDebugState.TOOL_ISSUE,
                order_by=SessionOrder.CREATED_AT_ASC,
            )
        )
        session_failures = await store.list_sessions(
            SessionQuery(
                debug_state=SessionDebugState.SESSION_FAILURE,
                order_by=SessionOrder.CREATED_AT_ASC,
            )
        )
        interruptions = await store.list_sessions(
            SessionQuery(
                debug_state=SessionDebugState.INTERRUPTION,
                order_by=SessionOrder.CREATED_AT_ASC,
            )
        )
        needs_attention = await store.list_sessions(
            SessionQuery(
                debug_state=SessionDebugState.NEEDS_ATTENTION,
                order_by=SessionOrder.CREATED_AT_ASC,
                limit=2,
                include_total_count=True,
            )
        )

        assert [session.id for session in tool_issues.sessions] == [
            "debug_query_tool_failure",
            "debug_query_tool_blocked",
        ]
        assert [session.id for session in session_failures.sessions] == [
            "debug_query_session_failure"
        ]
        assert [session.id for session in interruptions.sessions] == ["debug_query_interruption"]
        assert [session.id for session in needs_attention.sessions] == [
            "debug_query_tool_failure",
            "debug_query_tool_blocked",
        ]
        assert needs_attention.total_count == 4
        assert needs_attention.next_cursor is not None

        second_attention_page = await store.list_sessions(
            SessionQuery(
                debug_state=SessionDebugState.NEEDS_ATTENTION,
                order_by=SessionOrder.CREATED_AT_ASC,
                limit=2,
                cursor=needs_attention.next_cursor,
                include_total_count=True,
            )
        )
        assert [session.id for session in second_attention_page.sessions] == [
            "debug_query_session_failure",
            "debug_query_interruption",
        ]
        assert second_attention_page.total_count == 4
        assert second_attention_page.next_cursor is None
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

        owner_sessions = (
            await store.list_sessions(
                SessionQuery(labels={"owner": "org_123"}, order_by=SessionOrder.CREATED_AT_ASC)
            )
        ).sessions
        project_sessions = (
            await store.list_sessions(
                SessionQuery(labels={"project": "ap_q2"}, order_by=SessionOrder.CREATED_AT_ASC)
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
        causal_records = await store.query_events(EventQuery(causal_budget_id="job_build"))
        cursor_records = await store.query_events(
            EventQuery(after_sequence=all_records[1].sequence, limit=10)
        )
        event_summary = await store.summarize_events("sess_builder")

        assert [record.sequence for record in all_records] == [1, 2, 3, 4]
        assert [record.sequence for record in desc_records] == [4, 3]
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


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_query_events_with_multiple_event_types(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_multi_type",
                messages=[Message.text("user", "go")],
            ),
            identity=_identity(),
        )
        await store.append_events(
            "sess_multi_type",
            [
                Event(
                    id="event_delta",
                    type=EventType.MODEL_TEXT_DELTA,
                    session_id="sess_multi_type",
                    payload={"delta": "noise"},
                ),
                Event(
                    id="event_model",
                    type=EventType.MODEL_COMPLETED,
                    session_id="sess_multi_type",
                    payload={"finish_reason": "stop"},
                ),
                Event(
                    id="event_tool",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="sess_multi_type",
                    tool_name="echo",
                    payload={"tool_call_id": "call_1"},
                ),
                Event(
                    id="event_model_2",
                    type=EventType.MODEL_COMPLETED,
                    session_id="sess_multi_type",
                    payload={"finish_reason": "stop"},
                ),
            ],
        )

        usage_types = (EventType.MODEL_COMPLETED, EventType.TOOL_CALL_STARTED)
        multi = await store.query_events(
            EventQuery(session_id="sess_multi_type", event_types=usage_types, limit=10)
        )
        assert [record.event.id for record in multi] == [
            "event_model",
            "event_tool",
            "event_model_2",
        ]
        assert [record.sequence for record in multi] == sorted(record.sequence for record in multi)

        # `event_types` without a session filter exercises the type-index path.
        unscoped = await store.query_events(EventQuery(event_types=usage_types, limit=10))
        assert [record.event.id for record in unscoped] == [
            "event_model",
            "event_tool",
            "event_model_2",
        ]

        # Combines with the sequence cursor.
        after_first = await store.query_events(
            EventQuery(
                session_id="sess_multi_type",
                event_types=usage_types,
                after_sequence=multi[0].sequence,
                limit=10,
            )
        )
        assert [record.event.id for record in after_first] == ["event_tool", "event_model_2"]

        # A single-element `event_types` matches the singular `event_type` query.
        singular = await store.query_events(
            EventQuery(
                session_id="sess_multi_type",
                event_type=EventType.MODEL_COMPLETED,
                limit=10,
            )
        )
        plural_single = await store.query_events(
            EventQuery(
                session_id="sess_multi_type",
                event_types=(EventType.MODEL_COMPLETED,),
                limit=10,
            )
        )
        assert [record.event.id for record in plural_single] == [
            record.event.id for record in singular
        ]

        await _close_store(store)

    asyncio.run(run_store_operations())


def test_event_query_event_types_validation() -> None:
    with pytest.raises(ValidationError, match="not both"):
        EventQuery(
            event_type=EventType.MODEL_COMPLETED,
            event_types=(EventType.TOOL_CALL_STARTED,),
        )
    with pytest.raises(ValidationError, match="sequence of event types"):
        EventQuery(event_types="model.completed")
    with pytest.raises(ValidationError, match="duplicates"):
        EventQuery(event_types=(EventType.MODEL_COMPLETED, "model.completed"))
    normalized = EventQuery(event_types=("model.completed", EventType.TOOL_CALL_STARTED))
    assert normalized.event_types == (
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
    )


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
        # Messages are frozen: stored transcripts cannot be corrupted through
        # references the caller still holds.
        with pytest.raises(ValidationError):
            user_message.content[0].text = "mutated"  # type: ignore[misc]
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

        with pytest.raises(ValidationError):
            transcript[0].content[0].text = "changed after load"  # type: ignore[misc]
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

        with pytest.raises(ValidationError):
            user_page.records[0].message.content[0].text = "changed"  # type: ignore[misc]
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


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_delete_session_cascades_and_is_idempotent(store_factory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run() -> None:
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
        # All related data is gone, including label rows.
        assert (await store.list_sessions(SessionQuery(labels={"team": "drop"}))).sessions == []
        # The kept session is untouched, and the id can be reused after a clean delete.
        assert await store.load("sess_keep") is not None
        await store.create(_lifecycle_request("sess_drop"), identity=_identity())
        assert await store.load("sess_drop") is not None
        # Idempotent: deleting a missing session is a no-op.
        await store.delete_session("sess_never_existed")

        await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_delete_rejects_in_flight_sessions(store_factory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run() -> None:
        for index, status in enumerate((SessionStatus.RUNNING, SessionStatus.INTERRUPTING)):
            session_id = f"sess_inflight_{index}"
            await store.create(_lifecycle_request(session_id), identity=_identity())
            await store.update_status(session_id, status)
            with pytest.raises(ValueError, match="interrupt it first"):
                await store.delete_session(session_id)
            assert await store.load(session_id) is not None
        await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_delete_parent_nulls_child_parent(store_factory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run() -> None:
        await store.create(_lifecycle_request("sess_parent"), identity=_identity())
        await store.create(
            _lifecycle_request("sess_child", parent="sess_parent"), identity=_identity()
        )
        await store.delete_session("sess_parent")
        child = await store.load("sess_child")
        assert child is not None
        assert child.parent_session_id is None
        await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_update_labels_replaces_and_filters(store_factory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run() -> None:
        created = await store.create(
            _lifecycle_request("sess_labeled", labels={"team": "research", "stage": "draft"}),
            identity=_identity(),
        )
        await store.update_status("sess_labeled", SessionStatus.COMPLETED)
        updated = await store.update_labels("sess_labeled", {"stage": "review"})
        # Full replacement, not a merge: the old "team" label is gone.
        assert updated.labels == {"stage": "review"}
        assert updated.updated_at >= created.updated_at
        # Updating labels must not change the session's status.
        assert updated.status == SessionStatus.COMPLETED
        reloaded = await store.load("sess_labeled")
        assert reloaded is not None
        assert reloaded.labels == {"stage": "review"}
        assert reloaded.status == SessionStatus.COMPLETED
        # The replacement is reflected in subsequent label queries.
        matched = (await store.list_sessions(SessionQuery(labels={"stage": "review"}))).sessions
        assert [session.id for session in matched] == ["sess_labeled"]
        stale = (await store.list_sessions(SessionQuery(labels={"team": "research"}))).sessions
        assert stale == []
        # An empty dict clears every label.
        cleared = await store.update_labels("sess_labeled", {})
        assert cleared.labels == {}
        assert (await store.list_sessions(SessionQuery(labels={"stage": "review"}))).sessions == []
        # Reserved cayu: labels are rejected on update, same as on create.
        with pytest.raises(ValueError, match="reserved"):
            await store.update_labels("sess_labeled", {"cayu:internal": "x"})
        with pytest.raises(KeyError, match="Session not found"):
            await store.update_labels("sess_missing", {"k": "v"})
        await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_update_metadata_replaces(store_factory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run() -> None:
        await store.create(
            _lifecycle_request("sess_meta", metadata={"a": 1, "keep": False}),
            identity=_identity(),
        )
        await store.update_status("sess_meta", SessionStatus.COMPLETED)
        updated = await store.update_metadata("sess_meta", {"b": [1, 2]})
        assert updated.metadata == {"b": [1, 2]}
        # Updating metadata must not change the session's status.
        assert updated.status == SessionStatus.COMPLETED
        reloaded = await store.load("sess_meta")
        assert reloaded is not None
        assert reloaded.metadata == {"b": [1, 2]}
        with pytest.raises(KeyError, match="Session not found"):
            await store.update_metadata("sess_missing", {"k": "v"})
        await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_cursor_pagination_is_stable_across_orders(store_factory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run() -> None:
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
            # Same order, no duplicates, no skips.
            assert collected == expected_ids, order
        # total_count is opt-in: omitted by default.
        assert (await store.list_sessions(SessionQuery(limit=1))).total_count is None
        await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_cursor_survives_concurrent_insert(store_factory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run() -> None:
        for index in range(4):
            await store.create(_lifecycle_request(f"sess_{index}"), identity=_identity())
        order = SessionOrder.CREATED_AT_ASC
        first = await store.list_sessions(SessionQuery(order_by=order, limit=2))
        seen = [session.id for session in first.sessions]
        # A session inserted mid-pagination must not cause a duplicate or a skip of
        # the rows we have not yet passed.
        await store.create(_lifecycle_request("sess_inserted"), identity=_identity())
        cursor = first.next_cursor
        while cursor is not None:
            page = await store.list_sessions(SessionQuery(order_by=order, limit=2, cursor=cursor))
            seen.extend(session.id for session in page.sessions)
            cursor = page.next_cursor
        assert len(seen) == len(set(seen)), seen  # no duplicates
        # Every original session appears exactly once.
        assert {"sess_0", "sess_1", "sess_2", "sess_3"} <= set(seen)
        await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_reject_invalid_cursor(store_factory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run() -> None:
        await store.create(_lifecycle_request("sess_only"), identity=_identity())
        # Not valid base64/JSON.
        with pytest.raises(ValueError, match="[Cc]ursor"):
            await store.list_sessions(SessionQuery(cursor="!!!not-a-cursor"))
        # Structurally valid but the encoded sort value is not a timestamp: every
        # store must reject it rather than silently returning a wrong page.
        forged = base64.urlsafe_b64encode(b'["not-a-timestamp","sess_only"]').decode("ascii")
        with pytest.raises(ValueError, match="[Cc]ursor"):
            await store.list_sessions(SessionQuery(cursor=forged))
        await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_cursor_pagination_empty_result(store_factory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run() -> None:
        await store.create(_lifecycle_request("sess_only"), identity=_identity())
        # A filter that matches nothing: empty first page, no next cursor, zero total.
        page = await store.list_sessions(
            SessionQuery(labels={"absent": "1"}, limit=2, include_total_count=True)
        )
        assert page.sessions == []
        assert page.next_cursor is None
        assert page.total_count == 0
        await _close_store(store)

    asyncio.run(run())


def _force_timestamps(store: InMemorySessionStore, stamps: dict[str, datetime]) -> None:
    for session_id, stamp in stamps.items():
        store._sessions[session_id] = store._sessions[session_id].model_copy(
            update={"created_at": stamp, "updated_at": stamp}
        )


def test_inmemory_cursor_pagination_with_tied_timestamps() -> None:
    # Two tie-groups at distinct timestamps: paging across ties must not skip/duplicate,
    # AND ASC vs DESC must produce DIFFERENT orders (so a reversed primary-sort comparison,
    # not just a reversed tiebreak, would fail this test).
    store = InMemorySessionStore()

    async def run() -> None:
        for index in range(5):
            await store.create(_lifecycle_request(f"sess_{index}"), identity=_identity())
        early = datetime(2026, 1, 1, tzinfo=UTC)
        late = datetime(2026, 6, 1, tzinfo=UTC)
        _force_timestamps(
            store,
            {"sess_0": early, "sess_1": early, "sess_2": early, "sess_3": late, "sess_4": late},
        )

        async def page_through(order: SessionOrder) -> list[str]:
            collected: list[str] = []
            cursor: str | None = None
            while True:
                page = await store.list_sessions(
                    SessionQuery(order_by=order, limit=2, cursor=cursor)
                )
                collected.extend(session.id for session in page.sessions)
                if page.next_cursor is None:
                    break
                cursor = page.next_cursor
            return collected

        for order in SessionOrder:
            full = (await store.list_sessions(SessionQuery(order_by=order, limit=100))).sessions
            expected = [session.id for session in full]
            assert await page_through(order) == expected, order
            assert len(expected) == len(set(expected)) == 5

        asc = [
            s.id
            for s in (
                await store.list_sessions(
                    SessionQuery(order_by=SessionOrder.CREATED_AT_ASC, limit=100)
                )
            ).sessions
        ]
        desc = [
            s.id
            for s in (
                await store.list_sessions(
                    SessionQuery(order_by=SessionOrder.CREATED_AT_DESC, limit=100)
                )
            ).sessions
        ]
        assert asc == ["sess_0", "sess_1", "sess_2", "sess_3", "sess_4"]
        assert desc == ["sess_3", "sess_4", "sess_0", "sess_1", "sess_2"]
        assert asc != desc

    asyncio.run(run())


def test_inmemory_cursor_survives_mid_stream_insert() -> None:
    # A row inserted AFTER the cursor position (but before the end) must appear on a
    # later page — not be skipped. Forces created_at so the insert lands mid-stream.
    store = InMemorySessionStore()

    async def run() -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        for index in range(4):
            await store.create(_lifecycle_request(f"sess_{index}"), identity=_identity())
        _force_timestamps(
            store,
            {f"sess_{i}": base.replace(minute=i * 2) for i in range(4)},  # minutes 0,2,4,6
        )
        order = SessionOrder.CREATED_AT_ASC
        first = await store.list_sessions(SessionQuery(order_by=order, limit=2))
        seen = [s.id for s in first.sessions]  # sess_0(0), sess_1(2)
        # Insert a row at minute 3 — strictly after the cursor (sess_1@2), before sess_2@4.
        await store.create(_lifecycle_request("sess_mid"), identity=_identity())
        _force_timestamps(store, {"sess_mid": base.replace(minute=3)})
        cursor = first.next_cursor
        while cursor is not None:
            page = await store.list_sessions(SessionQuery(order_by=order, limit=2, cursor=cursor))
            seen.extend(s.id for s in page.sessions)
            cursor = page.next_cursor
        assert seen == ["sess_0", "sess_1", "sess_mid", "sess_2", "sess_3"]

    asyncio.run(run())


def test_session_query_rejects_cursor_with_offset() -> None:
    SessionQuery(cursor="abc")  # cursor alone is fine
    SessionQuery(offset=5)  # offset alone is fine
    with pytest.raises(ValidationError, match="cursor and a non-zero offset"):
        SessionQuery(cursor="abc", offset=5)


def test_decode_session_cursor_rejects_naive_datetimes() -> None:
    from cayu.runtime.sessions import decode_session_cursor

    # A UTC-aware cursor round-trips.
    aware = base64.urlsafe_b64encode(b'["2026-01-02T03:04:05+00:00","sess_ok"]').decode("ascii")
    value, session_id = decode_session_cursor(aware)
    assert session_id == "sess_ok"
    assert value.tzinfo is not None

    # A hand-forged cursor with a naive sort value must be rejected up front so it
    # never reaches the timezone-aware comparisons in the keyset walk.
    naive = base64.urlsafe_b64encode(b'["2026-01-02T03:04:05","sess_naive"]').decode("ascii")
    with pytest.raises(ValueError, match="Invalid session cursor."):
        decode_session_cursor(naive)


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_delete_session_guards_in_flight_status(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_running",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.update_status("sess_running", SessionStatus.RUNNING)

        # A running session cannot be deleted; the status guard is evaluated
        # atomically inside the delete so a concurrent transition cannot slip in.
        with pytest.raises(ValueError, match="Cannot delete a session while it is"):
            await store.delete_session("sess_running")
        assert await store.load("sess_running") is not None

        # Deleting a missing session stays an idempotent no-op.
        await store.delete_session("sess_absent")

        # Once interrupted, delete succeeds.
        await store.update_status("sess_running", SessionStatus.INTERRUPTED)
        await store.delete_session("sess_running")
        assert await store.load("sess_running") is None
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_stores_update_status_delegates_to_transition_machine(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_update_status",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        # update_status is the unconditional setter: every source status is allowed,
        # including moving "backwards" from RUNNING to PENDING.
        running = await store.update_status("sess_update_status", SessionStatus.RUNNING)
        assert running.status == SessionStatus.RUNNING
        back = await store.update_status("sess_update_status", SessionStatus.PENDING)
        assert back.status == SessionStatus.PENDING

        # A missing session still surfaces as KeyError through the transition path.
        with pytest.raises(KeyError):
            await store.update_status("sess_missing", SessionStatus.RUNNING)
        await _close_store(store)

    asyncio.run(run_store_operations())


def test_in_memory_store_event_indexes_stay_consistent_across_delete() -> None:
    # The InMemory store keeps per-session and per-type indexes so summaries and
    # scoped/budget queries stop scanning the global event list. Deleting a session
    # must evict its rows from every index so later scoped queries never resurrect them.
    store = InMemorySessionStore()

    async def run() -> None:
        for name in ("sess_a", "sess_b"):
            await store.create(
                RunRequest(
                    agent_name=name,
                    session_id=name,
                    causal_budget_id=f"job_{name}",
                    messages=[Message.text("user", "hi")],
                ),
                identity=_identity(),
            )
        await store.append_events(
            "sess_a",
            [
                Event(
                    id="a_started",
                    type=EventType.SESSION_STARTED,
                    session_id="sess_a",
                    agent_name="sess_a",
                    timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                ),
                Event(
                    id="a_model",
                    type=EventType.MODEL_COMPLETED,
                    session_id="sess_a",
                    agent_name="sess_a",
                    timestamp=datetime(2026, 1, 1, 12, 1, tzinfo=UTC),
                    payload={"finish_reason": "stop"},
                ),
            ],
        )
        await store.append_event(
            "sess_b",
            Event(
                id="b_model",
                type=EventType.MODEL_COMPLETED,
                session_id="sess_b",
                agent_name="sess_b",
                timestamp=datetime(2026, 1, 1, 12, 2, tzinfo=UTC),
                payload={"finish_reason": "stop"},
            ),
        )

        # Per-session index feeds summarize_events and session_id/session_ids queries.
        summary = await store.summarize_events("sess_a")
        assert summary.total_events == 2
        session_a = await store.query_events(EventQuery(session_id="sess_a"))
        assert [record.event.id for record in session_a] == ["a_started", "a_model"]
        merged = await store.query_events(EventQuery(session_ids=("sess_b", "sess_a"), limit=10))
        assert [record.event.id for record in merged] == ["a_started", "a_model", "b_model"]

        # Per-type index feeds the model-completed budget read path.
        model_completed = await store.query_events(
            EventQuery(event_type=EventType.MODEL_COMPLETED, limit=10)
        )
        assert [record.event.id for record in model_completed] == ["a_model", "b_model"]

        # Internal indexes are populated for both sessions before delete.
        assert set(store._session_event_records) == {"sess_a", "sess_b"}
        assert set(store._type_event_records) == {"model.completed", "session.started"}

        await store.delete_session("sess_a")

        # Deleted session is evicted from both indexes; empty type buckets are dropped.
        assert set(store._session_event_records) == {"sess_b"}
        assert set(store._type_event_records) == {"model.completed"}
        remaining_model = await store.query_events(
            EventQuery(event_type=EventType.MODEL_COMPLETED, limit=10)
        )
        assert [record.event.id for record in remaining_model] == ["b_model"]
        remaining_merged = await store.query_events(
            EventQuery(session_ids=("sess_a", "sess_b"), limit=10)
        )
        assert [record.event.id for record in remaining_merged] == ["b_model"]
        assert await store.query_events(EventQuery(session_id="sess_a")) == []
        await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_store_transcripts_do_not_alias_caller_messages(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run() -> None:
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_isolation",
                messages=[Message.text("user", "hello")],
            ),
            identity=_identity(),
        )

        appended = Message.tool_call(
            tool_call_id="call_1",
            tool_name="echo",
            arguments={"nested": {"value": "original"}},
        )
        await store.append_transcript_messages(session.id, [appended])

        # Producer-side: mutating the appended message afterwards must not
        # rewrite stored history.
        appended_call = appended.content[0]
        assert isinstance(appended_call, ToolCallPart)
        appended_call.arguments["nested"]["value"] = "producer-mutated"
        loaded = await store.load_transcript(session.id)
        call = loaded[-1].content[0]
        assert isinstance(call, ToolCallPart)
        assert call.arguments == {"nested": {"value": "original"}}

        # Consumer-side: mutating a loaded message must not corrupt the store.
        call.arguments["nested"]["value"] = "consumer-mutated"
        reloaded = await store.load_transcript(session.id)
        call = reloaded[-1].content[0]
        assert isinstance(call, ToolCallPart)
        assert call.arguments == {"nested": {"value": "original"}}
        await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_query_transcript_records_do_not_alias_stored_messages(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run() -> None:
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_query_isolation",
                messages=[Message.text("user", "hello")],
            ),
            identity=_identity(),
        )
        await store.append_transcript_messages(
            session.id,
            [
                Message(
                    role="assistant",
                    content=(
                        ThinkingPart(
                            text="reason",
                            provider_state={"nested": {"value": "original"}},
                        ),
                        ToolCallPart(
                            tool_call_id="call_1",
                            tool_name="echo",
                            arguments={"nested": {"value": "original"}},
                        ),
                    ),
                )
            ],
        )

        page = await store.query_transcript(
            TranscriptQuery(session_id=session.id, include_thinking=True)
        )
        thinking = page.records[-1].message.content[0]
        assert isinstance(thinking, ThinkingPart)
        assert thinking.provider_state is not None
        thinking.provider_state["nested"]["value"] = "mutated"

        repage = await store.query_transcript(
            TranscriptQuery(session_id=session.id, include_thinking=True)
        )
        thinking = repage.records[-1].message.content[0]
        assert isinstance(thinking, ThinkingPart)
        assert thinking.provider_state == {"nested": {"value": "original"}}

        # The thinking-stripped path isolates via the filter's rebuild — a
        # surviving tool-call payload must not alias stored state either.
        stripped = await store.query_transcript(
            TranscriptQuery(session_id=session.id, include_thinking=False)
        )
        call = stripped.records[-1].message.content[0]
        assert isinstance(call, ToolCallPart)
        call.arguments["nested"]["value"] = "mutated"

        restripped = await store.query_transcript(
            TranscriptQuery(session_id=session.id, include_thinking=False)
        )
        call = restripped.records[-1].message.content[0]
        assert isinstance(call, ToolCallPart)
        assert call.arguments == {"nested": {"value": "original"}}
        await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_fork_transcripts_do_not_alias_source_transcripts(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run() -> None:
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_fork_isolation_source",
                messages=[Message.text("user", "hello")],
            ),
            identity=_identity(),
        )
        await store.append_transcript_messages(
            source.id,
            [
                Message.tool_call(
                    tool_call_id="call_1",
                    tool_name="echo",
                    arguments={"nested": {"value": "original"}},
                )
            ],
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)

        fork = await store.create_fork(
            source_session_id=source.id,
            fork=Session(
                id="sess_fork_isolation_child",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
                parent_session_id=source.id,
                status=SessionStatus.COMPLETED,
            ),
            source_statuses={SessionStatus.COMPLETED},
            transcript_cursor=None,
            checkpoint_transform=None,
        )

        fork_transcript = await store.load_transcript(fork.id)
        call = fork_transcript[-1].content[0]
        assert isinstance(call, ToolCallPart)
        call.arguments["nested"]["value"] = "mutated-via-fork"

        source_transcript = await store.load_transcript(source.id)
        call = source_transcript[-1].content[0]
        assert isinstance(call, ToolCallPart)
        assert call.arguments == {"nested": {"value": "original"}}
        await _close_store(store)

    asyncio.run(run())


def _make_store(store_factory: StoreFactory, tmp_path) -> SessionStore:
    if store_factory is SQLiteSessionStore:
        return SQLiteSessionStore(tmp_path / "sessions.sqlite")
    return store_factory()


async def _close_store(store: SessionStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()
