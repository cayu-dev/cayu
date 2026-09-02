from __future__ import annotations

import asyncio
import base64
import sqlite3
from contextvars import Context
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError
from tests.core._workload_secret_support import FakeProvider

from cayu.core.agents import AgentSpec
from cayu.core.events import Event, EventType
from cayu.core.messages import Message, MessageRole, ProviderStatePart, TextPart
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.runtime._child_session_notifications import (
    CHILD_SESSION_NOTIFICATION_INTENT_KEY,
    ChildSessionLifecycleOccurrenceSource,
    ChildSessionLifecycleState,
    ChildSessionNotificationFreshness,
)
from cayu.runtime.app import CayuApp
from cayu.runtime.child_session_context import (
    ChildSessionContextContribution,
    ChildSessionContextContributor,
    ChildSessionContextCoverage,
    ChildSessionContextCoverageState,
    ChildSessionContextEntry,
    ChildSessionContextOccurrence,
    ChildSessionContextProjection,
    ChildSessionContextTruncationReason,
    ChildSessionResultReference,
)
from cayu.runtime.child_session_results import (
    ChildSessionResultProjection,
    ChildSessionResultUnavailable,
    project_terminal_child_session_result,
)
from cayu.runtime.context_counting import ContextCountingConfig, ContextCountingMode
from cayu.runtime.public_authority import (
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
)
from cayu.runtime.sessions import (
    LATEST_TRANSCRIPT_TEXT_MAX_PARTS,
    ChildSessionLifecycleQuery,
    InMemorySessionStore,
    ModelCompletionStageRequest,
    ResumeRequest,
    RunRequest,
    Session,
    SessionIdentity,
    SessionModelCompletionStageConflict,
    SessionRunFenced,
    SessionStatus,
    SessionStore,
    TranscriptTextReadLimitExceeded,
)
from cayu.storage import migrations as schema_migrations
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.child_sessions import ChildSessionResultTool


def _alias_codec() -> PublicAuthorityAliasCodec:
    key = base64.urlsafe_b64encode(bytes([41]) * 32).decode("ascii").rstrip("=")
    return PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(
            active_key_id="child-context-test",
            keys={"child-context-test": SecretStr(key)},
        )
    )


def _new_store(kind: str, database: Path) -> SessionStore:
    codec = _alias_codec()
    if kind == "memory":
        return InMemorySessionStore(public_authority_alias_codec=codec)
    return SQLiteSessionStore(database, public_authority_alias_codec=codec)


async def _close_store(store: SessionStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


def _message_text(message: Message) -> str:
    return "".join(part.text for part in message.content if type(part) is TextPart)


@pytest.mark.parametrize(
    "overrides",
    [
        {"state": ChildSessionLifecycleState.RUNNING},
        {"result_chars": 0},
        {"result_text": "too long", "result_chars": 8, "max_chars": 1},
        {"result_text": "short", "result_chars": 5, "max_chars": 8, "result_truncated": True},
    ],
)
def test_child_result_projection_enforces_terminal_character_bound(
    overrides: dict[str, object],
) -> None:
    codec = _alias_codec()
    material: dict[str, object] = {
        "parent_session_id": codec.encode("parent", field_name="session_id"),
        "child_session_id": codec.encode("child", field_name="session_id"),
        "terminal_occurrence_id": "cayu_child_occurrence_v1_" + "a" * 64,
        "state": ChildSessionLifecycleState.COMPLETED,
        "result_text": "result",
        "result_truncated": False,
        "result_chars": 6,
        "max_chars": 6,
    }
    material.update(overrides)
    with pytest.raises(ValidationError):
        ChildSessionResultProjection.model_validate(material)


def test_sqlite_child_lifecycle_candidate_read_uses_bounded_page_index(tmp_path: Path) -> None:
    database = tmp_path / "child-lifecycle-plan.sqlite"
    store = SQLiteSessionStore(database, public_authority_alias_codec=_alias_codec())

    async def close() -> None:
        await store.close()

    asyncio.run(close())
    connection = sqlite3.connect(database)
    try:
        plan = connection.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT child_session_id FROM cayu_child_session_lifecycle_candidates "
            "WHERE parent_session_id = ? "
            "ORDER BY priority, sort_at, child_session_id LIMIT ?",
            ("parent", 2),
        ).fetchall()
        transcript_plan = connection.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT sequence FROM cayu_transcript_messages "
            "WHERE session_id = ? AND role = ? "
            "ORDER BY session_order DESC LIMIT 1",
            ("parent", str(MessageRole.ASSISTANT)),
        ).fetchall()
    finally:
        connection.close()
    assert any("idx_cayu_child_lifecycle_candidates_page" in str(row[3]) for row in plan)
    assert any(
        "idx_cayu_transcript_messages_session_role_order" in str(row[3])
        for row in transcript_plan
    )


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_latest_transcript_text_rejects_structurally_unbounded_messages(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = _new_store(store_kind, tmp_path / "bounded-text.sqlite")
        try:
            child = await _create_session(store, "bounded-text-child")
            message = Message(
                role=MessageRole.ASSISTANT,
                content=(
                    *(
                        ProviderStatePart(provider="test", state={})
                        for _ in range(LATEST_TRANSCRIPT_TEXT_MAX_PARTS + 1)
                    ),
                    TextPart(text="outside the inspection bound"),
                ),
            )
            await store.append_transcript_messages(child.id, [message])
            with pytest.raises(TranscriptTextReadLimitExceeded, match="content-part"):
                await store.load_latest_transcript_text(
                    child.id,
                    role=MessageRole.ASSISTANT,
                    max_chars=32,
                )
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("corruption", ["table", "view", "trigger", "index"])
def test_sqlite_revision_79_rejects_same_named_schema_conflicts(
    tmp_path: Path,
    corruption: str,
) -> None:
    database = tmp_path / f"child-lifecycle-{corruption}.sqlite"
    store = SQLiteSessionStore(database, public_authority_alias_codec=_alias_codec())
    asyncio.run(store.close())
    connection = sqlite3.connect(database)
    try:
        if corruption == "table":
            connection.execute("DROP TABLE cayu_child_session_lifecycle_candidates")
            connection.execute(
                "CREATE TABLE cayu_child_session_lifecycle_candidates ("
                "child_session_id TEXT PRIMARY KEY, parent_session_id TEXT NOT NULL, "
                "priority INTEGER NOT NULL, sort_at TEXT NOT NULL)"
            )
        elif corruption == "view":
            connection.execute("DROP VIEW cayu_child_session_lifecycle_canonical")
            connection.execute(
                "CREATE VIEW cayu_child_session_lifecycle_canonical AS "
                "SELECT id AS child_session_id, parent_session_id, 1 AS priority, "
                "created_at AS sort_at FROM cayu_sessions WHERE 0"
            )
        elif corruption == "trigger":
            connection.execute("DROP TRIGGER cayu_index_child_lifecycle_event_insert")
            connection.execute(
                "CREATE TRIGGER cayu_index_child_lifecycle_event_insert "
                "AFTER INSERT ON cayu_events BEGIN SELECT 1; END"
            )
        else:
            connection.execute("DROP INDEX idx_cayu_events_child_lifecycle")
            connection.execute(
                "CREATE INDEX idx_cayu_events_child_lifecycle "
                "ON cayu_events(session_id, event_type, sequence)"
            )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="bounded child-lifecycle projection"):
        SQLiteSessionStore(
            database,
            schema_mode=schema_migrations.SchemaMode.VALIDATE,
            public_authority_alias_codec=_alias_codec(),
        )


async def _create_session(
    store: SessionStore,
    session_id: str,
    *,
    parent_session_id: str | None = None,
    status: SessionStatus = SessionStatus.PENDING,
) -> Session:
    session = await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            parent_session_id=parent_session_id,
            messages=[],
        ),
        identity=SessionIdentity(provider_name="test", model="test-model"),
    )
    if status is SessionStatus.PENDING:
        return session
    running = await store.transition_status(
        session.id,
        from_statuses={SessionStatus.PENDING},
        to_status=SessionStatus.RUNNING,
    )
    await store.append_event(
        session.id,
        Event(
            id=f"{session.id}-started",
            type=EventType.SESSION_STARTED,
            session_id=session.id,
        ),
    )
    if status is SessionStatus.RUNNING:
        return running
    terminal_event_type = {
        SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
        SessionStatus.FAILED: EventType.SESSION_FAILED,
        SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
    }[status]
    await store.append_event(
        session.id,
        Event(
            id=f"{session.id}-terminal",
            type=terminal_event_type,
            session_id=session.id,
        ),
    )
    return await store.transition_status(
        session.id,
        from_statuses={SessionStatus.RUNNING},
        to_status=status,
    )


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_child_context_projects_independent_terminal_and_running_children(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = _new_store(store_kind, tmp_path / "sessions.sqlite")
        try:
            parent = await _create_session(
                store,
                "parent",
                status=SessionStatus.RUNNING,
            )
            completed = await _create_session(
                store,
                "completed-child",
                parent_session_id=parent.id,
                status=SessionStatus.RUNNING,
            )
            await store.append_transcript_messages(
                completed.id,
                [Message.text(MessageRole.ASSISTANT, "untrusted child result")],
            )
            await store.append_event(
                completed.id,
                Event(
                    id="completed-child-terminal",
                    type=EventType.SESSION_COMPLETED,
                    session_id=completed.id,
                ),
            )
            await store.transition_status(
                completed.id,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.COMPLETED,
            )
            running = await _create_session(
                store,
                "running-child",
                parent_session_id=parent.id,
                status=SessionStatus.RUNNING,
            )

            contribution = await ChildSessionContextContributor().build(
                session_store=store,
                session=parent,
            )

            assert contribution.projection is not None
            assert contribution.message is not None
            assert contribution.message.role is MessageRole.USER
            message_text = _message_text(contribution.message)
            assert "untrusted child result" not in message_text
            assert contribution.stage_binding is not None
            entries = contribution.projection.entries
            assert [entry.state.value for entry in entries] == ["completed", "running"]
            assert entries[0].freshness == "fresh"
            assert entries[0].result_reference is not None
            assert entries[1].freshness == "current"
            assert entries[1].result_reference is None
            assert f'"child_session_id":"{completed.id}"' not in message_text
            assert f'"child_session_id":"{running.id}"' not in message_text
            assert contribution.projection.coverage.state is (
                ChildSessionContextCoverageState.COMPLETE
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_terminal_notification_consumes_only_at_dispatch_fence(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        database = tmp_path / "sessions.sqlite"
        store = _new_store(store_kind, database)
        try:
            parent = await _create_session(store, "parent", status=SessionStatus.RUNNING)
            await _create_session(
                store,
                "terminal-child",
                parent_session_id=parent.id,
                status=SessionStatus.COMPLETED,
            )
            contributor = ChildSessionContextContributor()
            first = await contributor.build(session_store=store, session=parent)
            assert first.stage_binding is not None
            stage_result = await store.prepare_model_completion_stage(
                parent.id,
                request=ModelCompletionStageRequest(
                    stage_id="mstep_child_notice:dispatch:0",
                    logical_step_id="mstep_child_notice",
                    dispatch_ordinal=0,
                    intent={
                        "interaction_id": "interaction-child-notice",
                        "model_step_id": "mstep_child_notice",
                        "model_attempt_id": "matt_child_notice",
                        "request_fingerprint": "a" * 64,
                        CHILD_SESSION_NOTIFICATION_INTENT_KEY: (
                            first.stage_binding.model_dump(mode="json")
                        ),
                    },
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=parent.run_epoch,
                expected_transcript_cursor=0,
            )

            before_dispatch = await contributor.build(session_store=store, session=parent)
            assert before_dispatch.stage_binding == first.stage_binding

            if store_kind == "sqlite":
                await _close_store(store)
                store = _new_store(store_kind, database)
                recovered_parent = await store.load(parent.id)
                assert recovered_parent is not None
                recovered = await contributor.build(
                    session_store=store,
                    session=recovered_parent,
                )
                assert recovered.stage_binding == first.stage_binding
                parent = recovered_parent

            dispatch = await store.mark_model_completion_stage_dispatched(
                parent.id,
                stage=stage_result.stage,
                consume_child_session_notifications=False,
            )
            after_pre_count_dispatch = await contributor.build(
                session_store=store,
                session=parent,
            )
            assert after_pre_count_dispatch.stage_binding == first.stage_binding
            assert (
                await store.mark_model_completion_stage_dispatched(
                    parent.id,
                    stage=stage_result.stage,
                    consume_child_session_notifications=False,
                )
            ) == dispatch
            assert (
                await store.mark_model_completion_stage_dispatched(
                    parent.id,
                    stage=stage_result.stage,
                    consume_child_session_notifications=True,
                )
            ) == dispatch

            if store_kind == "sqlite":
                await _close_store(store)
                store = _new_store(store_kind, database)
                recovered_parent = await store.load(parent.id)
                assert recovered_parent is not None
                parent = recovered_parent

            after_dispatch = await contributor.build(session_store=store, session=parent)
            assert after_dispatch.projection is None
            lifecycle = await store.query_child_session_lifecycle(
                ChildSessionLifecycleQuery(
                    parent_session_id=parent.id,
                    max_children_inspected=8,
                )
            )
            assert lifecycle.entries[0].freshness is (ChildSessionNotificationFreshness.CONSUMED)
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_consumption_does_not_cross_a_recreated_child_incarnation(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = _new_store(store_kind, tmp_path / "sessions.sqlite")
        try:
            parent = await _create_session(store, "parent", status=SessionStatus.RUNNING)
            original = await _create_session(
                store,
                "reused-child",
                parent_session_id=parent.id,
                status=SessionStatus.COMPLETED,
            )
            contributor = ChildSessionContextContributor()
            original_contribution = await contributor.build(
                session_store=store,
                session=parent,
            )
            assert original_contribution.stage_binding is not None
            prepared = await store.prepare_model_completion_stage(
                parent.id,
                request=ModelCompletionStageRequest(
                    stage_id="mstep_original_child:dispatch:0",
                    logical_step_id="mstep_original_child",
                    dispatch_ordinal=0,
                    intent={
                        "interaction_id": "interaction-original-child",
                        "model_step_id": "mstep_original_child",
                        "model_attempt_id": "matt_original_child",
                        "request_fingerprint": "e" * 64,
                        CHILD_SESSION_NOTIFICATION_INTENT_KEY: (
                            original_contribution.stage_binding.model_dump(mode="json")
                        ),
                    },
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=parent.run_epoch,
                expected_transcript_cursor=0,
            )
            await store.mark_model_completion_stage_dispatched(
                parent.id,
                stage=prepared.stage,
            )
            await store.delete_session(original.id)

            def create_replacement_task() -> asyncio.Task[Session]:
                return asyncio.create_task(
                    _create_session(
                        store,
                        original.id,
                        parent_session_id=parent.id,
                        status=SessionStatus.COMPLETED,
                    )
                )

            replacement = await Context().run(create_replacement_task)
            assert replacement.instance_id != original.instance_id
            replacement_contribution = await contributor.build(
                session_store=store,
                session=parent,
            )
            assert replacement_contribution.projection is not None
            assert len(replacement_contribution.projection.entries) == 1
            assert replacement_contribution.projection.entries[0].freshness == "fresh"
            assert replacement_contribution.stage_binding is not None
            assert replacement_contribution.stage_binding.claims[0].child_session_instance_id == (
                replacement.instance_id
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_old_fresh_terminal_is_not_starved_by_newer_consumed_children(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = _new_store(store_kind, tmp_path / "sessions.sqlite")
        try:
            parent = await _create_session(store, "parent", status=SessionStatus.RUNNING)
            old_child = await _create_session(
                store,
                "old-child",
                parent_session_id=parent.id,
                status=SessionStatus.RUNNING,
            )
            await _create_session(
                store,
                "newer-consumed-a",
                parent_session_id=parent.id,
                status=SessionStatus.COMPLETED,
            )
            await _create_session(
                store,
                "newer-consumed-b",
                parent_session_id=parent.id,
                status=SessionStatus.COMPLETED,
            )
            contributor = ChildSessionContextContributor(max_children_inspected=8)
            initial = await contributor.build(session_store=store, session=parent)
            assert initial.stage_binding is not None
            prepared = await store.prepare_model_completion_stage(
                parent.id,
                request=ModelCompletionStageRequest(
                    stage_id="mstep_consume_newer:dispatch:0",
                    logical_step_id="mstep_consume_newer",
                    dispatch_ordinal=0,
                    intent={
                        "interaction_id": "interaction-consume-newer",
                        "model_step_id": "mstep_consume_newer",
                        "model_attempt_id": "matt_consume_newer",
                        "request_fingerprint": "b" * 64,
                        CHILD_SESSION_NOTIFICATION_INTENT_KEY: (
                            initial.stage_binding.model_dump(mode="json")
                        ),
                    },
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=parent.run_epoch,
                expected_transcript_cursor=0,
            )
            await store.mark_model_completion_stage_dispatched(
                parent.id,
                stage=prepared.stage,
            )

            await store.append_event(
                old_child.id,
                Event(
                    id="old-child-terminal",
                    type=EventType.SESSION_COMPLETED,
                    session_id=old_child.id,
                ),
            )
            await store.transition_status(
                old_child.id,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.COMPLETED,
            )

            page = await store.query_child_session_lifecycle(
                ChildSessionLifecycleQuery(
                    parent_session_id=parent.id,
                    max_children_inspected=1,
                )
            )
            assert page.has_more is True
            assert len(page.entries) == 1
            assert page.entries[0].child_session_id == old_child.id
            assert page.entries[0].freshness is ChildSessionNotificationFreshness.FRESH
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_dispatch_rejects_a_superseded_terminal_occurrence(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = _new_store(store_kind, tmp_path / "sessions.sqlite")
        try:
            parent = await _create_session(store, "parent", status=SessionStatus.RUNNING)
            child = await _create_session(
                store,
                "child",
                parent_session_id=parent.id,
                status=SessionStatus.COMPLETED,
            )
            contributor = ChildSessionContextContributor()
            contribution = await contributor.build(session_store=store, session=parent)
            assert contribution.stage_binding is not None
            prepared = await store.prepare_model_completion_stage(
                parent.id,
                request=ModelCompletionStageRequest(
                    stage_id="mstep_stale_occurrence:dispatch:0",
                    logical_step_id="mstep_stale_occurrence",
                    dispatch_ordinal=0,
                    intent={
                        "interaction_id": "interaction-stale-occurrence",
                        "model_step_id": "mstep_stale_occurrence",
                        "model_attempt_id": "matt_stale_occurrence",
                        "request_fingerprint": "c" * 64,
                        CHILD_SESSION_NOTIFICATION_INTENT_KEY: (
                            contribution.stage_binding.model_dump(mode="json")
                        ),
                    },
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=parent.run_epoch,
                expected_transcript_cursor=0,
            )
            await store.append_event(
                child.id,
                Event(
                    id="child-terminal-replacement",
                    type=EventType.SESSION_COMPLETED,
                    session_id=child.id,
                ),
            )

            with pytest.raises(SessionModelCompletionStageConflict):
                await store.mark_model_completion_stage_dispatched(
                    parent.id,
                    stage=prepared.stage,
                )

            page = await store.query_child_session_lifecycle(
                ChildSessionLifecycleQuery(
                    parent_session_id=parent.id,
                    max_children_inspected=8,
                )
            )
            assert len(page.entries) == 1
            assert page.entries[0].freshness is ChildSessionNotificationFreshness.FRESH
            assert page.entries[0].occurrence.source_id == "child-terminal-replacement"
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_stale_parent_run_epoch_cannot_consume_terminal_notification(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = _new_store(store_kind, tmp_path / "sessions.sqlite")
        try:
            parent = await _create_session(store, "parent", status=SessionStatus.RUNNING)
            await _create_session(
                store,
                "child",
                parent_session_id=parent.id,
                status=SessionStatus.COMPLETED,
            )
            contribution = await ChildSessionContextContributor().build(
                session_store=store,
                session=parent,
            )
            assert contribution.stage_binding is not None
            prepared = await store.prepare_model_completion_stage(
                parent.id,
                request=ModelCompletionStageRequest(
                    stage_id="mstep_stale_owner:dispatch:0",
                    logical_step_id="mstep_stale_owner",
                    dispatch_ordinal=0,
                    intent={
                        "interaction_id": "interaction-stale-owner",
                        "model_step_id": "mstep_stale_owner",
                        "model_attempt_id": "matt_stale_owner",
                        "request_fingerprint": "d" * 64,
                        CHILD_SESSION_NOTIFICATION_INTENT_KEY: (
                            contribution.stage_binding.model_dump(mode="json")
                        ),
                    },
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=parent.run_epoch,
                expected_transcript_cursor=0,
            )
            interrupted = await store.transition_status(
                parent.id,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.INTERRUPTED,
            )
            resumed = await store.transition_status(
                parent.id,
                from_statuses={SessionStatus.INTERRUPTED},
                to_status=SessionStatus.RUNNING,
            )
            assert resumed.run_epoch > interrupted.run_epoch

            with pytest.raises((SessionRunFenced, SessionModelCompletionStageConflict)):
                await store.mark_model_completion_stage_dispatched(
                    parent.id,
                    stage=prepared.stage,
                )

            page = await store.query_child_session_lifecycle(
                ChildSessionLifecycleQuery(
                    parent_session_id=parent.id,
                    max_children_inspected=8,
                )
            )
            assert page.entries[0].freshness is ChildSessionNotificationFreshness.FRESH
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_admitted_state_and_typed_projection_limits(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = _new_store(store_kind, tmp_path / "sessions.sqlite")
        try:
            parent = await _create_session(store, "parent", status=SessionStatus.RUNNING)
            await _create_session(store, "admitted-a", parent_session_id=parent.id)
            await _create_session(store, "admitted-b", parent_session_id=parent.id)

            contribution = await ChildSessionContextContributor(
                max_children_inspected=8,
                max_entries=1,
            ).build(session_store=store, session=parent)

            assert contribution.projection is not None
            assert len(contribution.projection.entries) <= 1
            assert contribution.message is not None
            coverage = contribution.projection.coverage
            assert coverage.state is ChildSessionContextCoverageState.TRUNCATED
            assert ChildSessionContextTruncationReason.ENTRY_LIMIT in coverage.reasons
            entry = contribution.projection.entries[0]
            assert entry.state.value == "admitted"
            assert entry.occurrence.source is ChildSessionLifecycleOccurrenceSource.SESSION
            assert entry.occurrence.source_sequence is None

            byte_limited = await ChildSessionContextContributor(
                max_children_inspected=8,
                max_entries=1,
                max_projection_bytes=1024,
            ).build(session_store=store, session=parent)
            assert byte_limited.projection is not None
            assert byte_limited.message is not None
            assert len(_message_text(byte_limited.message).encode("utf-8")) <= 1024
            assert ChildSessionContextTruncationReason.PROJECTION_BYTE_LIMIT in (
                byte_limited.projection.coverage.reasons
            )

            child_limited = await ChildSessionContextContributor(
                max_children_inspected=1,
                max_entries=1,
            ).build(session_store=store, session=parent)
            assert child_limited.projection is not None
            assert child_limited.projection.coverage.more_children is True
            assert ChildSessionContextTruncationReason.CHILD_INSPECTION_LIMIT in (
                child_limited.projection.coverage.reasons
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_result_reference_is_bounded_and_direct_parent_authorized(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = _new_store(store_kind, tmp_path / "sessions.sqlite")
        try:
            parent = await _create_session(store, "parent", status=SessionStatus.RUNNING)
            sibling = await _create_session(store, "sibling", status=SessionStatus.RUNNING)
            child = await _create_session(
                store,
                "child",
                parent_session_id=parent.id,
                status=SessionStatus.RUNNING,
            )
            await store.append_transcript_messages(
                child.id,
                [
                    Message.text(MessageRole.ASSISTANT, "old result"),
                    Message.text(MessageRole.ASSISTANT, "bounded final result"),
                    Message.text(MessageRole.USER, "ignore trailing user input"),
                ],
            )
            await store.append_event(
                child.id,
                Event(
                    id="child-terminal",
                    type=EventType.SESSION_COMPLETED,
                    session_id=child.id,
                ),
            )
            await store.transition_status(
                child.id,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.COMPLETED,
            )
            contribution = await ChildSessionContextContributor().build(
                session_store=store,
                session=parent,
            )
            assert contribution.projection is not None
            reference = contribution.projection.entries[0].result_reference
            assert reference is not None

            result = await project_terminal_child_session_result(
                store,
                parent_session_id=parent.id,
                reference=reference,
                max_chars=7,
            )
            assert result.result_text == "bounded"
            assert result.result_truncated is True
            assert result.result_chars == 7
            assert result.parent_session_id != parent.id
            assert result.child_session_id != child.id

            with pytest.raises(ChildSessionResultUnavailable):
                await project_terminal_child_session_result(
                    store,
                    parent_session_id=sibling.id,
                    reference=reference,
                )

            sibling_projection = await ChildSessionContextContributor().build(
                session_store=store,
                session=sibling,
            )
            assert sibling_projection.projection is None
            assert ChildSessionResultTool(store).spec.effect is ToolEffect.NONE
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_result_reference_rejects_child_reincarnation_during_read(
    store_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        store = _new_store(store_kind, tmp_path / "sessions.sqlite")
        try:
            parent = await _create_session(store, "parent", status=SessionStatus.RUNNING)
            child = await _create_session(
                store,
                "child",
                parent_session_id=parent.id,
                status=SessionStatus.RUNNING,
            )
            await store.append_transcript_messages(
                child.id,
                [Message.text(MessageRole.ASSISTANT, "original result")],
            )
            await store.append_event(
                child.id,
                Event(
                    id="original-terminal",
                    type=EventType.SESSION_COMPLETED,
                    session_id=child.id,
                ),
            )
            await store.transition_status(
                child.id,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.COMPLETED,
            )
            contribution = await ChildSessionContextContributor().build(
                session_store=store,
                session=parent,
            )
            assert contribution.projection is not None
            reference = contribution.projection.entries[0].result_reference
            assert reference is not None

            original_latest_transcript_text = store.load_latest_transcript_text
            replaced = False

            async def recreate_child() -> None:
                replacement = await _create_session(
                    store,
                    child.id,
                    parent_session_id=parent.id,
                    status=SessionStatus.RUNNING,
                )
                await store.append_transcript_messages(
                    replacement.id,
                    [Message.text(MessageRole.ASSISTANT, "replacement result")],
                )
                await store.append_event(
                    replacement.id,
                    Event(
                        id="replacement-terminal",
                        type=EventType.SESSION_COMPLETED,
                        session_id=replacement.id,
                    ),
                )
                await store.transition_status(
                    replacement.id,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.COMPLETED,
                )

            def recreate_child_task() -> asyncio.Task[None]:
                return asyncio.create_task(recreate_child())

            async def replace_child_before_query(
                session_id: str,
                *,
                role: MessageRole,
                max_chars: int,
            ) -> tuple[str, bool] | None:
                nonlocal replaced
                if not replaced:
                    replaced = True
                    await store.delete_session(child.id)
                    await Context().run(recreate_child_task)
                return await original_latest_transcript_text(
                    session_id,
                    role=role,
                    max_chars=max_chars,
                )

            monkeypatch.setattr(
                store,
                "load_latest_transcript_text",
                replace_child_before_query,
            )
            with pytest.raises(ChildSessionResultUnavailable, match="authority changed"):
                await project_terminal_child_session_result(
                    store,
                    parent_session_id=parent.id,
                    reference=reference,
                )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_result_reference_identity_and_tool_schema_are_bounded() -> None:
    with pytest.raises(ValueError, match="canonical public session alias"):
        ChildSessionResultReference(
            child_session_id="not-an-alias",
            terminal_occurrence_id="cayu_child_occurrence_v1_" + "a" * 64,
        )
    with pytest.raises(ValueError, match="canonical child occurrence"):
        ChildSessionResultReference(
            child_session_id=_alias_codec().encode("child", field_name="session_id"),
            terminal_occurrence_id="cayu_child_occurrence_v1_" + "é" * 64,
        )
    with pytest.raises(ValueError):
        ChildSessionResultReference(
            child_session_id="x" * 257,
            terminal_occurrence_id="cayu_child_occurrence_v1_" + "a" * 64,
        )
    with pytest.raises(ValueError):
        ChildSessionResultReference(
            child_session_id="cayu_authority_v1_test_session_id_tag",
            terminal_occurrence_id="x" * 129,
        )

    schema = ChildSessionResultTool(InMemorySessionStore()).spec.input_schema
    reference_schema = schema["properties"]["reference"]["properties"]
    assert reference_schema["child_session_id"]["maxLength"] == 256
    assert reference_schema["terminal_occurrence_id"]["maxLength"] == 128


def test_child_context_models_reject_contradictory_public_projections() -> None:
    codec = _alias_codec()
    parent_id = codec.encode("parent", field_name="session_id")
    child_id = codec.encode("child", field_name="session_id")
    occurrence_id = "cayu_child_occurrence_v1_" + "a" * 64
    occurrence = ChildSessionContextOccurrence(
        id=occurrence_id,
        source=ChildSessionLifecycleOccurrenceSource.EVENT,
        source_type=str(EventType.SESSION_COMPLETED),
        source_sequence=1,
    )
    with pytest.raises(ValueError, match="containing child occurrence"):
        ChildSessionContextEntry(
            parent_session_id=parent_id,
            child_session_id=child_id,
            relationship="session_fork",
            state=ChildSessionLifecycleState.COMPLETED,
            occurrence=occurrence,
            freshness="fresh",
            terminal_result_available=True,
            result_reference=ChildSessionResultReference(
                child_session_id=codec.encode("other-child", field_name="session_id"),
                terminal_occurrence_id=occurrence_id,
            ),
        )
    with pytest.raises(ValueError, match="must be current"):
        running_occurrence = occurrence.model_copy(
            update={"source_type": str(EventType.SESSION_STARTED)}
        )
        ChildSessionContextEntry(
            parent_session_id=parent_id,
            child_session_id=child_id,
            relationship="session_fork",
            state=ChildSessionLifecycleState.RUNNING,
            occurrence=running_occurrence,
            freshness="fresh",
            terminal_result_available=False,
        )
    with pytest.raises(ValueError, match="cannot exceed inspected"):
        ChildSessionContextCoverage(
            state=ChildSessionContextCoverageState.TRUNCATED,
            inspected_child_count=1,
            rendered_entry_count=1,
            unavailable_child_count=1,
            reasons=(ChildSessionContextTruncationReason.SOURCE_OCCURRENCE_UNAVAILABLE,),
        )

    entry = ChildSessionContextEntry(
        parent_session_id=parent_id,
        child_session_id=child_id,
        relationship="session_fork",
        state=ChildSessionLifecycleState.RUNNING,
        occurrence=running_occurrence,
        freshness="current",
        terminal_result_available=False,
    )
    projection = ChildSessionContextProjection(
        parent_session_id=parent_id,
        entries=(entry,),
        coverage=ChildSessionContextCoverage(
            state=ChildSessionContextCoverageState.COMPLETE,
            inspected_child_count=1,
            rendered_entry_count=1,
            unavailable_child_count=0,
        ),
    )
    with pytest.raises(ValueError, match="exactly render"):
        ChildSessionContextContribution(
            projection=projection,
            message=Message.text(MessageRole.USER, "wrong projection"),
        )


def test_child_context_contributor_configuration_is_validated_and_immutable() -> None:
    with pytest.raises(ValueError, match="inspection bounds"):
        ChildSessionContextContributor(max_children_inspected=0)
    with pytest.raises(ValueError, match="entry bounds"):
        ChildSessionContextContributor(max_entries=17)
    with pytest.raises(ValueError, match="byte bounds"):
        ChildSessionContextContributor(max_projection_bytes=1023)

    contributor = ChildSessionContextContributor(max_entries=4)
    field_name = "max_entries"
    with pytest.raises(FrozenInstanceError):
        setattr(contributor, field_name, 16)


def test_runtime_composes_current_child_projection_only_for_model_request() -> None:
    class CreateChildrenTool(Tool):
        spec = ToolSpec(
            name="create_children",
            description="Create independent child sessions.",
            input_schema={"type": "object", "properties": {}},
            parallel_safe=False,
            effect=ToolEffect.EXTERNAL,
        )

        def __init__(self, store: SessionStore) -> None:
            super().__init__()
            self._store = store

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            await _create_session(
                self._store,
                "runtime-completed-child",
                parent_session_id=ctx.session_id,
                status=SessionStatus.COMPLETED,
            )
            await self._store.append_transcript_messages(
                "runtime-completed-child",
                [Message.text(MessageRole.ASSISTANT, "never inline this child output")],
            )
            await _create_session(
                self._store,
                "runtime-running-child",
                parent_session_id=ctx.session_id,
                status=SessionStatus.RUNNING,
            )
            return ToolResult(content="Independent children were dispatched.")

    async def run():
        store = InMemorySessionStore(public_authority_alias_codec=_alias_codec())
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call-create-children",
                        name="create_children",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("parent complete"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="test-model"),
            tools=[CreateChildrenTool(store)],
            child_session_context=ChildSessionContextContributor(),
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="runtime-parent",
                    messages=[Message.text(MessageRole.USER, "coordinate")],
                )
            )
        ]
        transcript = await store.load_transcript("runtime-parent")
        lifecycle = await store.query_child_session_lifecycle(
            ChildSessionLifecycleQuery(
                parent_session_id="runtime-parent",
                max_children_inspected=8,
            )
        )
        return provider, events, transcript, lifecycle

    provider, events, transcript, lifecycle = asyncio.run(run())

    assert events[-1].type is EventType.SESSION_COMPLETED
    assert len(provider.requests) == 2
    second_request_text = "\n".join(
        _message_text(message) for message in provider.requests[1].messages
    )
    assert '<cayu_child_session_notifications version="1">' in second_request_text
    assert '"state":"completed"' in second_request_text
    assert '"state":"running"' in second_request_text
    assert "never inline this child output" not in second_request_text
    assert all(
        '<cayu_child_session_notifications version="1">' not in _message_text(message)
        for message in transcript
    )
    assert [entry.freshness.value for entry in lifecycle.entries] == [
        "current",
        "consumed",
    ]


def test_provider_count_exposure_does_not_consume_terminal_child_notification() -> None:
    class CountingFakeProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="call-create-child",
                            name="create_child",
                            arguments={},
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [
                        ModelStreamEvent.text_delta("resumed"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                ]
            )
            self.count_requests: list[ModelRequest] = []

        async def count_input_tokens(self, request: ModelRequest):
            self.count_requests.append(request)
            return None

    class CreateChildTool(Tool):
        spec = ToolSpec(
            name="create_child",
            description="Create one completed child.",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL,
        )

        def __init__(self, store: SessionStore) -> None:
            super().__init__()
            self._store = store

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            await _create_session(
                self._store,
                "count-exposed-child",
                parent_session_id=ctx.session_id,
                status=SessionStatus.COMPLETED,
            )
            return ToolResult(content="created")

    async def run() -> tuple[
        CountingFakeProvider,
        ChildSessionNotificationFreshness,
        ChildSessionNotificationFreshness,
        str,
    ]:
        store = InMemorySessionStore(public_authority_alias_codec=_alias_codec())
        provider = CountingFakeProvider()
        app = CayuApp(
            session_store=store,
            context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="test-model"),
            tools=[CreateChildTool(store)],
            child_session_context=ChildSessionContextContributor(),
        )
        events = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="count-exposed-parent",
                messages=[Message.text(MessageRole.USER, "create it")],
            )
        )
        async for event in events:
            if len(provider.count_requests) == 2 and event.type is EventType.MODEL_STARTED:
                break
        await events.aclose()
        lifecycle = await store.query_child_session_lifecycle(
            ChildSessionLifecycleQuery(
                parent_session_id="count-exposed-parent",
                max_children_inspected=8,
            )
        )
        parent = await store.load("count-exposed-parent")
        assert parent is not None
        active_stage = await store.load_active_model_completion_stage(parent.id)
        assert parent.status is SessionStatus.INTERRUPTED
        assert active_stage is None
        _ = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=parent.id,
                    messages=[Message.text(MessageRole.USER, "continue")],
                )
            )
        ]
        after_resume = await store.query_child_session_lifecycle(
            ChildSessionLifecycleQuery(
                parent_session_id=parent.id,
                max_children_inspected=8,
            )
        )
        resumed_request_text = "\n".join(
            _message_text(message) for message in provider.requests[-1].messages
        )
        return (
            provider,
            lifecycle.entries[0].freshness,
            after_resume.entries[0].freshness,
            resumed_request_text,
        )

    provider, freshness_before_resume, freshness_after_resume, resumed_request_text = asyncio.run(
        run()
    )
    assert len(provider.requests) == 2
    assert len(provider.count_requests) == 3
    assert freshness_before_resume is ChildSessionNotificationFreshness.FRESH
    assert freshness_after_resume is ChildSessionNotificationFreshness.CONSUMED
    assert '<cayu_child_session_notifications version="1">' in resumed_request_text
