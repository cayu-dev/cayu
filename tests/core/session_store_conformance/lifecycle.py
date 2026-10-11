"""Shared session-store lifecycle contract cases."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any

import pytest
from tests.core._execution_profile_fixtures import (
    create_admitted_session,
)
from tests.core.completion_result_resolver_conformance import (
    assert_completion_result_resolver_session_publication_conformance,
)
from tests.core.session_store_conformance.support import (
    _close_store,
    _identity,
    _load_raw_session_operation_record,
    _open_store,
    _reopen_store,
    _set_raw_session_operation_record,
    _TransitionReplayConformanceProvider,
)

import cayu.sessions.base as sessions_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import (
    Event,
    EventType,
    event_with_runtime_envelope_authority,
)
from cayu.messages import (
    Message,
    MessageRole,
    ProviderStatePart,
)
from cayu.runtime._invocation_lifecycle import (
    ReleaseInvocationCommand,
    _release_invocation_command_with_cleanup_authority,
)
from cayu.runtime.authority import SessionRunFenced
from cayu.sessions import _checkpoint_preservation as checkpoint_preservation
from cayu.sessions import _interaction_receipts as interaction_receipts
from cayu.sessions._interaction_transitions import (
    InteractionTransitionSpec,
)
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalOutcome,
    build_invocation_terminal_decision,
    checkpoint_with_invocation_terminal_decision,
    invocation_terminal_decision_from_checkpoint,
    invocation_terminal_event_id,
    settled_invocation_terminal_decision_from_checkpoint,
)
from cayu.sessions._model_completion_publication import (
    ModelCompletionStageDisposition,
)
from cayu.sessions._model_stage import (
    ModelCompletionStageRequest,
    SessionModelCompletionStageConflict,
)
from cayu.sessions._operation_records import (
    SessionOperationPublication,
)
from cayu.sessions._terminal_evidence import (
    _checkpoint_with_session_run_operation,
)
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.forks import (
    fork_session_invocation,
)
from cayu.sessions.interactions import INTERACTION_LIFECYCLE_EVENT_TYPES
from cayu.sessions.messaging import (
    EnqueueSessionMessageRequest,
    SessionMessageDeliveryMode,
)
from cayu.sessions.queries import SessionOrder, SessionQuery
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.requests import (
    RunRequest,
)
from cayu.sessions.store import SessionStore
from cayu.sessions.transcript_queries import TranscriptQuery
from cayu.tools.base import (
    DurableToolOperationConflict,
)

pytestmark = pytest.mark.postgres


def test_session_store_conformance_blocks_delete_during_incomplete_recovery_claim(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_recovery_claim_delete_conformance",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            claimed_at = datetime.now(UTC)
            claim_id = "recovery-delete-conformance"
            await store.checkpoint(
                created.id,
                {
                    "incomplete_session_recovery_claim": {
                        "version": 1,
                        "claim_id": claim_id,
                        "claimed_at": claimed_at.isoformat(),
                        "claim_expires_at": (claimed_at + timedelta(minutes=5)).isoformat(),
                    }
                },
            )

            with pytest.raises(
                ValueError,
                match=f"incomplete-session recovery claim {claim_id} is active",
            ):
                await store.delete_session(created.id)
            assert await store.load(created.id) is not None

            await store.checkpoint(
                created.id,
                {
                    "incomplete_session_recovery_claim": {
                        "version": 1,
                        "claim_id": claim_id,
                        "claimed_at": (claimed_at - timedelta(minutes=10)).isoformat(),
                        "claim_expires_at": (claimed_at - timedelta(minutes=5)).isoformat(),
                    }
                },
            )
            await store.delete_session(created.id)
            assert await store.load(created.id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_fences_completion_result_event_publication(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            await assert_completion_result_resolver_session_publication_conformance(
                store,
                store_kind=session_store_case[0],
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_blocks_delete_until_budget_settlement_is_delivered(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_pending_budget_settlement_delete",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            reservation_id = "bres_" + "1" * 32
            reserved = Event(
                type=EventType.BUDGET_RESERVED,
                session_id=created.id,
                payload={"reservation_id": reservation_id},
            )
            await store.claim_budget_reservation_identity(
                reservation_id=reservation_id,
                publication_session_id=created.id,
                publication_id=reserved.id,
            )
            await store.append_event(created.id, reserved)
            reserved_claim = await store.claim_persisted_event_side_effect(
                session_id=created.id,
                event_id=reserved.id,
            )
            assert reserved_claim is not None
            await store.mark_persisted_event_side_effect_delivered(reserved_claim)

            with pytest.raises(
                ValueError,
                match="budget settlement audit event is pending",
            ):
                await store.delete_session(created.id)

            released = Event(
                type=EventType.BUDGET_RESERVATION_RELEASED,
                session_id=created.id,
                payload={"reservation_id": reservation_id},
            )
            await store.append_event(created.id, released)
            with pytest.raises(
                ValueError,
                match="budget settlement audit event is pending",
            ):
                await store.delete_session(created.id)

            released_claim = await store.claim_persisted_event_side_effect(
                session_id=created.id,
                event_id=released.id,
            )
            assert released_claim is not None
            await store.mark_persisted_event_side_effect_delivered(released_claim)
            await store.delete_session(created.id)
            assert await store.load(created.id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_blocks_delete_during_terminal_publication(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_terminal_publication_delete_conformance",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            operation_id = "terminal-publication-delete-conformance"
            running = await store.transition_status_and_checkpoint(
                created.id,
                from_statuses={SessionStatus.COMPLETED},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda current_session, checkpoint: (
                    _checkpoint_with_session_run_operation(
                        checkpoint=checkpoint,
                        current_session=current_session,
                        operation_id=operation_id,
                    )
                ),
            )
            assert running.run_epoch == completed.run_epoch + 1
            await store.update_status(created.id, SessionStatus.INTERRUPTED)

            with pytest.raises(
                ValueError,
                match=f"terminal publication {operation_id} is incomplete",
            ):
                await store.delete_session(created.id)
            assert await store.load(created.id) is not None

            await store.append_event(
                created.id,
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=created.id,
                    payload={"session_run_operation_id": operation_id},
                ),
            )
            await store.checkpoint(created.id, {})
            await store.delete_session(created.id)
            assert await store.load(created.id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_blocks_delete_until_markerless_terminal_evidence_is_safe(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            pending_interrupt = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_pending_interrupt_terminal_delete",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            await store.update_status(
                pending_interrupt.id,
                SessionStatus.INTERRUPTED,
            )
            await store.checkpoint(
                pending_interrupt.id,
                {
                    "pending_session_interrupt": {
                        "reason": "operator request",
                        "interruption_type": "operator_requested",
                        "interruption_request_id": "interrupt-delete-conformance",
                    }
                },
            )
            with pytest.raises(
                ValueError,
                match="pending interruption terminal publication is incomplete",
            ):
                await store.delete_session(pending_interrupt.id)
            await store.append_event(
                pending_interrupt.id,
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=pending_interrupt.id,
                    payload={
                        "reason": "operator request",
                        "interruption_type": "operator_requested",
                        "interruption_request_id": "interrupt-delete-conformance",
                    },
                ),
            )
            with pytest.raises(
                ValueError,
                match="pending interruption terminal publication is incomplete",
            ):
                await store.delete_session(pending_interrupt.id)
            await store.checkpoint(pending_interrupt.id, {})
            await store.delete_session(pending_interrupt.id)
            assert await store.load(pending_interrupt.id) is None

            pending_action = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_pending_action_terminal_delete",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            await store.update_status(pending_action.id, SessionStatus.INTERRUPTED)
            await store.checkpoint(
                pending_action.id,
                {"pending_tool_round": {"recovery_marker": True}},
            )
            with pytest.raises(
                ValueError,
                match="terminal publication evidence is incomplete",
            ):
                await store.delete_session(pending_action.id)
            await store.append_event(
                pending_action.id,
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=pending_action.id,
                ),
            )
            await store.delete_session(pending_action.id)
            assert await store.load(pending_action.id) is None

            markerless = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_markerless_current_terminal_delete",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            await store.update_status(markerless.id, SessionStatus.INTERRUPTED)
            await store.append_event(
                markerless.id,
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=markerless.id,
                ),
            )
            await store.update_status(markerless.id, SessionStatus.RUNNING)
            await store.append_event(
                markerless.id,
                Event(
                    type=EventType.SESSION_RESUMED,
                    session_id=markerless.id,
                ),
            )
            await store.update_status(markerless.id, SessionStatus.INTERRUPTED)
            with pytest.raises(
                ValueError,
                match="terminal publication evidence is incomplete",
            ):
                await store.delete_session(markerless.id)
            await store.append_event(
                markerless.id,
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=markerless.id,
                ),
            )
            await store.delete_session(markerless.id)
            assert await store.load(markerless.id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_blocks_fork_and_delete_with_active_model_stage(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_active_model_stage_control_plane"
            source_message = Message.text("user", "preserve the staged provider boundary")
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[source_message],
                ),
                identity=_identity(),
            )
            await store.append_transcript_messages(created.id, [source_message])
            running = await store.transition_status(
                created.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            prepared = await store.prepare_model_completion_stage(
                created.id,
                request=ModelCompletionStageRequest(
                    stage_id="control-plane-stage",
                    logical_step_id="control-plane-step",
                    dispatch_ordinal=0,
                    intent={"request_fingerprint": "control-plane"},
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=1,
            )
            interrupted = await store.update_status(
                created.id,
                SessionStatus.INTERRUPTED,
            )
            await store.append_event(
                created.id,
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=created.id,
                ),
            )
            fork = Session(
                id="sess_active_model_stage_control_plane_fork",
                agent_name=interrupted.agent_name,
                provider_name=interrupted.provider_name,
                model=interrupted.model,
                parent_session_id=interrupted.id,
                causal_budget_id=interrupted.causal_budget_id,
                invocation=fork_session_invocation(interrupted),
                status=interrupted.status,
            )

            with pytest.raises(ValueError, match="model-completion stage is active"):
                await store.create_fork(
                    source_session_id=interrupted.id,
                    fork=fork,
                    source_statuses={SessionStatus.INTERRUPTED},
                    transcript_cursor=None,
                    checkpoint_transform=None,
                    expected_source_run_epoch=interrupted.run_epoch,
                )
            with pytest.raises(ValueError, match="model-completion stage is active"):
                await store.delete_session(interrupted.id)
            assert await store.load(interrupted.id) is not None
            assert await store.load_active_model_completion_stage(interrupted.id) is not None

            await store.abandon_model_completion_stage(
                interrupted.id,
                stage_id=prepared.stage.stage_id,
                preparation_digest=prepared.stage.preparation_digest,
                expected_run_epoch=interrupted.run_epoch,
            )
            created_fork = await store.create_fork(
                source_session_id=interrupted.id,
                fork=fork,
                source_statuses={SessionStatus.INTERRUPTED},
                transcript_cursor=None,
                checkpoint_transform=None,
                expected_source_run_epoch=interrupted.run_epoch,
            )
            assert created_fork.id == fork.id
            await store.delete_session(interrupted.id)
            assert await store.load(interrupted.id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_atomically_fences_checkpoint_owner(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_atomic_checkpoint_fence_conformance",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            original_checkpoint = {"owner": "expired", "preserved": {"value": 1}}
            await store.checkpoint(created.id, original_checkpoint)

            def replace_owner(
                current: Session,
                checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any]:
                assert current.run_epoch == completed.run_epoch
                assert checkpoint == original_checkpoint
                assert checkpoint is not None
                updated = dict(checkpoint)
                updated["owner"] = "replacement"
                return updated

            fenced = await store.fence_run_and_transform_checkpoint(
                created.id,
                statuses={SessionStatus.COMPLETED},
                checkpoint_transform=replace_owner,
            )
            assert fenced.run_epoch == completed.run_epoch + 1
            persisted = await store.load(created.id)
            assert persisted is not None
            assert persisted.run_epoch == fenced.run_epoch
            assert await store.load_checkpoint(created.id) == {
                "owner": "replacement",
                "preserved": {"value": 1},
            }
            await store.release_run_fence(created.id)

            before_rejected_fence = await store.load(created.id)
            before_rejected_checkpoint = await store.load_checkpoint(created.id)
            assert before_rejected_fence is not None

            def reject_fence(
                _current: Session,
                _checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any]:
                raise RuntimeError("checkpoint owner changed")

            with pytest.raises(RuntimeError, match="checkpoint owner changed"):
                await store.fence_run_and_transform_checkpoint(
                    created.id,
                    statuses={SessionStatus.COMPLETED},
                    checkpoint_transform=reject_fence,
                )
            assert await store.load(created.id) == before_rejected_fence
            assert await store.load_checkpoint(created.id) == before_rejected_checkpoint

            def cancel_fence(
                _current: Session,
                _checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any]:
                raise asyncio.CancelledError("cancel atomic fence")

            with pytest.raises(asyncio.CancelledError, match="cancel atomic fence"):
                await store.fence_run_and_transform_checkpoint(
                    created.id,
                    statuses={SessionStatus.COMPLETED},
                    checkpoint_transform=cancel_fence,
                )
            assert await store.load(created.id) == before_rejected_fence
            assert await store.load_checkpoint(created.id) == before_rejected_checkpoint

            fenced_after_cancel = await store.fence_run_and_transform_checkpoint(
                created.id,
                statuses={SessionStatus.COMPLETED},
                checkpoint_transform=lambda _session, checkpoint: checkpoint,
            )
            assert fenced_after_cancel.run_epoch == before_rejected_fence.run_epoch + 1
            await store.release_run_fence(created.id)
            before_rejected_fence = await store.load(created.id)
            before_rejected_checkpoint = await store.load_checkpoint(created.id)
            assert before_rejected_fence is not None

            def omit_replacement(
                _current: Session,
                _checkpoint: dict[str, Any] | None,
            ) -> None:
                return None

            with pytest.raises(
                ValueError,
                match="Fenced checkpoint transform must return a checkpoint",
            ):
                await store.fence_run_and_transform_checkpoint(
                    created.id,
                    statuses={SessionStatus.COMPLETED},
                    checkpoint_transform=omit_replacement,
                )
            assert await store.load(created.id) == before_rejected_fence
            assert await store.load_checkpoint(created.id) == before_rejected_checkpoint
        finally:
            await _close_store(store)

    asyncio.run(run())


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


def test_session_store_conformance_rejects_future_checkpoint_before_marker_projection(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_future_cascade_projection"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            with checkpoint_preservation._invocation_lifecycle_authority_mutation_scope():
                await store.checkpoint(
                    session_id,
                    {
                        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION + 1,
                        "pending_interruption_cascade": {
                            "attempt_id": "must-not-be-interpreted",
                        },
                    },
                )

            with pytest.raises(CheckpointCompatibilityError) as caught:
                await CayuApp(
                    session_store=store,
                    enable_logging=False,
                ).interruption_cascade_status(session_id)

            assert caught.value.reason == "checkpoint_schema_version_too_new"
            assert "must-not-be-interpreted" not in str(caught.value)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_failed_transition_admission_is_atomic(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "sess_failed_transition_admission_atomic"
        duplicate_event_id = "evt_duplicate_transition_admission"
        try:
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            await store.checkpoint(session_id, {"state": "before"})
            existing = Event(
                id=duplicate_event_id,
                type=EventType.SESSION_STARTED,
                session_id=session_id,
            )
            await store.append_event(session_id, existing)
            original = await store.load(session_id)
            assert original is not None

            with pytest.raises(ValueError, match="Event already exists for session"):
                await store.transition_status_and_checkpoint(
                    session_id,
                    from_statuses={SessionStatus.PENDING},
                    to_status=SessionStatus.RUNNING,
                    checkpoint_transform=lambda _session, _checkpoint: {"state": "after"},
                    interaction_started_event=Event(
                        id=duplicate_event_id,
                        type=EventType.INTERACTION_STARTED,
                        session_id=session_id,
                        interaction_id="interaction-failed-transition-admission",
                    ),
                    interaction_source_messages=[Message.text("user", "must not persist")],
                )

            current = await store.load(session_id)
            assert current == original
            assert await store.load_checkpoint(session_id) == {"state": "before"}
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == [existing]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_create_atomically_claims_and_admits_first_interaction(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            source: list[Message] = []
            started = Event(
                id="evt_atomic_create_interaction",
                type=EventType.INTERACTION_STARTED,
                session_id="sess_atomic_create_interaction",
                interaction_id="interaction-atomic-create",
            )
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_atomic_create_interaction",
                    messages=source,
                ),
                identity=_identity(),
                interaction_started_event=started,
                interaction_source_messages=source,
            )

            assert session.status is SessionStatus.RUNNING
            assert session.run_epoch == 1
            store = await _reopen_store(session_store_case, store)
            assert [
                record.event.id
                for record in await store.query_events(EventQuery(session_id=session.id))
            ] == [started.id]
            assert (
                await store.query_transcript(TranscriptQuery(session_id=session.id))
            ).records == []
            deferred = await store.load_deferred_interaction_input(session.id)
            assert deferred is not None
            assert deferred.interaction_id == "interaction-atomic-create"
            assert deferred.source_messages == []
            checkpoint = await store.load_checkpoint(session.id)
            assert checkpoint == {
                CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
                "initial_transcript_pending": {
                    "version": 1,
                    "interaction_id": "interaction-atomic-create",
                },
            }
            assert await store.materialize_deferred_interaction_input(
                session.id,
                interaction_id="interaction-atomic-create",
            )
            assert await store.load_deferred_interaction_input(session.id) is None
            assert await store.load_checkpoint(session.id) == checkpoint
        finally:
            await store.release_run_fence("sess_atomic_create_interaction")
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_create_atomically_stages_pending_checkpoint(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            assert store.supports_pending_session_initial_checkpoint is True
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_atomic_pending_checkpoint",
                    messages=[Message.text("user", "prepared work")],
                ),
                identity=_identity(),
                checkpoint_transform=lambda current, checkpoint: {
                    "prepared": current.id,
                    "prior": checkpoint,
                },
            )
            assert session.status is SessionStatus.PENDING
            assert session.run_epoch == 0
            store = await _reopen_store(session_store_case, store)
            assert await store.load_checkpoint(session.id) == {
                "prepared": session.id,
                "prior": None,
            }

            def reject_create(_current: Session, _checkpoint: dict[str, Any] | None):
                raise RuntimeError("reject staged checkpoint")

            with pytest.raises(RuntimeError, match="reject staged checkpoint"):
                await store.create(
                    RunRequest(
                        agent_name="assistant",
                        session_id="sess_failed_atomic_pending_checkpoint",
                        messages=[],
                    ),
                    identity=_identity(),
                    checkpoint_transform=reject_create,
                )
            assert await store.load("sess_failed_atomic_pending_checkpoint") is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_atomically_initializes_branch_operation_records(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        source_id = f"discovery-view-source-{session_store_case[0]}"
        child_id = f"discovery-view-child-{session_store_case[0]}"
        failed_create_id = f"discovery-view-failed-create-{session_store_case[0]}"
        failed_fork_id = f"discovery-view-failed-fork-{session_store_case[0]}"
        operation_key = "branch-initial-state"

        def initialize(current: Session) -> dict[str, dict[str, Any]]:
            return {
                operation_key: {
                    "schema_version": 1,
                    "session_id": current.id,
                    "root_invocation_id": current.invocation.root_invocation_id,
                    "revision": 0,
                }
            }

        def reject(_current: Session) -> dict[str, dict[str, Any]]:
            raise RuntimeError("reject initial operation records")

        try:
            assert store.supports_atomic_session_operation_initialization is True
            source = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[],
                ),
                identity=_identity(),
                operation_initializer=initialize,
            )
            source_state = await store.load_session_operation(
                source.id,
                operation_key,
            )
            assert source_state == {
                "schema_version": 1,
                "session_id": source.id,
                "root_invocation_id": source.invocation.root_invocation_id,
                "revision": 0,
            }

            with pytest.raises(RuntimeError, match="reject initial operation records"):
                await store.create(
                    RunRequest(
                        agent_name="assistant",
                        session_id=failed_create_id,
                        messages=[],
                    ),
                    identity=_identity(),
                    operation_initializer=reject,
                )
            assert await store.load(failed_create_id) is None

            child_template = Session(
                id=child_id,
                agent_name=source.agent_name,
                provider_name=source.provider_name,
                model=source.model,
                parent_session_id=source.id,
                causal_budget_id=source.causal_budget_id,
                invocation=fork_session_invocation(source),
                status=source.status,
                metadata=source.metadata,
            )
            child = await store.create_fork(
                source_session_id=source.id,
                fork=child_template,
                source_statuses={SessionStatus.PENDING},
                transcript_cursor=None,
                checkpoint_transform=None,
                expected_source_run_epoch=source.run_epoch,
                operation_initializer=initialize,
            )
            child_state = await store.load_session_operation(
                child.id,
                operation_key,
            )
            assert child_state == {
                "schema_version": 1,
                "session_id": child.id,
                "root_invocation_id": child.invocation.root_invocation_id,
                "revision": 0,
            }
            assert child_state["session_id"] != source_state["session_id"]
            assert await store.load_session_operation(source.id, operation_key) == source_state

            failed_fork = child_template.model_copy(update={"id": failed_fork_id})
            with pytest.raises(RuntimeError, match="reject initial operation records"):
                await store.create_fork(
                    source_session_id=source.id,
                    fork=failed_fork,
                    source_statuses={SessionStatus.PENDING},
                    transcript_cursor=None,
                    checkpoint_transform=None,
                    expected_source_run_epoch=source.run_epoch,
                    operation_initializer=reject,
                )
            assert await store.load(failed_fork_id) is None

            store = await _reopen_store(session_store_case, store)
            assert (
                await store.load_session_operation(
                    source.id,
                    operation_key,
                )
                == source_state
            )
            assert (
                await store.load_session_operation(
                    child.id,
                    operation_key,
                )
                == child_state
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_serializes_operation_compare_and_set_contention(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"operation-contention-{session_store_case[0]}"
        operation_key = "tool-view-contention"
        try:
            created = await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
                operation_initializer=lambda _current: {
                    operation_key: {"revision": 0, "writer": None}
                },
            )
            expected = await store.load_session_operation(created.id, operation_key)
            assert expected == {"revision": 0, "writer": None}

            async def publish(writer: str) -> None:
                desired = {"revision": 1, "writer": writer}

                def compare_and_set(_current, checkpoint, current_record):
                    if current_record != expected:
                        raise DurableToolOperationConflict("simulated compare-and-set loss")
                    return SessionOperationPublication(
                        checkpoint={} if checkpoint is None else checkpoint,
                        operation_records={operation_key: desired},
                    )

                await store.publish_session_operation(
                    created.id,
                    idempotency_key=operation_key,
                    operation_transform=compare_and_set,
                    events=[],
                )

            outcomes = await asyncio.gather(
                publish("first"),
                publish("second"),
                return_exceptions=True,
            )
            assert sum(outcome is None for outcome in outcomes) == 1
            [conflict] = [
                outcome for outcome in outcomes if isinstance(outcome, DurableToolOperationConflict)
            ]
            assert str(conflict) == "simulated compare-and-set loss"
            winning = await store.load_session_operation(created.id, operation_key)
            assert winning in (
                {"revision": 1, "writer": "first"},
                {"revision": 1, "writer": "second"},
            )

            store = await _reopen_store(session_store_case, store)
            assert await store.load_session_operation(created.id, operation_key) == winning
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_transition_atomically_claims_and_admits_interaction(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_atomic_resume_interaction",
                    messages=[],
                ),
                identity=_identity(),
            )
            source = [Message.text("user", "resume source")]
            started = Event(
                id="evt_atomic_resume_interaction",
                type=EventType.INTERACTION_STARTED,
                session_id=session.id,
                interaction_id="interaction-atomic-resume",
            )
            transitioned = await store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda _session, checkpoint: checkpoint,
                interaction_started_event=started,
                interaction_source_messages=source,
            )

            assert transitioned.status is SessionStatus.RUNNING
            assert transitioned.run_epoch == 1
            store = await _reopen_store(session_store_case, store)
            events = await store.query_events(
                EventQuery(session_id=session.id, interaction_id="interaction-atomic-resume")
            )
            assert [record.event.id for record in events] == [started.id]
            transcript = await store.query_transcript(
                TranscriptQuery(session_id=session.id, limit=10)
            )
            assert [record.message for record in transcript.records] == source
            assert [record.interaction_id for record in transcript.records] == [
                "interaction-atomic-resume"
            ]
            assert await store.load_deferred_interaction_input(session.id) is None
        finally:
            await store.release_run_fence("sess_atomic_resume_interaction")
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_initial_transcript_publication_clears_authority_marker(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "sess_initial_transcript_authority"
        interaction_id = "interaction-initial-authority"
        source = [Message.text("user", "start")]
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=source,
                ),
                identity=_identity(),
                interaction_started_event=Event(
                    id="evt_initial_transcript_authority",
                    type=EventType.INTERACTION_STARTED,
                    session_id=session_id,
                    interaction_id=interaction_id,
                ),
                interaction_source_messages=source,
            )
            assert await store.load_checkpoint(session.id) is not None

            runtime_marker = Message(
                role=MessageRole.ASSISTANT,
                content=(
                    ProviderStatePart(
                        provider="openai",
                        state={"type": "runtime-test-marker"},
                    ),
                ),
            )
            final = [Message.text("system", "authoritative"), *source, runtime_marker]
            with pytest.raises(ValueError, match="runtime_suffix_count"):
                await store.replace_initial_transcript_messages(
                    session.id,
                    source,
                    final,
                    interaction_id=interaction_id,
                    runtime_suffix_count=len(final) + 1,
                )
            await store.replace_initial_transcript_messages(
                session.id,
                source,
                final,
                interaction_id=interaction_id,
                runtime_suffix_count=1,
            )

            store = await _reopen_store(session_store_case, store)
            assert await store.load_checkpoint(session.id) == {
                CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION
            }
            transcript = await store.query_transcript(
                TranscriptQuery(session_id=session.id, limit=10)
            )
            assert [record.message for record in transcript.records] == final
            assert [record.interaction_id for record in transcript.records] == [
                None,
                interaction_id,
                interaction_id,
            ]
        finally:
            await store.release_run_fence(session_id)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rejects_future_checkpoint_before_initial_transcript_write(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "sess_future_initial_transcript"
        interaction_id = "interaction-future-initial-transcript"
        source = [Message.text("user", "start")]
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=source,
                ),
                identity=_identity(),
                interaction_started_event=Event(
                    id="evt_future_initial_transcript",
                    type=EventType.INTERACTION_STARTED,
                    session_id=session_id,
                    interaction_id=interaction_id,
                ),
                interaction_source_messages=source,
            )
            checkpoint = await store.load_checkpoint(session.id)
            assert checkpoint is not None
            checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION + 1
            checkpoint["private_checkpoint_detail"] = "must-not-be-reported"
            with checkpoint_preservation._invocation_lifecycle_authority_mutation_scope():
                await store.checkpoint(session.id, checkpoint)

            runtime_store = CayuApp(
                session_store=store,
                enable_logging=False,
            )._runtime_session_store
            with pytest.raises(CheckpointCompatibilityError) as caught:
                await runtime_store.replace_initial_transcript_messages(
                    session.id,
                    source,
                    [Message.text("system", "authoritative"), *source],
                    interaction_id=interaction_id,
                )

            assert caught.value.reason == "checkpoint_schema_version_too_new"
            assert "must-not-be-reported" not in str(caught.value)
            assert (
                await store.query_transcript(TranscriptQuery(session_id=session.id))
            ).records == []
            deferred = await store.load_deferred_interaction_input(session.id)
            assert deferred is not None
            assert deferred.interaction_id == interaction_id
            assert await store.load_checkpoint(session.id) == checkpoint
        finally:
            await store.release_run_fence(session_id)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_transition_can_defer_continued_interaction_input(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_atomic_continue_interaction",
                    messages=[],
                ),
                identity=_identity(),
            )
            source = [Message.text("user", "continue after recovery")]
            await store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda _session, checkpoint: checkpoint,
                interaction_source_messages=source,
                continued_interaction_id="interaction-existing",
                defer_interaction_source=True,
            )

            store = await _reopen_store(session_store_case, store)
            assert await store.query_events(EventQuery(session_id=session.id)) == []
            assert (
                await store.query_transcript(TranscriptQuery(session_id=session.id))
            ).records == []
            deferred = await store.load_deferred_interaction_input(session.id)
            assert deferred is not None
            assert deferred.interaction_id == "interaction-existing"
            assert deferred.source_messages == source
            assert await store.materialize_deferred_interaction_input(
                session.id,
                interaction_id="interaction-existing",
            )
            transcript = await store.query_transcript(
                TranscriptQuery(session_id=session.id, limit=10)
            )
            assert [record.interaction_id for record in transcript.records] == [
                "interaction-existing"
            ]
        finally:
            await store.release_run_fence("sess_atomic_continue_interaction")
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_transition_extends_matching_deferred_interaction_input(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_atomic_extend_deferred_interaction",
                    messages=[],
                ),
                identity=_identity(),
            )
            first_source = [Message.text("user", "continue after recovery")]
            await store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda _session, checkpoint: checkpoint,
                interaction_source_messages=first_source,
                continued_interaction_id="interaction-existing",
                defer_interaction_source=True,
            )
            await store.update_status(session.id, SessionStatus.FAILED)
            await store.release_run_fence(session.id)
            store = await _reopen_store(session_store_case, store)

            combined_source = [
                *first_source,
                Message.text("user", "retry recovery"),
            ]
            await store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.FAILED},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda _session, checkpoint: checkpoint,
                interaction_source_messages=combined_source,
                continued_interaction_id="interaction-existing",
                defer_interaction_source=True,
            )

            store = await _reopen_store(session_store_case, store)
            deferred = await store.load_deferred_interaction_input(session.id)
            assert deferred is not None
            assert deferred.interaction_id == "interaction-existing"
            assert deferred.source_messages == combined_source
            assert (
                await store.query_transcript(TranscriptQuery(session_id=session.id))
            ).records == []
        finally:
            await store.release_run_fence("sess_atomic_extend_deferred_interaction")
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_interaction_transition_is_atomic_and_reconstructable(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_interaction_transition_conformance"
            interaction_id = "interaction-transition-conformance"
            started = Event(
                id="evt_interaction_transition_started",
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                identity=_identity(),
                interaction_started_event=started,
                interaction_source_messages=[Message.text("user", "start")],
            )
            failed = Event(
                id="evt_interaction_transition_failed",
                type=EventType.INTERACTION_FAILED,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            published = await store.publish_interaction_transition(
                session_id,
                event=failed,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.FAILED,
            )
            failed_transition = InteractionTransitionSpec(
                event=failed,
                from_statuses=(SessionStatus.RUNNING,),
                to_status=SessionStatus.FAILED,
            )
            assert published.status_changed is True
            assert published.replayed is False
            assert published.session.status is SessionStatus.FAILED
            assert published.event == failed

            loaded_receipt = await store.load_interaction_transition_receipt(
                session_id,
                transition=failed_transition,
            )
            assert loaded_receipt is not None
            assert loaded_receipt.status_changed is True
            assert loaded_receipt.replayed is True
            assert loaded_receipt.session == published.session
            assert loaded_receipt.transition == failed_transition

            store = await _reopen_store(session_store_case, store)
            replayed = await store.publish_interaction_transition(
                session_id,
                event=failed,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.FAILED,
            )
            assert replayed.status_changed is True
            assert replayed.replayed is True
            assert replayed.session.status is SessionStatus.FAILED
            assert replayed.event == failed
            records = await store.query_events(
                EventQuery(session_id=session_id, event_id=failed.id)
            )
            assert [record.event for record in records] == [failed]

            with pytest.raises(ValueError, match="different data"):
                await store.publish_interaction_transition(
                    session_id,
                    event=failed.model_copy(update={"payload": {"changed": True}}),
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.FAILED,
                )
            with pytest.raises(ValueError, match="different data"):
                await store.load_interaction_transition_receipt(
                    session_id,
                    transition=failed_transition.model_copy(
                        update={"event": failed.model_copy(update={"payload": {"changed": True}})}
                    ),
                )

            receipt_storage_key = (
                "__cayu_interaction_transition_v1__:"
                + sha256(failed.id.encode("utf-8")).hexdigest()
            )
            receipt = await _load_raw_session_operation_record(
                session_store_case,
                store,
                session_id=session_id,
                storage_key=receipt_storage_key,
            )
            assert receipt is not None
            receipt["session"]["model"] = "drifted-model"
            await _set_raw_session_operation_record(
                session_store_case,
                store,
                session_id=session_id,
                storage_key=receipt_storage_key,
                record=receipt,
            )
            with pytest.raises(
                RuntimeError,
                match="Stored interaction-transition receipt is invalid",
            ):
                await store.load_interaction_transition_receipt(
                    session_id,
                    transition=failed_transition,
                )
            with pytest.raises(
                RuntimeError,
                match="Stored interaction-transition receipt is invalid",
            ):
                await store.publish_interaction_transition(
                    session_id,
                    event=failed,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.FAILED,
                )
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("runtime_envelope", [True, False])
def test_session_store_conformance_terminal_decision_publishes_exact_event_pair_atomically(
    session_store_case,
    runtime_envelope: bool,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"terminal-decision-pair-{session_store_case[0]}"
        interaction_id = f"terminal-decision-interaction-{session_store_case[0]}"
        try:
            provider = ScriptedModelProvider([])
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
            admitted = await create_admitted_session(
                store,
                request=RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                provider_name=provider.name,
                model="scripted-model",
                interaction_id=interaction_id,
                provider=provider,
                app=app,
            )
            observed_at = datetime(2026, 9, 3, 12, tzinfo=UTC)
            interruption_request_id = "terminal-decision-interrupt-request"
            interaction_event_id = invocation_terminal_event_id(
                outcome=InvocationTerminalOutcome.INTERRUPTED,
                session_id=session_id,
                session_instance_id=admitted.session.instance_id,
                run_epoch=admitted.session.run_epoch,
                interaction_id=interaction_id,
                source_id=interruption_request_id,
                event_kind="interaction",
            )
            terminal_event_id = invocation_terminal_event_id(
                outcome=InvocationTerminalOutcome.INTERRUPTED,
                session_id=session_id,
                session_instance_id=admitted.session.instance_id,
                run_epoch=admitted.session.run_epoch,
                interaction_id=interaction_id,
                source_id=interruption_request_id,
                event_kind="session",
            )
            terminal_payload = {
                "interruption_type": "operator_requested",
                "interruption_request_id": interruption_request_id,
                "reason": "stop",
            }
            decision = build_invocation_terminal_decision(
                outcome=InvocationTerminalOutcome.INTERRUPTED,
                session_id=session_id,
                session_instance_id=admitted.session.instance_id,
                run_epoch=admitted.session.run_epoch,
                profile_interaction_id=interaction_id,
                interaction_id=interaction_id,
                execution_profile_fingerprint=(
                    admitted.active_invocation_profile.profile.fingerprint
                ),
                interaction_event_id=interaction_event_id,
                terminal_event_id=terminal_event_id,
                observed_at=observed_at,
                terminal_payload=terminal_payload,
                interruption_request_id=interruption_request_id,
            )

            with checkpoint_preservation._invocation_lifecycle_authority_mutation_scope():
                await store.publish_checkpoint_and_events(
                    session_id,
                    checkpoint_transform=lambda _session, checkpoint: (
                        checkpoint_with_invocation_terminal_decision(checkpoint, decision)
                    ),
                    events=[],
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=admitted.session.run_epoch,
                )

            interaction_event = Event(
                id=interaction_event_id,
                type=EventType.INTERACTION_INTERRUPTED,
                session_id=session_id,
                interaction_id=interaction_id,
                timestamp=observed_at,
            )
            terminal_event = Event(
                id=terminal_event_id,
                type=EventType.SESSION_INTERRUPTED,
                session_id=session_id,
                timestamp=observed_at,
                payload=terminal_payload,
            )
            if runtime_envelope:
                terminal_event = event_with_runtime_envelope_authority(terminal_event, "session_id")
            published = await store.publish_interaction_transition(
                session_id,
                event=interaction_event,
                from_statuses={SessionStatus.RUNNING, SessionStatus.INTERRUPTING},
                to_status=SessionStatus.INTERRUPTED,
                terminal_event=terminal_event,
                terminal_decision=decision,
            )
            assert published.session.status is SessionStatus.INTERRUPTED
            assert published.event == interaction_event
            assert published.terminal_event == terminal_event
            assert (
                invocation_terminal_decision_from_checkpoint(
                    await store.load_checkpoint(session_id)
                )
                is None
            )
            assert (
                settled_invocation_terminal_decision_from_checkpoint(
                    await store.load_checkpoint(session_id)
                )
                == decision
            )
            receipt_storage_key = (
                "__cayu_interaction_transition_v1__:"
                + sha256(interaction_event_id.encode("utf-8")).hexdigest()
            )
            raw_receipt = await _load_raw_session_operation_record(
                session_store_case,
                store,
                session_id=session_id,
                storage_key=receipt_storage_key,
            )
            assert raw_receipt is not None
            assert raw_receipt["schema_version"] == 6
            assert raw_receipt["terminal_decision"] == decision.model_dump(mode="json")
            assert [
                record.event
                for record in await store.query_events(
                    EventQuery(session_id=session_id, order_by=EventOrder.SEQUENCE_ASC)
                )
                if record.event.id in {interaction_event_id, terminal_event_id}
            ] == [interaction_event, terminal_event]

            store = await _reopen_store(session_store_case, store)
            replayed = await store.publish_interaction_transition(
                session_id,
                event=interaction_event,
                from_statuses={SessionStatus.RUNNING, SessionStatus.INTERRUPTING},
                to_status=SessionStatus.INTERRUPTED,
                terminal_event=terminal_event,
                terminal_decision=decision,
            )
            assert replayed.replayed is True
            assert replayed.session == published.session
            assert replayed.terminal_event == terminal_event
            terminal_receipt = await _load_raw_session_operation_record(
                session_store_case,
                store,
                session_id=session_id,
                storage_key=interaction_receipts._invocation_terminal_event_storage_key(
                    terminal_event.id
                ),
            )
            if runtime_envelope:
                assert terminal_receipt is not None
                assert terminal_receipt["event"] == terminal_event.model_dump(mode="json")
                assert terminal_receipt["session_instance_id"] == admitted.session.instance_id
                assert terminal_receipt[
                    "active_profile"
                ] == admitted.active_invocation_profile.model_dump(mode="json")
            else:
                assert terminal_receipt is None

            stale_failure = build_invocation_terminal_decision(
                outcome=InvocationTerminalOutcome.FAILED,
                session_id=session_id,
                session_instance_id=admitted.session.instance_id,
                run_epoch=admitted.session.run_epoch,
                profile_interaction_id=interaction_id,
                interaction_id=interaction_id,
                execution_profile_fingerprint=(
                    admitted.active_invocation_profile.profile.fingerprint
                ),
                interaction_event_id="stale-failure:interaction_failed",
                terminal_event_id="stale-failure:session_failed",
                observed_at=observed_at,
                terminal_payload={"runtime_task_failure_id": "stale-failure"},
                task_id="stale-task",
                runtime_task_failure_id="stale-failure",
                task_error_payload={"error": "stale"},
                turn_completed_payload={"status": "failed"},
            )
            with pytest.raises(SessionRunFenced):
                await store.publish_interaction_transition(
                    session_id,
                    event=Event(
                        id="stale-failure:interaction_failed",
                        type=EventType.INTERACTION_FAILED,
                        session_id=session_id,
                        interaction_id=interaction_id,
                        timestamp=observed_at,
                    ),
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.FAILED,
                    terminal_event=Event(
                        id="stale-failure:session_failed",
                        type=EventType.SESSION_FAILED,
                        session_id=session_id,
                        timestamp=observed_at,
                        payload={"runtime_task_failure_id": "stale-failure"},
                    ),
                    terminal_decision=stale_failure,
                )
            release = _release_invocation_command_with_cleanup_authority(
                ReleaseInvocationCommand(
                    session_id=session_id,
                    expected_session_instance_id=admitted.session.instance_id,
                    expected_run_epoch=admitted.session.run_epoch,
                    expected_active_profile=admitted.active_invocation_profile,
                    terminal_session_event=terminal_event,
                )
            )
            if runtime_envelope:
                released = await store.apply_invocation_lifecycle_command(release)
                assert released.session.run_epoch == admitted.session.run_epoch + 1
                assert released.replayed is False
                replayed_release = await store.apply_invocation_lifecycle_command(release)
                assert replayed_release.replayed is True
                assert replayed_release.session == released.session
            else:
                with pytest.raises(
                    SessionRunFenced, match="exact durable terminal session evidence"
                ):
                    await store.apply_invocation_lifecycle_command(release)
        finally:
            await store.release_run_fence(session_id)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_interrupt_transition_rejects_stale_latest_interaction(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"interrupt-latest-interaction-{session_store_case[0]}"
        interaction_id = f"interrupt-latest-interaction-id-{session_store_case[0]}"
        try:
            provider = ScriptedModelProvider([])
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
            admitted = await create_admitted_session(
                store,
                request=RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                provider_name=provider.name,
                model="scripted-model",
                interaction_id=interaction_id,
                provider=provider,
                app=app,
            )
            lifecycle_records = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                )
            )
            assert len(lifecycle_records) == 1
            stale_event_id = lifecycle_records[0].event.id
            newer_event = Event(
                id=f"newer-interaction-event-{session_store_case[0]}",
                type=EventType.INTERACTION_RESUMED,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            await store.append_events(session_id, [newer_event])
            checkpoint_before = await store.load_checkpoint(session_id)

            with pytest.raises(
                SessionRunFenced,
                match="latest interaction changed",
            ):
                await store.transition_status_and_checkpoint(
                    session_id,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.INTERRUPTING,
                    checkpoint_transform=lambda _session, checkpoint: {
                        **(checkpoint or {}),
                        "must_not_commit": True,
                    },
                    expected_latest_interaction_event_id=stale_event_id,
                )

            current = await store.load(session_id)
            assert current is not None
            assert current.instance_id == admitted.session.instance_id
            assert current.status is SessionStatus.RUNNING
            assert current.run_epoch == admitted.session.run_epoch
            assert await store.load_checkpoint(session_id) == checkpoint_before
        finally:
            await store.release_run_fence(session_id)
            await _close_store(store)

    asyncio.run(run())


def test_invocation_terminal_decision_rejects_non_derived_event_identity() -> None:
    with pytest.raises(
        ValueError,
        match="event identities conflict with its authority",
    ):
        build_invocation_terminal_decision(
            outcome=InvocationTerminalOutcome.INTERRUPTED,
            session_id="terminal-decision-session",
            session_instance_id="terminal-decision-instance",
            run_epoch=1,
            profile_interaction_id="terminal-decision-interaction",
            interaction_id="terminal-decision-interaction",
            execution_profile_fingerprint="0" * 64,
            interaction_event_id="forged-interaction-event",
            terminal_event_id="forged-terminal-event",
            observed_at=datetime(2026, 9, 3, 12, tzinfo=UTC),
            terminal_payload={"reason": "stop", "session_run_operation_id": "a" * 64},
            interruption_request_id="terminal-decision-request",
        )


def test_session_store_conformance_terminal_decision_rejects_profile_drift(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"terminal-decision-profile-drift-{session_store_case[0]}"
        interaction_id = f"terminal-decision-profile-interaction-{session_store_case[0]}"
        try:
            provider = ScriptedModelProvider([])
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
            admitted = await create_admitted_session(
                store,
                request=RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                provider_name=provider.name,
                model="scripted-model",
                interaction_id=interaction_id,
                provider=provider,
                app=app,
            )
            observed_at = datetime(2026, 9, 3, 13, tzinfo=UTC)
            interruption_request_id = "terminal-decision-profile-drift-request"
            interaction_event_id = invocation_terminal_event_id(
                outcome=InvocationTerminalOutcome.INTERRUPTED,
                session_id=session_id,
                session_instance_id=admitted.session.instance_id,
                run_epoch=admitted.session.run_epoch,
                interaction_id=interaction_id,
                source_id=interruption_request_id,
                event_kind="interaction",
            )
            terminal_event_id = invocation_terminal_event_id(
                outcome=InvocationTerminalOutcome.INTERRUPTED,
                session_id=session_id,
                session_instance_id=admitted.session.instance_id,
                run_epoch=admitted.session.run_epoch,
                interaction_id=interaction_id,
                source_id=interruption_request_id,
                event_kind="session",
            )
            terminal_payload = {
                "interruption_type": "operator_requested",
                "interruption_request_id": interruption_request_id,
                "reason": "stop",
            }
            forged_decision = build_invocation_terminal_decision(
                outcome=InvocationTerminalOutcome.INTERRUPTED,
                session_id=session_id,
                session_instance_id=admitted.session.instance_id,
                run_epoch=admitted.session.run_epoch,
                profile_interaction_id=interaction_id,
                interaction_id=interaction_id,
                execution_profile_fingerprint="0" * 64,
                interaction_event_id=interaction_event_id,
                terminal_event_id=terminal_event_id,
                observed_at=observed_at,
                terminal_payload=terminal_payload,
                interruption_request_id=interruption_request_id,
            )

            with checkpoint_preservation._invocation_lifecycle_authority_mutation_scope():
                await store.publish_checkpoint_and_events(
                    session_id,
                    checkpoint_transform=lambda _session, checkpoint: (
                        checkpoint_with_invocation_terminal_decision(
                            checkpoint,
                            forged_decision,
                        )
                    ),
                    events=[],
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=admitted.session.run_epoch,
                )

            with pytest.raises(SessionRunFenced):
                await store.publish_interaction_transition(
                    session_id,
                    event=Event(
                        id=interaction_event_id,
                        type=EventType.INTERACTION_INTERRUPTED,
                        session_id=session_id,
                        interaction_id=interaction_id,
                        timestamp=observed_at,
                    ),
                    from_statuses={SessionStatus.RUNNING, SessionStatus.INTERRUPTING},
                    to_status=SessionStatus.INTERRUPTED,
                    terminal_event=Event(
                        id=terminal_event_id,
                        type=EventType.SESSION_INTERRUPTED,
                        session_id=session_id,
                        timestamp=observed_at,
                        payload=terminal_payload,
                    ),
                    terminal_decision=forged_decision,
                )

            unchanged = await store.load(session_id)
            assert unchanged is not None
            assert unchanged.status is SessionStatus.RUNNING
            assert (
                invocation_terminal_decision_from_checkpoint(
                    await store.load_checkpoint(session_id)
                )
                == forged_decision
            )
            assert not await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_types=(
                        EventType.INTERACTION_INTERRUPTED,
                        EventType.SESSION_INTERRUPTED,
                    ),
                    order_by=EventOrder.SEQUENCE_ASC,
                )
            )
        finally:
            await store.release_run_fence(session_id)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_interaction_failure_atomically_settles_model_stage(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_interaction_model_stage_settlement"
            interaction_id = "interaction-model-stage-settlement"
            stage_id = "model-step:settlement:attempt-1"
            started = Event(
                id="evt_interaction_model_stage_started",
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            running = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                identity=_identity(),
                interaction_started_event=started,
                interaction_source_messages=[Message.text("user", "start")],
            )
            staged = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=stage_id,
                    logical_step_id="model-step:settlement",
                    dispatch_ordinal=0,
                    intent={
                        "interaction_id": interaction_id,
                        "model_attempt_id": "model-attempt:settlement:1",
                        "request_fingerprint": "a" * 64,
                    },
                    reservation_ids=("budget-reservation:settlement",),
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            settlement_request = sessions_module.model_completion_stage_settlement_request(
                staged.stage,
                interaction_id=interaction_id,
                disposition=(ModelCompletionStageDisposition.PROVIDER_EFFECT_OUTCOME_UNKNOWN),
                reason_code="provider_stream_failed",
                execution_profile_fingerprint=None,
                settlement_run_epoch=running.run_epoch,
                settled_reservation_ids=("budget-reservation:settlement",),
            )
            failed = Event(
                id="evt_interaction_model_stage_failed",
                type=EventType.INTERACTION_FAILED,
                session_id=session_id,
                interaction_id=interaction_id,
            )

            with pytest.raises(SessionModelCompletionStageConflict):
                await store.publish_interaction_transition(
                    session_id,
                    event=failed,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.FAILED,
                    model_completion_stage_settlement=settlement_request.model_copy(
                        update={"stage_id": "different-stage"}
                    ),
                )
            with pytest.raises(SessionModelCompletionStageConflict):
                await store.publish_interaction_transition(
                    session_id,
                    event=failed,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.FAILED,
                    model_completion_stage_settlement=settlement_request.model_copy(
                        update={"settled_reservation_ids": ()}
                    ),
                )
            unchanged = await store.load(session_id)
            assert unchanged is not None and unchanged.status is SessionStatus.RUNNING
            assert await store.load_active_model_completion_stage(session_id) is not None
            assert (
                await store.query_events(EventQuery(session_id=session_id, event_id=failed.id))
                == []
            )

            await store.release_run_fence(session_id)
            recovered_owner = await store.load(session_id)
            assert recovered_owner is not None
            assert recovered_owner.run_epoch > staged.stage.source_run_epoch
            settlement_request = sessions_module.model_completion_stage_settlement_request(
                staged.stage,
                interaction_id=interaction_id,
                disposition=(ModelCompletionStageDisposition.PROVIDER_EFFECT_OUTCOME_UNKNOWN),
                reason_code="provider_stream_failed",
                execution_profile_fingerprint=None,
                settlement_run_epoch=recovered_owner.run_epoch,
                settled_reservation_ids=("budget-reservation:settlement",),
            )

            published = await store.publish_interaction_transition(
                session_id,
                event=failed,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.FAILED,
                model_completion_stage_settlement=settlement_request,
            )
            transition = InteractionTransitionSpec(
                event=failed,
                from_statuses=(SessionStatus.RUNNING,),
                to_status=SessionStatus.FAILED,
                model_completion_stage_settlement=settlement_request,
            )
            assert published.session.status is SessionStatus.FAILED
            assert await store.load_active_model_completion_stage(session_id) is None
            settlement = await store.load_model_completion_stage_settlement(
                session_id,
                stage_id,
            )
            assert settlement is not None
            assert settlement.disposition is (
                ModelCompletionStageDisposition.PROVIDER_EFFECT_OUTCOME_UNKNOWN
            )
            assert settlement.model_attempt_id == "model-attempt:settlement:1"
            assert settlement.interaction_id == interaction_id
            assert settlement.source_run_epoch == running.run_epoch
            assert settlement.settlement_run_epoch == recovered_owner.run_epoch
            assert settlement.settled_reservation_ids == ("budget-reservation:settlement",)

            store = await _reopen_store(session_store_case, store)
            receipt = await store.load_interaction_transition_receipt(
                session_id,
                transition=transition,
            )
            assert receipt is not None and receipt.transition == transition
            replayed = await store.publish_interaction_transition(
                session_id,
                event=failed,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.FAILED,
                model_completion_stage_settlement=settlement_request,
            )
            assert replayed.replayed is True
            assert (
                await store.load_model_completion_stage_settlement(
                    session_id,
                    stage_id,
                )
                == settlement
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_receipt_lookup_rejects_non_event_transition_conflicts(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)

        async def publish_and_reject_conflict(
            *,
            suffix: str,
            event_type: EventType,
            stored_from_statuses: set[SessionStatus],
            stored_to_status: SessionStatus,
            stored_queue_guard: bool,
            conflicting_from_statuses: tuple[SessionStatus, ...],
            conflicting_to_status: SessionStatus,
            conflicting_queue_guard: bool,
        ) -> None:
            session_id = f"sess_interaction_receipt_conflict_{suffix}"
            interaction_id = f"interaction-receipt-conflict-{suffix}"
            started = Event(
                id=f"evt_interaction_receipt_conflict_started_{suffix}",
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                identity=_identity(),
                interaction_started_event=started,
                interaction_source_messages=[Message.text("user", "start")],
            )
            event = Event(
                id=f"evt_interaction_receipt_conflict_{suffix}",
                type=event_type,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            await store.publish_interaction_transition(
                session_id,
                event=event,
                from_statuses=stored_from_statuses,
                to_status=stored_to_status,
                only_if_no_queued_messages=stored_queue_guard,
            )
            conflicting = InteractionTransitionSpec(
                event=event,
                from_statuses=conflicting_from_statuses,
                to_status=conflicting_to_status,
                only_if_no_queued_messages=conflicting_queue_guard,
            )
            with pytest.raises(ValueError, match="different data"):
                await store.load_interaction_transition_receipt(
                    session_id,
                    transition=conflicting,
                )

        try:
            await publish_and_reject_conflict(
                suffix="source_statuses",
                event_type=EventType.INTERACTION_FAILED,
                stored_from_statuses={SessionStatus.RUNNING},
                stored_to_status=SessionStatus.FAILED,
                stored_queue_guard=False,
                conflicting_from_statuses=(SessionStatus.INTERRUPTING,),
                conflicting_to_status=SessionStatus.FAILED,
                conflicting_queue_guard=False,
            )
            await publish_and_reject_conflict(
                suffix="target_status",
                event_type=EventType.INTERACTION_PAUSED,
                stored_from_statuses={SessionStatus.RUNNING},
                stored_to_status=SessionStatus.FAILED,
                stored_queue_guard=False,
                conflicting_from_statuses=(SessionStatus.RUNNING,),
                conflicting_to_status=SessionStatus.INTERRUPTED,
                conflicting_queue_guard=False,
            )
            await publish_and_reject_conflict(
                suffix="queue_guard",
                event_type=EventType.INTERACTION_COMPLETED,
                stored_from_statuses={SessionStatus.RUNNING},
                stored_to_status=SessionStatus.COMPLETED,
                stored_queue_guard=True,
                conflicting_from_statuses=(SessionStatus.RUNNING,),
                conflicting_to_status=SessionStatus.COMPLETED,
                conflicting_queue_guard=False,
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("lost_acknowledgements", [1, 2])
def test_runtime_conformance_replays_exact_interaction_transition_after_ack_loss(
    session_store_case,
    lost_acknowledgements: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = (
            f"interaction-transition-ack-loss-{lost_acknowledgements}-{session_store_case[0]}"
        )
        store_type = type(store)
        original_publish = store_type.publish_interaction_transition
        attempted_events: list[Event] = []
        remaining = lost_acknowledgements
        try:

            async def publish_then_lose_ack(self, session_id: str, **kwargs):
                nonlocal remaining
                event = kwargs["event"]
                if event.type == EventType.INTERACTION_COMPLETED:
                    attempted_events.append(event.model_copy(deep=True))
                result = await original_publish(self, session_id, **kwargs)
                if event.type == EventType.INTERACTION_COMPLETED and remaining > 0:
                    remaining -= 1
                    raise ConnectionError("interaction transition acknowledgement lost")
                return result

            monkeypatch.setattr(
                store_type,
                "publish_interaction_transition",
                publish_then_lose_ack,
            )
            provider = _TransitionReplayConformanceProvider()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))

            emitted = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "complete")],
                    )
                )
            ]

            assert emitted[-1].type == EventType.SESSION_COMPLETED
            assert remaining == 0
            assert len(attempted_events) == lost_acknowledgements + 1
            assert all(event == attempted_events[0] for event in attempted_events)
            assert len(provider.requests) == 1

            monkeypatch.setattr(
                store_type,
                "publish_interaction_transition",
                original_publish,
            )
            store = await _reopen_store(session_store_case, store)
            session = await store.load(session_id)
            durable = await store.load_events(session_id)
            assert session is not None
            assert session.status is SessionStatus.COMPLETED
            terminal_indexes = [
                index
                for index, event in enumerate(durable)
                if event.type == EventType.INTERACTION_COMPLETED
            ]
            assert len(terminal_indexes) == 1
            assert all(event.interaction_id is None for event in durable[terminal_indexes[0] + 1 :])
            assert sum(event.type == EventType.SESSION_COMPLETED for event in durable) == 1
        finally:
            await store.release_run_fence(session_id)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_interaction_receipts_do_not_bypass_run_fencing(
    session_store_case,
) -> None:
    async def fence_in_child(
        store: SessionStore,
        session_id: str,
        status: SessionStatus,
    ) -> None:
        fenced = await store.fence_stalled_run(
            session_id,
            statuses={status},
            inactive_for_seconds=0,
        )
        assert fenced is not None

    async def run() -> None:
        store = await _open_store(session_store_case)
        transition_session_id = "sess_stale_interaction_transition_replay"
        delivery_session_id = "sess_stale_interaction_delivery_replay"
        try:
            transition_interaction_id = "interaction-stale-transition"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=transition_session_id,
                    messages=[Message.text("user", "start")],
                ),
                identity=_identity(),
                interaction_started_event=Event(
                    id="evt_stale_transition_started",
                    type=EventType.INTERACTION_STARTED,
                    session_id=transition_session_id,
                    interaction_id=transition_interaction_id,
                ),
                interaction_source_messages=[Message.text("user", "start")],
            )
            failed = Event(
                id="evt_stale_transition_failed",
                type=EventType.INTERACTION_FAILED,
                session_id=transition_session_id,
                interaction_id=transition_interaction_id,
            )
            await store.publish_interaction_transition(
                transition_session_id,
                event=failed,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.FAILED,
            )
            await asyncio.create_task(
                fence_in_child(store, transition_session_id, SessionStatus.FAILED)
            )
            with pytest.raises(SessionRunFenced):
                await store.publish_interaction_transition(
                    transition_session_id,
                    event=failed,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.FAILED,
                )

            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=delivery_session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            await store.transition_status(
                delivery_session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            queued = await store.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=delivery_session_id,
                    idempotency_key="stale-delivery",
                    content="continue",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                )
            )
            delivery_id = "delivery-stale-replay"
            delivered = await store.deliver_queued_session_messages(
                delivery_session_id,
                include_on_idle=False,
                delivery_id=delivery_id,
                eligible_through=queued.message.ordering_key,
                interaction_id="interaction-stale-delivery",
                interaction_started_event=Event(
                    id="evt_stale_delivery_started",
                    type=EventType.INTERACTION_STARTED,
                    session_id=delivery_session_id,
                    interaction_id="interaction-stale-delivery",
                ),
            )
            assert delivered.messages
            await asyncio.create_task(
                fence_in_child(store, delivery_session_id, SessionStatus.RUNNING)
            )
            with pytest.raises(SessionRunFenced):
                await store.deliver_queued_session_messages(
                    delivery_session_id,
                    include_on_idle=False,
                    delivery_id=delivery_id,
                    eligible_through=queued.message.ordering_key,
                    interaction_id="interaction-stale-delivery",
                    interaction_started_event=Event(
                        id="evt_stale_delivery_started",
                        type=EventType.INTERACTION_STARTED,
                        session_id=delivery_session_id,
                        interaction_id="interaction-stale-delivery",
                    ),
                )
        finally:
            await store.release_run_fence(transition_session_id)
            await store.release_run_fence(delivery_session_id)
            await _close_store(store)

    asyncio.run(run())
