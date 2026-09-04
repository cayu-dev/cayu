"""Store-atomic terminalization of a proven pristine interrupted invocation.

This is deliberately an allowlist: unfamiliar evidence requires ordinary exact-
profile recovery. No executable registration is consulted or reconstructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Any

from cayu._validation import canonical_durable_json_bytes
from cayu.core.events import Event, EventType, event_with_runtime_envelope_authority
from cayu.core.messages import Message
from cayu.runtime._durable_operation_ownership import DurableOperationOwnership
from cayu.runtime._invocation_lifecycle import (
    ReleaseInvocationCommand,
    checkpoint_with_invocation_lifecycle_receipt,
    require_released_invocation_command_authority,
)
from cayu.runtime._invocation_terminal_decision import (
    InvocationTerminalOutcome,
    checkpoint_after_invocation_terminal_decision,
    invocation_terminal_decision_from_checkpoint,
)
from cayu.runtime.checkpoints import decode_runtime_checkpoint
from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint
from cayu.runtime.interactions import InteractionStatus, InteractionSummaryEvidence
from cayu.runtime.sessions import (
    ZERO_WORK_INTERRUPTION_OPERATION_KEY,
    InteractionTransitionSpec,
    Session,
    SessionExecutionSource,
    SessionStatus,
    _incomplete_recovery_claim_from_checkpoint,
    _interaction_transition_receipt_record,
    _interaction_transition_storage_key,
    _invocation_terminal_event_storage_key,
    _load_interaction_transition_receipt,
    _load_invocation_terminal_event_receipt,
)

RECEIPT_KEY = ZERO_WORK_INTERRUPTION_OPERATION_KEY
MAX_EVIDENCE_ITEMS = 16
_ALLOWED_CHECKPOINT_KEYS = frozenset(
    {
        "checkpoint_schema_version",
        "active_invocation_execution_profile",
        "invocation_lifecycle_receipt",
        "invocation_terminal_decision",
        "pending_session_interrupt",
        "pending_interruption_cascade",
        "recovery_plan_execution",
        "incomplete_session_recovery_claim",
    }
)


@dataclass(frozen=True)
class ZeroWorkInterruptionRequest:
    session: Session
    checkpoint: dict[str, Any] | None
    inactive_for_seconds: int | None
    commit: bool = False
    recovery_ownership: DurableOperationOwnership | None = None

    def __post_init__(self) -> None:
        if type(self.session) is not Session or type(self.commit) is not bool:
            raise TypeError("Zero-work recovery requires exact session and commit values.")
        if self.inactive_for_seconds is not None and (
            type(self.inactive_for_seconds) is not int or self.inactive_for_seconds < 0
        ):
            raise ValueError("Recovery inactivity must be a nonnegative duration.")
        from cayu._validation import copy_durable_json_object

        object.__setattr__(self, "session", self.session.model_copy(deep=True))
        if self.checkpoint is not None:
            object.__setattr__(
                self, "checkpoint", copy_durable_json_object(self.checkpoint, "checkpoint")
            )


@dataclass(frozen=True)
class ZeroWorkInterruptionPublication:
    session: Session
    checkpoint: dict[str, Any]
    events: tuple[Event, ...]
    operations: dict[str, dict[str, Any]]
    replayed: bool = False


def prepare_zero_work_interruption(
    request: ZeroWorkInterruptionRequest,
    *,
    session: Session,
    checkpoint: dict[str, Any] | None,
    events: list[Event],
    messages: list[Message],
    operations: dict[str, dict[str, Any]],
    blocked: bool,
    now: datetime,
) -> ZeroWorkInterruptionPublication | None:
    """Called only with a bounded snapshot under the backend's writer lock."""
    if (
        blocked
        or session.id != request.session.id
        or session.instance_id != request.session.instance_id
    ):
        return None
    prior = operations.get(RECEIPT_KEY)
    if prior is not None:
        material = {key: value for key, value in prior.items() if key != "record_digest"}
        if (
            prior.get("version") != 1
            or prior.get("record_digest")
            != sha256(canonical_durable_json_bytes(material, "receipt")).hexdigest()
        ):
            raise RuntimeError("Zero-work interruption receipt is invalid.")
        # The complete exact event material, incarnation and terminal epoch are
        # checked even on acknowledgement-loss replay. A successor cannot be
        # mistaken for the old terminalization.
        if (
            prior.get("session_instance_id") != session.instance_id
            or prior.get("terminal_run_epoch") != session.run_epoch
            or session.status is not SessionStatus.INTERRUPTED
            or request.session.run_epoch not in {session.run_epoch, session.run_epoch - 1}
        ):
            return None
        source_digest = sha256(
            canonical_durable_json_bytes(request.checkpoint, "checkpoint")
        ).hexdigest()
        if (
            request.session.run_epoch == session.run_epoch - 1
            and prior.get("source_checkpoint_sha256") != source_digest
            and canonical_durable_json_bytes(request.checkpoint, "checkpoint")
            != canonical_durable_json_bytes(checkpoint, "checkpoint")
        ):
            return None
        active = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if active is None:
            raise RuntimeError("Zero-work interruption lost its released invocation.")
        require_released_invocation_command_authority(
            session,
            checkpoint,
            session_id=session.id,
            session_instance_id=session.instance_id,
            active_profile=active,
        )
        replay_checkpoint_keys = {
            "checkpoint_schema_version",
            "active_invocation_execution_profile",
            "invocation_lifecycle_receipt",
            "settled_invocation_terminal_decision",
            "recovery_plan_execution",
        }
        if not checkpoint or set(checkpoint) - replay_checkpoint_keys:
            return None
        terminal_events = tuple(Event.model_validate(e) for e in prior["events"])
        if len(terminal_events) != 2 or set(operations) != {
            RECEIPT_KEY,
            _interaction_transition_storage_key(terminal_events[0].id),
            _invocation_terminal_event_storage_key(terminal_events[1].id),
        }:
            return None
        if any(
            event.type
            not in {
                EventType.INTERACTION_STARTED,
                EventType.SESSION_STARTED,
                EventType.INTERACTION_INTERRUPTED,
                EventType.SESSION_INTERRUPTED,
                EventType.RECOVERY_PLAN_ITEM_EXECUTED,
            }
            for event in events
        ):
            return None
        transition = _load_interaction_transition_receipt(
            operations[_interaction_transition_storage_key(terminal_events[0].id)]
        )
        terminal_receipt = _load_invocation_terminal_event_receipt(
            operations[_invocation_terminal_event_storage_key(terminal_events[1].id)]
        )
        if (
            transition.event != terminal_events[0]
            or transition.terminal_event != terminal_events[1]
            or transition.invocation_session_instance_id != session.instance_id
            or transition.invocation_active_profile != active
            or terminal_receipt.event != terminal_events[1]
            or terminal_receipt.session_instance_id != session.instance_id
            or terminal_receipt.active_profile != active
        ):
            raise RuntimeError("Zero-work interruption receipts have conflicting authority.")
        durable = {e.id: e.model_dump(mode="json") for e in events}
        if any(durable.get(e.id) != e.model_dump(mode="json") for e in terminal_events):
            raise RuntimeError("Zero-work interruption receipt lost its terminal evidence.")
        return ZeroWorkInterruptionPublication(session, checkpoint or {}, terminal_events, {}, True)
    if (
        blocked
        or operations
        or len(events) > MAX_EVIDENCE_ITEMS
        or len(messages) > MAX_EVIDENCE_ITEMS
        or session.status is not SessionStatus.INTERRUPTING
        or session != request.session
        or canonical_durable_json_bytes(checkpoint, "checkpoint")
        != canonical_durable_json_bytes(request.checkpoint, "checkpoint")
        or session.invocation.source
        not in {SessionExecutionSource.SDK_RUN, SessionExecutionSource.HTTP_RUN}
        or session.parent_session_id is not None
    ):
        return None
    if request.inactive_for_seconds is not None and session.last_activity_at > now - timedelta(
        seconds=request.inactive_for_seconds
    ):
        return None
    # Do not let a historical upcaster discard unknown work before the proof.
    if not checkpoint or set(checkpoint) - _ALLOWED_CHECKPOINT_KEYS:
        return None
    checkpoint = decode_runtime_checkpoint(checkpoint, session_id=session.id)
    if not checkpoint or set(checkpoint) - _ALLOWED_CHECKPOINT_KEYS:
        return None
    prior_claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
    if prior_claim is not None and prior_claim[1] > now:
        return None
    marker = checkpoint.get("recovery_plan_execution")
    if marker is not None:
        if type(marker) is not dict or request.recovery_ownership is None:
            return None
        current_ownership = DurableOperationOwnership.model_validate(marker.get("ownership"))
        expected = request.recovery_ownership
        if (
            current_ownership.claim_id != expected.claim_id
            or current_ownership.generation != expected.generation
            or current_ownership.owner_id != expected.owner_id
            or current_ownership.operation_id != expected.operation_id
            or current_ownership.state.value != "active"
            or current_ownership.lease_expires_at is None
            or current_ownership.lease_expires_at <= now
        ):
            return None
    elif request.recovery_ownership is not None:
        return None
    active = active_invocation_execution_profile_from_checkpoint(checkpoint)
    decision = invocation_terminal_decision_from_checkpoint(checkpoint)
    intent = checkpoint.get("pending_session_interrupt")
    if (
        active is None
        or decision is None
        or type(intent) is not dict
        or active.session_id != session.id
        or active.run_epoch != session.run_epoch
        or decision.outcome is not InvocationTerminalOutcome.INTERRUPTED
        or decision.session_id != session.id
        or decision.session_instance_id != session.instance_id
        or decision.run_epoch != session.run_epoch
        or decision.execution_profile_fingerprint != active.profile.fingerprint
        or decision.profile_interaction_id != active.interaction_id
        or decision.interaction_id != active.interaction_id
        or decision.interaction_event_id is None
        or decision.task_id is not None
        or decision.terminal_payload != intent
        or decision.interruption_request_id != intent.get("interruption_request_id")
        or intent.get("interruption_type") != "operator_requested"
        or "provider_cancellation_failures" in intent
    ):
        return None
    ledger = checkpoint.get("invocation_lifecycle_receipt", {})
    receipts = ledger.get("receipts", []) if type(ledger) is dict else []
    if not receipts or any(r.get("kind") not in {"create", "admit", "rebind"} for r in receipts):
        return None
    if any(m.role not in {"system", "user"} for m in messages):
        return None
    starts = [e for e in events if e.type == EventType.INTERACTION_STARTED]
    if (
        len(starts) != 1
        or starts[0].interaction_id != active.interaction_id
        or any(
            e.type not in {EventType.INTERACTION_STARTED, EventType.SESSION_STARTED} for e in events
        )
    ):
        return None
    start_evidence = InteractionSummaryEvidence.model_validate(starts[0].payload)
    if (
        len(events) > 2
        or start_evidence.status is not InteractionStatus.ACTIVE
        or start_evidence.start_event_id != starts[0].id
        or start_evidence.model_step_count != 0
        or start_evidence.tool_call_count != 0
        or start_evidence.targeted_tool_grant_count is not None
        or start_evidence.queued_interaction_profile_handoff is not None
        or decision.observed_at < start_evidence.started_at
    ):
        return None
    user_indices = [index for index, message in enumerate(messages) if message.role == "user"]
    interaction_payload = InteractionSummaryEvidence(
        status=InteractionStatus.INTERRUPTED,
        start_event_id=starts[0].id,
        start_event_sequence=start_evidence.start_event_sequence,
        source_transcript_start=user_indices[0] if user_indices else None,
        source_transcript_end=user_indices[-1] if user_indices else None,
        started_at=start_evidence.started_at,
        completed_at=decision.observed_at,
        active_duration_ms=max(
            0,
            int(
                (
                    min(session.last_activity_at, decision.observed_at) - start_evidence.started_at
                ).total_seconds()
                * 1000
            ),
        ),
        wall_duration_ms=max(
            0, int((decision.observed_at - start_evidence.started_at).total_seconds() * 1000)
        ),
    ).model_dump(mode="json")
    cascade = checkpoint.get("pending_interruption_cascade")
    if cascade is not None and (
        type(cascade) is not dict or cascade.get("interrupt_payload") != intent
    ):
        return None
    terminal_events = tuple(
        event_with_runtime_envelope_authority(
            Event(
                id=event_id,
                type=event_type,
                session_id=session.id,
                interaction_id=(
                    active.interaction_id
                    if event_type is EventType.INTERACTION_INTERRUPTED
                    else None
                ),
                agent_name=session.agent_name,
                environment_name=session.environment_name,
                timestamp=decision.observed_at,
                payload=(
                    interaction_payload
                    if event_type is EventType.INTERACTION_INTERRUPTED
                    else decision.terminal_payload
                ),
            ),
            *(
                ("session_id", "interaction_id")
                if event_type is EventType.INTERACTION_INTERRUPTED
                else ("session_id",)
            ),
        )
        for event_id, event_type in (
            (decision.interaction_event_id, EventType.INTERACTION_INTERRUPTED),
            (decision.terminal_event_id, EventType.SESSION_INTERRUPTED),
        )
    )
    updated_checkpoint = checkpoint_after_invocation_terminal_decision(
        checkpoint, expected=decision
    )
    assert updated_checkpoint is not None
    updated_checkpoint.pop("incomplete_session_recovery_claim", None)
    updated_checkpoint.pop("pending_session_interrupt", None)
    updated_checkpoint.pop("pending_interruption_cascade", None)
    # A released invocation retains its old immutable profile while the session
    # epoch advances. No owner or hook is granted executable authority.
    updated = session.model_copy(
        update={
            "status": SessionStatus.INTERRUPTED,
            "run_epoch": session.run_epoch + 1,
            "updated_at": now,
            "last_activity_at": now,
        },
        deep=True,
    )
    transition = _interaction_transition_receipt_record(
        session=updated.model_copy(update={"run_epoch": session.run_epoch}),
        event=terminal_events[0],
        from_statuses={SessionStatus.INTERRUPTING},
        to_status=SessionStatus.INTERRUPTED,
        only_if_no_queued_messages=False,
        model_completion_stage_settlement=None,
        terminal_event=terminal_events[1],
        terminal_decision=decision,
        status_changed=True,
        invocation_session_instance_id=session.instance_id,
        invocation_active_profile=active,
    )
    release = ReleaseInvocationCommand(
        session_id=session.id,
        expected_session_instance_id=session.instance_id,
        expected_run_epoch=session.run_epoch,
        expected_active_profile=active,
        settlement_transition=InteractionTransitionSpec(
            event=terminal_events[0],
            from_statuses=(SessionStatus.INTERRUPTING,),
            to_status=SessionStatus.INTERRUPTED,
            terminal_event=terminal_events[1],
            terminal_decision=decision,
        ),
    )
    updated_checkpoint = checkpoint_with_invocation_lifecycle_receipt(
        updated_checkpoint,
        release,
        active_profile=active,
        result_session=updated,
    )
    records = {
        _interaction_transition_storage_key(terminal_events[0].id): transition,
        RECEIPT_KEY: {
            "version": 1,
            "session_instance_id": session.instance_id,
            "source_run_epoch": session.run_epoch,
            "terminal_run_epoch": updated.run_epoch,
            "source_checkpoint_sha256": sha256(
                canonical_durable_json_bytes(request.checkpoint, "checkpoint")
            ).hexdigest(),
            "interruption_request_id": decision.interruption_request_id,
            "events": [e.model_dump(mode="json") for e in terminal_events],
        },
    }
    records[RECEIPT_KEY]["record_digest"] = sha256(
        canonical_durable_json_bytes(records[RECEIPT_KEY], "receipt")
    ).hexdigest()
    return ZeroWorkInterruptionPublication(updated, updated_checkpoint, terminal_events, records)
