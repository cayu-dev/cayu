"""Inspect exact terminal evidence without acquiring recovery ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cayu._validation import (
    copy_json_value,
)
from cayu.approvals.user_input import (
    AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY,
    USER_INPUT_SUPERSESSION_INTENT_KEY,
    UserInputPauseState,
    pending_user_input_interruption_payload,
    user_input_lifecycle_authority_from_checkpoint,
)
from cayu.events import (
    Event,
    EventType,
)
from cayu.providers._credential_boundary import copy_provider_cancellation_failures
from cayu.runtime import _approval_support as approval_support
from cayu.runtime._approval_support import _pending_approval_and_round_for_atomic_claim
from cayu.runtime._interruption_coordinator import (
    _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY,
)
from cayu.runtime._run_limits import (
    SessionUsageTracker,
)
from cayu.runtime._session_control import (
    SessionControl,
)
from cayu.runtime._user_input_recovery_evidence import UserInputRecoveryEvidence
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions._checkpoint_preservation import (
    _invocation_lifecycle_authority_read_scope,
)
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._terminal_evidence import (
    _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
    _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
    _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
    _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
    TERMINAL_EVIDENCE_EVENT_TYPES,
    TERMINAL_EVIDENCE_QUERY_LIMIT,
    _session_run_operation_from_checkpoint,
    _SessionRunOperation,
    classify_current_terminal_evidence,
    interruption_request_id_from_payload,
    require_interruption_event_matches_pending_marker,
)
from cayu.sessions.base import (
    SessionRuntimePublicationConflict,
    SessionStore,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.records import (
    Session,
    SessionStatus,
)
from cayu.vaults.redaction import SecretRedactor

_TERMINAL_EVENT_TYPE_BY_STATUS = {
    SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
    SessionStatus.FAILED: EventType.SESSION_FAILED,
    SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
}


def _provider_cancellation_interrupt_payload(
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return one exact reconstructed provider-cancellation interrupt marker."""

    if checkpoint is None:
        return None
    marker = checkpoint.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
    if marker is None:
        return None
    if type(marker) is not dict:
        raise ValueError("Pending session interrupt checkpoint must be an object.")
    payload = copy_json_value(marker, "pending_session_interrupt")
    failures = payload.get("provider_cancellation_failures")
    if failures is None:
        return None
    copied_failures = copy_provider_cancellation_failures(failures)
    if not copied_failures:
        raise ValueError("Provider cancellation interruption diagnostics cannot be empty.")
    interruption_type = payload.get("interruption_type")
    if type(interruption_type) is not str or interruption_type not in (
        _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
        _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
    ):
        raise ValueError("Provider cancellation interruption type is invalid.")
    if interruption_request_id_from_payload(payload) is None:
        raise ValueError("Provider cancellation interruption request identity is invalid.")
    payload["provider_cancellation_failures"] = [dict(item) for item in copied_failures]
    return payload


@dataclass(frozen=True)
class _TerminalEvidenceInspection:
    event: Event | None
    pending_interrupt_payload: dict[str, Any] | None
    pending_action_interrupt_payload: dict[str, Any] | None
    run_operation: _SessionRunOperation | None
    terminal_event_required: bool


class TerminalEvidenceReader:
    """Inspect exact terminal evidence without acquiring recovery ownership."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        session_control: SessionControl[SessionUsageTracker],
        user_input_evidence: UserInputRecoveryEvidence,
        secret_redactor: SecretRedactor,
    ) -> None:
        self._session_store = session_store
        self._session_control = session_control
        self._user_input_evidence = user_input_evidence
        self._secret_redactor = secret_redactor

    async def inspect(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
    ) -> _TerminalEvidenceInspection:
        expected_event_type = _TERMINAL_EVENT_TYPE_BY_STATUS.get(session.status)
        if expected_event_type is None:
            raise ValueError(f"Session is not terminal: {session.status}.")
        run_operation = _session_run_operation_from_checkpoint(checkpoint)

        pending_interrupt_payload: dict[str, Any] | None = None
        pending_interrupt_request_id: str | None = None
        if checkpoint is not None and _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY in checkpoint:
            marker = checkpoint[_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY]
            if type(marker) is not dict:
                raise ValueError("Pending session interrupt checkpoint must be an object.")
            pending_interrupt_payload = copy_json_value(marker, "pending_session_interrupt")
            provider_interrupt_payload = _provider_cancellation_interrupt_payload(checkpoint)
            if provider_interrupt_payload is not None:
                pending_interrupt_payload = provider_interrupt_payload
            if session.status != SessionStatus.INTERRUPTED:
                raise RuntimeError(
                    "Terminal evidence is contradictory: a non-interrupted session retains "
                    "a pending interruption marker."
                )
            pending_interrupt_request_id = interruption_request_id_from_payload(
                pending_interrupt_payload
            )
            if pending_interrupt_request_id is None:
                raise RuntimeError(
                    "Terminal evidence is not repairable: the pending interruption marker "
                    "has no stable request identity."
                )
            await self._user_input_evidence.validated_user_input_supersession_interrupt_payload(
                session=session,
                pending_interrupt_payload=pending_interrupt_payload,
            )

        pending_approval = pending_approval_reader.pending_approval_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        pending_user_input, _resolution_intent = user_input_lifecycle_authority_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            current_run_epoch=session.run_epoch,
            runtime_session=session,
        )
        pending_tool_round = pending_round_reader.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            runtime_session=session,
        )
        approval_owns_tool_round = False
        if pending_approval is not None and pending_tool_round is not None:
            _pending_approval_and_round_for_atomic_claim(
                checkpoint,
                approval_id=pending_approval.approval_id,
                tool_round_id=pending_approval.tool_round_id,
                gating_tool_call_id=pending_approval.tool_call_id,
                redactor=self._secret_redactor,
                runtime_session=session,
            )
            approval_owns_tool_round = True
        pending_actions = tuple(
            action
            for action in (
                pending_approval,
                pending_user_input,
                None if approval_owns_tool_round else pending_tool_round,
            )
            if action is not None
        )
        if len(pending_actions) > 1:
            raise RuntimeError(
                "Terminal evidence is not repairable: the checkpoint contains "
                "conflicting pending actions."
            )
        if pending_user_input is not None:
            pause_state = await self._user_input_evidence.classify_pause(
                session=session,
                checkpoint=checkpoint,
                input_id=pending_user_input.input_id,
            )
            if pause_state not in {
                UserInputPauseState.ACTIVE,
                UserInputPauseState.ANSWERING,
            }:
                raise SessionRuntimePublicationConflict(
                    "Terminal user-input evidence has ambiguous pause authority."
                )
        pending_action_interrupt_payload: dict[str, Any] | None = None
        if pending_approval is not None and session.status == SessionStatus.INTERRUPTED:
            pending_action_interrupt_payload = {
                "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                "model_step_id": pending_approval.model_step_id,
                "model_attempt_id": pending_approval.model_attempt_id,
                "tool_round_id": pending_approval.tool_round_id,
                **approval_support.bounded_pending_approval_event_payload(
                    pending_approval,
                    redactor=self._secret_redactor,
                ),
            }
        elif pending_user_input is not None and session.status == SessionStatus.INTERRUPTED:
            pending_action_interrupt_payload = {
                "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                "model_step_id": pending_user_input.model_step_id,
                "model_attempt_id": pending_user_input.model_attempt_id,
                "tool_round_id": pending_user_input.tool_round_id,
                **pending_user_input_interruption_payload(pending_user_input),
            }
        elif pending_tool_round is not None and session.status == SessionStatus.INTERRUPTED:
            pending_action_interrupt_payload = {
                **pending_rounds.pending_tool_round_identity(pending_tool_round).payload(),
                "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                "reason": "terminal_event_evidence_repaired",
                "recovered": True,
            }

        evidence_records = await self._session_store.query_events(
            EventQuery(
                session_id=session.id,
                event_types=TERMINAL_EVIDENCE_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=TERMINAL_EVIDENCE_QUERY_LIMIT,
            )
        )
        classification = classify_current_terminal_evidence(
            evidence_events=tuple(record.event for record in evidence_records),
            expected_event_type=expected_event_type,
            run_operation_id=(None if run_operation is None else run_operation.operation_id),
            interruption_request_id=pending_interrupt_request_id,
        )
        terminal_events = classification.events
        if classification.run_operation_conflict:
            raise RuntimeError(
                "Terminal evidence is contradictory: the interruption event and "
                "pending run operation have different identities."
            )
        if any(event.type != expected_event_type for event in terminal_events):
            raise RuntimeError(
                "Terminal evidence is contradictory: the durable event type does not "
                f"match session status {session.status.value}."
            )
        if len(terminal_events) > 1:
            raise RuntimeError(
                "Terminal evidence is contradictory: more than one terminal event exists "
                "for the current run."
            )

        existing_event = None if not terminal_events else terminal_events[0].model_copy(deep=True)
        exact_interrupt_marker_retained = pending_interrupt_payload is not None and (
            "provider_cancellation_failures" in pending_interrupt_payload
            or pending_interrupt_payload.get("terminal_publication_repair") is True
            or USER_INPUT_SUPERSESSION_INTENT_KEY in pending_interrupt_payload
            or AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY in pending_interrupt_payload
        )
        if existing_event is not None and exact_interrupt_marker_retained:
            require_interruption_event_matches_pending_marker(
                existing_event,
                pending_interrupt_payload,
            )
        return _TerminalEvidenceInspection(
            event=existing_event,
            pending_interrupt_payload=pending_interrupt_payload,
            pending_action_interrupt_payload=pending_action_interrupt_payload,
            run_operation=run_operation,
            terminal_event_required=(
                run_operation is not None
                or pending_interrupt_payload is not None
                or pending_action_interrupt_payload is not None
                or classification.latest_lifecycle_event_type != EventType.SESSION_FORKED
            ),
        )

    async def has_completed_queued_predecessor(self, session: Session, event: Event) -> bool:
        """Authenticate a completed interaction in an interrupted queue-bearing run."""
        if (
            session.status is not SessionStatus.INTERRUPTED
            or event.type is not EventType.INTERACTION_COMPLETED
        ):
            return False
        with _invocation_lifecycle_authority_read_scope():
            checkpoint = await self._session_store.load_checkpoint(session.id)
        active = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if (
            active is None
            or active.session_id != session.id
            or active.interaction_id != event.interaction_id
        ):
            return False
        receipt = await self._session_store.load_historical_interaction_settlement(
            session.id,
            expected_session_instance_id=session.instance_id,
            expected_event=event,
            expected_profile=active.profile,
        )
        if (
            receipt is None
            or receipt.status_changed
            or not receipt.transition.only_if_no_queued_messages
            or receipt.transition.to_status is not SessionStatus.COMPLETED
            or receipt.session.status is not SessionStatus.RUNNING
            or receipt.session.run_epoch != active.run_epoch
        ):
            return False
        from cayu.sessions._invocation_lifecycle import (
            _require_released_invocation_command_receipt,
        )

        # Session terminal events are session-scoped, not interaction-scoped.
        # The native release receipt binds this terminal session to the exact
        # predecessor interaction/profile/epoch instead.
        _require_released_invocation_command_receipt(
            session,
            checkpoint,
            session_id=session.id,
            session_instance_id=session.instance_id,
            active_profile=active,
        )
        inspection = await self.inspect(session=session, checkpoint=checkpoint)
        return (
            inspection.event is not None
            and inspection.event.type is EventType.SESSION_INTERRUPTED
            and inspection.event.interaction_id is None
            and inspection.pending_interrupt_payload is None
            and inspection.run_operation is None
        )
