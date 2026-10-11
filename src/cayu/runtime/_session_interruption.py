"""Explicit session interruption composed below ordinary turn execution."""

from __future__ import annotations

import sys
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from cayu.approvals.tools import PendingToolApproval, resolution_actor_payload
from cayu.approvals.user_input import (
    AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY,
    USER_INPUT_SUPERSESSION_INTENT_KEY,
    AmbiguousPendingUserInput,
    PendingUserInput,
    ambiguous_pending_user_input_from_checkpoint,
    user_input_lifecycle_authority_from_checkpoint,
    user_input_supersession_intent_for,
)
from cayu.events import Event, EventType
from cayu.observability.hooks import RuntimeHookPhase
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _session_request_boundary as session_request_boundary
from cayu.runtime._durable_tool_round import _environment_name as _environment_name
from cayu.runtime._environment_lifecycle import EnvironmentLifecycle
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._interruption_coordinator import (
    _INTERRUPTIBLE_SESSION_STATUSES,
    _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY,
    BackgroundInterruptionCoordinator,
    _copy_interruption_cascade_retry_request,
    _interruption_cascade_retry_event_payload,
    interruption_cascade_suppressed,
)
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._model_completion_recovery import ModelCompletionRecovery
from cayu.runtime._recovery_ownership import RecoveryOwnership
from cayu.runtime._run_limits import SessionUsageTracker
from cayu.runtime._session_control import INTERRUPT_REQUESTED_SESSION_STATUSES, SessionControl
from cayu.runtime._session_finalization import (
    _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
    SessionFinalization,
    _checkpoint_with_pending_session_interrupt,
    _close_async_iterator,
    _collect_through_event_type,
    _runtime_interruption_event,
)
from cayu.runtime._terminal_event_publication import TerminalEventPublication
from cayu.runtime._terminal_evidence_finalization import TerminalEvidenceFinalization
from cayu.runtime._terminal_evidence_reader import TerminalEvidenceReader
from cayu.runtime._user_input_recovery_evidence import UserInputRecoveryEvidence
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions._checkpoint_mutations import SessionRuntimePublicationConflict
from cayu.sessions._checkpoint_preservation import _invocation_lifecycle_authority_mutation_scope
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._foreground_child_checkpoint import (
    ForegroundChildWait,
    foreground_child_state_from_checkpoint,
)
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalDecision,
    InvocationTerminalOutcome,
    build_invocation_terminal_decision,
    invocation_terminal_decision_from_checkpoint,
    invocation_terminal_event_id,
    settled_invocation_terminal_decision_from_checkpoint,
)
from cayu.sessions._model_stage import SessionModelCompletionDispatchAlreadyAuthorized
from cayu.sessions._terminal_evidence import (
    interruption_request_id_from_payload,
    require_interruption_event_matches_pending_marker,
)
from cayu.sessions.base import SessionRunFenced
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
    INTERACTION_TERMINAL_EVENT_TYPES,
)
from cayu.sessions.invocation import SessionExecutionSource
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.recovery import (
    _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY,
    _incomplete_recovery_claim_from_checkpoint,
)
from cayu.sessions.requests import InterruptSessionRequest
from cayu.sessions.store import SessionStore
from cayu.vaults.redaction import SecretRedactor


def _store_time_checkpoint_with_claimed_pending_session_interrupt(
    payload: dict[str, Any],
    *,
    claim_id: str,
    claim_lease: timedelta,
    include_interruption_cascade: bool = True,
    cascade_created_at: datetime | None = None,
    expected_interrupted_user_input: PendingUserInput | None = None,
    expected_ambiguous_user_input: AmbiguousPendingUserInput | None = None,
    expected_foreground_child_wait: ForegroundChildWait | None = None,
    expected_interrupted_approval: PendingToolApproval | None = None,
    terminal_decision: InvocationTerminalDecision | None = None,
    redactor: SecretRedactor | None = None,
):
    """Install a supersession claim using the transition transaction's clock."""

    base_transform = _checkpoint_with_pending_session_interrupt(
        payload,
        include_interruption_cascade=include_interruption_cascade,
        cascade_created_at=cascade_created_at,
        expected_interrupted_user_input=expected_interrupted_user_input,
        expected_ambiguous_user_input=expected_ambiguous_user_input,
        expected_foreground_child_wait=expected_foreground_child_wait,
        expected_interrupted_approval=expected_interrupted_approval,
        terminal_decision=terminal_decision,
        redactor=redactor,
    )

    def transform(
        session: Session,
        checkpoint: dict[str, Any] | None,
        store_now: datetime,
    ) -> dict[str, Any]:
        updated = base_transform(session, checkpoint)
        pending_payload = updated.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
        if type(pending_payload) is not dict or not (
            USER_INPUT_SUPERSESSION_INTENT_KEY in pending_payload
            or AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY in pending_payload
        ):
            return updated
        existing_claim = _incomplete_recovery_claim_from_checkpoint(updated)
        if (
            existing_claim is not None
            and existing_claim[0] != claim_id
            and existing_claim[1] > store_now
        ):
            raise SessionRuntimePublicationConflict(
                "Terminal finalization ownership changed before supersession."
            )
        claim_expires_at = store_now + claim_lease
        updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = {
            "version": 1,
            "claim_id": claim_id,
            "claimed_at": store_now.isoformat(),
            "claim_expires_at": claim_expires_at.isoformat(),
        }
        return updated

    return transform


class SessionInterruption:
    """Own explicit stops, exact interruption evidence and terminal delivery."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        environment_lifecycle: EnvironmentLifecycle,
        session_control: SessionControl[SessionUsageTracker],
        recovery_ownership: RecoveryOwnership,
        terminal_evidence: TerminalEvidenceReader,
        terminal_finalization: TerminalEvidenceFinalization,
        session_finalization: SessionFinalization,
        terminal_event_publication: TerminalEventPublication,
        background_interruption_coordinator: BackgroundInterruptionCoordinator,
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
        get_registered_agent: Callable[[str], runtime_records.RegisteredAgentState],
        get_registered_provider: Callable[[str | None], runtime_records.RegisteredProvider],
        get_registered_environment_for_session: Callable[
            [str | None], runtime_records.RegisteredEnvironment | None
        ],
        model_completion_recovery: ModelCompletionRecovery,
        user_input_evidence: UserInputRecoveryEvidence,
    ) -> None:
        self.session_store = session_store
        self._event_writer = event_writer
        self._environment_lifecycle = environment_lifecycle
        self._session_control = session_control
        self._recovery_ownership = recovery_ownership
        self._terminal_evidence = terminal_evidence
        self._terminal_finalization = terminal_finalization
        self._session_finalization = session_finalization
        self._terminal_event_publication = terminal_event_publication
        self._background_interruption_coordinator = background_interruption_coordinator
        self._secret_redactor = secret_redactor
        self._clock = clock
        self._get_registered_agent = get_registered_agent
        self._get_registered_provider = get_registered_provider
        self._get_registered_environment_for_session = get_registered_environment_for_session
        self._model_completion_recovery = model_completion_recovery
        self._user_input_evidence = user_input_evidence

    async def interrupt_session(
        self,
        request: InterruptSessionRequest,
        *,
        store_resolved_session_id: str | None = None,
    ) -> AsyncGenerator[Event, None]:
        request = session_request_boundary.prepare_interrupt_session_request(
            request,
            redactor=self._secret_redactor,
            store_resolved_session_id=store_resolved_session_id,
        )
        loaded_session = await self.session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")
        for field_name in (
            "agent_name",
            "provider_name",
            "model",
            "environment_name",
        ):
            session_request_boundary.require_secret_free_session_authority(
                getattr(loaded_session, field_name),
                field_name=field_name,
                redactor=self._secret_redactor,
            )
        interrupted_pending_user_input = None
        interrupted_ambiguous_user_input = None
        interrupted_foreground_wait = None
        interrupted_pending_approval = None
        accepted_delegated_gate_kind: str | None = None
        if loaded_session.status == SessionStatus.INTERRUPTED:
            interrupted_checkpoint = await self.session_store.load_checkpoint(loaded_session.id)
            interrupted_pending_approval = pending_approval_reader.pending_approval_from_checkpoint(
                interrupted_checkpoint
            )
            interrupted_foreground_wait, _ = foreground_child_state_from_checkpoint(
                interrupted_checkpoint
            )
            if interrupted_foreground_wait is not None:
                from cayu.runtime._foreground_gate_continuation import load_gate_request

                accepted_gate = await load_gate_request(
                    self.session_store,
                    parent=loaded_session,
                    wait=interrupted_foreground_wait,
                )
                if accepted_gate is not None:
                    accepted_delegated_gate_kind = accepted_gate["kind"]
                if accepted_delegated_gate_kind == "approval":
                    # This is an accepted execution waiting on a child, not an
                    # unclaimed approval that can be closed as a denied call.
                    # The exact wait/interaction stop fences late completion;
                    # retain the accepted gate as non-executable closure evidence.
                    interrupted_pending_approval = None
                closed_interaction = await self.session_store.query_events(
                    EventQuery(
                        session_id=loaded_session.id,
                        interaction_id=interrupted_foreground_wait.parent_effect.interaction_id,
                        event_types=tuple(INTERACTION_TERMINAL_EVENT_TYPES),
                        limit=1,
                    )
                )
                if closed_interaction:
                    interrupted_foreground_wait = None
            interrupted_ambiguous_user_input = ambiguous_pending_user_input_from_checkpoint(
                interrupted_checkpoint
            )
            if interrupted_ambiguous_user_input is None:
                interrupted_pending_user_input, _interrupted_resolution_intent = (
                    user_input_lifecycle_authority_from_checkpoint(
                        interrupted_checkpoint,
                        redactor=self._secret_redactor,
                        consume_on_rejection=True,
                        current_run_epoch=loaded_session.run_epoch,
                        runtime_session=loaded_session,
                    )
                )
            if accepted_delegated_gate_kind == "input":
                # The answer is already accepted and executing. Stop its exact
                # delegated interaction; do not supersede or re-answer the gate.
                interrupted_pending_user_input = None
            if interrupted_pending_user_input is not None:
                await self._user_input_evidence.require_exact_user_input_open_receipt(
                    session=loaded_session,
                    pending=interrupted_pending_user_input,
                )
            pending_interrupt_payload = await self._background_interruption_coordinator.load_pending_session_interrupt_payload(
                loaded_session.id,
                default={},
            )
            retained_supersession = (
                await self._user_input_evidence.validated_user_input_supersession_interrupt_payload(
                    session=loaded_session,
                    pending_interrupt_payload=pending_interrupt_payload,
                )
            )
            if retained_supersession is not None:
                if (
                    interrupted_pending_user_input is not None
                    or interrupted_ambiguous_user_input is not None
                ):
                    raise SessionRuntimePublicationConflict(
                        "Interrupted user-input supersession retains conflicting pause authority."
                    )
                repaired_event = await self._terminal_finalization.repair_and_join(
                    session=loaded_session,
                    expected_payload=retained_supersession,
                )
                if not interruption_cascade_suppressed():
                    self._session_finalization.schedule_background_interruption_cascade(
                        parent_session_id=loaded_session.id,
                        interrupt_payload=repaired_event.payload,
                        create_if_missing=False,
                    )
                yield repaired_event
                return
        if (
            loaded_session.status == SessionStatus.INTERRUPTED
            and interrupted_pending_user_input is None
            and interrupted_ambiguous_user_input is None
            and interrupted_foreground_wait is None
            and interrupted_pending_approval is None
        ):
            existing_interrupt_event = (
                await self._session_control.wait_for_active_interrupted_event(loaded_session.id)
            )
            if existing_interrupt_event is not None:
                retry_event: Event | None = None
                retry_request: dict[str, Any] | None = None
                if not interruption_cascade_suppressed():
                    marker = await self._background_interruption_coordinator.load_pending_interruption_cascade(
                        loaded_session.id
                    )
                    if (
                        marker is not None
                        and await self._background_interruption_coordinator.cascade_status(
                            loaded_session.id
                        )
                        == "failed"
                    ):
                        retry_request = {
                            "retry_request_id": str(uuid4()),
                            "reason": request.reason,
                            "metadata": request.metadata,
                            "requested_by": resolution_actor_payload(request.requested_by),
                        }
                        retry_event = await self._event_writer.emit(
                            _runtime_interruption_event(
                                Event(
                                    type=EventType.SESSION_INTERRUPTION_CASCADE_RETRY_REQUESTED,
                                    session_id=loaded_session.id,
                                    agent_name=loaded_session.agent_name,
                                    environment_name=loaded_session.environment_name,
                                    payload={
                                        "interruption_type": (
                                            _INTERRUPTION_TYPE_OPERATOR_REQUESTED
                                        ),
                                        "attempt_id": marker["attempt_id"],
                                        "previous_generation": marker.get("generation", 0),
                                        **_interruption_cascade_retry_event_payload(
                                            _copy_interruption_cascade_retry_request(retry_request)
                                        ),
                                    },
                                )
                            )
                        )
                    self._session_finalization.schedule_background_interruption_cascade(
                        parent_session_id=loaded_session.id,
                        interrupt_payload=existing_interrupt_event.payload,
                        create_if_missing=False,
                        retry_request=retry_request,
                    )
                yield existing_interrupt_event
                if retry_event is not None:
                    yield retry_event
                return
            raise RuntimeError(
                f"Session is interrupted but has no session.interrupted event: {loaded_session.id}"
            )

        adopted_user_input_interrupt_payload: dict[str, Any] | None = None
        if loaded_session.status == SessionStatus.INTERRUPTING:
            pending_interrupt_payload = await self._background_interruption_coordinator.load_pending_session_interrupt_payload(
                loaded_session.id,
                default={},
            )
            existing_interrupt_event = (
                await self._session_control.wait_for_active_interrupted_event(
                    loaded_session.id,
                    interruption_request_id=interruption_request_id_from_payload(
                        pending_interrupt_payload
                    ),
                )
            )
            if existing_interrupt_event is not None:
                checkpoint = await self.session_store.load_checkpoint(loaded_session.id)
                settled_decision = settled_invocation_terminal_decision_from_checkpoint(checkpoint)
                if settled_decision is not None and settled_decision.interaction_event_id is None:
                    if settled_decision.terminal_payload != pending_interrupt_payload:
                        raise SessionRuntimePublicationConflict(
                            "Settled closed-interaction interruption conflicts with its "
                            "pending terminal payload."
                        )
                    require_interruption_event_matches_pending_marker(
                        existing_interrupt_event,
                        pending_interrupt_payload,
                    )
                    (
                        loaded_session,
                        existing_interrupt_event,
                    ) = await self._session_finalization.publish_closed_interaction_terminal_decision(
                        session=loaded_session,
                        decision=settled_decision,
                        terminal_event=existing_interrupt_event,
                    )
                    await self._session_finalization.clear_pending_session_interrupt(
                        loaded_session.id,
                        expected_payload=pending_interrupt_payload,
                        expected_run_epoch=loaded_session.run_epoch,
                    )
                if not interruption_cascade_suppressed():
                    self._session_finalization.schedule_background_interruption_cascade(
                        parent_session_id=loaded_session.id,
                        interrupt_payload=existing_interrupt_event.payload,
                        create_if_missing=False,
                    )
                yield existing_interrupt_event
                return
            adopted_user_input_interrupt_payload = (
                await self._user_input_evidence.validated_user_input_supersession_interrupt_payload(
                    session=loaded_session,
                    pending_interrupt_payload=pending_interrupt_payload,
                )
            )
            if adopted_user_input_interrupt_payload is None:
                raise TimeoutError(f"Session interruption is still finalizing: {loaded_session.id}")
        interruptible_statuses = (
            _INTERRUPTIBLE_SESSION_STATUSES | {SessionStatus.INTERRUPTED}
            if (
                interrupted_pending_user_input is not None
                or interrupted_ambiguous_user_input is not None
                or interrupted_foreground_wait is not None
                or interrupted_pending_approval is not None
            )
            else _INTERRUPTIBLE_SESSION_STATUSES
        )
        if (
            adopted_user_input_interrupt_payload is None
            and loaded_session.status not in interruptible_statuses
        ):
            raise ValueError(f"Session cannot be interrupted from status: {loaded_session.status}")
        terminal_interaction_publication_supported = (
            self._session_finalization.supports_terminal_interaction_publication_protocol()
        )
        if (
            loaded_session.invocation.source is SessionExecutionSource.TASK
            and not terminal_interaction_publication_supported
        ):
            # Current task-backed entrances require this capability before
            # admission. Keep the interruption boundary fail-closed for a
            # directly task-created session reconstructed from external state,
            # without imposing the new extension on unrelated sessions.
            self._session_finalization.require_terminal_interaction_publication_protocol()
        registered_agent = self._get_registered_agent(loaded_session.agent_name)
        try:
            registered_provider = self._get_registered_provider(loaded_session.provider_name)
        except (KeyError, RuntimeError):
            # A live run owns its already-frozen provider and can still be
            # interrupted when the application registry has changed. If no live
            # owner remains and durable provider work requires cancellation, the
            # recovery boundary below fails closed instead of re-resolving the
            # mutable registry after the interrupt transition.
            registered_provider = None
        registered_environment = self._get_registered_environment_for_session(
            loaded_session.environment_name
        )

        interrupt_payload = (
            adopted_user_input_interrupt_payload
            if adopted_user_input_interrupt_payload is not None
            else {
                "reason": request.reason,
                "metadata": request.metadata,
                "requested_by": resolution_actor_payload(request.requested_by),
                "interruption_type": _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
                "interruption_request_id": str(uuid4()),
            }
        )
        finalization = self._terminal_finalization.interruption(
            session_id=loaded_session.id,
            adopting_pending=adopted_user_input_interrupt_payload is not None,
        )
        interruption_request_id = interruption_request_id_from_payload(interrupt_payload)
        if interruption_request_id is None:
            raise SessionRuntimePublicationConflict(
                "Pending session interruption has no request identity."
            )

        async def prepare_interrupt_terminal_decision() -> tuple[
            InvocationTerminalDecision | None,
            str | None,
        ]:
            if not terminal_interaction_publication_supported or (
                loaded_session.status is not SessionStatus.RUNNING
                and interrupted_foreground_wait is None
                and interrupted_pending_user_input is None
                and interrupted_pending_approval is None
            ):
                return None, None
            interrupt_checkpoint = await self.session_store.load_checkpoint(loaded_session.id)
            if invocation_terminal_decision_from_checkpoint(interrupt_checkpoint) is None:
                active_model = await self._model_completion_recovery.load_model_completion_boundary(
                    loaded_session
                )
                if active_model is not None and (
                    await self._model_completion_recovery.has_recoverable_provider_operation(
                        active_model.stage
                    )
                ):
                    return None, None
            if interrupted_pending_approval is not None:
                if (
                    pending_approval_reader.pending_approval_from_checkpoint(interrupt_checkpoint)
                    != interrupted_pending_approval
                    or pending_approval_reader.approval_resolution_intent_from_checkpoint(
                        interrupt_checkpoint
                    )
                    is not None
                ):
                    raise SessionRunFenced("Approval stop lost its exact unclaimed pause.")
                interrupt_payload[approval_support.APPROVAL_INTERRUPT_CLOSE_INTENT_KEY] = (
                    approval_support.approval_interrupt_close_intent(interrupted_pending_approval)
                )
            if interrupted_pending_user_input is not None:
                decision_input, decision_resolution = (
                    user_input_lifecycle_authority_from_checkpoint(
                        interrupt_checkpoint, current_run_epoch=loaded_session.run_epoch
                    )
                )
                if decision_input is None or decision_input != interrupted_pending_user_input:
                    raise SessionRunFenced("User-input stop lost its exact pending action.")
                interrupt_payload[USER_INPUT_SUPERSESSION_INTENT_KEY] = (
                    user_input_supersession_intent_for(
                        decision_input, resolution_intent=decision_resolution
                    ).model_dump(mode="json", exclude_none=True)
                )
            decision_wait, _ = foreground_child_state_from_checkpoint(interrupt_checkpoint)
            if (
                decision_wait is None
                and interrupted_pending_approval is None
                and interrupted_pending_user_input is None
                and pending_round_reader.pending_tool_round_from_checkpoint(interrupt_checkpoint)
                is not None
                and invocation_terminal_decision_from_checkpoint(interrupt_checkpoint) is None
            ):
                # A running tool may leave an uncertain external effect when
                # interrupted. Let its owner settle the round before electing
                # terminal closure; unresolved rounds retain their interaction.
                return None, None
            if decision_wait is not None:
                intent = decision_wait.parent_effect
                if (
                    intent.session_id != loaded_session.id
                    or intent.session_instance_id != loaded_session.instance_id
                ):
                    raise SessionRunFenced(
                        "Interruption decision has foreign foreground wait evidence."
                    )
                interrupt_payload.update(
                    model_step_id=intent.model_step_id,
                    model_attempt_id=intent.model_attempt_id,
                    tool_round_id=intent.tool_round_id,
                    tool_call_id=intent.tool_call_id,
                )
            interrupt_active_profile = active_invocation_execution_profile_from_checkpoint(
                interrupt_checkpoint
            )
            latest_interaction_records = await self.session_store.query_events(
                EventQuery(
                    session_id=loaded_session.id,
                    event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                )
            )
            latest_interaction = (
                latest_interaction_records[0].event if latest_interaction_records else None
            )
            current_interaction_id = (
                None if latest_interaction is None else latest_interaction.interaction_id
            )
            if interrupt_active_profile is None or current_interaction_id is None:
                return None, None
            latest_interaction_event_id = latest_interaction_records[0].event.id
            interaction_id = current_interaction_id
            observed_at = self._clock()
            assert latest_interaction is not None
            interaction_event_id = (
                None
                if latest_interaction.type in INTERACTION_TERMINAL_EVENT_TYPES
                else invocation_terminal_event_id(
                    outcome=InvocationTerminalOutcome.INTERRUPTED,
                    session_id=loaded_session.id,
                    session_instance_id=loaded_session.instance_id,
                    run_epoch=loaded_session.run_epoch,
                    interaction_id=interaction_id,
                    source_id=interruption_request_id,
                    event_kind="interaction",
                )
            )
            terminal_event_id = invocation_terminal_event_id(
                outcome=InvocationTerminalOutcome.INTERRUPTED,
                session_id=loaded_session.id,
                session_instance_id=loaded_session.instance_id,
                run_epoch=loaded_session.run_epoch,
                interaction_id=interaction_id,
                source_id=interruption_request_id,
                event_kind="session",
            )
            return (
                build_invocation_terminal_decision(
                    outcome=InvocationTerminalOutcome.INTERRUPTED,
                    session_id=loaded_session.id,
                    session_instance_id=loaded_session.instance_id,
                    run_epoch=loaded_session.run_epoch,
                    profile_interaction_id=interrupt_active_profile.interaction_id,
                    interaction_id=interaction_id,
                    execution_profile_fingerprint=(interrupt_active_profile.profile.fingerprint),
                    interaction_event_id=interaction_event_id,
                    predecessor_interaction_event_id=(
                        latest_interaction.id
                        if latest_interaction.type in INTERACTION_TERMINAL_EVENT_TYPES
                        else None
                    ),
                    terminal_event_id=terminal_event_id,
                    observed_at=observed_at,
                    terminal_payload=interrupt_payload,
                    interruption_request_id=interruption_request_id,
                ),
                latest_interaction_event_id,
            )

        interrupt_terminal_decision: InvocationTerminalDecision | None = None
        interrupt_latest_interaction_event_id: str | None = None
        cascade_suppressed = interruption_cascade_suppressed()
        finalization.begin_request()

        async def clear_locally_claimed_adopted_interrupt(
            event: Event,
        ) -> None:
            """Settle an adopted marker after its exact terminal event is observed."""

            if adopted_user_input_interrupt_payload is None:
                return
            require_interruption_event_matches_pending_marker(
                event,
                interrupt_payload,
            )
            claim_id = finalization.claim_id
            if claim_id is None:
                raise RuntimeError("User-input supersession finalization lost its durable owner.")
            await finalization.await_operation(
                lambda: self._session_finalization.clear_claimed_pending_interrupt_if_retained(
                    session_id=loaded_session.id,
                    claim_id=claim_id,
                    expected_payload=interrupt_payload,
                ),
                operation_name="Observed interruption marker cleanup",
            )

        try:
            if adopted_user_input_interrupt_payload is None:
                assert finalization.claim_id is not None
                for terminal_decision_attempt in range(3):
                    (
                        interrupt_terminal_decision,
                        interrupt_latest_interaction_event_id,
                    ) = await prepare_interrupt_terminal_decision()
                    try:
                        transition_kwargs: dict[str, Any] = {}
                        if terminal_interaction_publication_supported:
                            transition_kwargs = {
                                "expected_latest_interaction_event_id": (
                                    interrupt_latest_interaction_event_id
                                ),
                                "require_no_active_model_completion_dispatch": (
                                    interrupt_terminal_decision is not None
                                ),
                            }
                        with _invocation_lifecycle_authority_mutation_scope():
                            session = await self.session_store.transition_status_and_checkpoint(
                                loaded_session.id,
                                from_statuses=interruptible_statuses,
                                to_status=SessionStatus.INTERRUPTING,
                                store_time_checkpoint_transform=(
                                    _store_time_checkpoint_with_claimed_pending_session_interrupt(
                                        interrupt_payload,
                                        claim_id=finalization.claim_id,
                                        claim_lease=self._recovery_ownership.claim_lease_duration,
                                        include_interruption_cascade=not cascade_suppressed,
                                        cascade_created_at=self._clock(),
                                        expected_interrupted_user_input=(
                                            interrupted_pending_user_input
                                        ),
                                        expected_ambiguous_user_input=(
                                            interrupted_ambiguous_user_input
                                        ),
                                        terminal_decision=interrupt_terminal_decision,
                                        expected_foreground_child_wait=interrupted_foreground_wait,
                                        expected_interrupted_approval=interrupted_pending_approval,
                                        redactor=self._secret_redactor,
                                    )
                                ),
                                **transition_kwargs,
                            )
                        break
                    except SessionModelCompletionDispatchAlreadyAuthorized:
                        # The dispatch fence won first. Move the session to the
                        # interrupting state, but do not publish a decisive
                        # terminal winner until the admitted provider operation
                        # has quiesced or been reconciled.
                        interrupt_terminal_decision = None
                        try:
                            with _invocation_lifecycle_authority_mutation_scope():
                                session = await self.session_store.transition_status_and_checkpoint(
                                    loaded_session.id,
                                    from_statuses=interruptible_statuses,
                                    to_status=SessionStatus.INTERRUPTING,
                                    store_time_checkpoint_transform=(
                                        _store_time_checkpoint_with_claimed_pending_session_interrupt(
                                            interrupt_payload,
                                            claim_id=finalization.claim_id,
                                            claim_lease=self._recovery_ownership.claim_lease_duration,
                                            include_interruption_cascade=(not cascade_suppressed),
                                            cascade_created_at=self._clock(),
                                            expected_interrupted_user_input=(
                                                interrupted_pending_user_input
                                            ),
                                            expected_ambiguous_user_input=(
                                                interrupted_ambiguous_user_input
                                            ),
                                            terminal_decision=None,
                                            expected_foreground_child_wait=(
                                                interrupted_foreground_wait
                                            ),
                                            expected_interrupted_approval=interrupted_pending_approval,
                                            redactor=self._secret_redactor,
                                        )
                                    ),
                                    expected_latest_interaction_event_id=(
                                        interrupt_latest_interaction_event_id
                                    ),
                                )
                        except SessionRunFenced:
                            concurrent_session = await self.session_store.load(loaded_session.id)
                            concurrent_checkpoint = await self.session_store.load_checkpoint(
                                loaded_session.id
                            )
                            if (
                                terminal_decision_attempt == 2
                                or concurrent_session is None
                                or concurrent_session.instance_id != loaded_session.instance_id
                                or concurrent_session.run_epoch != loaded_session.run_epoch
                                or concurrent_session.status is not SessionStatus.RUNNING
                                or invocation_terminal_decision_from_checkpoint(
                                    concurrent_checkpoint
                                )
                                is not None
                            ):
                                raise
                            continue
                        break
                    except SessionRunFenced:
                        concurrent_session = await self.session_store.load(loaded_session.id)
                        concurrent_checkpoint = await self.session_store.load_checkpoint(
                            loaded_session.id
                        )
                        if (
                            terminal_decision_attempt == 2
                            or concurrent_session is None
                            or concurrent_session.instance_id != loaded_session.instance_id
                            or concurrent_session.run_epoch != loaded_session.run_epoch
                            or concurrent_session.status is not SessionStatus.RUNNING
                            or invocation_terminal_decision_from_checkpoint(concurrent_checkpoint)
                            is not None
                        ):
                            raise
            else:
                session = loaded_session
                repaired_event = await finalization.claim_or_join(
                    session=session, expected_payload=interrupt_payload
                )
                if repaired_event is not None:
                    await finalization.finish_preparation(None)
                    if not interruption_cascade_suppressed():
                        self._session_finalization.schedule_background_interruption_cascade(
                            parent_session_id=session.id,
                            interrupt_payload=repaired_event.payload,
                            create_if_missing=False,
                        )
                    yield repaired_event
                    return
            session = await finalization.prepare_dispatch(
                session=session, interruption_request_id=interruption_request_id
            )
            self._session_control.signal_interrupt(session.id)
            active_work_signalled = self._session_control.cancel_active_runs(session.id)
            if finalization.handed_to_active_run and not active_work_signalled:
                raise RuntimeError(
                    "Terminal finalization handoff lost its live interruption target."
                )
            if active_work_signalled:
                existing_interrupt_event = (
                    await self._session_control.wait_for_active_interrupted_event(
                        session.id,
                        interruption_request_id=interruption_request_id,
                    )
                )
                if existing_interrupt_event is not None:
                    await clear_locally_claimed_adopted_interrupt(existing_interrupt_event)
                    await finalization.finish_preparation(None)
                    yield existing_interrupt_event
                    return
                raise TimeoutError(f"Session interruption is still finalizing: {session.id}")
            provider_operation_profile = await finalization.await_operation(
                lambda: self._model_completion_recovery.cancel_provider_operation_for_interruption(
                    session,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                ),
                operation_name="Offline provider interruption preparation",
            )
            provider_operation_addressed = provider_operation_profile is not None
            # Offline provider settlement retains the original interaction for
            # explicit provider recovery. Do not elect an interaction close just
            # because cancellation was attempted; a prior durable decision still
            # wins when terminal finalization reads it below.
            if loaded_session.status == SessionStatus.RUNNING and not provider_operation_addressed:
                existing_interrupt_event = await finalization.await_operation(
                    lambda: self._session_control.wait_for_active_interrupted_event(
                        session.id,
                        interruption_request_id=interruption_request_id,
                    ),
                    operation_name="Offline interruption live-owner observation",
                )
                if existing_interrupt_event is not None:
                    await clear_locally_claimed_adopted_interrupt(existing_interrupt_event)
                    await finalization.finish_preparation(None)
                    yield existing_interrupt_event
                    return
                raise TimeoutError(f"Session interruption is still finalizing: {session.id}")
            if provider_operation_profile is not None:
                session = await finalization.await_operation(
                    lambda: self._recovery_ownership.claim_provider_operation_interruption(
                        session,
                        provider_operation_profile.active_profile,
                        interruption_request_id=interruption_request_id,
                        invocation_context=provider_operation_profile.invocation_context,
                    ),
                    operation_name="Offline provider interruption claim",
                )
                invocation_context = (
                    provider_operation_profile.invocation_context.with_rebound_session(
                        session,
                        active_profile=provider_operation_profile.active_profile.model_copy(
                            update={"run_epoch": session.run_epoch}
                        ),
                    )
                )
            else:
                invocation_context = None
        except SessionRuntimePublicationConflict as preparation_failure:
            await finalization.finish_preparation(preparation_failure)
            raise
        except ValueError:
            reconciliation_failure: BaseException | None = None
            try:
                reloaded_session = await self.session_store.load(loaded_session.id)
                if reloaded_session is None:
                    raise KeyError(f"Session not found: {loaded_session.id}") from None
                if reloaded_session.status in INTERRUPT_REQUESTED_SESSION_STATUSES:
                    self._session_control.signal_interrupt(reloaded_session.id)
                    persisted_interrupt_payload = await self._background_interruption_coordinator.load_pending_session_interrupt_payload(
                        reloaded_session.id,
                        default={},
                    )
                    persisted_interruption_request_id = interruption_request_id_from_payload(
                        persisted_interrupt_payload
                    )
                    if persisted_interruption_request_id is not None:
                        existing_interrupt_event = (
                            await self._session_control.wait_for_active_interrupted_event(
                                reloaded_session.id,
                                interruption_request_id=persisted_interruption_request_id,
                            )
                        )
                        if existing_interrupt_event is not None:
                            require_interruption_event_matches_pending_marker(
                                existing_interrupt_event,
                                persisted_interrupt_payload,
                            )
                    elif reloaded_session.status is SessionStatus.INTERRUPTED:
                        reloaded_checkpoint = await self.session_store.load_checkpoint(
                            reloaded_session.id
                        )
                        terminal_inspection = await self._terminal_evidence.inspect(
                            session=reloaded_session,
                            checkpoint=reloaded_checkpoint,
                        )
                        existing_interrupt_event = terminal_inspection.event
                    else:
                        raise SessionRuntimePublicationConflict(
                            "Concurrent interruption transition retained no durable request "
                            "identity."
                        ) from None
                    if existing_interrupt_event is not None:
                        await clear_locally_claimed_adopted_interrupt(existing_interrupt_event)
                        if not interruption_cascade_suppressed():
                            self._session_finalization.schedule_background_interruption_cascade(
                                parent_session_id=reloaded_session.id,
                                interrupt_payload=existing_interrupt_event.payload,
                                create_if_missing=False,
                            )
                        await finalization.finish_preparation(None)
                        yield existing_interrupt_event
                        return
                    if reloaded_session.status is SessionStatus.INTERRUPTED:
                        reloaded_checkpoint = await self.session_store.load_checkpoint(
                            reloaded_session.id
                        )
                        current_pending_user_input, _current_resolution_intent = (
                            user_input_lifecycle_authority_from_checkpoint(
                                reloaded_checkpoint,
                                redactor=self._secret_redactor,
                                consume_on_rejection=True,
                                current_run_epoch=reloaded_session.run_epoch,
                                runtime_session=reloaded_session,
                            )
                        )
                        if current_pending_user_input is not None:
                            raise SessionRuntimePublicationConflict(
                                "Interrupted user-input pause changed before supersession."
                            ) from None
                    if self._session_control.has_active_tasks(reloaded_session.id):
                        raise TimeoutError(
                            f"Session interruption is still finalizing: {reloaded_session.id}"
                        ) from None
                    if reloaded_session.status == SessionStatus.INTERRUPTING:
                        raise TimeoutError(
                            f"Session interruption is still finalizing: {reloaded_session.id}"
                        ) from None
                    raise RuntimeError(
                        f"Session is interrupted but has no session.interrupted event: "
                        f"{reloaded_session.id}"
                    ) from None
                raise
            except BaseException as failure:
                reconciliation_failure = failure
                raise
            finally:
                await finalization.finish_preparation(reconciliation_failure)
        except BaseException as preparation_failure:
            await finalization.finish_preparation(preparation_failure)
            raise

        async def release_offline_provider_interruption(
            authoritative_failure: BaseException | None,
            *,
            operation: str,
        ) -> None:
            if provider_operation_profile is None:
                return
            await self._recovery_ownership.run_cleanup_steps(
                authoritative_failure=authoritative_failure,
                steps=(
                    (
                        operation,
                        lambda: (
                            self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                                session_id=session.id,
                                execution_profile=provider_operation_profile.active_profile.profile,
                                invocation_context=invocation_context,
                            )
                        ),
                    ),
                ),
            )

        offline_transition_claim_id = finalization.transition_claim_id
        offline_terminal_decision: InvocationTerminalDecision | None = None
        if (
            provider_operation_profile is not None
            or interrupt_terminal_decision is not None
            or adopted_user_input_interrupt_payload is not None
        ):
            offline_decision_checkpoint = await finalization.await_operation(
                lambda: self.session_store.load_checkpoint(session.id),
                operation_name="Offline interruption terminal-decision read",
            )
            offline_terminal_decision = invocation_terminal_decision_from_checkpoint(
                offline_decision_checkpoint
            )
        try:
            if offline_terminal_decision is None:
                if invocation_context is not None:
                    restored_interaction = (
                        await self._session_finalization.activate_latest_open_interaction(
                            session.id
                        )
                    )
                    if restored_interaction != invocation_context.active_profile.interaction_id:
                        raise SessionRunFenced(
                            "Offline interruption lost its original open interaction."
                        )
                (
                    session,
                    _interaction_event,
                    _,
                ) = await finalization.await_operation(
                    lambda: self._session_finalization.publish_sibling_interaction_transition(
                        session=session,
                        invocation_context=invocation_context,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                        to_status=SessionStatus.INTERRUPTED,
                        execution_profile=(
                            None
                            if provider_operation_profile is None
                            else provider_operation_profile.active_profile.profile
                        ),
                        finalize_unsettled_cancellation=offline_transition_claim_id is None,
                        expected_recovery_claim_id=offline_transition_claim_id,
                    ),
                    operation_name="Offline interruption interaction transition",
                )
                if _interaction_event is not None:
                    yield _interaction_event
        except BaseException as transition_failure:
            try:
                await release_offline_provider_interruption(
                    transition_failure,
                    operation="failed offline interruption run-fence release",
                )
            finally:
                await finalization.release(transition_failure)
                finalization.end_request()
            raise
        try:
            payload = await finalization.await_operation(
                lambda: (
                    self._background_interruption_coordinator.load_pending_session_interrupt_payload(
                        session.id,
                        default={
                            "reason": request.reason,
                            "metadata": request.metadata,
                            "requested_by": resolution_actor_payload(request.requested_by),
                            "interruption_type": _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
                            "interruption_request_id": interrupt_payload["interruption_request_id"],
                        },
                    )
                ),
                operation_name="Offline interruption payload read",
            )
        except BaseException as payload_failure:
            try:
                await release_offline_provider_interruption(
                    payload_failure,
                    operation="offline interruption payload run-fence release",
                )
            finally:
                await finalization.release(payload_failure)
                finalization.end_request()
            raise
        user_input_supersession_retained = (
            USER_INPUT_SUPERSESSION_INTENT_KEY in payload
            or AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY in payload
        )

        async def finalize_terminal_interruption(session: Session) -> AsyncGenerator[Event, None]:
            owned_terminal_stream: AsyncIterator[Event] | None = None
            try:
                if offline_terminal_decision is not None:
                    offline_active_profile = active_invocation_execution_profile_from_checkpoint(
                        offline_decision_checkpoint
                    )
                    if offline_active_profile is None:
                        raise SessionRunFenced("Offline interruption lost its invocation profile.")
                    terminal_profile = offline_active_profile.profile
                    if invocation_context is not None:
                        if terminal_profile != invocation_context.profile:
                            raise SessionRunFenced(
                                "Offline interruption changed its frozen execution profile."
                            )
                        terminal_profile = invocation_context.profile
                    async for (
                        decided_event
                    ) in self._session_finalization.handle_session_interrupted(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                        execution_profile=terminal_profile,
                        invocation_context=invocation_context,
                    ):
                        # ``interrupt_session`` historically exposes the
                        # session-scoped terminal stream. The interaction event
                        # is nevertheless committed in the same transaction and
                        # remains available through durable event queries.
                        if decided_event.type is not EventType.INTERACTION_INTERRUPTED:
                            yield decided_event
                    return
                existing_interrupt_event = await self._session_control.latest_interrupted_event(
                    session.id,
                    interruption_request_id=interruption_request_id_from_payload(payload),
                )
                if existing_interrupt_event is not None:
                    if user_input_supersession_retained:
                        require_interruption_event_matches_pending_marker(
                            existing_interrupt_event,
                            payload,
                        )
                        if finalization.claim_id is None:
                            raise RuntimeError(
                                "User-input supersession finalization lost its durable owner."
                            )
                        await (
                            self._session_finalization.clear_claimed_pending_interrupt_if_retained(
                                session_id=session.id,
                                claim_id=finalization.claim_id,
                                expected_payload=payload,
                            )
                        )
                    else:
                        await self._session_finalization.clear_pending_session_interrupt(session.id)
                    if not cascade_suppressed:
                        self._session_finalization.schedule_background_interruption_cascade(
                            parent_session_id=session.id,
                            interrupt_payload=existing_interrupt_event.payload,
                            create_if_missing=False,
                        )
                    yield existing_interrupt_event
                    return
                await self._emit_active_turn_completed_if_needed(
                    session=session,
                    status=SessionStatus.INTERRUPTED,
                    invocation_context=invocation_context,
                )
                terminal_event_publisher: Callable[[Event], Awaitable[Event]] | None = None
                if user_input_supersession_retained:
                    if finalization.claim_id is None:
                        raise RuntimeError(
                            "User-input supersession finalization lost its durable owner."
                        )
                    owned_claim_id = finalization.claim_id

                    async def publish_owned_terminal_event(event: Event) -> Event:
                        return await self._session_finalization.publish_terminal_event_under_finalization_claim(
                            event=event,
                            session=session,
                            claim_id=owned_claim_id,
                            expected_payload=payload,
                        )

                    terminal_event_publisher = publish_owned_terminal_event
                owned_terminal_stream = self._terminal_event_publication.emit(
                    event=_runtime_interruption_event(
                        Event(
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=_environment_name(registered_environment),
                            payload=payload,
                        )
                    ),
                    phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=(
                        None
                        if provider_operation_profile is None
                        else provider_operation_profile.active_profile.profile
                    ),
                    invocation_context=invocation_context,
                    terminal_event_publisher=terminal_event_publisher,
                )
                terminal_prefix, interrupted_event = await _collect_through_event_type(
                    owned_terminal_stream,
                    EventType.SESSION_INTERRUPTED,
                    missing_message="Session interruption produced no terminal event.",
                )
                if user_input_supersession_retained:
                    require_interruption_event_matches_pending_marker(
                        interrupted_event,
                        payload,
                    )
                    if finalization.claim_id is None:
                        raise RuntimeError(
                            "User-input supersession finalization lost its durable owner."
                        )
                    await self._session_finalization.clear_claimed_pending_interrupt_if_retained(
                        session_id=session.id,
                        claim_id=finalization.claim_id,
                        expected_payload=payload,
                    )
                else:
                    await self._session_finalization.clear_pending_session_interrupt(session.id)
                if not cascade_suppressed:
                    self._session_finalization.schedule_background_interruption_cascade(
                        parent_session_id=session.id,
                        interrupt_payload=interrupted_event.payload,
                        create_if_missing=False,
                    )
                for event in terminal_prefix:
                    yield event
                async for event in owned_terminal_stream:
                    yield event
            except BaseException as terminal_failure:
                if owned_terminal_stream is not None:
                    await self._recovery_ownership.run_cleanup_steps(
                        authoritative_failure=terminal_failure,
                        steps=(
                            (
                                "offline interruption terminal stream close",
                                lambda: _close_async_iterator(owned_terminal_stream),
                            ),
                        ),
                    )
                raise

        try:
            if user_input_supersession_retained:
                async with finalization.finalize(
                    session=session,
                    expected_payload=payload,
                    finalization=finalize_terminal_interruption,
                    terminal_decision=offline_terminal_decision,
                ) as owned:
                    async for event in owned:
                        yield event
            else:
                async for event in finalize_terminal_interruption(session):
                    yield event
        finally:
            try:
                await release_offline_provider_interruption(
                    sys.exception(),
                    operation="offline provider interruption run-fence release",
                )
            finally:
                try:
                    await finalization.release(sys.exception())
                finally:
                    finalization.end_request()
        return

    async def _emit_active_turn_completed_if_needed(
        self,
        *,
        session: Session,
        status: SessionStatus,
        invocation_context: InvocationContext | None = None,
    ) -> Event | None:
        for active_run in self._session_control.active_runs(session.id):
            if (
                active_run.turn_registered_agent is None
                or active_run.turn_started_at is None
                or active_run.turn_usage_tracker is None
            ):
                continue
            events = await self._session_finalization.emit_turn_completed_once(
                session=session,
                registered_agent=active_run.turn_registered_agent,
                environment_name=active_run.turn_environment_name,
                status=status,
                run_started_at=active_run.turn_started_at,
                usage_tracker=active_run.turn_usage_tracker,
                active_run=active_run,
                invocation_context=invocation_context,
            )
            return events[-1]
        return None
