from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from cayu._exception_state import pop_exception_state
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
)
from cayu.collaboration.access import CollaborationAccessContext
from cayu.events import (
    Event,
    EventType,
)
from cayu.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileIdentity,
)
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _session_request_boundary as session_request_boundary
from cayu.runtime._delegated_event_stream import _close_delegated_event_stream
from cayu.runtime._environment_lifecycle import (
    EnvironmentLifecycle,
)
from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait
from cayu.runtime._foreground_gate_continuation import ForegroundGatePolicyOwner
from cayu.runtime._incomplete_session_recovery import IncompleteSessionRecovery
from cayu.runtime._interruption_coordinator import (
    BackgroundInterruptionCoordinator,
    suppress_interruption_cascade,
)
from cayu.runtime._invocation_lifecycle import (
    InvocationContext,
)
from cayu.runtime._model_completion_contracts import (
    ModelCompletionManualRecoveryRequired,
    model_completion_recovery_context_from_stage,
)
from cayu.runtime._model_errors import (
    detach_billing_identity_cancellation_group,
)
from cayu.runtime._recovery_coordinator import (
    RecoveryCoordinator,
)
from cayu.runtime._run_limits import (
    SessionUsageTracker,
)
from cayu.runtime._session_control import (
    SessionControl,
)
from cayu.runtime._session_engine import (
    SessionEngine,
    _detach_billing_cancellation_for_public,
    _detach_provider_cancellation_for_public,
)
from cayu.runtime._session_finalization import (
    _INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE,
    _INTERACTION_TRANSITION_RUN_FENCE_ATTRIBUTE,
    SessionFinalization,
    _close_async_iterator,
    _is_interaction_transition_run_fence,
    _pop_interaction_transition_cancellation_outcome,
    _run_interaction_transition_cancellation_cleanup_steps,
)
from cayu.runtime._task_store_operation_boundary import (
    raise_task_store_operation_failure,
)
from cayu.runtime._terminal_evidence_reader import TerminalEvidenceReader
from cayu.runtime._work_attempt_invocation import (
    _authenticated_work_attempt_invocation,
    _WorkAttemptRecoveryAlreadyActive,
)
from cayu.runtime._work_attempt_session_mutation import (
    settle_work_attempt_owned_operation,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileMismatchError,
)
from cayu.runtime.provider_operations import (
    ProviderOperationEvidenceError,
    load_recoverable_provider_operation,
)
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
)
from cayu.sessions._foreground_child_checkpoint import (
    ForegroundChildResumeRequest,
    ForegroundChildTerminal,
    ForegroundChildWait,
    ForegroundParentContinuation,
)
from cayu.sessions._invocation_lifecycle import (
    invocation_lifecycle_receipt_history_present,
)
from cayu.sessions.base import (
    InteractionTransitionSpec,
    ModelCompletionManualRecoveryRequest,
    SessionRunFenced,
    SessionStore,
    _activate_session_interaction,
    _activate_session_run_fence,
    _current_session_interaction_id,
    _deactivate_session_interaction,
    _deactivate_session_run_fence,
    _latest_session_invocation_interaction_is_settled,
    _set_session_interaction_recovered_active_through,
    execution_profile_adoption_request_fingerprint,
)
from cayu.sessions.cleanup import RecoveryCleanupSupervisor
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
    INTERACTION_TERMINAL_EVENT_TYPES,
)
from cayu.sessions.queries import SessionOrder, SessionQuery
from cayu.sessions.records import (
    Session,
    SessionStatus,
)
from cayu.sessions.recovery import (
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionRecoveryResult,
    IncompleteSessionsRecoveryPage,
    IncompleteSessionsRecoveryRequest,
    RecoveryBlockerCode,
    StartupRecoveryBlockedSession,
    StartupRecoveryResult,
    copy_incomplete_session_recovery_request,
    copy_incomplete_sessions_recovery_request,
)
from cayu.sessions.requests import ResumeRequest
from cayu.tasks.admission import (
    WorkAttemptAdmission,
    WorkAttemptExecutionClaimLost,
    WorkAttemptRecoveryRequired,
)
from cayu.tasks.contracts import TaskCompletionDecisionRequired
from cayu.vaults.redaction import SecretRedactor

if TYPE_CHECKING:
    from cayu.runtime._session_continuation_resume import _ResumeAdmissionHandoff

logger = logging.getLogger(__name__)


class SessionRecovery:
    """Coordinate continuation, startup, and abandoned-session recovery above execution."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        engine: SessionEngine,
        recovery_coordinator: RecoveryCoordinator,
        incomplete_recovery: IncompleteSessionRecovery,
        background_interruption_coordinator: BackgroundInterruptionCoordinator,
        environment_lifecycle: EnvironmentLifecycle,
        foreground_gate_policy_owner: ForegroundGatePolicyOwner,
        get_registered_agent: Callable[[str], runtime_records.RegisteredAgentState],
        recovery_cleanup_supervisor: RecoveryCleanupSupervisor,
        require_participant_execution: Callable[
            [Session, CollaborationAccessContext | None], Awaitable[None]
        ],
        secret_redactor: SecretRedactor,
        session_control: SessionControl[SessionUsageTracker],
        session_finalization: SessionFinalization,
        terminal_evidence: TerminalEvidenceReader,
        effective_retry_policy: Callable[[RetryPolicy | None], RetryPolicy],
    ) -> None:
        self.session_store = session_store
        self._engine = engine
        self._recovery_coordinator = recovery_coordinator
        self._incomplete_recovery = incomplete_recovery
        self._background_interruption_coordinator = background_interruption_coordinator
        self._environment_lifecycle = environment_lifecycle
        self._foreground_gate_policy_owner = foreground_gate_policy_owner
        self._get_registered_agent = get_registered_agent
        self._recovery_cleanup_supervisor = recovery_cleanup_supervisor
        self._require_participant_execution = require_participant_execution
        self._secret_redactor = secret_redactor
        self._session_control = session_control
        self._session_finalization = session_finalization
        self._terminal_evidence = terminal_evidence
        self._effective_retry_policy = effective_retry_policy
        self._startup_recovery_result = StartupRecoveryResult()

    async def resume_pending_interruption_cascade(
        self,
        session_id: str,
        interrupting_inactive_for_seconds: int | None = None,
    ) -> bool:
        """Resume exactly one durable descendant-interruption cascade."""

        if interrupting_inactive_for_seconds is not None and (
            type(interrupting_inactive_for_seconds) is not int
            or not 0 <= interrupting_inactive_for_seconds <= MAX_DURABLE_JSON_INTEGER
        ):
            raise ValueError(
                "interrupting_inactive_for_seconds must be a non-negative durable integer."
            )
        session = await self._engine.require_session(session_id)
        if session.status not in {
            SessionStatus.INTERRUPTING,
            SessionStatus.INTERRUPTED,
        } or (
            session.status is SessionStatus.INTERRUPTING
            and interrupting_inactive_for_seconds is None
        ):
            return False
        marker = await self._background_interruption_coordinator.load_pending_interruption_cascade(
            session.id
        )
        if marker is None:
            return False
        if session.status is SessionStatus.INTERRUPTING:
            (
                requires_completion_decision,
                admission_failure,
            ) = await self._engine.verifier_aware_task_execution_outcome(
                None,
                session_id=session.id,
                admit_session=False,
            )
            if admission_failure is not None:
                raise_task_store_operation_failure(admission_failure)
            if requires_completion_decision:
                return False

        already_scheduled = self._background_interruption_coordinator.is_admitted(session.id)
        if session.status is SessionStatus.INTERRUPTING:
            recovery_session_id = session.id

            async def admit_before_recovery_mutation() -> None:
                (
                    requires_completion_decision,
                    admission_failure,
                ) = await self._engine.verifier_aware_task_execution_outcome(
                    None,
                    session_id=recovery_session_id,
                )
                if admission_failure is not None:
                    raise_task_store_operation_failure(admission_failure)
                if requires_completion_decision:
                    raise TaskCompletionDecisionRequired(
                        "Contracted tasks require the verifier-aware execution entrance."
                    ) from None

            with suppress_interruption_cascade():
                recovery = await self._incomplete_recovery.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id=session.id,
                        inactive_for_seconds=interrupting_inactive_for_seconds,
                        reason="interruption_cascade_operator_recovery",
                        metadata={"source": "recovery_plan"},
                    ),
                    before_mutation=admit_before_recovery_mutation,
                )
            session = await self._engine.require_session(session.id)
            if session.status is not SessionStatus.INTERRUPTED:
                raise RuntimeError(
                    "Could not finalize the interruption-cascade parent during recovery: "
                    f"{recovery.actions!r}."
                )
        if already_scheduled:
            task = self._session_finalization.schedule_background_interruption_cascade(
                parent_session_id=session.id,
                interrupt_payload=marker["interrupt_payload"],
                create_if_missing=False,
            )
            if task is not None:
                await asyncio.shield(task)
        else:
            await self._background_interruption_coordinator.run_cascade(
                parent_session_id=session.id,
                interrupt_payload=marker["interrupt_payload"],
                create_if_missing=False,
            )
        return not already_scheduled

    async def resume_pending_interruption_cascades(
        self,
        *,
        interrupting_inactive_for_seconds: int | None = None,
        startup_preflight: Callable[[str, int], Awaitable[tuple[RecoveryBlockerCode, ...] | None]]
        | None = None,
    ) -> int:
        """Resume durable descendant interruption work left by an earlier process.

        Both ``interrupting`` and ``interrupted`` parents are inspected. An
        ``interrupting`` parent is finalized only when
        ``interrupting_inactive_for_seconds`` is supplied and the store can fence that
        inactive run. Work remains checkpointed until traversal succeeds, so
        another restart can retry it safely. Returns the number of roots scheduled.
        """

        if interrupting_inactive_for_seconds is not None and (
            type(interrupting_inactive_for_seconds) is not int
            or not 0 <= interrupting_inactive_for_seconds <= MAX_DURABLE_JSON_INTEGER
        ):
            raise ValueError(
                "interrupting_inactive_for_seconds must be a non-negative durable integer."
            )

        scheduled = 0
        blocked_ids: set[str] = set()
        blocked: list[StartupRecoveryBlockedSession] = []
        deferred_ids: set[str] = set()
        skipped_ids: set[str] = set()
        sweep_count = self._startup_recovery_result.sweep_count + 1

        def report(*, status: Literal["running", "completed", "failed"] = "running") -> None:
            self._startup_recovery_result = StartupRecoveryResult(
                completed=status == "completed",
                status=status,
                sweep_count=sweep_count,
                scheduled_roots=scheduled,
                deferred_session_count=len(deferred_ids),
                skipped_session_count=len(skipped_ids),
                blocked_session_count=len(blocked_ids),
                blocked_sessions=tuple(blocked),
                blocked_sessions_truncated=len(blocked_ids) > len(blocked),
            )

        def record_blocked(session_id: str, codes: tuple[RecoveryBlockerCode, ...]) -> None:
            if session_id in blocked_ids:
                return
            blocked_ids.add(session_id)
            if len(blocked) < 100:
                blocked.append(
                    StartupRecoveryBlockedSession(session_id=session_id, blocker_codes=codes)
                )
            logger.warning(
                "Startup recovery blocked session=%s blocker=%s",
                self._secret_redactor.redact_text(session_id),
                ",".join(code.value for code in codes),
            )
            report()

        report()
        try:
            admitted_parent_ids: set[str] = set()
            for status in (SessionStatus.INTERRUPTING, SessionStatus.INTERRUPTED):
                if (
                    status == SessionStatus.INTERRUPTING
                    and interrupting_inactive_for_seconds is None
                ):
                    continue
                cursor: str | None = None
                while True:
                    result = (
                        await self.session_store.list_sessions_with_pending_interruption_cascade(
                            SessionQuery(
                                status=status,
                                inactive_for_seconds=(
                                    interrupting_inactive_for_seconds
                                    if status == SessionStatus.INTERRUPTING
                                    else None
                                ),
                                limit=1000,
                                cursor=cursor,
                                order_by=SessionOrder.CREATED_AT_ASC,
                            )
                        )
                    )
                    for session in result.sessions:
                        if session.id in admitted_parent_ids:
                            continue
                        if (
                            session.status == SessionStatus.INTERRUPTING
                            and interrupting_inactive_for_seconds is None
                        ):
                            continue
                        try:
                            marker = await self._background_interruption_coordinator.load_pending_interruption_cascade(
                                session.id
                            )
                        except (TypeError, ValueError):
                            record_blocked(session.id, (RecoveryBlockerCode.INVALID_DURABLE_STATE,))
                            continue
                        if marker is None:
                            continue
                        if session.status == SessionStatus.INTERRUPTING:
                            (
                                requires_completion_decision,
                                admission_failure,
                            ) = await self._engine.verifier_aware_task_execution_outcome(
                                None,
                                session_id=session.id,
                                admit_session=False,
                            )
                            if admission_failure is not None:
                                del marker, result, session
                                raise_task_store_operation_failure(admission_failure)
                            if requires_completion_decision:
                                # Keep both the stale parent and its durable cascade marker
                                # untouched for the verifier-aware recovery owner.
                                continue
                        already_scheduled = self._background_interruption_coordinator.is_admitted(
                            session.id
                        )
                        if session.status == SessionStatus.INTERRUPTING:
                            if startup_preflight is not None:
                                assert interrupting_inactive_for_seconds is not None
                                codes = await startup_preflight(
                                    session.id, interrupting_inactive_for_seconds
                                )
                                if codes is None:
                                    skipped_ids.add(session.id)
                                    report()
                                    continue
                                if codes:
                                    if set(codes) <= {
                                        RecoveryBlockerCode.ACTIVE_RECOVERY_CLAIM,
                                        RecoveryBlockerCode.ACTIVE_TASK_CLAIM,
                                    }:
                                        deferred_ids.add(session.id)
                                        report()
                                    else:
                                        record_blocked(session.id, codes)
                                    continue
                            recovery_session_id = session.id

                            async def admit_before_recovery_mutation(
                                session_id: str = recovery_session_id,
                            ) -> None:
                                (
                                    requires_completion_decision,
                                    admission_failure,
                                ) = await self._engine.verifier_aware_task_execution_outcome(
                                    None,
                                    session_id=session_id,
                                )
                                if admission_failure is not None:
                                    raise_task_store_operation_failure(admission_failure)
                                if requires_completion_decision:
                                    raise TaskCompletionDecisionRequired(
                                        "Contracted tasks require the verifier-aware execution "
                                        "entrance."
                                    ) from None

                            try:
                                with suppress_interruption_cascade():
                                    recovery = await self._incomplete_recovery.recover_incomplete_session(
                                        IncompleteSessionRecoveryRequest(
                                            session_id=session.id,
                                            inactive_for_seconds=interrupting_inactive_for_seconds,
                                            reason="interruption_cascade_startup_recovery",
                                            metadata={
                                                "source": "resume_pending_interruption_cascades"
                                            },
                                        ),
                                        before_mutation=admit_before_recovery_mutation,
                                    )
                            except ExecutionProfileMismatchError:
                                # A registration can change after read-only planning. Isolate
                                # this typed per-session rejection; store and unknown failures propagate.
                                record_blocked(
                                    session.id, (RecoveryBlockerCode.REGISTRATION_INCOMPATIBLE,)
                                )
                                continue
                            except ModelCompletionManualRecoveryRequired:
                                record_blocked(
                                    session.id, (RecoveryBlockerCode.MODEL_EFFECT_OUTCOME_UNKNOWN,)
                                )
                                continue
                            except KeyError:
                                # A root deleted after preflight is no longer startup work.
                                if await self.session_store.load(session.id) is not None:
                                    raise
                                skipped_ids.add(session.id)
                                report()
                                continue
                            recovered_session = await self.session_store.load(session.id)
                            if recovered_session is None:
                                skipped_ids.add(session.id)
                                report()
                                continue
                            session = recovered_session
                            if session.status != SessionStatus.INTERRUPTED:
                                logger.warning(
                                    "Could not finalize interruption cascade parent %s during "
                                    "startup recovery: %s",
                                    session.id,
                                    recovery.message,
                                )
                                continue
                        admitted_parent_ids.add(session.id)
                        self._session_finalization.schedule_background_interruption_cascade(
                            parent_session_id=session.id,
                            interrupt_payload=marker["interrupt_payload"],
                            create_if_missing=False,
                        )
                        if not already_scheduled:
                            scheduled += 1
                            report()
                    if result.next_cursor is not None and result.next_cursor == cursor:
                        raise RuntimeError("Startup recovery returned a repeated session cursor.")
                    cursor = result.next_cursor
                    if cursor is None:
                        break
        except BaseException:
            report(status="failed")
            raise
        report(status="completed")
        return scheduled

    def get_startup_recovery_status(self) -> StartupRecoveryResult:
        return self._startup_recovery_result.model_copy(deep=True)

    async def recover_abandoned_execution(
        self,
        session: Session,
        *,
        participant_context: CollaborationAccessContext | None = None,
        replays_tool_calls: bool = False,
    ) -> tuple[Event, ...]:
        from cayu.runtime._abandoned_session_recovery import recover_abandoned_execution

        await self._require_participant_execution(session, participant_context)
        return await recover_abandoned_execution(
            store=self.session_store,
            session=session,
            locally_active=self._session_control.has_active_tasks(session.id),
            recover=lambda request: self.recover_incomplete_session(
                request, participant_context=participant_context
            ),
            settle_model_dispatch=self._settle_abandoned_model_dispatch,
            replays_tool_calls=replays_tool_calls,
        )

    async def _settle_abandoned_model_dispatch(self, session: Session) -> tuple[Event, ...]:
        """Interrupt an assistant model call whose owning process is gone.

        Only an ordinary dispatched call qualifies: no terminal response, no provider
        operation that can be resumed or inspected, and no task contract that needs an
        explicit completion decision. Everything else stays on the operator path.
        """

        from cayu.runtime.provider_operations import load_recoverable_provider_operation_start

        active = await self.session_store.load_active_model_completion_stage(session.id)
        if active is None:
            return ()
        stage = active.stage
        context = model_completion_recovery_context_from_stage(stage)
        if (
            stage.state != "in_flight"
            or stage.purpose != "assistant-turn"
            or stage.intent.get("provider_operation_start") is not None
            or context is None
            or context.task_id is not None
        ):
            return ()
        if (
            await self.session_store.load_model_completion_stage_dispatch(
                session.id, stage.stage_id
            )
            is None
            or await self.session_store.load_model_completion_stage_settlement(
                session.id, stage.stage_id
            )
            is not None
        ):
            return ()
        try:
            if (
                await load_recoverable_provider_operation(self.session_store, stage) is not None
                or await load_recoverable_provider_operation_start(self.session_store, stage)
                is not None
            ):
                return ()
        except ProviderOperationEvidenceError:
            return ()
        result = await self._engine.recover_model_completion_stage(
            ModelCompletionManualRecoveryRequest(
                session_id=session.id,
                stage_id=stage.stage_id,
                expected_run_epoch=session.run_epoch,
                expected_session_instance_id=session.instance_id,
                terminal_status=SessionStatus.INTERRUPTED,
                inactive_for_seconds=0,
            )
        )
        return result.budget_events

    async def recover_incomplete_session(
        self,
        request: IncompleteSessionRecoveryRequest,
        *,
        participant_context: CollaborationAccessContext | None = None,
        execution_to_wait: _ExternalExecutionToWait | None = None,
    ) -> IncompleteSessionRecoveryResult:
        request = copy_incomplete_session_recovery_request(request)
        (
            requires_completion_decision,
            admission_failure,
        ) = await self._engine.verifier_aware_task_execution_outcome(
            None,
            session_id=request.session_id,
            admit_session=False,
        )
        if admission_failure is not None:
            del request
            raise_task_store_operation_failure(admission_failure)
        if requires_completion_decision:
            del request
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        interaction_id = await self._session_finalization.activate_latest_open_interaction(
            request.session_id
        )
        active_through = (
            None
            if interaction_id is None
            else await self._latest_interaction_activity_at(
                request.session_id,
                interaction_id,
            )
        )
        if active_through is not None:
            _set_session_interaction_recovered_active_through(
                request.session_id,
                active_through,
            )
        try:

            async def admit_before_mutation() -> None:
                (
                    requires_completion_decision,
                    admission_failure,
                ) = await self._engine.verifier_aware_task_execution_outcome(
                    None,
                    session_id=request.session_id,
                )
                if admission_failure is not None:
                    raise_task_store_operation_failure(admission_failure)
                if requires_completion_decision:
                    raise TaskCompletionDecisionRequired(
                        "Contracted tasks require the verifier-aware execution entrance."
                    ) from None

            return await self._recover_incomplete_session_after_admission(
                request,
                interaction_id=interaction_id,
                active_through=active_through,
                before_mutation=admit_before_mutation,
                participant_context=participant_context,
                execution_to_wait=execution_to_wait,
            )
        finally:
            if interaction_id is not None:
                _deactivate_session_interaction(request.session_id)

    async def recover_work_attempt_session(
        self,
        request: IncompleteSessionRecoveryRequest,
        *,
        interaction_id: str,
        before_mutation: Callable[[], Awaitable[None]],
        admission: WorkAttemptAdmission | None = None,
    ) -> IncompleteSessionRecoveryResult:
        """Settle one contracted predecessor under authenticated attempt authority."""

        request = copy_incomplete_session_recovery_request(request)
        work_attempt = (
            None
            if admission is None or admission.execution_entry is None
            else _authenticated_work_attempt_invocation(admission)
        )
        # A recovery generation replaces only the execution owner. Its durable
        # WorkAttempt keeps the same interaction identity, so generic abandoned-
        # session settlement must not publish an interaction terminal event that
        # the replacement would then silently reopen.
        _deactivate_session_interaction(request.session_id)
        # Recovery is entered only after the task store has replaced the expired
        # execution generation.  Retire any copied predecessor epoch in this
        # caller before the owned settlement task is created; otherwise an
        # in-process replacement can carry the crashed worker's context token
        # into the new generation and reject its own post-settlement transition.
        _deactivate_session_run_fence(request.session_id)

        async def settle_predecessor() -> IncompleteSessionRecoveryResult:
            return await self._incomplete_recovery.recover_incomplete_session(
                request,
                before_mutation=before_mutation,
                preserve_interaction_id=interaction_id,
                _work_attempt=work_attempt,
            )

        try:
            return await settle_work_attempt_owned_operation(
                settle_predecessor,
                operation_name="work-attempt-predecessor-settlement",
                preserved_failure_types=(
                    _WorkAttemptRecoveryAlreadyActive,
                    WorkAttemptExecutionClaimLost,
                    WorkAttemptRecoveryRequired,
                ),
                redactor=self._secret_redactor,
            )
        finally:
            _deactivate_session_interaction(request.session_id)
            _deactivate_session_run_fence(request.session_id)

    async def _recover_incomplete_session_after_admission(
        self,
        request: IncompleteSessionRecoveryRequest,
        *,
        interaction_id: str | None,
        active_through: datetime | None,
        before_mutation: Callable[[], Awaitable[None]],
        participant_context: CollaborationAccessContext | None = None,
        execution_to_wait: _ExternalExecutionToWait | None = None,
    ) -> IncompleteSessionRecoveryResult:
        retained_invocation_context: InvocationContext | None = None

        def retain_invocation_context(context: InvocationContext) -> None:
            nonlocal retained_invocation_context
            if (
                retained_invocation_context is not None
                and retained_invocation_context is not context
            ):
                raise RuntimeError(
                    "Incomplete-session recovery produced conflicting live authority."
                )
            retained_invocation_context = context

        result = await self._incomplete_recovery.recover_incomplete_session(
            request,
            execution_to_wait=execution_to_wait,
            before_mutation=before_mutation,
            participant_context=participant_context,
            retain_open_interaction_invocation=(interaction_id is not None),
            retain_invocation_context=(
                retain_invocation_context if interaction_id is not None else None
            ),
        )
        if any(
            action in result.actions
            for action in (
                IncompleteSessionRecoveryAction.AMBIGUOUS_PENDING_USER_INPUT,
                IncompleteSessionRecoveryAction.PENDING_ALLOCATION_CLEANUP,
            )
        ):
            return result
        if interaction_id is not None:
            _activate_session_interaction(request.session_id, interaction_id)
        await before_mutation()
        reconciled = await self._reconcile_recovered_interaction(
            result,
            recovered_active_through=active_through,
            invocation_context=retained_invocation_context,
        )
        if interaction_id is not None and retained_invocation_context is not None:
            await self._release_reconciled_recovery_invocation(
                request.session_id,
                invocation_context=retained_invocation_context,
            )
        return reconciled

    async def _release_reconciled_recovery_invocation(
        self,
        session_id: str,
        *,
        invocation_context: InvocationContext | None,
    ) -> None:
        """Release a public recovery epoch after exact interaction settlement."""

        session = await self.session_store.load(session_id)
        checkpoint = await self.session_store.load_checkpoint(session_id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if (
            session is None
            or active_profile is None
            or active_profile.run_epoch != session.run_epoch
        ):
            return
        settlement = await self.session_store.load_invocation_settlement_transition(
            session_id,
            expected_session_instance_id=session.instance_id,
            expected_active_invocation_profile=active_profile,
        )
        if settlement is None:
            return
        if (
            invocation_context is None
            or invocation_context.binding.session_id != session.id
            or invocation_context.binding.session_instance_id != session.instance_id
            or invocation_context.binding.run_epoch != session.run_epoch
            or invocation_context.active_profile != active_profile
        ):
            raise RuntimeError(
                "Reconciled recovery release lacks its authenticated invocation context."
            )
        _activate_session_run_fence(session)
        await self._environment_lifecycle.release_run_fence_after_environment_cleanup(
            session_id=session_id,
            execution_profile=invocation_context.profile,
            invocation_context=invocation_context,
        )

    async def recover_incomplete_sessions(
        self,
        request: IncompleteSessionsRecoveryRequest,
    ) -> IncompleteSessionsRecoveryPage:
        request = copy_incomplete_sessions_recovery_request(request)
        active_through_by_session: dict[str, datetime] = {}

        async def admit_and_activate_interaction(session_id: str) -> None:
            (
                requires_completion_decision,
                admission_failure,
            ) = await self._engine.verifier_aware_task_execution_outcome(
                None,
                session_id=session_id,
                admit_session=False,
            )
            if admission_failure is not None:
                raise_task_store_operation_failure(admission_failure)
            if requires_completion_decision:
                raise TaskCompletionDecisionRequired(
                    "Contracted tasks require the verifier-aware execution entrance."
                ) from None
            interaction_id = await self._session_finalization.activate_latest_open_interaction(
                session_id
            )
            if interaction_id is not None:
                active_through_by_session[session_id] = await self._latest_interaction_activity_at(
                    session_id,
                    interaction_id,
                )
                _set_session_interaction_recovered_active_through(
                    session_id,
                    active_through_by_session[session_id],
                )

        async def admit_before_mutation(session_id: str) -> None:
            (
                requires_completion_decision,
                admission_failure,
            ) = await self._engine.verifier_aware_task_execution_outcome(
                None,
                session_id=session_id,
            )
            if admission_failure is not None:
                raise_task_store_operation_failure(admission_failure)
            if requires_completion_decision:
                raise TaskCompletionDecisionRequired(
                    "Contracted tasks require the verifier-aware execution entrance."
                ) from None

        async def deactivate_interaction(session_id: str) -> None:
            _deactivate_session_interaction(session_id)

        async def reconcile_result(
            result: IncompleteSessionRecoveryResult,
            invocation_context: InvocationContext | None,
        ) -> IncompleteSessionRecoveryResult:
            # Recovery cleanup can clear task-local interaction ownership. Restore
            # the still-open durable interaction before publishing its terminal
            # transition; the coordinator's after hook releases this ownership.
            await self._session_finalization.activate_latest_open_interaction(result.session_id)
            reconciled = await self._reconcile_recovered_interaction(
                result,
                recovered_active_through=active_through_by_session.get(result.session_id),
                invocation_context=invocation_context,
            )
            if invocation_context is not None:
                await self._release_reconciled_recovery_invocation(
                    result.session_id,
                    invocation_context=invocation_context,
                )
            return reconciled

        recovery = self._incomplete_recovery.recover_incomplete_sessions(
            request,
            before_recovery=admit_and_activate_interaction,
            before_mutation=admit_before_mutation,
            after_recovery=deactivate_interaction,
            reconcile_result=reconcile_result,
        )
        del request
        return await recovery

    async def _latest_interaction_activity_at(
        self,
        session_id: str,
        interaction_id: str,
    ) -> datetime:
        records = await self.session_store.query_events(
            EventQuery(
                session_id=session_id,
                interaction_id=interaction_id,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not records:
            raise RuntimeError(f"Interaction has no durable activity: {interaction_id}")
        return records[0].event.timestamp

    async def _publish_durable_recovery_interaction_transition(
        self,
        *,
        session: Session,
        recovered_active_through: datetime | None,
        invocation_context: InvocationContext | None,
    ) -> tuple[Session, Event | None, bool]:
        """Publish terminal recovery evidence under the applicable authority.

        A context is mandatory once the session has ever acquired invocation
        authority.  The context-free branch is retained only for a terminal
        session that was created and disposed before invocation admission.
        """

        if invocation_context is None:
            checkpoint = await self.session_store.load_checkpoint(session.id)
            active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            if active_profile is None and invocation_lifecycle_receipt_history_present(checkpoint):
                raise RuntimeError(
                    "Recovery interaction settlement lost durable invocation authority."
                )
            if active_profile is not None and not active_invocation_execution_profile_is_released(
                active_profile,
                session_id=session.id,
                run_epoch=session.run_epoch,
            ):
                raise RuntimeError("Recovery interaction settlement lost its invocation context.")
            execution_profile = None
        else:
            execution_profile = invocation_context.profile

        try:
            return await self._session_finalization.publish_interaction_transition(
                session=session,
                invocation_context=invocation_context,
                allow_released_invocation_authority=(invocation_context is None),
                agent_name=session.agent_name,
                environment_name=session.environment_name,
                to_status=session.status,
                from_statuses={session.status},
                recovered_active_through=recovered_active_through,
                execution_profile=execution_profile,
            )
        except asyncio.CancelledError as cancellation:
            if _is_interaction_transition_run_fence(cancellation):
                pop_exception_state(
                    cancellation,
                    _INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE,
                )
                pop_exception_state(
                    cancellation,
                    _INTERACTION_TRANSITION_RUN_FENCE_ATTRIBUTE,
                )
                raise

            transition_cancellation_outcome = _pop_interaction_transition_cancellation_outcome(
                cancellation
            )
            if transition_cancellation_outcome is None:
                transition_settled = False
                interaction_transition_failures: list[dict[str, Any]] = []
                interaction_transition: InteractionTransitionSpec | None = None
            else:
                (
                    transition_settled,
                    interaction_transition_failures,
                    interaction_transition,
                ) = transition_cancellation_outcome

            async def reconcile_terminal_interaction() -> None:
                if interaction_transition_failures and interaction_transition is not None:
                    committed = await self._session_finalization.record_durable_interaction_transition_cancellation(
                        session=session,
                        transition=interaction_transition,
                        failures=tuple(interaction_transition_failures),
                        agent_name=session.agent_name,
                        environment_name=session.environment_name,
                        expected_recovery_claim_id=None,
                    )
                    if committed:
                        return
                if transition_settled or _latest_session_invocation_interaction_is_settled(
                    session.id
                ):
                    return
                await self._session_finalization.publish_interaction_transition(
                    session=session,
                    invocation_context=invocation_context,
                    allow_released_invocation_authority=(invocation_context is None),
                    agent_name=session.agent_name,
                    environment_name=session.environment_name,
                    to_status=session.status,
                    from_statuses={session.status},
                    recovered_active_through=recovered_active_through,
                    execution_profile=execution_profile,
                )

            await _run_interaction_transition_cancellation_cleanup_steps(
                cancellation,
                supervisor=self._recovery_cleanup_supervisor,
                steps=(
                    (
                        "terminal recovery interaction reconciliation",
                        reconcile_terminal_interaction,
                    ),
                ),
            )
            raise

    async def _reconcile_recovered_interaction(
        self,
        result: IncompleteSessionRecoveryResult,
        *,
        recovered_active_through: datetime | None,
        invocation_context: InvocationContext | None,
    ) -> IncompleteSessionRecoveryResult:
        if any(
            action in result.actions
            for action in (
                IncompleteSessionRecoveryAction.TERMINALIZED_ZERO_WORK,
                IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,
                IncompleteSessionRecoveryAction.SKIPPED_UNREGISTERED_AGENT,
            )
        ):
            return result
        if any(
            event.type == EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED for event in result.events
        ):
            # Exact continuation is paused for an operator decision, not
            # terminally disposed. Keep the interaction open so the authorized
            # fallback can resume the same logical model step.
            return result
        if _current_session_interaction_id(result.session_id) is None:
            return result
        session = await self.session_store.load(result.session_id)
        if session is None:
            return result
        if result.status not in {
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.INTERRUPTED,
        }:
            return result
        interaction_id = _current_session_interaction_id(result.session_id)
        latest_records = await self.session_store.query_events(
            EventQuery(
                session_id=result.session_id,
                interaction_id=interaction_id,
                event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if latest_records and latest_records[0].event.type in INTERACTION_TERMINAL_EVENT_TYPES:
            expected_terminal_type = {
                SessionStatus.COMPLETED: EventType.INTERACTION_COMPLETED,
                SessionStatus.FAILED: EventType.INTERACTION_FAILED,
                SessionStatus.INTERRUPTED: EventType.INTERACTION_INTERRUPTED,
            }[result.status]
            if latest_records[0].event.type is not expected_terminal_type:
                if await self._terminal_evidence.has_completed_queued_predecessor(
                    session, latest_records[0].event
                ):
                    return result
                raise RuntimeError(
                    "Recovered interaction terminal evidence conflicts with session status."
                )
            # The real transition already committed under the live context in
            # the inner recovery path.  Do not reuse that now-released context
            # merely to replay a read-only acknowledgement.
            return result
        # Recovery already reconstructed and validated the exact live context.
        # Retain that lineage through terminal publication instead of consulting
        # mutable registrations or treating an equal durable profile as live authority.
        # A context-free transition is allowed only for a pre-admission session;
        # the publication helper positively proves that no invocation profile exists.
        _, interaction_event, _ = await self._publish_durable_recovery_interaction_transition(
            session=session,
            recovered_active_through=recovered_active_through,
            invocation_context=invocation_context,
        )
        if interaction_event is None:
            return result
        return result.model_copy(update={"events": (*result.events, interaction_event)})

    async def settle_foreground_child_terminal(
        self,
        wait: ForegroundChildWait,
        event: Event,
        *,
        before_mutation: Callable[[], Awaitable[None]],
    ) -> None:
        from cayu.runtime._foreground_child_terminal_settlement import (
            settle_foreground_child_terminal,
        )

        tool = self._get_registered_agent(wait.parent_effect.agent_name).tools.get(
            wait.parent_effect.tool_name
        )
        if tool is None or tool.child_session_recovery is None:
            raise SessionRunFenced(
                "Foreground terminal cleanup lacks its registered child matcher."
            )
        await settle_foreground_child_terminal(
            wait,
            event,
            store=self.session_store,
            matcher=tool.child_session_recovery,
            recover=self._incomplete_recovery.recover_incomplete_session,
            before_mutation=before_mutation,
        )

    async def resume_foreground_child(
        self,
        terminal: ForegroundChildTerminal,
        *,
        before_mutation: Callable[[], Awaitable[None]],
    ) -> None:
        """Continue the original pending round using its persisted run configuration."""
        await before_mutation()
        wait = terminal.wait
        from cayu.runtime._foreground_child_continuation import (
            load_attached_foreground_continuation,
        )

        parent = await self.session_store.load(wait.parent_effect.session_id)
        if parent is None:
            raise SessionRunFenced("Foreground continuation parent disappeared.")
        if await self._recovery_coordinator.resume_foreground_gate(
            parent, terminal, before_mutation=before_mutation
        ):
            return
        attached = await load_attached_foreground_continuation(parent, store=self.session_store)
        if attached is not None and attached.terminal == terminal:
            retained_profile = active_invocation_execution_profile_from_checkpoint(
                await self.session_store.load_checkpoint(parent.id)
            )
            if (
                retained_profile is None
                or retained_profile.interaction_id != wait.parent_effect.interaction_id
                or retained_profile.profile.fingerprint
                != wait.parent_effect.execution_profile_fingerprint
            ):
                raise SessionRunFenced(
                    "Attached foreground continuation lost its original profile."
                )

            async def require_attached_ownership() -> None:
                await before_mutation()
                current = await self.session_store.load(parent.id)
                if (
                    current is None
                    or await load_attached_foreground_continuation(
                        current, store=self.session_store
                    )
                    != attached
                ):
                    raise SessionRunFenced(
                        "Attached foreground continuation changed during recovery."
                    )

            if parent.status in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}:
                _deactivate_session_interaction(parent.id)
                _deactivate_session_run_fence(parent.id)
                try:
                    await self._incomplete_recovery.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(
                            session_id=parent.id,
                            inactive_for_seconds=0,
                            reason="Recovering the owned foreground child continuation.",
                        ),
                        before_mutation=require_attached_ownership,
                        preserve_interaction_id=wait.parent_effect.interaction_id,
                    )
                finally:
                    _deactivate_session_interaction(parent.id)
                    _deactivate_session_run_fence(parent.id)
            stream = self.resume_session(
                request=attached.request.model_copy(
                    update={
                        "loop_policies": await self._foreground_gate_policy_owner.for_attached(
                            self.session_store, parent=parent, continuation=attached
                        ),
                    }
                ),
                task_id=attached.task_id,
                start_event_payload_extra={},
                start_task_on_enter=False,
                required_foreground_wait=wait,
                required_foreground_terminal=terminal,
                required_foreground_continuation=attached,
                foreground_before_mutation=require_attached_ownership,
            )
            async with _close_delegated_event_stream(stream) as owned_stream:
                async for _ in owned_stream:
                    pass
            self._foreground_gate_policy_owner.release_attached(parent, attached)
            return
        checkpoint, pending = await pending_round_reader.load_pending_tool_round(
            self.session_store,
            wait.parent_effect.session_id,
        )
        if pending is None or pending.max_steps is None:
            raise RuntimeError("Foreground continuation lacks its original pending round.")
        if pending.tool_round_id != wait.parent_effect.tool_round_id:
            raise RuntimeError("Foreground continuation conflicts with its original tool round.")
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if (
            active_profile is None
            or active_profile.interaction_id != wait.parent_effect.interaction_id
            or (
                pending.interaction_id is not None
                and pending.interaction_id != active_profile.interaction_id
            )
        ):
            raise RuntimeError("Foreground continuation conflicts with its original interaction.")
        if (
            active_profile.profile.fingerprint != wait.parent_effect.execution_profile_fingerprint
            or (
                pending.execution_profile_fingerprint is not None
                and pending.execution_profile_fingerprint != active_profile.profile.fingerprint
            )
        ):
            raise RuntimeError("Foreground continuation conflicts with its original profile.")
        if pending.limits is None or pending.budget_limits is None:
            raise RuntimeError("Foreground continuation lacks its original limits.")
        request = ForegroundChildResumeRequest(
            session_id=wait.parent_effect.session_id,
            loop_policies=self._foreground_gate_policy_owner.for_wait(parent, wait),
            messages=[],
            metadata=pending.request_metadata,
            max_steps=pending.max_steps,
            limits=pending.limits,
            budget_limits=pending.budget_limits,
            retry_policy=pending.retry_policy,
            structured_output=pending.structured_output,
            thinking=pending.thinking,
        )
        stream = self.resume_session(
            request=request,
            task_id=pending.task_id,
            start_event_payload_extra={},
            start_task_on_enter=False,
            required_foreground_wait=wait,
            required_foreground_terminal=terminal,
            foreground_before_mutation=before_mutation,
        )
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for _ in owned_stream:
                pass

    async def resume_session(
        self,
        *,
        request: ResumeRequest,
        task_id: str | None,
        start_event_payload_extra: dict[str, Any],
        start_task_on_enter: bool,
        adoption_request_fingerprint: str | None = None,
        source_execution_profile: ExecutionProfileIdentity | None = None,
        required_execution_profile: ExecutionProfileIdentity | None = None,
        required_session_instance_fingerprint: str | None = None,
        required_task_session_instance_id: str | None = None,
        run_operation_id: str | None = None,
        terminal_event_id: str | None = None,
        queue_task_id: str | None = None,
        queued_dispatch_id: str | None = None,
        required_foreground_wait: ForegroundChildWait | None = None,
        required_foreground_terminal: ForegroundChildTerminal | None = None,
        required_foreground_continuation: ForegroundParentContinuation | None = None,
        foreground_before_mutation: Callable[[], Awaitable[None]] | None = None,
        continuation_handoff: _ResumeAdmissionHandoff | None = None,
        participant_context: CollaborationAccessContext | None = None,
        execution_to_wait: _ExternalExecutionToWait | None = None,
    ) -> AsyncGenerator[Event, None]:
        preparation = await self._engine.prepare_resume(
            request=request,
            task_id=task_id,
            start_event_payload_extra=start_event_payload_extra,
            start_task_on_enter=start_task_on_enter,
            adoption_request_fingerprint=adoption_request_fingerprint,
            source_execution_profile=source_execution_profile,
            required_execution_profile=required_execution_profile,
            required_session_instance_fingerprint=required_session_instance_fingerprint,
            required_task_session_instance_id=required_task_session_instance_id,
            run_operation_id=run_operation_id,
            terminal_event_id=terminal_event_id,
            queue_task_id=queue_task_id,
            queued_dispatch_id=queued_dispatch_id,
            required_foreground_wait=required_foreground_wait,
            required_foreground_terminal=required_foreground_terminal,
            required_foreground_continuation=required_foreground_continuation,
            foreground_before_mutation=foreground_before_mutation,
            continuation_handoff=continuation_handoff,
            participant_context=participant_context,
            execution_to_wait=execution_to_wait,
        )
        recovery_events = await self.recover_abandoned_execution(
            preparation.loaded_session,
            participant_context=participant_context,
            replays_tool_calls=True,
        )
        for event in recovery_events:
            yield event
        stream = self._engine.resume(preparation)
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield event

    async def resume(
        self,
        request: ResumeRequest,
        *,
        store_resolved_session_id: str | None = None,
        continuation_handoff: _ResumeAdmissionHandoff | None = None,
        participant_context: CollaborationAccessContext | None = None,
        execution_to_wait: _ExternalExecutionToWait | None = None,
    ) -> AsyncGenerator[Event, None]:
        request = session_request_boundary.prepare_resume_request(
            request,
            redactor=self._secret_redactor,
            store_resolved_session_id=store_resolved_session_id,
        )
        session = await self.session_store.load(request.session_id)
        if session is not None:
            await self._require_participant_execution(session, participant_context)
        from cayu.runtime._resume_configuration import inherit_resume_configuration

        # Bind adoption replay to the caller's request before resolving controls
        # from mutable predecessor evidence. Candidate-profile checks still bind
        # the effective controls used by the admitted invocation.
        adoption_request_fingerprint = (
            None
            if request.profile_adoption is None
            else execution_profile_adoption_request_fingerprint(
                request, redactor=self._secret_redactor
            )
        )
        configuration_differences: tuple[str, ...] = ()
        if session is not None:
            request, configuration_differences = await inherit_resume_configuration(
                self.session_store, session, request, self._effective_retry_policy(None)
            )
        task_id, task_session_instance_id = await self._engine.linked_resume_task_id(request)
        replayed_failure, replay_events = await self._engine.replay_runtime_task_failure_if_needed(
            session_id=request.session_id,
            task_id=task_id,
            task_worker_id=request.task_worker_id,
            task_handoff_id=request.task_handoff_id,
        )
        if replayed_failure:
            for event in replay_events:
                yield event
            return
        session_stream = self.resume_session(
            request=request,
            adoption_request_fingerprint=adoption_request_fingerprint,
            participant_context=participant_context,
            task_id=task_id,
            required_task_session_instance_id=task_session_instance_id,
            start_event_payload_extra={},
            start_task_on_enter=False,
            continuation_handoff=continuation_handoff,
            execution_to_wait=execution_to_wait,
        )
        session_id = request.session_id
        del request
        forwarded_stream = self._session_control.stream_with_out_of_band_events(
            session_id,
            session_stream,
        )
        billing_identity_cancellation: asyncio.CancelledError | None = None
        billing_identity_cancellation_group: BaseExceptionGroup | None = None
        propagated_cancellation: asyncio.CancelledError | None = None
        try:
            async for event in forwarded_stream:
                yield event
        except ExecutionProfileMismatchError as exc:
            if (
                configuration_differences
                and ExecutionProfileComponentClass.FINALIZATION in exc.changed_component_classes
            ):
                exc.add_note(
                    "Resume configuration differs: " + "; ".join(configuration_differences)
                )
            raise
        except asyncio.CancelledError as exc:
            billing_identity_cancellation = _detach_billing_cancellation_for_public(exc)
            if billing_identity_cancellation is None:
                detached_provider_cancellation = _detach_provider_cancellation_for_public(exc)
                if detached_provider_cancellation is None:
                    raise
                propagated_cancellation = detached_provider_cancellation
        except BaseExceptionGroup as exc:
            billing_identity_cancellation_group = detach_billing_identity_cancellation_group(exc)
            if billing_identity_cancellation_group is None:
                raise
        except GeneratorExit:
            await _close_async_iterator(forwarded_stream)
            raise
        del forwarded_stream, session_stream
        if billing_identity_cancellation is not None:
            raise billing_identity_cancellation
        if billing_identity_cancellation_group is not None:
            raise billing_identity_cancellation_group
        if propagated_cancellation is not None:
            raise propagated_cancellation
