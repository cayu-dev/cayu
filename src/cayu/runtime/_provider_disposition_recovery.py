"""Settle durable provider-operation decisions and resume their exact continuation."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import datetime
from typing import Any

from cayu.budgets._run_limit_accounting import (
    rebase_run_limit_accounting_context,
)
from cayu.budgets.base import (
    BudgetPolicy,
    copy_budget_policy,
    request_budget_limits_for_session,
)
from cayu.collaboration.access import CollaborationAccessContext
from cayu.events import (
    Event,
    EventType,
    copy_event,
)
from cayu.execution_units import (
    ModelAttemptIdentity,
    ModelStepIdentity,
)
from cayu.observability.hooks import RuntimeHookPhase
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._execution_profile_continuation import ExecutionProfileContinuation
from cayu.runtime._invocation_lifecycle import (
    InvocationContext,
    reconstruct_invocation_context,
)
from cayu.runtime._model_completion_contracts import (
    ModelCompletionRecoveryContext,
    model_completion_recovery_context_from_stage,
)
from cayu.runtime._model_errors import (
    _FallbackBillingCancellationStateCheckFailed,
    detach_billing_identity_cancellation_group,
)
from cayu.runtime._pending_tool_round_recovery import (
    RegisteredAgentResolver,
    RegisteredEnvironmentResolver,
)
from cayu.runtime._recovery_admission import (
    RecoveryAdmission,
    RecoveryMutationHook,
)
from cayu.runtime._recovery_ownership import (
    RecoveryOwnership,
    _recovery_abandonment_signal,
)
from cayu.runtime._recovery_requests import (
    ProviderOperationFailureRequest,
    RecoverySessionRunRequest,
    RecoveryTerminalEventRequest,
)
from cayu.runtime._run_limits import (
    RunLimitController,
)
from cayu.runtime._session_engine import SessionEngine
from cayu.runtime._session_finalization import (
    SessionFinalization,
)
from cayu.runtime._terminal_event_publication import TerminalEventPublication
from cayu.runtime.loop_policies import LoopPolicy
from cayu.runtime.provider_operations import (
    ProviderOperationEvidenceError,
    ProviderOperationPendingDisposition,
    ProviderOperationResolutionAction,
    ProviderOperationResolutionConflict,
    ProviderOperationResolutionResult,
    checkpoint_with_provider_operation_disposition_execution_owner,
    clear_pending_provider_operation_disposition,
    load_pending_provider_operation_disposition,
    provider_operation_resolution_outcome_event_id,
    validate_provider_operation_resolution_outcome_event,
)
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
)
from cayu.sessions.base import SessionRunFenced, SessionStore
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.interactions import (
    InteractionStatus,
    InteractionSummaryEvidence,
)
from cayu.sessions.records import (
    Session,
    SessionStatus,
    SessionStatusConflict,
)
from cayu.tools.exposure import (
    ResolvedToolExposureAuthority,
    tool_capability_ceiling_from_session_metadata,
    validate_resolved_tool_exposure_authority,
)


def _retried_model_step_tool_exposure_authority(
    authority: ResolvedToolExposureAuthority | None,
    registered_agent: runtime_records.RegisteredAgentState,
    session: Session,
) -> ResolvedToolExposureAuthority:
    """Return exact exposure authority for a same-model-step retry."""

    if authority is None:
        raise ProviderOperationEvidenceError(
            "Provider-operation fallback has no durable tool-exposure authority."
        )
    try:
        validated = validate_resolved_tool_exposure_authority(
            authority,
            registered_agent.tool_capabilities,
            catalogue_revision=registered_agent.tool_catalogue.revision,
        )
        capability_ceiling = tool_capability_ceiling_from_session_metadata(session.metadata)
    except (TypeError, ValueError) as exc:
        raise ProviderOperationEvidenceError(
            "Provider-operation fallback has invalid durable tool-exposure authority."
        ) from exc
    ceiling_names = frozenset(capability_ceiling.tool_names)
    if validated.ceiling_count != len(capability_ceiling.tool_names) or any(
        name not in ceiling_names for name in validated.tool_names
    ):
        raise ProviderOperationEvidenceError(
            "Provider-operation fallback tool exposure conflicts with the session capability "
            "ceiling."
        )
    return validated


class ProviderDispositionRecovery:
    """Settle durable provider-operation decisions and resume their exact continuation."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime],
        engine: SessionEngine,
        execution_profile_continuation: ExecutionProfileContinuation,
        loop_policies: tuple[LoopPolicy, ...],
        recovery_admission: RecoveryAdmission,
        recovery_ownership: RecoveryOwnership,
        require_participant_execution: Callable[
            [Session, CollaborationAccessContext | None], Awaitable[None]
        ],
        resolve_budget_policy: Callable[[], BudgetPolicy | None],
        resolve_registered_agent: RegisteredAgentResolver,
        resolve_registered_environment: RegisteredEnvironmentResolver,
        resolve_registered_provider: Callable[[str], runtime_records.RegisteredProvider],
        run_limit_controller: RunLimitController,
        runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
        session_finalization: SessionFinalization,
        session_store: SessionStore,
        terminal_event_publication: TerminalEventPublication,
    ) -> None:
        self._clock = clock
        self._engine = engine
        self._execution_profile_continuation = execution_profile_continuation
        self._loop_policies = loop_policies
        self._recovery_admission = recovery_admission
        self._recovery_ownership = recovery_ownership
        self._require_participant_execution = require_participant_execution
        self._resolve_budget_policy = resolve_budget_policy
        self._resolve_registered_agent = resolve_registered_agent
        self._resolve_registered_environment = resolve_registered_environment
        self._resolve_registered_provider = resolve_registered_provider
        self._run_limit_controller = run_limit_controller
        self._runtime_hooks = runtime_hooks
        self._session_finalization = session_finalization
        self._session_store = session_store
        self._terminal_event_publication = terminal_event_publication

    async def _claim_provider_operation_disposition_execution(
        self,
        *,
        pending: ProviderOperationPendingDisposition,
        task_worker_id: str | None,
        task_handoff_id: str | None,
    ) -> bool:
        """Atomically bind a pre-execution disposition to its first caller."""

        if pending.execution_claimed:
            if (
                pending.execution_task_worker_id,
                pending.execution_task_handoff_id,
            ) != (task_worker_id, task_handoff_id):
                raise ProviderOperationResolutionConflict(
                    "Provider-operation execution is owned by another task continuation."
                )
            return True

        def claim_execution(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            return checkpoint_with_provider_operation_disposition_execution_owner(
                checkpoint,
                expected=pending,
                task_worker_id=task_worker_id,
                task_handoff_id=task_handoff_id,
            )

        try:
            await self._session_store.transform_checkpoint(
                pending.session_id,
                claim_execution,
            )
        except ProviderOperationResolutionConflict:
            latest = await load_pending_provider_operation_disposition(
                self._session_store,
                pending.session_id,
            )
            if (
                latest is not None
                and latest[0].execution_claimed
                and latest[0].execution_task_worker_id == task_worker_id
                and latest[0].execution_task_handoff_id == task_handoff_id
            ):
                return False
            raise
        return True

    async def provider_operation_disposition_effect_is_durable(
        self,
        *,
        pending: ProviderOperationPendingDisposition,
        result: ProviderOperationResolutionResult,
        terminal_hook_authority: RecoveryTerminalEventRequest | None = None,
    ) -> bool:
        if pending.action is ProviderOperationResolutionAction.FAIL:
            terminal_event_id = provider_operation_resolution_outcome_event_id(
                result.record.resolution_id,
                "session_failed",
            )
            terminal_records = await self._session_store.query_events(
                EventQuery(
                    session_id=pending.session_id,
                    event_id=terminal_event_id,
                    limit=2,
                )
            )
            if not terminal_records:
                return False
            if len(terminal_records) != 1:
                raise ProviderOperationEvidenceError(
                    "Provider-operation failure has duplicate terminal evidence."
                )
            validate_provider_operation_resolution_outcome_event(
                terminal_records[0].event,
                resolution_event=result.event,
                outcome="session_failed",
                expected_execution_profile_fingerprint=pending.execution_profile_fingerprint,
            )
            for outcome in ("model_error", "interaction_failed"):
                event_id = provider_operation_resolution_outcome_event_id(
                    result.record.resolution_id,
                    outcome,
                )
                records = await self._session_store.query_events(
                    EventQuery(
                        session_id=pending.session_id,
                        event_id=event_id,
                        limit=2,
                    )
                )
                if len(records) != 1:
                    raise ProviderOperationEvidenceError(
                        "Provider-operation failure has incomplete durable evidence."
                    )
                validate_provider_operation_resolution_outcome_event(
                    records[0].event,
                    resolution_event=result.event,
                    outcome=outcome,
                    expected_execution_profile_fingerprint=(pending.execution_profile_fingerprint),
                )
            session = await self._session_store.load(pending.session_id)
            if session is None or session.status is not SessionStatus.FAILED:
                raise ProviderOperationEvidenceError(
                    "Provider-operation failure conflicts with durable session status."
                )
            if terminal_hook_authority is None:
                return False
            terminal_event = terminal_records[0].event
            if (
                terminal_hook_authority.event != terminal_event
                or terminal_hook_authority.phase is not RuntimeHookPhase.AFTER_SESSION_FAILED
                or terminal_hook_authority.session.id != session.id
                or terminal_hook_authority.session.instance_id != session.instance_id
                or terminal_hook_authority.session.run_epoch != session.run_epoch
                or terminal_hook_authority.session.status is not session.status
                or not terminal_hook_authority.terminal_event_already_durable
            ):
                raise ProviderOperationEvidenceError(
                    "Provider-operation terminal-hook authority conflicts with durable evidence."
                )
            return await self._terminal_event_publication.recovered_hooks_are_settled(
                terminal_hook_authority
            )

        if await self._provider_operation_fallback_terminal_outcome_is_durable(
            pending=pending,
            result=result,
        ):
            return True

        target_ordinal = pending.target_dispatch_ordinal
        if target_ordinal is None:
            raise ProviderOperationEvidenceError(
                "Fallback disposition lost its target dispatch ordinal."
            )
        target_stage_id = f"{pending.logical_step_id}:dispatch:{target_ordinal}"
        target_stage = await self._session_store.load_model_completion_stage(
            pending.session_id,
            target_stage_id,
        )
        if target_stage is not None:
            target_context = model_completion_recovery_context_from_stage(target_stage)
            if (
                target_stage.logical_step_id != pending.logical_step_id
                or target_stage.dispatch_ordinal != target_ordinal
                or target_context is None
                or target_context.execution_profile_fingerprint
                != pending.execution_profile_fingerprint
            ):
                raise ProviderOperationEvidenceError(
                    "Fallback disposition target stage has contradictory identity."
                )
            return True
        return False

    async def _provider_operation_fallback_terminal_outcome_is_durable(
        self,
        *,
        pending: ProviderOperationPendingDisposition,
        result: ProviderOperationResolutionResult,
    ) -> bool:
        """Recognize a typed pre-dispatch stop owned by this disposition."""

        session = await self._session_store.load(pending.session_id)
        if session is None or session.status is not SessionStatus.INTERRUPTED:
            return False
        resolution_records = await self._session_store.query_events(
            EventQuery(
                session_id=pending.session_id,
                event_id=result.event.id,
                limit=2,
            )
        )
        if len(resolution_records) != 1:
            raise ProviderOperationEvidenceError(
                "Fallback terminal outcome has missing or duplicate resolution evidence."
            )
        resolution_sequence = resolution_records[0].sequence
        interaction_records = await self._session_store.query_events(
            EventQuery(
                session_id=pending.session_id,
                interaction_id=result.event.interaction_id,
                event_type=EventType.INTERACTION_INTERRUPTED,
                after_sequence=resolution_sequence,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=2,
            )
        )
        terminal_records = await self._session_store.query_events(
            EventQuery(
                session_id=pending.session_id,
                event_type=EventType.SESSION_INTERRUPTED,
                after_sequence=resolution_sequence,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=2,
            )
        )
        if not interaction_records or not terminal_records:
            return False
        if len(interaction_records) != 1 or len(terminal_records) != 1:
            raise ProviderOperationEvidenceError(
                "Fallback terminal outcome has contradictory terminal evidence."
            )
        try:
            interaction = InteractionSummaryEvidence.model_validate(
                interaction_records[0].event.payload
            )
        except (TypeError, ValueError):
            raise ProviderOperationEvidenceError(
                "Fallback terminal outcome has malformed interaction evidence."
            ) from None
        if interaction.status is not InteractionStatus.INTERRUPTED:
            raise ProviderOperationEvidenceError(
                "Fallback terminal outcome has contradictory interaction status."
            )
        terminal_payload = terminal_records[0].event.payload
        interruption_type = terminal_payload.get("interruption_type")
        if interruption_type == "operator_requested":
            return True
        if (
            interruption_type != "limit_reached"
            and terminal_payload.get("terminal_evidence_repaired") is not True
        ):
            return False
        limit_records = await self._session_store.query_events(
            EventQuery(
                session_id=pending.session_id,
                event_type=EventType.SESSION_LIMIT_REACHED,
                after_sequence=resolution_sequence,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        return bool(limit_records)

    async def retire_completed_provider_operation_disposition(
        self,
        *,
        pending: ProviderOperationPendingDisposition,
        result: ProviderOperationResolutionResult,
        terminal_hook_authority: RecoveryTerminalEventRequest | None = None,
    ) -> bool:
        await self.settle_provider_operation_disposition_reservations(
            pending=pending,
            result=result,
        )
        if not await self.provider_operation_disposition_effect_is_durable(
            pending=pending,
            result=result,
            terminal_hook_authority=terminal_hook_authority,
        ):
            return False
        await clear_pending_provider_operation_disposition(self._session_store, pending)
        return True

    async def settle_provider_operation_disposition_reservations(
        self,
        *,
        pending: ProviderOperationPendingDisposition,
        result: ProviderOperationResolutionResult,
    ) -> tuple[Event, ...]:
        """Settle the source dispatch before replacement or terminalization."""

        stage = await self._session_store.load_model_completion_stage(
            pending.session_id,
            pending.stage_id,
        )
        if stage is None:
            raise ProviderOperationEvidenceError(
                "Provider-operation disposition lost its source stage."
            )
        recovery_context = model_completion_recovery_context_from_stage(stage)
        if pending.action is ProviderOperationResolutionAction.FALLBACK_RETRY:
            if recovery_context is None:
                raise ProviderOperationEvidenceError(
                    "Provider-operation fallback requires durable model-completion context."
                )
            session = await self._session_store.load(pending.session_id)
            if session is None:
                raise KeyError(f"Session not found: {pending.session_id}")
            registered_agent = self._resolve_registered_agent(session.agent_name)
            _retried_model_step_tool_exposure_authority(
                recovery_context.tool_exposure,
                registered_agent,
                session,
            )
        if not stage.reservation_ids:
            return ()
        if recovery_context is None:
            raise ProviderOperationEvidenceError(
                "Budgeted provider-operation disposition has no accounting context."
            )
        model_attempt_id = stage.intent.get("model_attempt_id")
        provider_name = stage.intent.get("provider_name")
        if type(model_attempt_id) is not str or type(provider_name) is not str:
            raise ProviderOperationEvidenceError(
                "Provider-operation disposition lost its dispatch identity."
            )
        session = await self._session_store.load(pending.session_id)
        if session is None:
            raise KeyError(f"Session not found: {pending.session_id}")
        reason = (
            "provider operation explicitly failed; usage unknown; charged reserved amount"
            if pending.action is ProviderOperationResolutionAction.FAIL
            else (
                "provider operation fallback accepted; original usage unknown; "
                "charged reserved amount"
            )
        )
        try:
            events = await (
                self._run_limit_controller.reconcile_unavailable_provider_operation_reservations(
                    reservation_ids=stage.reservation_ids,
                    recovery_contexts=recovery_context.budget_reservations,
                    session=session,
                    provider_name=provider_name,
                    model_attempt_identity=ModelAttemptIdentity(
                        model_step_id=stage.logical_step_id,
                        model_attempt_id=model_attempt_id,
                    ),
                    dispatch_id=stage.stage_id,
                    request_billing_identity=recovery_context.billing_identity,
                    reason=reason,
                    occurred_at=result.record.resolved_at,
                )
            )
        except (KeyError, NotImplementedError, TypeError, ValueError) as accounting_error:
            raise ProviderOperationEvidenceError(
                "Provider-operation disposition could not reconstruct its original budget "
                "reservation and pricing context."
            ) from accounting_error
        return tuple(events)

    async def finish_pending_provider_operation_disposition(
        self,
        *,
        pending: ProviderOperationPendingDisposition,
        result: ProviderOperationResolutionResult,
        invocation_context: InvocationContext | None = None,
        task_id: str | None = None,
        task_worker_id: str | None = None,
        task_handoff_id: str | None = None,
        after_admission: RecoveryMutationHook | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Finish one accepted disposition without replaying its old provider request."""

        loaded_session = await self._session_store.load(pending.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {pending.session_id}")
        await self._require_participant_execution(loaded_session, participant_context)
        if invocation_context is None:
            registered_agent = self._resolve_registered_agent(loaded_session.agent_name)
            registered_provider = self._resolve_registered_provider(loaded_session.provider_name)
            registered_environment = self._resolve_registered_environment(
                loaded_session.environment_name
            )
            budget_policy_snapshot = copy_budget_policy(self._resolve_budget_policy())
        else:
            if (
                type(invocation_context) is not InvocationContext
                or invocation_context.binding.session_id != loaded_session.id
                or invocation_context.binding.session_instance_id != loaded_session.instance_id
                or invocation_context.binding.run_epoch != loaded_session.run_epoch
            ):
                raise RuntimeError(
                    "Provider-operation disposition lost exact invocation authority."
                )
            registered_agent = invocation_context.registered_agent
            registered_provider = invocation_context.registered_provider
            registered_environment = invocation_context.registered_environment
            budget_policy_snapshot = invocation_context.budget_policy
        checkpoint = await self._session_store.load_checkpoint(loaded_session.id)
        recovery_context: ModelCompletionRecoveryContext | None = None
        if pending.action is ProviderOperationResolutionAction.FALLBACK_RETRY:
            stage = await self._session_store.load_model_completion_stage(
                pending.session_id,
                pending.stage_id,
            )
            if stage is None:
                raise RuntimeError("Resolved provider-operation stage is missing.")
            recovery_context = model_completion_recovery_context_from_stage(stage)
            if recovery_context is None:
                raise RuntimeError(
                    "Provider-operation fallback requires durable model-completion context."
                )
        execution_profile_snapshot = await self._execution_profile_continuation.validate(
            session=loaded_session,
            checkpoint=checkpoint,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            budget_policy=budget_policy_snapshot,
            request_budget_limits=(
                () if recovery_context is None else recovery_context.budget_limits
            ),
            structured_output=(
                None if recovery_context is None else recovery_context.structured_output
            ),
            thinking=None if recovery_context is None else recovery_context.thinking,
            max_steps=None if recovery_context is None else recovery_context.max_steps,
            limits=None if recovery_context is None else recovery_context.limits,
            retry_policy=(None if recovery_context is None else recovery_context.retry_policy),
            invocation_semantics_available=recovery_context is not None,
            require_open_interaction=not (
                pending.action is ProviderOperationResolutionAction.FAIL
                and loaded_session.status is SessionStatus.FAILED
            ),
        )
        if invocation_context is None:
            invocation_context = reconstruct_invocation_context(
                runtime_hooks=self._runtime_hooks,
                loop_policies=self._loop_policies,
                session=loaded_session,
                execution_profile_snapshot=execution_profile_snapshot,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                budget_policy=budget_policy_snapshot,
            )
        elif invocation_context.active_profile != execution_profile_snapshot:
            raise RuntimeError("Provider-operation disposition substituted its execution profile.")
        else:
            execution_profile_snapshot = invocation_context.active_profile

        for settlement_event in await self.settle_provider_operation_disposition_reservations(
            pending=pending,
            result=result,
        ):
            yield settlement_event

        if await self.retire_completed_provider_operation_disposition(
            pending=pending,
            result=result,
        ):
            return

        if pending.action is ProviderOperationResolutionAction.FALLBACK_RETRY:
            if loaded_session.status is not SessionStatus.INTERRUPTED:
                raise SessionStatusConflict(
                    "Fallback retry can continue only from an interrupted session."
                )
        elif loaded_session.status not in {
            SessionStatus.INTERRUPTED,
            SessionStatus.FAILED,
        }:
            raise SessionStatusConflict("Fail resolution requires interrupted provider work.")
        if pending.action is ProviderOperationResolutionAction.FAIL:
            if after_admission is not None:
                await after_admission()
            if not await self._claim_provider_operation_disposition_execution(
                pending=pending,
                task_worker_id=task_worker_id,
                task_handoff_id=task_handoff_id,
            ):
                return
            refreshed = await load_pending_provider_operation_disposition(
                self._session_store,
                pending.session_id,
            )
            if refreshed is None:
                return
            pending, result = refreshed
            async for event in self._session_finalization.fail_recovered_provider_operation(
                ProviderOperationFailureRequest(
                    resolution_event=result.event,
                    session=loaded_session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile_snapshot.profile,
                    task_id=task_id,
                    task_worker_id=task_worker_id,
                    task_handoff_id=task_handoff_id,
                    legacy_resolution_without_profile=(
                        result.record.execution_profile_fingerprint is None
                    ),
                    invocation_context=invocation_context,
                )
            ):
                yield event
            failed_session = await self._recovery_ownership.require_session(pending.session_id)
            terminal_event_id = provider_operation_resolution_outcome_event_id(
                result.record.resolution_id,
                "session_failed",
            )
            terminal_records = await self._session_store.query_events(
                EventQuery(
                    session_id=pending.session_id,
                    event_id=terminal_event_id,
                    limit=2,
                )
            )
            if len(terminal_records) != 1:
                raise ProviderOperationEvidenceError(
                    "Provider-operation failure has incomplete terminal evidence."
                )
            terminal_hook_authority = RecoveryTerminalEventRequest(
                event=copy_event(terminal_records[0].event),
                phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                session=failed_session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=invocation_context,
                terminal_event_already_durable=True,
                yield_durable_terminal_event=False,
            )
            if not await self.retire_completed_provider_operation_disposition(
                pending=pending,
                result=result,
                terminal_hook_authority=terminal_hook_authority,
            ):
                raise RuntimeError(
                    "Provider-operation failure disposition has no durable terminal outcome."
                )
            return

        if recovery_context is None:
            raise RuntimeError(
                "Provider-operation fallback requires durable model-completion context."
            )
        session: Session | None = None
        fallback_stream: AsyncGenerator[Event, None] | None = None
        authoritative_failure: BaseException | None = None
        detached_billing_failure: BaseExceptionGroup | None = None
        billing_state_check_failure: RuntimeError | None = None
        try:

            def claim_fallback_execution(
                _session: Session,
                current_checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any]:
                try:
                    return checkpoint_with_provider_operation_disposition_execution_owner(
                        current_checkpoint,
                        expected=pending,
                        task_worker_id=task_worker_id,
                        task_handoff_id=task_handoff_id,
                    )
                except ProviderOperationResolutionConflict:
                    raise SessionRunFenced(
                        "Provider-operation fallback execution ownership changed."
                    ) from None

            (
                session,
                resumed_event,
            ) = await self._recovery_admission.transition_recovery_session_to_running(
                loaded_session,
                checkpoint=checkpoint,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile_snapshot=execution_profile_snapshot,
                checkpoint_transform=claim_fallback_execution,
                preserve_open_interaction_on_failure=True,
                after_admission=after_admission,
                invocation_context=invocation_context,
            )
            # Admission has already advanced the durable epoch. Cleanup must
            # own that epoch before any fallible read or public event yield.
            invocation_context = invocation_context.with_rebound_session(
                session,
                active_profile=execution_profile_snapshot.model_copy(
                    update={"run_epoch": session.run_epoch}
                ),
            )
            refreshed = await load_pending_provider_operation_disposition(
                self._session_store,
                pending.session_id,
            )
            if refreshed is None:
                raise ProviderOperationEvidenceError(
                    "Fallback execution claim lost its pending disposition."
                )
            pending, result = refreshed
            if resumed_event is not None:
                yield resumed_event

            fallback_stream = self.run_pending_provider_operation_fallback(
                pending=pending,
                participant_context=participant_context,
                result=result,
                session=session,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                execution_profile_snapshot=execution_profile_snapshot,
                recovery_context=recovery_context,
                budget_policy=budget_policy_snapshot,
                release_run_fence_on_cleanup=False,
                task_worker_id=task_worker_id,
                task_handoff_id=task_handoff_id,
                invocation_context=invocation_context,
            )
            try:
                async for event in fallback_stream:
                    yield event
            finally:
                await fallback_stream.aclose()
        except BaseExceptionGroup as exc:
            detached_billing_failure = detach_billing_identity_cancellation_group(exc)
            if detached_billing_failure is None:
                authoritative_failure = exc
                raise
            authoritative_failure = detached_billing_failure
        except _FallbackBillingCancellationStateCheckFailed:
            billing_state_check_failure = RuntimeError(
                "Session interruption state check failed after provider billing cancellation"
            )
            authoritative_failure = billing_state_check_failure
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            if session is not None:
                await self._recovery_admission.cleanup_entrypoint_handoff(
                    stream=None,
                    session_id=session.id,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    authoritative_failure=authoritative_failure,
                    finalize_abandoned=False,
                    release_run_fence=True,
                    execution_profile=execution_profile_snapshot.profile,
                    invocation_context=invocation_context,
                )
        if detached_billing_failure is not None:
            authoritative_failure = None
            fallback_stream = None
            del registered_provider, registered_environment
            raise detached_billing_failure from None
        if billing_state_check_failure is not None:
            authoritative_failure = None
            fallback_stream = None
            del registered_provider, registered_environment
            raise billing_state_check_failure from None

    async def run_pending_provider_operation_fallback(
        self,
        *,
        pending: ProviderOperationPendingDisposition,
        result: ProviderOperationResolutionResult,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
        recovery_context: ModelCompletionRecoveryContext,
        budget_policy: BudgetPolicy | None,
        release_run_fence_on_cleanup: bool,
        task_worker_id: str | None = None,
        task_handoff_id: str | None = None,
        invocation_context: InvocationContext | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Run the accepted fallback from an already fenced running session."""

        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or registered_agent is not invocation_context.registered_agent
            or registered_provider is not invocation_context.registered_provider
            or registered_environment is not invocation_context.registered_environment
            or execution_profile_snapshot.profile is not invocation_context.profile
            or budget_policy is not invocation_context.budget_policy
        ):
            raise RuntimeError(
                "Provider-operation fallback substituted frozen invocation authority."
            )

        transcript = await self._session_store.load_transcript(session.id)
        recovery_events = await self._session_store.load_events(session.id)
        continued_accounting = recovery_context.run_limit_accounting
        if continued_accounting is not None:
            continued_accounting = rebase_run_limit_accounting_context(
                continued_accounting,
                session_id=session.id,
                limits=recovery_context.limits,
                budget_limits=request_budget_limits_for_session(
                    limits=recovery_context.budget_limits,
                    agent_name=registered_agent.spec.name,
                    causal_budget_id=session.causal_budget_id,
                ),
                events=recovery_events,
                reset_run_limits=False,
                reset_budgets=False,
                now=self._clock(),
            )
        if pending.source_step > recovery_context.max_steps:
            raise ProviderOperationEvidenceError(
                "Provider-operation fallback step exceeds its durable run limit."
            )
        session_stream = self._engine.continue_run(
            RecoverySessionRunRequest(
                session=session,
                participant_context=participant_context,
                messages=transcript,
                messages_to_append=[],
                max_steps=recovery_context.max_steps,
                limits=recovery_context.limits,
                budget_limits=recovery_context.budget_limits,
                retry_policy=recovery_context.retry_policy,
                structured_output=recovery_context.structured_output,
                thinking=recovery_context.thinking,
                request_metadata=recovery_context.request_metadata,
                task_id=recovery_context.task_id,
                task_worker_id=task_worker_id,
                task_handoff_id=task_handoff_id,
                start_event_type=None,
                start_event_payload={},
                start_task_on_enter=False,
                release_run_fence_on_exit=False,
                run_limit_accounting=continued_accounting,
                initial_model_step_identity=ModelStepIdentity(
                    model_step_id=pending.logical_step_id,
                ),
                initial_model_step_number=pending.source_step,
                initial_model_step_tool_exposure=(
                    _retried_model_step_tool_exposure_authority(
                        recovery_context.tool_exposure,
                        registered_agent,
                        session,
                    )
                ),
                preserve_failure_until_initial_provider_dispatch=True,
                invocation_context=(
                    invocation_context
                    if invocation_context is not None
                    else reconstruct_invocation_context(
                        runtime_hooks=self._runtime_hooks,
                        loop_policies=self._loop_policies,
                        session=session,
                        execution_profile_snapshot=execution_profile_snapshot,
                        registered_agent=registered_agent,
                        registered_provider=registered_provider,
                        registered_environment=registered_environment,
                        budget_policy=copy_budget_policy(budget_policy),
                        request_loop_policies=(),
                    )
                ),
            )
        )
        authoritative_failure: BaseException | None = None
        replacement_dispatch_durable = False
        limit_outcome_started = False
        try:
            async for event in session_stream:
                if event.type is EventType.PROVIDER_OPERATION_STARTING:
                    if not await self.retire_completed_provider_operation_disposition(
                        pending=pending,
                        result=result,
                    ):
                        raise RuntimeError(
                            "Provider-operation fallback started without its exact durable stage."
                        )
                    replacement_dispatch_durable = True
                elif event.type is EventType.SESSION_LIMIT_REACHED:
                    limit_outcome_started = True
                yield event
            if not await self.retire_completed_provider_operation_disposition(
                pending=pending,
                result=result,
            ):
                raise RuntimeError("Provider-operation fallback has no exact durable target stage.")
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            if (
                limit_outcome_started
                and isinstance(authoritative_failure, GeneratorExit)
                and not replacement_dispatch_durable
            ):
                # The limit decision is already durable and cannot lead to
                # provider dispatch. Finish its typed terminal evidence during
                # stream cleanup so recovery neither duplicates the limit event
                # nor mistakes this accepted disposition for pending work.
                async for _event in session_stream:
                    pass
                if not await self.retire_completed_provider_operation_disposition(
                    pending=pending,
                    result=result,
                ):
                    raise RuntimeError(
                        "Provider-operation fallback limit has no durable terminal outcome."
                    )
            cleanup = (
                self._recovery_admission.cleanup_entrypoint_handoff
                if release_run_fence_on_cleanup
                else self._recovery_admission.cleanup_recovery_handoff
            )
            await cleanup(
                stream=session_stream,
                session_id=session.id,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=(
                    replacement_dispatch_durable
                    and _recovery_abandonment_signal(authoritative_failure) is not None
                ),
                release_run_fence=release_run_fence_on_cleanup,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=invocation_context,
            )
