"""Shared model-completion recovery below session orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from cayu._task_wait import await_shielded_task_outcome, unexpected_child_cancellation_error
from cayu._validation import canonical_durable_json_bytes, require_clean_nonblank
from cayu.approvals.tools import PendingToolApproval, PendingToolCallApproval, ToolApprovalDecision
from cayu.approvals.user_input import (
    USER_INPUT_SUPERSESSION_INTENT_KEY,
    PendingUserInput,
    pending_user_input_identity,
    user_input_lifecycle_authority_from_checkpoint,
)
from cayu.budgets.base import BudgetPolicy, copy_budget_policy
from cayu.context.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputSpec,
    StructuredOutputStrategy,
)
from cayu.events import Event, EventType, copy_event, event_with_runtime_payload_authority
from cayu.execution_profiles import event_with_execution_profile_authority
from cayu.execution_units import ModelAttemptIdentity, ToolRoundIdentity
from cayu.memory.evidence import ContextExposureEvidenceKind, ContextExposureState
from cayu.messages import Message, MessageRole, ToolCallPart, ToolResultPart
from cayu.providers.operations import (
    ProviderOperationAdapter,
    ProviderOperationMode,
    ProviderOperationSnapshot,
    ProviderOperationStatus,
)
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _resume_ledger as resume_ledger
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _structured_output_tool_round as structured_output_tool_round
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._approval_support import _pending_approval_for_atomic_claim
from cayu.runtime._assistant_model_publication import AssistantModelPublication
from cayu.runtime._durable_tool_round import _environment_name
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._execution_profile_admission import ModelFailoverProfileResolution
from cayu.runtime._execution_profile_continuation import ExecutionProfileContinuation
from cayu.runtime._interruption_coordinator import _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY
from cayu.runtime._invocation_lifecycle import InvocationContext, reconstruct_invocation_context
from cayu.runtime._memory_evidence import (
    close_context_exposure_without_provider_effect,
    close_unrecoverable_context_exposure,
    recover_context_exposure,
)
from cayu.runtime._model_completion_contracts import (
    ModelCompletionBoundaryReconciliation,
    ModelCompletionManualRecoveryRequired,
    ModelCompletionRecoveryContext,
    model_completion_recovery_context_from_stage,
)
from cayu.runtime._model_execution_selection import ModelExecutionSelection
from cayu.runtime._model_failover_stage import model_failover_target_for_stored_stage
from cayu.runtime._model_step_executor import ModelStepExecutor
from cayu.runtime._run_limits import BorrowedAutomaticCompactionOutcomeUnknown, RunLimitController
from cayu.runtime._structured_output_tool_round import has_recoverable_structured_output_round
from cayu.runtime._user_input_recovery_evidence import UserInputRecoveryEvidence
from cayu.runtime.loop_policies import LoopPolicy
from cayu.runtime.provider_operations import (
    ProviderOperationEvidenceError,
    ProviderOperationRecoveryResult,
    ProviderOperationRecoveryStatus,
    RecoverableProviderOperation,
    RecoverableProviderOperationStart,
    load_recoverable_provider_operation,
    load_recoverable_provider_operation_start,
)
from cayu.runtime.stop_policy import StopLimit
from cayu.sessions import _model_completion_publication as model_completion_publication
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalOutcome,
    invocation_terminal_decision_from_checkpoint,
    invocation_terminal_decision_matches_recovery_profile,
)
from cayu.sessions._terminal_evidence import (
    _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
    _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
    _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
)
from cayu.sessions.base import (
    RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS,
    ActiveModelCompletionStage,
    ModelCompletionStage,
    RuntimePublicationReceipt,
    SessionStore,
    _model_completion_stage_promotion_statuses,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.records import EventRecord, Session, SessionStatus
from cayu.tools import _argument_publication as tool_argument_publication
from cayu.tools.catalogue import CALL_TOOL_NAME
from cayu.tools.gateway import gateway_lifecycle_matches_outer_call
from cayu.vaults.redaction import SecretRedactor


def _receiptless_pause_event_identity(
    event: Event,
) -> tuple[Literal["approval", "user-input"], str]:
    """Return one exact pause identity without accepting ordinary tool evidence."""

    has_approval_id = "approval_id" in event.payload
    has_input_id = "input_id" in event.payload
    if has_approval_id == has_input_id:
        raise RuntimeError(
            "Receipt-less tool evidence must carry exactly one approval_id or input_id."
        )
    field_name = "approval_id" if has_approval_id else "input_id"
    try:
        pause_id = require_clean_nonblank(event.payload[field_name], field_name)
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("Receipt-less tool evidence has a malformed pause identity.") from None
    return ("approval" if has_approval_id else "user-input"), pause_id


def _receiptless_exact_execution_evidence(
    records: list[EventRecord],
    *,
    identity: ToolRoundIdentity,
    expected_call_ids: set[str],
) -> list[tuple[int, Event]]:
    """Select exact round evidence while rejecting partial or conflicting identity."""

    evidence: list[tuple[int, Event]] = []
    for record in records:
        event = record.event
        tool_call_id = event.payload.get("tool_call_id")
        event_matches_call = type(tool_call_id) is str and tool_call_id in expected_call_ids
        event_matches_round = event.payload.get("tool_round_id") == identity.tool_round_id
        event_matches_attempt = (
            event.payload.get("model_step_id") == identity.model_step_id
            and event.payload.get("model_attempt_id") == identity.model_attempt_id
        )
        if not (event_matches_call or event_matches_round or event_matches_attempt):
            continue
        try:
            event_identity = ToolRoundIdentity.model_validate(
                {
                    "model_step_id": event.payload.get("model_step_id"),
                    "model_attempt_id": event.payload.get("model_attempt_id"),
                    "tool_round_id": event.payload.get("tool_round_id"),
                }
            )
        except (TypeError, ValueError):
            if any(
                event.payload.get(field_name) == expected
                for field_name, expected in (
                    ("model_step_id", identity.model_step_id),
                    ("model_attempt_id", identity.model_attempt_id),
                    ("tool_round_id", identity.tool_round_id),
                )
            ):
                raise RuntimeError(
                    "Durable tool lifecycle evidence has a partial execution identity."
                ) from None
            # Provider call ids can be reused. An unscoped historical event
            # is not evidence for this round and cannot contradict exact
            # round-owned terminal material.
            continue
        if event_identity != identity:
            if (
                event_matches_call
                and event_identity.tool_round_id != identity.tool_round_id
                and (
                    event_identity.model_step_id,
                    event_identity.model_attempt_id,
                )
                != (
                    identity.model_step_id,
                    identity.model_attempt_id,
                )
            ):
                continue
            raise RuntimeError(
                "Durable tool lifecycle evidence conflicts with its source model step."
            )
        evidence.append((record.sequence, event))
    return evidence


_MODEL_BOUNDARY_TOOL_TERMINAL_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }
)


@dataclass(frozen=True)
class ProviderOperationInterruptionAuthority:
    """Frozen provider-operation authority retained across offline interruption."""

    active_profile: ActiveInvocationExecutionProfile
    invocation_context: InvocationContext


class ModelCompletionRecovery:
    """Reconcile durable model work without session-engine or recovery orchestration."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        run_limit_controller: RunLimitController,
        secret_redactor: SecretRedactor,
        resolve_registered_agent: Callable[[str], runtime_records.RegisteredAgentState],
        resolve_registered_provider: Callable[[str], runtime_records.RegisteredProvider],
        resolve_registered_environment: Callable[
            [str | None], runtime_records.RegisteredEnvironment | None
        ],
        resolve_budget_policy: Callable[[], BudgetPolicy | None],
        execution_profile_continuation: ExecutionProfileContinuation,
        model_step_executor: ModelStepExecutor,
        assistant_model_publication: AssistantModelPublication,
        user_input_evidence: UserInputRecoveryEvidence,
        runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
        loop_policies: tuple[LoopPolicy, ...],
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._run_limit_controller = run_limit_controller
        self._secret_redactor = secret_redactor
        self._resolve_registered_agent = resolve_registered_agent
        self._resolve_registered_provider = resolve_registered_provider
        self._resolve_registered_environment = resolve_registered_environment
        self._resolve_budget_policy = resolve_budget_policy
        self._execution_profile_continuation = execution_profile_continuation
        self._model_step_executor = model_step_executor
        self._assistant_model_publication = assistant_model_publication
        self._user_input_evidence = user_input_evidence
        self._runtime_hooks = runtime_hooks
        self._loop_policies = loop_policies

    async def recoverable_provider_operation(
        self,
        stage: ModelCompletionStage,
        *,
        registered_provider: runtime_records.RegisteredProvider | None = None,
    ) -> (
        tuple[
            RecoverableProviderOperation | RecoverableProviderOperationStart,
            runtime_records.RegisteredProvider,
        ]
        | None
    ):
        try:
            operation = await load_recoverable_provider_operation(self._session_store, stage)
            if operation is None:
                operation = await load_recoverable_provider_operation_start(
                    self._session_store,
                    stage,
                )
        except ProviderOperationEvidenceError as evidence_error:
            raise ModelCompletionManualRecoveryRequired(
                "Provider-operation recovery cannot continue because provider output already "
                "crossed Cayu's durable acceptance boundary."
            ) from evidence_error
        if operation is None:
            return None
        if registered_provider is None:
            try:
                registered_provider = self._resolve_registered_provider(operation.provider)
            except KeyError:
                return None
        elif registered_provider.name != operation.provider:
            session = await self._session_store.load(stage.session_id)
            target = (
                None
                if session is None
                else model_failover_target_for_stored_stage(session=session, stage=stage)
            )
            if (
                target is None
                or session is None
                or registered_provider.name != session.provider_name
                or target.provider_name != operation.provider
            ):
                return None
            registered_provider = self._resolve_registered_provider(target.provider_name)
        provider = registered_provider.provider
        if (
            provider.provider_operation_mode is not ProviderOperationMode.BACKGROUND
            or not isinstance(
                provider.provider_operations,
                ProviderOperationAdapter,
            )
        ):
            return None
        return operation, registered_provider

    async def provider_operation_execution_scope(
        self,
        *,
        session: Session,
        stage: ModelCompletionStage,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        recovered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        invocation_context: InvocationContext | None,
    ) -> tuple[InvocationContext, ModelExecutionSelection | None]:
        """Validate root continuation and retain its separately selected collaborators."""
        if registered_provider.name != session.provider_name:
            raise RuntimeError("Provider-operation recovery substituted its root provider.")
        if invocation_context is not None and (
            registered_agent is not invocation_context.registered_agent
            or registered_provider is not invocation_context.registered_provider
            or registered_environment is not invocation_context.registered_environment
        ):
            raise RuntimeError(
                "Provider-operation recovery substituted frozen invocation authority."
            )
        checkpoint = await self._session_store.load_checkpoint(session.id)
        active = active_invocation_execution_profile_from_checkpoint(checkpoint)
        binding = None if active is None else active.profile.model_failover
        providers = (
            ()
            if binding is None
            else tuple(
                registered_provider
                if target.provider_name == registered_provider.name
                else self._resolve_registered_provider(target.provider_name)
                for target in binding.plan.candidates
            )
        )
        budget_policy = (
            copy_budget_policy(self._resolve_budget_policy())
            if invocation_context is None
            else invocation_context.budget_policy
        )
        snapshot = await self._execution_profile_continuation.validate(
            session=session,
            checkpoint=checkpoint,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            request_loop_policies=None,
            budget_policy=budget_policy,
        )
        # The validator checked every candidate. Do not replace those collaborators
        # by a second registry lookup after its asynchronous continuation boundary.
        if any(
            provider.name != registered_provider.name
            and self._resolve_registered_provider(provider.name) is not provider
            for provider in providers
        ):
            raise RuntimeError(
                "Provider-operation recovery registrations changed during validation."
            )
        if invocation_context is None:
            invocation_context = reconstruct_invocation_context(
                runtime_hooks=self._runtime_hooks,
                loop_policies=self._loop_policies,
                session=session,
                execution_profile_snapshot=snapshot,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                budget_policy=budget_policy,
            )
        elif invocation_context.active_profile != snapshot:
            raise RuntimeError("Provider-operation recovery substituted its execution profile.")
        target = model_failover_target_for_stored_stage(session=session, stage=stage)
        if target is None:
            if recovered_provider is not registered_provider:
                raise RuntimeError("Provider-operation recovery substituted its provider.")
            return invocation_context, None
        if binding is None or active is None or snapshot.profile != active.profile:
            raise RuntimeError("Provider-operation recovery lost its admitted candidate plan.")
        index = next(
            i
            for i, candidate in enumerate(binding.plan.candidates)
            if (candidate.provider_name, candidate.model) == (target.provider_name, target.model)
        )
        selection = ModelExecutionSelection(
            invocation_context=invocation_context,
            resolution=ModelFailoverProfileResolution(
                plan=binding.plan,
                candidate_profiles=tuple(item.as_profile() for item in binding.candidate_profiles),
                registered_providers=providers,
            ),
            candidate_index=index,
        )
        selection.require_recovery_scope(
            session=session,
            stage=stage,
            invocation_context=invocation_context,
            registered_provider=recovered_provider,
        )
        return invocation_context, selection

    async def has_recoverable_provider_operation(self, stage: ModelCompletionStage) -> bool:
        """Read-only eligibility for this owner's exact background recovery path.

        The operation loader validates durable model/start/cursor identities;
        a scanner never receives a provider connection or dispatch authority.
        The elected recovery owner must resolve and validate them again.
        """
        return await self.recoverable_provider_operation(stage) is not None

    async def load_model_completion_boundary(
        self,
        session: Session,
    ) -> ActiveModelCompletionStage | None:
        """Load and validate durable model work without consulting a provider."""

        active = await self._session_store.load_active_model_completion_stage(session.id)
        if active is not None:
            self.validate_active_model_completion_stage(session, active.stage)
        return active

    async def preflight_model_completion_boundary(
        self,
        session: Session,
        *,
        registered_provider: runtime_records.RegisteredProvider | None = None,
        active_stage: ActiveModelCompletionStage | None = None,
    ) -> bool:
        """Reject a non-terminal dispatch and report terminal work to promote."""

        if active_stage is None:
            active = await self.load_model_completion_boundary(session)
        else:
            if type(active_stage) is not ActiveModelCompletionStage:
                raise TypeError("active_stage must be an ActiveModelCompletionStage.")
            active = active_stage.model_copy(deep=True)
        if active is None:
            return False
        stage = active.stage
        self.validate_active_model_completion_stage(session, stage)
        if stage.state == "in_flight":
            if (
                await self.recoverable_provider_operation(
                    stage,
                    registered_provider=registered_provider,
                )
                is not None
            ):
                return True
            if (
                await self._session_store.load_model_completion_stage_dispatch(
                    session.id,
                    stage.stage_id,
                )
                is None
            ):
                return True
            raise ModelCompletionManualRecoveryRequired(
                "The active model-completion dispatch has no durable terminal response. "
                "Its provider outcome and linked budget reservations require "
                "CayuApp.recover_model_completion_stage(...) before retrying: "
                f"{stage.stage_id}"
            )
        return True

    async def cancel_provider_operation_for_interruption(
        self,
        session: Session,
        *,
        registered_agent: runtime_records.RegisteredAgentState | None = None,
        registered_provider: runtime_records.RegisteredProvider | None = None,
        registered_environment: runtime_records.RegisteredEnvironment | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> ProviderOperationInterruptionAuthority | None:
        """Address one durable in-flight provider operation without a live worker."""

        if registered_agent is None and registered_provider is not None:
            raise ValueError("Frozen provider-operation cancellation requires the original agent.")

        active = await self._session_store.load_active_model_completion_stage(session.id)
        if active is None or active.stage.state != "in_flight":
            return None
        stage = active.stage
        self.validate_active_model_completion_stage(session, stage)
        if registered_agent is not None and registered_provider is None:
            raise ModelCompletionManualRecoveryRequired(
                "Provider-operation cancellation requires the original provider registration."
            )
        recoverable = await self.recoverable_provider_operation(
            stage,
            registered_provider=registered_provider,
        )
        if recoverable is None:
            return None
        operation, recovered_provider = recoverable
        if isinstance(operation, RecoverableProviderOperationStart):
            # There is no durable provider operation identity to cancel yet.
            # The normal boundary reconciler may replay an exact-idempotent
            # start; an unsupported start remains explicitly ambiguous.
            return None
        if registered_agent is None or registered_provider is None:
            try:
                registered_agent = self._resolve_registered_agent(session.agent_name)
                registered_provider = self._resolve_registered_provider(session.provider_name)
                registered_environment = self._resolve_registered_environment(
                    session.environment_name
                )
            except KeyError as registration_error:
                raise ModelCompletionManualRecoveryRequired(
                    "Provider-operation cancellation requires the original agent, provider, "
                    "and environment registrations."
                ) from registration_error
        (
            invocation_context,
            model_execution_selection,
        ) = await self.provider_operation_execution_scope(
            session=session,
            stage=stage,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            recovered_provider=recovered_provider,
            registered_environment=registered_environment,
            invocation_context=invocation_context,
        )
        cancellation = await self.cancel_provider_operation(
            session,
            stage,
            operation,
            registered_agent,
            recovered_provider,
            registered_environment,
            invocation_context,
            model_execution_selection,
        )
        if cancellation is not None and cancellation.status is ProviderOperationStatus.COMPLETED:
            recovered = await self.recover_provider_operation(
                session,
                stage,
                operation,
                registered_agent,
                recovered_provider,
                registered_environment,
                invocation_context,
                model_execution_selection,
            )
            if recovered.status is not ProviderOperationRecoveryStatus.RECONCILED:
                raise ModelCompletionManualRecoveryRequired(
                    "Provider completion won cancellation but could not be reconciled."
                )
        return ProviderOperationInterruptionAuthority(
            active_profile=invocation_context.active_profile,
            invocation_context=invocation_context,
        )

    async def reconcile_model_completion_boundary(
        self,
        session: Session,
        *,
        invocation_context: InvocationContext | None = None,
        registered_agent: runtime_records.RegisteredAgentState | None = None,
        registered_provider: runtime_records.RegisteredProvider | None = None,
        registered_environment: runtime_records.RegisteredEnvironment | None = None,
    ) -> ModelCompletionBoundaryReconciliation:
        """Promote terminal model evidence or verify an already-published boundary."""

        if invocation_context is not None:
            if type(invocation_context) is not InvocationContext:
                raise TypeError("invocation_context must be an authenticated InvocationContext.")
            invocation_context._validate()
            if (
                invocation_context.binding.session_id != session.id
                or invocation_context.binding.session_instance_id != session.instance_id
                or invocation_context.binding.run_epoch != session.run_epoch
            ):
                raise RuntimeError(
                    "Model-completion reconciliation lost exact invocation authority."
                )
            for supplied, owned, field_name in (
                (registered_agent, invocation_context.registered_agent, "agent"),
                (registered_provider, invocation_context.registered_provider, "provider"),
                (
                    registered_environment,
                    invocation_context.registered_environment,
                    "environment",
                ),
            ):
                if supplied is not None and supplied is not owned:
                    raise RuntimeError(
                        f"Model-completion reconciliation substituted its registered {field_name}."
                    )
            registered_agent = invocation_context.registered_agent
            registered_provider = invocation_context.registered_provider
            registered_environment = invocation_context.registered_environment

        if (registered_agent is None) != (registered_provider is None):
            raise ValueError(
                "Frozen model-completion reconciliation requires both agent and provider."
            )

        # Fail closed for this session's own accounting before scanning the
        # shared ledger. The global pass may retain rows owned by another
        # session-store publication domain without blocking this boundary.
        await self._run_limit_controller.recover_pending_budget_settlements(session_id=session.id)
        await self._run_limit_controller.recover_pending_budget_settlements()
        active = await self._session_store.load_active_model_completion_stage(session.id)
        state: Literal[
            "none",
            "prepared_abandoned",
            "promoted",
            "already_promoted",
            "provider_operation_pending",
            "provider_operation_unavailable",
            "provider_operation_reconciled",
        ] = "none"
        recovery_events: tuple[Event, ...] = ()
        if active is not None:
            stage = active.stage
            self.validate_active_model_completion_stage(session, stage)
            try:
                recovery_events = tuple(
                    await self._run_limit_controller.reconcile_borrowed_automatic_compaction_budget_authority(
                        session=session,
                        stage=stage,
                    )
                )
            except BorrowedAutomaticCompactionOutcomeUnknown as outcome_unknown:
                raise ModelCompletionManualRecoveryRequired(
                    str(outcome_unknown)
                ) from outcome_unknown
            if (
                stage.state == "in_flight"
                and await self.recoverable_provider_operation(
                    stage,
                    registered_provider=registered_provider,
                )
                is None
                and await self._session_store.load_model_completion_stage_dispatch(
                    session.id,
                    stage.stage_id,
                )
                is None
            ):
                await close_context_exposure_without_provider_effect(
                    store=self._session_store,
                    session_id=session.id,
                    stage_id=stage.stage_id,
                    stage_intent=stage.intent,
                    evidence_ref_suffix="dispatch-receipt-absent",
                )
                recovery_context = model_completion_recovery_context_from_stage(stage)
                budget_recovery_contexts = (
                    () if recovery_context is None else recovery_context.budget_reservations
                )
                if stage.purpose == "auxiliary-inference":
                    from cayu.runtime._auxiliary_inference_contract import (
                        auxiliary_budget_recovery_contexts,
                    )

                    budget_recovery_contexts = auxiliary_budget_recovery_contexts(stage)
                budget_dispatch_id = stage.stage_id
                if stage.purpose in {"context-compaction", "auxiliary-inference"}:
                    model_attempt_id = stage.intent.get("model_attempt_id")
                    if type(model_attempt_id) is not str:
                        raise ModelCompletionManualRecoveryRequired(
                            "Receipt-less model-stage recovery lost its budget dispatch identity."
                        )
                    budget_dispatch_id = require_clean_nonblank(
                        model_attempt_id,
                        "model_attempt_id",
                    )
                release_events = await (
                    self._run_limit_controller.release_pre_provider_dispatch_reservations(
                        reservation_ids=stage.reservation_ids,
                        recovery_contexts=budget_recovery_contexts,
                        dispatch_id=budget_dispatch_id,
                    )
                )
                await self._session_store.abandon_model_completion_stage(
                    session.id,
                    stage_id=stage.stage_id,
                    preparation_digest=stage.preparation_digest,
                    expected_run_epoch=session.run_epoch,
                    stage_source_run_epoch=stage.source_run_epoch,
                )
                active = None
                state = "prepared_abandoned"
                recovery_events = tuple(
                    {event.id: event for event in (*recovery_events, *release_events)}.values()
                )
        if active is not None:
            stage = active.stage
            if (
                stage.purpose == "auxiliary-inference"
                and stage.state == "in_flight"
                and invocation_context is not None
                and invocation_context.recovery_claim_id is not None
            ):
                await self._complete_unknown_auxiliary_stage(
                    session=session,
                    stage=stage,
                    invocation=invocation_context,
                )
                active = await self._session_store.load_active_model_completion_stage(session.id)
                if active is None or active.stage.stage_id != stage.stage_id:
                    raise RuntimeError("Auxiliary recovery lost its active completion boundary.")
                stage = active.stage
            if stage.purpose != "auxiliary-inference" and session.status in {
                SessionStatus.COMPLETED,
                SessionStatus.INTERRUPTED,
            }:
                raise ModelCompletionManualRecoveryRequired(
                    "A terminal session retains an active model-completion stage. "
                    "Provider output and linked reservations require the runtime-owned "
                    "CayuApp.recover_model_completion_stage(...) operation before the "
                    f"stage can be cleared: {stage.stage_id}"
                )
            if stage.state == "in_flight":
                recoverable = await self.recoverable_provider_operation(
                    stage,
                    registered_provider=registered_provider,
                )
                if recoverable is None:
                    await close_unrecoverable_context_exposure(
                        store=self._session_store,
                        session_id=session.id,
                        stage_id=stage.stage_id,
                        stage_intent=stage.intent,
                    )
                    raise ModelCompletionManualRecoveryRequired(
                        "The active model-completion dispatch has no durable terminal response. "
                        "Its provider outcome and linked budget reservations require "
                        "CayuApp.recover_model_completion_stage(...) before retrying: "
                        f"{stage.stage_id}"
                    )
                operation, recovered_provider = recoverable
                if registered_agent is None or registered_provider is None:
                    try:
                        registered_agent = self._resolve_registered_agent(session.agent_name)
                        registered_provider = self._resolve_registered_provider(
                            session.provider_name
                        )
                        registered_environment = self._resolve_registered_environment(
                            session.environment_name
                        )
                    except KeyError as registration_error:
                        raise ModelCompletionManualRecoveryRequired(
                            "Provider-operation recovery requires the original agent, provider, "
                            "and environment registrations."
                        ) from registration_error
                (
                    invocation_context,
                    model_execution_selection,
                ) = await self.provider_operation_execution_scope(
                    session=session,
                    stage=stage,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    recovered_provider=recovered_provider,
                    registered_environment=registered_environment,
                    invocation_context=invocation_context,
                )
                if isinstance(operation, RecoverableProviderOperationStart):
                    recovered = await self.recover_provider_operation_start(
                        session,
                        stage,
                        operation,
                        registered_agent,
                        recovered_provider,
                        registered_environment,
                        invocation_context,
                        model_execution_selection,
                    )
                else:
                    recovered = await self.recover_provider_operation(
                        session,
                        stage,
                        operation,
                        registered_agent,
                        recovered_provider,
                        registered_environment,
                        invocation_context,
                        model_execution_selection,
                    )
                recovery_events = tuple(
                    {event.id: event for event in (*recovery_events, *recovered.events)}.values()
                )
                if recovered.status is ProviderOperationRecoveryStatus.PENDING:
                    transcript = await self._session_store.load_transcript(session.id)
                    return ModelCompletionBoundaryReconciliation(
                        state="provider_operation_pending",
                        session=session,
                        transcript_cursor=len(transcript),
                        recovery_events=recovery_events,
                    )
                if recovered.status is ProviderOperationRecoveryStatus.UNAVAILABLE:
                    transcript = await self._session_store.load_transcript(session.id)
                    return ModelCompletionBoundaryReconciliation(
                        state="provider_operation_unavailable",
                        session=session,
                        transcript_cursor=len(transcript),
                        recovery_events=recovery_events,
                    )
                if recovered.status is not ProviderOperationRecoveryStatus.RECONCILED:
                    raise RuntimeError("Provider-operation recovery returned an unknown state.")
                active = await self._session_store.load_active_model_completion_stage(session.id)
                if active is not None:
                    raise RuntimeError(
                        "Reconciled provider operation retained an active model-completion stage."
                    )
                loaded_session = await self._session_store.load(session.id)
                if loaded_session is None:
                    raise KeyError(f"Session not found: {stage.session_id}")
                session = loaded_session
                state = "provider_operation_reconciled"
            elif session.status not in _model_completion_stage_promotion_statuses(stage):
                raise ModelCompletionManualRecoveryRequired(
                    "The completed model stage cannot be promoted from the current session "
                    f"status ({session.status.value}); expected {stage.source_status.value}."
                )
            else:
                if stage.purpose == "auxiliary-inference":
                    from cayu.runtime._auxiliary_inference_contract import (
                        auxiliary_budget_recovery_contexts,
                        validate_auxiliary_publication,
                    )

                    publication = stage.publication
                    if publication is None:
                        raise ModelCompletionManualRecoveryRequired(
                            "Auxiliary terminal stage lost its publication."
                        )
                    validate_auxiliary_publication(publication, session_id=session.id, stage=stage)
                    budget_events = (
                        await self._run_limit_controller.reconcile_model_completion_settlements(
                            publication.events[0],
                            reservation_ids=stage.reservation_ids,
                        )
                    )
                    await (
                        self._run_limit_controller.require_model_completion_reservation_settlements(
                            reservation_ids=stage.reservation_ids,
                            recovery_contexts=auxiliary_budget_recovery_contexts(stage),
                            dispatch_id=stage.intent["model_attempt_id"],
                        )
                    )
                    recovery_events = tuple(
                        {event.id: event for event in (*recovery_events, *budget_events)}.values()
                    )
                if stage.purpose == "context-compaction" and stage.reservation_ids:
                    recovery_context = model_completion_recovery_context_from_stage(stage)
                    pricing_provider_name = stage.intent.get("pricing_provider_name")
                    requested_model = stage.intent.get("requested_model")
                    model_attempt_id = stage.intent.get("model_attempt_id")
                    if recovery_context is None or not all(
                        type(value) is str
                        for value in (
                            pricing_provider_name,
                            requested_model,
                            model_attempt_id,
                        )
                    ):
                        raise ModelCompletionManualRecoveryRequired(
                            "Completed context-compaction recovery lost its exact budget authority."
                        )
                    assert isinstance(pricing_provider_name, str)
                    assert isinstance(requested_model, str)
                    assert isinstance(model_attempt_id, str)
                    budget_events = await self._run_limit_controller.reconcile_completed_automatic_compaction_reservations(
                        session=session,
                        stage=stage,
                        recovery_contexts=recovery_context.budget_reservations,
                        pricing_provider_name=pricing_provider_name,
                        model=requested_model,
                        model_attempt_identity=ModelAttemptIdentity(
                            model_step_id=stage.logical_step_id,
                            model_attempt_id=model_attempt_id,
                        ),
                    )
                    recovery_events = tuple(
                        {event.id: event for event in (*recovery_events, *budget_events)}.values()
                    )
                await recover_context_exposure(
                    store=self._session_store,
                    session_id=session.id,
                    stage_id=stage.stage_id,
                    stage_intent=stage.intent,
                    state=ContextExposureState.COMPLETED,
                    evidence_kind=ContextExposureEvidenceKind.RECOVERY_COMPLETION,
                    evidence_ref=f"model-stage:{stage.stage_id}:completed",
                )
                session = await self._promote_completed_model_stage(
                    session=session,
                    stage_id=stage.stage_id,
                )
                if stage.purpose == "auxiliary-inference":
                    assert stage.publication is not None
                    terminal_events = await self._event_writer.fan_out_persisted(
                        list(stage.publication.events)
                    )
                    recovery_events = tuple(
                        {event.id: event for event in (*recovery_events, *terminal_events)}.values()
                    )
                state = "promoted"

        checkpoint = await self._session_store.load_checkpoint(session.id)
        pointer = model_completion_publication.model_step_publication_from_checkpoint(checkpoint)
        tool_receipt: RuntimePublicationReceipt | None = None
        pending_round = pending_round_reader.pending_tool_round_from_checkpoint(checkpoint)
        pending_approval = pending_approval_reader.pending_approval_from_checkpoint(checkpoint)
        pending_user_input, _resolution_intent = user_input_lifecycle_authority_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            current_run_epoch=session.run_epoch,
            runtime_session=session,
        )
        if pending_approval is not None and pending_user_input is not None:
            raise RuntimeError(
                "The checkpoint contains conflicting pending approval and user-input pauses."
            )
        if pending_round is not None and pending_user_input is not None:
            raise RuntimeError(
                "The checkpoint contains conflicting pending tool-round and pause markers."
            )
        if pending_round is not None and pending_approval is not None:
            _pending_approval_for_atomic_claim(
                checkpoint,
                approval_id=pending_approval.approval_id,
                tool_round_id=pending_approval.tool_round_id,
                gating_tool_call_id=pending_approval.tool_call_id,
                redactor=self._secret_redactor,
                runtime_session=session,
            )
        if pointer is None:
            if active is not None:
                if active.stage.purpose == "context-compaction":
                    raise ModelCompletionManualRecoveryRequired(
                        "The completed context compaction was promoted without a durable "
                        "context checkpoint; its completion evidence prevents provider "
                        "redispatch."
                    )
                raise RuntimeError(
                    "Promoted model completion did not publish its durable model-step pointer."
                )
            if pending_round is not None and pending_round.source_model_step_id is not None:
                raise RuntimeError(
                    "A pending tool round exists without a durable source model-step pointer."
                )
            return ModelCompletionBoundaryReconciliation(
                state=state,
                session=session,
                recovery_events=recovery_events,
            )

        completion_event, completed_stage = await self.reconcile_published_model_budget(
            session=session,
            pointer=pointer,
        )

        transcript_window = await self._session_store.load_transcript_window(
            session.id,
            start_index=(
                pointer.transcript_end_cursor
                if pointer.assistant_message_deferred
                else max(0, pointer.transcript_end_cursor - 1)
            ),
            limit=2,
        )
        if transcript_window.cursor < pointer.transcript_end_cursor:
            raise RuntimeError("The durable model-step pointer extends beyond the transcript.")
        if pointer.assistant_message_published and (
            not transcript_window.records
            or transcript_window.records[0].index != pointer.transcript_end_cursor - 1
            or transcript_window.records[0].message.role != MessageRole.ASSISTANT
        ):
            raise RuntimeError(
                "The durable model-step pointer does not identify its assistant message."
            )

        if pending_round is not None:
            if (
                pointer.tool_round_id != pending_round.tool_round_id
                or pending_round.source_model_step_id != pointer.logical_step_id
                or pending_round.source_transcript_cursor != pointer.source_transcript_cursor
            ):
                raise RuntimeError(
                    "The pending tool round conflicts with its durable source model step."
                )
        elif pointer.tool_round_id is not None:
            pending_pause = pending_approval if pending_approval is not None else pending_user_input
            assistant_message = None
            if pointer.assistant_message_deferred and pending_pause is not None:
                assistant_message = getattr(
                    pending_pause,
                    "quarantined_assistant_message",
                    None,
                )
            elif transcript_window.records:
                assistant_message = transcript_window.records[0].message
            assistant_call_parts = tuple(
                part
                for part in (() if assistant_message is None else assistant_message.content)
                if type(part) is ToolCallPart
            )
            assistant_calls = tuple(
                (part.tool_call_id, part.tool_name, part.arguments) for part in assistant_call_parts
            )
            retired_pause = (
                pending_pause is None
                and pointer.assistant_message_deferred
                and transcript_window.cursor == pointer.transcript_end_cursor
                and await self._model_pause_retired_by_terminal_decision(
                    session=session,
                    checkpoint=checkpoint,
                    pointer=pointer,
                    completed_stage=completed_stage,
                )
            )
            if retired_pause:
                # The elected stop atomically retired an unpublished human gate.
                # Preserve model accounting, but never recreate executable work.
                pass
            elif transcript_window.cursor == pointer.transcript_end_cursor:
                if pending_pause is None:
                    raise RuntimeError(
                        "The latest model completion requires a pending tool round, but its "
                        "durable marker is missing."
                    )
                if (
                    pending_pause.agent_name != session.agent_name
                    or pending_pause.environment_name != session.environment_name
                ):
                    raise RuntimeError(
                        "The pending tool pause conflicts with its durable source model step."
                    )
                pending_calls = tuple(
                    (call.tool_call_id, call.tool_name, call.arguments)
                    for call in pending_pause.tool_calls
                )
                pending_target = (
                    pending_pause.tool_call_id,
                    pending_pause.tool_name,
                    pending_pause.arguments,
                )
                if (
                    not (
                        pointer.assistant_message_published
                        or (
                            pointer.assistant_message_deferred
                            and getattr(
                                pending_pause,
                                "assistant_message_state",
                                None,
                            )
                            == "quarantined"
                        )
                    )
                    or not assistant_calls
                    or assistant_calls != pending_calls
                    or pending_calls.count(pending_target) != 1
                ):
                    raise RuntimeError(
                        "The pending tool pause conflicts with its durable source model step."
                    )
            else:
                if pending_pause is not None:
                    raise RuntimeError(
                        "A pending tool pause remains after its source model step advanced."
                    )
                next_record = (
                    transcript_window.records[1] if len(transcript_window.records) > 1 else None
                )
                tool_results = (
                    tuple(
                        (part.tool_call_id, part.tool_name)
                        for part in next_record.message.content
                        if type(part) is ToolResultPart
                    )
                    if next_record is not None
                    and next_record.index
                    == pointer.transcript_end_cursor + int(pointer.assistant_message_deferred)
                    and next_record.message.role == MessageRole.TOOL
                    else ()
                )
                expected_results = tuple(
                    (tool_call_id, tool_name)
                    for tool_call_id, tool_name, _arguments in assistant_calls
                )
                if (
                    not (pointer.assistant_message_published or pointer.assistant_message_deferred)
                    or not expected_results
                    or next_record is None
                    or len(tool_results) != len(next_record.message.content)
                    or tool_results != expected_results
                ):
                    raise RuntimeError(
                        "The transcript after the durable model step does not exactly close "
                        "its assistant tool calls."
                    )
                assistant_identities = {
                    (
                        part.model_step_id,
                        part.model_attempt_id,
                        part.tool_round_id,
                    )
                    for part in assistant_call_parts
                }
                if len(assistant_identities) != 1:
                    raise RuntimeError(
                        "The assistant tool calls do not share one complete execution identity."
                    )
                model_step_id, model_attempt_id, tool_round_id = next(iter(assistant_identities))
                if (
                    type(model_step_id) is not str
                    or type(model_attempt_id) is not str
                    or type(tool_round_id) is not str
                ):
                    raise RuntimeError(
                        "The assistant tool-call identity is incomplete at recovery."
                    )
                if (
                    model_step_id != pointer.logical_step_id
                    or tool_round_id != pointer.tool_round_id
                ):
                    raise RuntimeError(
                        "The assistant tool-call identity conflicts with its durable model step."
                    )
                tool_publication_id = f"tool-round:{pointer.tool_round_id}"
                tool_receipt = await self._session_store.load_runtime_publication_receipt(
                    session.id,
                    tool_publication_id,
                )
                if tool_receipt is None:
                    if not await self._tool_result_tail_has_durable_lifecycle_provenance(
                        session=session,
                        identity=ToolRoundIdentity(
                            model_step_id=model_step_id,
                            model_attempt_id=model_attempt_id,
                            tool_round_id=tool_round_id,
                        ),
                        assistant_call_parts=assistant_call_parts,
                        tool_result_message=next_record.message,
                        arguments_deferred=pointer.assistant_message_deferred,
                    ):
                        raise RuntimeError(
                            "The transcript contains tool results without durable tool-round "
                            "publication provenance."
                        )
                elif (
                    tool_receipt.publication_id != tool_publication_id
                    or tool_receipt.kind != "tool-round"
                    or tool_receipt.transcript_start_cursor != pointer.transcript_end_cursor
                    or tool_receipt.transcript_end_cursor
                    != pointer.transcript_end_cursor
                    + (2 if pointer.assistant_message_deferred else 1)
                    or tool_receipt.intent.get("round_id") != pointer.tool_round_id
                    or tool_receipt.intent.get("model_step_id") != model_step_id
                    or tool_receipt.intent.get("model_attempt_id") != model_attempt_id
                    or tool_receipt.intent.get("tool_round_id") != tool_round_id
                    or tool_receipt.intent.get("tool_call_ids")
                    != [tool_call_id for tool_call_id, _tool_name, _arguments in assistant_calls]
                ):
                    raise RuntimeError(
                        "The durable tool-round publication receipt conflicts with its "
                        "source model step."
                    )

        structured_events: tuple[Event, ...] = ()
        if (
            invocation_context is not None
            and invocation_context.work_attempt is not None
            and pending_round is None
            and pointer.tool_round_id is not None
            and transcript_window.cursor > pointer.transcript_end_cursor
        ):
            governed_semantics = invocation_context.work_attempt.admission.run_semantics
            if governed_semantics is None:
                raise RuntimeError("Closed structured output requires governed run semantics.")
            structured_events = await self.load_closed_structured_output_events(
                session,
                completed_stage,
                invocation_context,
                None if tool_receipt is None else tool_receipt.model_copy(deep=True),
                expected_spec=governed_semantics.structured_output,
            )
        await self._event_writer.fan_out_persisted([completion_event])
        return ModelCompletionBoundaryReconciliation(
            state=("already_promoted" if state == "none" else state),
            session=session,
            pointer=pointer,
            completion_event=copy_event(completion_event),
            pending_tool_round=pending_round,
            transcript_cursor=transcript_window.cursor,
            recovery_events=recovery_events,
            completed_stage=completed_stage.model_copy(deep=True),
            closed_tool_receipt=(
                None if tool_receipt is None else tool_receipt.model_copy(deep=True)
            ),
            structured_output_events=structured_events,
        )

    async def _model_pause_retired_by_terminal_decision(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        pointer: model_completion_publication.ModelStepPublicationCheckpoint,
        completed_stage: ModelCompletionStage,
    ) -> bool:
        """Recognize only an exact elected stop of this unpublished model round."""
        if session.status is not SessionStatus.INTERRUPTING or checkpoint is None:
            return False
        decision = invocation_terminal_decision_from_checkpoint(checkpoint)
        profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        payload = checkpoint.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
        if (
            decision is None
            or profile is None
            or decision.outcome is not InvocationTerminalOutcome.INTERRUPTED
            or type(payload) is not dict
            or payload != decision.terminal_payload
            or payload.get("interruption_type") != _INTERRUPTION_TYPE_OPERATOR_REQUESTED
            or not invocation_terminal_decision_matches_recovery_profile(
                decision,
                session_id=session.id,
                session_instance_id=session.instance_id,
                current_run_epoch=session.run_epoch,
                interaction_id=profile.interaction_id,
                execution_profile_fingerprint=profile.profile.fingerprint,
            )
        ):
            return False
        approval_intent = payload.get(approval_support.APPROVAL_INTERRUPT_CLOSE_INTENT_KEY)
        input_intent = payload.get(USER_INPUT_SUPERSESSION_INTENT_KEY)
        if (approval_intent is None) == (input_intent is None):
            return False
        if input_intent is not None:
            if (
                await self._user_input_evidence.validated_user_input_supersession_interrupt_payload(
                    session=session, pending_interrupt_payload=payload
                )
                is None
            ):
                return False
            intent = input_intent
        else:
            intent = approval_intent
            if (
                type(intent) is not dict
                or set(intent)
                != {
                    "approval_id",
                    "tool_call_id",
                    "tool_round_id",
                    "model_step_id",
                    "model_attempt_id",
                }
                or any(type(value) is not str or not value.strip() for value in intent.values())
            ):
                return False
        publication = completed_stage.publication
        if publication is None:
            return False
        source_round = pending_round_reader.pending_tool_round_from_checkpoint(
            {
                operation.key: operation.value
                for operation in publication.mutation.operations
                if operation.key == pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY
                and operation.action == "set"
            }
        )
        return (
            type(intent) is dict
            and source_round is not None
            and source_round.assistant_message_state == "quarantined"
            and source_round.source_run_epoch == completed_stage.source_run_epoch
            and source_round.source_model_step_id == pointer.logical_step_id
            and source_round.source_transcript_cursor == pointer.source_transcript_cursor
            and source_round.tool_round_id == pointer.tool_round_id
            and sum(
                call.tool_call_id == intent.get("tool_call_id") for call in source_round.tool_calls
            )
            == 1
            and intent.get("tool_round_id") == pointer.tool_round_id
            and intent.get("model_step_id") == pointer.logical_step_id
            and intent.get("model_attempt_id") == completed_stage.intent.get("model_attempt_id")
            and publication.interaction_id == decision.interaction_id
        )

    async def reconcile_published_model_budget(
        self,
        *,
        session: Session,
        pointer: model_completion_publication.ModelStepPublicationCheckpoint,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[Event, ModelCompletionStage]:
        """Settle immutable published evidence without admitting provider or tool work."""
        receipt = await self._session_store.load_runtime_publication_receipt(
            session.id, pointer.logical_step_id
        )
        if receipt is None:
            raise RuntimeError("The durable model-step pointer has no publication receipt.")
        if (
            receipt.kind != "model-step"
            or receipt.transcript_start_cursor != pointer.source_transcript_cursor
            or receipt.transcript_end_cursor != pointer.transcript_end_cursor
            or receipt.appended_event_ids != (pointer.completion_event_id,)
            or receipt.referenced_events
        ):
            raise RuntimeError(
                "The durable model-step pointer conflicts with its publication receipt."
            )
        event_records = await self._session_store.query_events(
            EventQuery(session_id=session.id, event_id=pointer.completion_event_id, limit=1)
        )
        if len(event_records) != 1:
            raise RuntimeError("The durable model-step pointer has no exact completion event.")
        completion_event = event_records[0].event
        if (
            completion_event.type != EventType.MODEL_COMPLETED
            or completion_event.payload.get("step_classification") != pointer.classification
            or completion_event.payload.get("transcript_cursor") != pointer.transcript_end_cursor
        ):
            raise RuntimeError(
                "The durable model-step pointer conflicts with its completion event."
            )
        completed_stage = await self._session_store.load_model_completion_stage(
            session.id, pointer.stage_id
        )
        if (
            completed_stage is None
            or completed_stage.state != "completed"
            or completed_stage.logical_step_id != pointer.logical_step_id
            or completed_stage.publication is None
            or completed_stage.publication.events != (completion_event,)
        ):
            raise RuntimeError(
                "The durable model-step pointer conflicts with its completion stage."
            )
        if completed_stage.reservation_ids:
            if before_mutation is not None:
                await before_mutation()
            await self._run_limit_controller.reconcile_model_completion_settlements(
                completion_event, reservation_ids=completed_stage.reservation_ids
            )
        return completion_event, completed_stage

    async def load_closed_structured_output_events(
        self,
        session: Session,
        stage: ModelCompletionStage,
        context: InvocationContext,
        expected_receipt: RuntimePublicationReceipt | None,
        *,
        expected_spec: StructuredOutputSpec | None,
    ) -> tuple[Event, ...]:
        """Read exact structured evidence for an already reconciled tool tail."""
        context.require_runtime_authority()
        if expected_spec is None:
            return ()
        if expected_spec.strategy is not StructuredOutputStrategy.TOOL:
            return ()
        if stage.publication is None:
            raise RuntimeError("Closed structured output lost its source publication.")
        pending = pending_round_reader.pending_tool_round_from_checkpoint(
            {
                operation.key: operation.value
                for operation in stage.publication.mutation.operations
                if operation.action == "set"
            }
        )
        if pending is None:
            raise RuntimeError("Closed structured output lost its source round.")
        if not any(call.tool_name == STRUCTURED_OUTPUT_TOOL_NAME for call in pending.tool_calls):
            return ()
        if not has_recoverable_structured_output_round(pending):
            raise RuntimeError("Closed structured output lacks authoritative validation.")
        spec, validation = pending.structured_output, pending.structured_output_validation
        assert spec is not None and validation is not None
        assert pending.model_step is not None and pending.structured_output_attempt is not None
        if spec != expected_spec:
            raise RuntimeError("Closed structured output changed its source specification.")
        receipt = await self._session_store.load_runtime_publication_receipt(
            session.id, f"tool-round:{pending.tool_round_id}"
        )
        if (
            type(receipt) is not RuntimePublicationReceipt
            or expected_receipt is None
            or receipt != expected_receipt
            or receipt.session_id != session.id
            or receipt.kind != "tool-round"
            or receipt.interaction_id != context.binding.interaction_id
        ):
            raise RuntimeError("Closed structured output lost its exact interaction receipt.")
        auxiliary = receipt.intent.get("auxiliary")
        if (
            type(auxiliary) is not dict
            or any(
                type(auxiliary.get(key)) is not int for key in ("schema_version", "step", "attempt")
            )
            or type(auxiliary.get("valid")) is not bool
            or type(auxiliary.get("retry_scheduled")) is not bool
        ):
            raise RuntimeError("Closed structured output has malformed validation authority.")
        retry = auxiliary["retry_scheduled"]
        if retry and (validation.valid or pending.structured_output_attempt > spec.max_retries):
            raise RuntimeError("Closed structured output retry conflicts with its policy.")
        identity = pending_rounds.pending_tool_round_identity(pending)
        expected: list[Event] = []
        expected.append(
            structured_output_tool_round._structured_output_validating_event(
                session=session,
                registered_agent=context.registered_agent,
                environment_name=_environment_name(context.registered_environment),
                spec=spec,
                step=pending.model_step,
                attempt=pending.structured_output_attempt,
                tool_round_identity=identity,
            )
        )
        kinds = [
            EventType.STRUCTURED_OUTPUT_VALIDATED
            if validation.valid
            else EventType.STRUCTURED_OUTPUT_FAILED
        ]
        if retry:
            kinds.append(EventType.STRUCTURED_OUTPUT_RETRY)
        for kind in kinds:
            expected.append(
                structured_output_tool_round._structured_output_event(
                    event_type=kind,
                    session=session,
                    registered_agent=context.registered_agent,
                    environment_name=_environment_name(context.registered_environment),
                    spec=spec,
                    validation=validation,
                    step=pending.model_step,
                    attempt=pending.structured_output_attempt,
                    redactor=self._secret_redactor,
                    tool_round_identity=identity,
                )
            )
        if auxiliary != {
            "schema_version": 1,
            "kind": "structured-output-validation",
            "step": pending.model_step,
            "attempt": pending.structured_output_attempt,
            "valid": validation.valid,
            "retry_scheduled": retry,
            "event_ids": [event.id for event in expected],
        } or receipt.appended_event_ids != tuple(event.id for event in expected):
            raise RuntimeError("Closed structured output receipt conflicts with its source result.")
        records: list[Event] = []
        for event in expected:
            expected_event = event_with_execution_profile_authority(event, context.profile)
            rows = await self._session_store.query_events(
                EventQuery(session_id=session.id, event_id=event.id, limit=2)
            )
            if len(rows) != 1:
                raise RuntimeError("Closed structured output event is missing or ambiguous.")
            actual = rows[0].event
            if (
                actual.id != event.id
                or actual.type != event.type
                or actual.session_id != session.id
                or actual.interaction_id != receipt.interaction_id
                or actual.agent_name != event.agent_name
                or actual.environment_name != event.environment_name
                or canonical_durable_json_bytes(actual.payload, "structured output evidence")
                != canonical_durable_json_bytes(
                    expected_event.payload, "structured output evidence"
                )
            ):
                raise RuntimeError("Closed structured output event conflicts with its receipt.")
            if actual.type in kinds:
                records.append(copy_event(actual))
        return tuple(records)

    async def _tool_result_tail_has_durable_lifecycle_provenance(
        self,
        *,
        session: Session,
        identity: ToolRoundIdentity,
        assistant_call_parts: tuple[ToolCallPart, ...],
        tool_result_message: Message,
        arguments_deferred: bool,
    ) -> bool:
        """Verify pause-resume results published outside the ordinary round protocol.

        Approval and user-input continuations atomically append their grouped
        result while clearing their pause checkpoint. They do not create an
        ordinary tool-round receipt, so recovery reconstructs their exact
        transcript message from bounded, round-scoped terminal evidence.
        """

        pending_calls = [
            PendingToolCallApproval(
                tool_call_id=part.tool_call_id,
                tool_name=part.tool_name,
                arguments=part.arguments,
            )
            for part in assistant_call_parts
        ]
        lifecycle_records = await self._session_store.query_events(
            EventQuery(
                session_id=session.id,
                event_types=(
                    EventType.TOOL_CALL_STARTED,
                    *_MODEL_BOUNDARY_TOOL_TERMINAL_EVENT_TYPES,
                ),
                limit=RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS + 1,
                order_by=EventOrder.SEQUENCE_DESC,
            )
        )
        resume_records = await self._session_store.query_events(
            EventQuery(
                session_id=session.id,
                event_type=EventType.SESSION_RESUMED,
                limit=RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS + 1,
                order_by=EventOrder.SEQUENCE_DESC,
            )
        )
        interruption_records = await self._session_store.query_events(
            EventQuery(
                session_id=session.id,
                event_type=EventType.SESSION_INTERRUPTED,
                limit=RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS + 1,
                order_by=EventOrder.SEQUENCE_DESC,
            )
        )
        expected_call_ids = {call.tool_call_id for call in pending_calls}
        lifecycle_evidence = _receiptless_exact_execution_evidence(
            lifecycle_records,
            identity=identity,
            expected_call_ids=expected_call_ids,
        )
        resume_evidence = _receiptless_exact_execution_evidence(
            resume_records,
            identity=identity,
            expected_call_ids=expected_call_ids,
        )
        interruption_evidence = _receiptless_exact_execution_evidence(
            interruption_records,
            identity=identity,
            expected_call_ids=expected_call_ids,
        )
        if len(lifecycle_evidence) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
            raise RuntimeError("Durable tool lifecycle evidence exceeds the recovery bound.")
        if len(resume_evidence) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
            raise RuntimeError("Durable pause-resume evidence exceeds the recovery bound.")
        if len(interruption_evidence) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
            raise RuntimeError("Durable pause-origin evidence exceeds the recovery bound.")
        if not resume_evidence:
            return False

        pending_by_call_id = {call.tool_call_id: call for call in pending_calls}
        if len(pending_by_call_id) != len(pending_calls):
            raise RuntimeError("The assistant tool-call tail contains duplicate call identities.")

        pause_kind: Literal["approval", "user-input"] | None = None
        pause_id: str | None = None
        resume_tool_call_id: str | None = None
        approval_decision: str | None = None
        for _sequence, event in resume_evidence:
            if (
                event.session_id != session.id
                or event.agent_name != session.agent_name
                or event.environment_name != session.environment_name
                or not identity.matches_payload(event.payload)
            ):
                raise RuntimeError(
                    "Durable pause-continuation anchor conflicts with its source model step."
                )
            event_pause_kind, event_pause_id = _receiptless_pause_event_identity(event)
            if pause_kind is None:
                pause_kind = event_pause_kind
                pause_id = event_pause_id
            elif pause_kind != event_pause_kind or pause_id != event_pause_id:
                raise RuntimeError(
                    "Durable pause-continuation evidence has conflicting pause identity."
                )
            tool_call_id = event.payload.get("tool_call_id")
            if type(tool_call_id) is not str:
                raise RuntimeError(
                    "Durable pause-continuation anchor has no valid tool-call identity."
                )
            if tool_call_id not in pending_by_call_id:
                raise RuntimeError(
                    "Durable pause-continuation anchor conflicts with its assistant tool call."
                )
            if event.tool_name is not None:
                raise RuntimeError(
                    "Durable pause-continuation anchor contains an unexpected tool name."
                )
            if resume_tool_call_id is None:
                resume_tool_call_id = tool_call_id
            elif resume_tool_call_id != tool_call_id:
                raise RuntimeError(
                    "Durable pause-continuation anchors identify different tool calls."
                )
            if event_pause_kind == "approval":
                decision = event.payload.get("decision")
                if decision not in {
                    ToolApprovalDecision.APPROVE.value,
                    ToolApprovalDecision.DENY.value,
                }:
                    raise RuntimeError("Durable approval continuation has no valid decision.")
                if approval_decision is None:
                    approval_decision = decision
                elif approval_decision != decision:
                    raise RuntimeError(
                        "Durable approval continuations contain conflicting decisions."
                    )
            elif event.payload.get("interruption_type") != _INTERRUPTION_TYPE_USER_INPUT_REQUIRED:
                raise RuntimeError(
                    "Durable user-input continuation has the wrong interruption type."
                )

        if pause_kind is None or pause_id is None or resume_tool_call_id is None:
            raise RuntimeError("Durable pause-continuation anchor has no pause identity.")

        user_input_open_receipt = None
        user_input_close_receipt = None
        if pause_kind == "user-input":
            user_input_open_receipt = await self._session_store.load_runtime_publication_receipt(
                session.id,
                f"user-input-open:{pause_id}",
            )
            user_input_close_receipt = await self._session_store.load_runtime_publication_receipt(
                session.id,
                f"user-input-close:{pause_id}",
            )
            if (
                user_input_open_receipt is None
                or user_input_close_receipt is None
                or user_input_open_receipt.kind != "user-input-open"
                or user_input_close_receipt.kind != "user-input-close"
                or user_input_open_receipt.intent.get("input_id") != pause_id
                or user_input_close_receipt.intent.get("input_id") != pause_id
                or user_input_open_receipt.interaction_id
                != user_input_open_receipt.intent.get("source_interaction_id")
                or user_input_close_receipt.interaction_id != user_input_open_receipt.interaction_id
            ):
                raise RuntimeError(
                    "Durable user-input continuation has no exact publication authority."
                )

        expected_interruption_type = (
            _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED
            if pause_kind == "approval"
            else _INTERRUPTION_TYPE_USER_INPUT_REQUIRED
        )
        expected_origin_field = "approval" if pause_kind == "approval" else "user_input"
        expected_calls = [
            {
                "tool_call_id": call.tool_call_id,
                "tool_name": call.tool_name,
                "arguments": call.arguments,
            }
            for call in pending_calls
        ]
        origin_sequences: list[int] = []
        for sequence, event in interruption_evidence:
            if event.payload.get("interruption_type") != expected_interruption_type:
                continue
            if (
                event.session_id != session.id
                or event.agent_name != session.agent_name
                or event.environment_name != session.environment_name
                or not identity.matches_payload(event.payload)
            ):
                raise RuntimeError(
                    "Durable pause-origin evidence conflicts with its source model step."
                )
            try:
                pause_checkpoint, origin_arguments_quarantined = (
                    tool_argument_publication.pause_checkpoint_validation_view(
                        event.payload.get(expected_origin_field),
                        pause_kind=pause_kind,
                    )
                )
                if origin_arguments_quarantined:
                    projected_calls = pause_checkpoint.get("tool_calls")
                    if type(projected_calls) is not list or len(projected_calls) != len(
                        pending_calls
                    ):
                        raise ValueError("Projected pause tool calls conflict with the round.")
                    for projected_call, pending_call in zip(
                        projected_calls,
                        pending_calls,
                        strict=True,
                    ):
                        if type(projected_call) is not dict:
                            raise ValueError("Projected pause tool calls conflict with the round.")
                        typed_projected_call = cast("dict[str, Any]", projected_call)
                        projected_tool_name = typed_projected_call.get("tool_name")
                        gateway_transcript_alias = (
                            pending_call.tool_name == CALL_TOOL_NAME
                            and type(projected_tool_name) is str
                            and projected_tool_name != CALL_TOOL_NAME
                        )
                        if (
                            projected_tool_name != pending_call.tool_name
                            and not gateway_transcript_alias
                        ):
                            raise ValueError("Projected pause tool calls conflict with the round.")
                        typed_projected_call["tool_call_id"] = pending_call.tool_call_id
                        if gateway_transcript_alias:
                            typed_projected_call["tool_name"] = CALL_TOOL_NAME
                    gating_pending_call = pending_by_call_id[resume_tool_call_id]
                    if gating_pending_call.tool_name == CALL_TOOL_NAME:
                        pause_checkpoint["tool_name"] = CALL_TOOL_NAME
                    pause_checkpoint.update(
                        {
                            "tool_round_id": identity.tool_round_id,
                            "model_step_id": identity.model_step_id,
                            "model_attempt_id": identity.model_attempt_id,
                            "tool_call_id": resume_tool_call_id,
                            ("approval_id" if pause_kind == "approval" else "input_id"): pause_id,
                        }
                    )
                    assistant_publication = pause_checkpoint.get("assistant_publication")
                    if type(assistant_publication) is dict:
                        safe_assistant_message = assistant_publication.get("message")
                        if type(safe_assistant_message) is not dict:
                            raise ValueError(
                                "Projected pause has no safe assistant publication evidence."
                            )
                        pause_checkpoint["assistant_message_state"] = "quarantined"
                        pause_checkpoint["quarantined_assistant_message"] = safe_assistant_message
                if pause_kind == "approval":
                    pending_pause = PendingToolApproval.model_validate(pause_checkpoint)
                    origin_pause_id = pending_pause.approval_id
                else:
                    assert user_input_open_receipt is not None
                    assert user_input_close_receipt is not None
                    pause_checkpoint.update(
                        {
                            "session_id": user_input_open_receipt.intent.get("session_id"),
                            "session_instance_id": user_input_open_receipt.intent.get(
                                "session_instance_id"
                            ),
                            "source_interaction_id": user_input_open_receipt.intent.get(
                                "source_interaction_id"
                            ),
                            "source_run_epoch": user_input_open_receipt.intent.get(
                                "source_run_epoch"
                            ),
                            "execution_profile_fingerprint": (
                                user_input_open_receipt.intent.get("execution_profile_fingerprint")
                            ),
                        }
                    )
                    pending_pause = PendingUserInput.model_validate(pause_checkpoint)
                    origin_pause_id = pending_pause.input_id
                    await self._user_input_evidence.require_exact_user_input_open_receipt(
                        session=session,
                        input_id=pending_pause.input_id,
                    )
                    await self._user_input_evidence.exact_user_input_close_event(
                        session=session,
                        input_id=pending_pause.input_id,
                        receipt=user_input_close_receipt,
                    )
                    pause_identity = pending_user_input_identity(pending_pause)
                    if any(
                        user_input_open_receipt.intent.get(key) != value
                        or user_input_close_receipt.intent.get(key) != value
                        for key, value in pause_identity.items()
                        if key != "pause_digest"
                    ) or user_input_open_receipt.intent.get(
                        "pause_digest"
                    ) != user_input_close_receipt.intent.get("pause_digest"):
                        raise ValueError(
                            "User-input publication receipts conflict with the pause origin."
                        )
            except (TypeError, ValueError):
                raise RuntimeError(
                    "Durable pause-origin evidence contains an invalid pause checkpoint."
                ) from None
            origin_identity = ToolRoundIdentity(
                model_step_id=pending_pause.model_step_id,
                model_attempt_id=pending_pause.model_attempt_id,
                tool_round_id=pending_pause.tool_round_id,
            )
            origin_calls = [
                {
                    "tool_call_id": call.tool_call_id,
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                }
                for call in pending_pause.tool_calls
            ]
            comparable_origin_calls = origin_calls
            comparable_expected_calls = expected_calls
            if arguments_deferred or origin_arguments_quarantined:
                comparable_origin_calls = [
                    {
                        "tool_call_id": call["tool_call_id"],
                        "tool_name": call["tool_name"],
                    }
                    for call in origin_calls
                ]
                comparable_expected_calls = [
                    {
                        "tool_call_id": call["tool_call_id"],
                        "tool_name": call["tool_name"],
                    }
                    for call in expected_calls
                ]
            if (
                origin_pause_id != pause_id
                or origin_identity != identity
                or pending_pause.agent_name != session.agent_name
                or pending_pause.environment_name != session.environment_name
                or pending_pause.tool_call_id != resume_tool_call_id
                or canonical_durable_json_bytes(
                    comparable_origin_calls,
                    "pause_origin_tool_calls",
                )
                != canonical_durable_json_bytes(
                    comparable_expected_calls,
                    "assistant_tool_calls",
                )
            ):
                raise RuntimeError("Durable pause-origin evidence conflicts with its continuation.")
            origin_sequences.append(sequence)

        if not origin_sequences:
            raise RuntimeError("Receipt-less tool evidence has no authoritative pause origin.")
        if min(origin_sequences) >= min(sequence for sequence, _event in resume_evidence):
            raise RuntimeError("Durable pause-continuation evidence precedes its pause origin.")
        if lifecycle_evidence and min(sequence for sequence, _event in lifecycle_evidence) <= min(
            sequence for sequence, _event in resume_evidence
        ):
            raise RuntimeError("Durable tool lifecycle evidence precedes its pause continuation.")

        started_by_call_id: dict[str, tuple[int, Event]] = {}
        terminal_by_call_id: dict[str, tuple[int, Event]] = {}
        for sequence, event in lifecycle_evidence:
            if (
                event.session_id != session.id
                or event.agent_name != session.agent_name
                or event.environment_name != session.environment_name
                or not identity.matches_payload(event.payload)
            ):
                raise RuntimeError(
                    "Durable tool lifecycle evidence conflicts with its source model step."
                )
            event_pause_kind, event_pause_id = _receiptless_pause_event_identity(event)
            if pause_kind != event_pause_kind or pause_id != event_pause_id:
                raise RuntimeError(
                    "Durable pause-continuation evidence has conflicting pause identity."
                )
            tool_call_id = event.payload.get("tool_call_id")
            if type(tool_call_id) is not str:
                raise RuntimeError(
                    "Durable tool lifecycle evidence has no valid tool-call identity."
                )
            pending_call = pending_by_call_id.get(tool_call_id)
            if pending_call is None:
                raise RuntimeError(
                    "Durable tool lifecycle evidence conflicts with its assistant tool call."
                )
            gateway_outer_call = False
            if event.tool_name != pending_call.tool_name:
                gateway_outer_call = gateway_lifecycle_matches_outer_call(
                    effective_tool_name=event.tool_name,
                    event_payload=event.payload,
                    outer_tool_name=pending_call.tool_name,
                    outer_arguments=pending_call.arguments,
                )
            if event.tool_name != pending_call.tool_name and not gateway_outer_call:
                raise RuntimeError(
                    "Durable tool lifecycle evidence conflicts with its assistant tool call."
                )
            expected_idempotency_key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_round_id=identity.tool_round_id,
                tool_call_id=tool_call_id,
                approval_id=pause_id if event_pause_kind == "approval" else None,
                pause_id=pause_id if event_pause_kind == "user-input" else None,
            )
            if event.payload.get("idempotency_key") != expected_idempotency_key:
                raise RuntimeError(
                    "Durable pause-continuation evidence has a conflicting idempotency key."
                )
            if event.type == EventType.TOOL_CALL_STARTED:
                if tool_call_id in started_by_call_id:
                    raise RuntimeError(
                        "Durable pause-continuation evidence contains duplicate started events."
                    )
                if not gateway_outer_call and not (
                    tool_argument_publication.started_arguments_match_private_call(
                        event.payload,
                        private_arguments=pending_call.arguments,
                    )
                ):
                    raise RuntimeError(
                        "Durable pause-continuation started arguments conflict with "
                        "the assistant tool call."
                    )
                started_by_call_id[tool_call_id] = (sequence, event)
                continue
            if event.type not in _MODEL_BOUNDARY_TOOL_TERMINAL_EVENT_TYPES:
                continue
            if tool_call_id in terminal_by_call_id:
                raise RuntimeError(
                    "Durable tool lifecycle evidence contains duplicate terminal results."
                )
            terminal_by_call_id[tool_call_id] = (sequence, event)

        if set(terminal_by_call_id) != set(pending_by_call_id):
            return False

        outcomes: list[runtime_records.ToolCallOutcome] = []
        for pending_call in pending_calls:
            terminal_sequence, terminal = terminal_by_call_id[pending_call.tool_call_id]
            started = started_by_call_id.get(pending_call.tool_call_id)
            outcome = resume_ledger.tool_call_outcome_from_terminal_event(
                event=terminal,
                pending_tool_call=pending_call,
            )
            limit_skip = (
                terminal.type is EventType.TOOL_CALL_FAILED
                and terminal.payload.get("reason") == "limit_reached"
            )
            if limit_skip:
                structured = outcome.result.structured
                if (
                    started is not None
                    or pause_kind != "approval"
                    or approval_decision != ToolApprovalDecision.APPROVE.value
                    or not outcome.result.is_error
                    or not isinstance(structured, Mapping)
                    or structured.get("skipped") is not True
                    or structured.get("reason") != "limit_reached"
                    or structured.get("tool_call_id") != pending_call.tool_call_id
                    or structured.get("tool_name") != pending_call.tool_name
                    or not identity.matches_payload(structured)
                    or structured.get("limit") != terminal.payload.get("limit")
                    or structured.get("limit") not in {limit.value for limit in StopLimit}
                    or "maximum" not in structured
                    or "actual" not in structured
                ):
                    raise RuntimeError(
                        "Durable limit skip conflicts with never-dispatched evidence."
                    )
            if (
                terminal.type
                in {
                    EventType.TOOL_CALL_COMPLETED,
                    EventType.TOOL_CALL_FAILED,
                }
                and started is None
                and not (
                    terminal.type is EventType.TOOL_CALL_FAILED
                    and (
                        terminal.payload.get("registration_state") == "unregistered_at_policy_plan"
                        or limit_skip
                    )
                )
            ):
                raise RuntimeError("Durable executed tool evidence has no preceding started event.")
            if (
                terminal.payload.get("registration_state") == "unregistered_at_policy_plan"
                and started is not None
            ):
                raise RuntimeError(
                    "Durable unregistered-at-plan evidence contains a started event."
                )
            if terminal.type == EventType.TOOL_CALL_APPROVAL_DENIED and started is not None:
                raise RuntimeError(
                    "Durable approval-denied tool evidence contains a started event."
                )
            if started is not None and started[0] >= terminal_sequence:
                raise RuntimeError(
                    "Durable pause-continuation terminal evidence precedes its start."
                )
            if (
                pause_kind == "approval"
                and approval_decision == ToolApprovalDecision.DENY.value
                and terminal.type
                not in {
                    EventType.TOOL_CALL_BLOCKED,
                    EventType.TOOL_CALL_APPROVAL_DENIED,
                }
            ):
                raise RuntimeError(
                    "Durable denied approval evidence contains an executed tool result."
                )
            if (
                pause_kind == "approval"
                and approval_decision == ToolApprovalDecision.DENY.value
                and started is not None
            ):
                raise RuntimeError("Durable denied approval evidence contains a started event.")
            if (
                (
                    pause_kind == "approval"
                    and approval_decision == ToolApprovalDecision.APPROVE.value
                )
                or pause_kind == "user-input"
            ) and terminal.type == EventType.TOOL_CALL_APPROVAL_DENIED:
                raise RuntimeError(
                    "Durable pause-continuation decision conflicts with its terminal result."
                )
            completed = terminal.type == EventType.TOOL_CALL_COMPLETED
            if completed == outcome.result.is_error:
                raise RuntimeError(
                    "Durable terminal tool evidence conflicts with its result status."
                )
            outcomes.append(outcome)

        expected_messages = transcript_helpers.tool_result_messages(
            outcomes,
            tool_round_identity=identity,
        )
        if canonical_durable_json_bytes(
            [message.model_dump(mode="json") for message in expected_messages],
            "expected_tool_result_messages",
        ) != canonical_durable_json_bytes(
            [tool_result_message.model_dump(mode="json")],
            "tool_result_messages",
        ):
            raise RuntimeError(
                "The tool-result transcript conflicts with its durable terminal evidence."
            )
        return True

    async def _complete_unknown_auxiliary_stage(
        self,
        *,
        session: Session,
        stage: ModelCompletionStage,
        invocation: InvocationContext,
    ) -> None:
        """Retain unknown consumption without manufacturing a response or redispatch."""
        from cayu.budgets.billing import BillingIdentity
        from cayu.events import event_with_runtime_nested_payload_authority
        from cayu.runtime._auxiliary_inference_contract import (
            AUXILIARY_ATTRIBUTION_AUTHORITY_PATHS,
            auxiliary_budget_recovery_contexts,
            auxiliary_terminal_publication,
        )
        from cayu.runtime._durable_model_terminalization import terminalization_plan_owner
        from cayu.sessions.base import ModelCompletionStageRecoveryFence

        invocation._validate()
        if invocation.recovery_claim_id is None:
            raise ModelCompletionManualRecoveryRequired("Auxiliary recovery requires a live claim.")
        binding = invocation.binding
        if (
            stage.purpose != "auxiliary-inference"
            or stage.intent.get("session_instance_id") != binding.session_instance_id
            or stage.intent.get("interaction_id") != binding.interaction_id
            or stage.intent.get("provider_name") != binding.provider_name
            or stage.intent.get("requested_model") != binding.model
            or stage.intent.get("execution_profile_fingerprint") != invocation.profile.fingerprint
        ):
            raise ModelCompletionManualRecoveryRequired(
                "Auxiliary recovery invocation conflicts with its stage."
            )
        identity = ModelAttemptIdentity.model_validate(
            {
                "model_step_id": stage.intent.get("model_step_id"),
                "model_attempt_id": stage.intent.get("model_attempt_id"),
            }
        )
        pricing_provider = stage.intent.get("pricing_provider_name")
        if type(pricing_provider) is not str:
            raise ModelCompletionManualRecoveryRequired(
                "Auxiliary recovery lost its pricing provider."
            )
        pricing_provider = require_clean_nonblank(pricing_provider, "pricing_provider_name")
        raw_billing = stage.intent.get("billing_identity")
        billing_identity = (
            None if raw_billing is None else BillingIdentity.model_validate(raw_billing)
        )
        event = Event(
            type=EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED,
            session_id=session.id,
            interaction_id=binding.interaction_id,
            agent_name=binding.agent_name,
            environment_name=binding.environment_name,
            payload={
                **identity.payload(),
                "auxiliary_inference": stage.intent["auxiliary_inference"],
                "provider_name": binding.provider_name,
                "requested_model": binding.model,
                "provider": pricing_provider,
                "model": binding.model,
                "execution_profile_fingerprint": invocation.profile.fingerprint,
                "auxiliary_outcome": "outcome_unknown",
                "attempt": stage.dispatch_ordinal + 1,
                "usage_status": "missing",
            },
        )
        event = event_with_execution_profile_authority(
            event_with_runtime_payload_authority(event, "model_step_id", "model_attempt_id"),
            invocation.profile,
        )
        event = await self._run_limit_controller.recover_model_completion_budget_evidence(
            event_with_runtime_nested_payload_authority(
                event, *AUXILIARY_ATTRIBUTION_AUTHORITY_PATHS
            ),
            reservation_ids=stage.reservation_ids,
            recovery_contexts=auxiliary_budget_recovery_contexts(stage),
            session=session,
            provider_name=pricing_provider,
            model_attempt_identity=identity,
            dispatch_id=identity.model_attempt_id,
            request_billing_identity=billing_identity,
        )
        await self._session_store.complete_recovered_model_completion_stage(
            session.id,
            stage_id=stage.stage_id,
            publication=auxiliary_terminal_publication(stage, event),
            recovery_fence=ModelCompletionStageRecoveryFence(
                expected_session_instance_id=binding.session_instance_id,
                expected_run_epoch=binding.run_epoch,
                expected_active_profile=invocation.active_profile,
                recovery_claim_id=invocation.recovery_claim_id,
                preparation_digest=stage.preparation_digest,
                recovery_plan_ownership=terminalization_plan_owner(),
            ),
        )

    @staticmethod
    def validate_active_model_completion_stage(session: Session, stage) -> None:
        if stage.session_id != session.id:
            raise RuntimeError("The active model-completion stage belongs to another session.")
        if stage.source_run_epoch > session.run_epoch:
            raise RuntimeError(
                "The active model-completion stage was prepared by a future run epoch."
            )

    async def _promote_completed_model_stage(
        self,
        *,
        session: Session,
        stage_id: str,
    ) -> Session:
        async def commit_once():
            return await self._session_store.promote_model_completion_stage(
                session.id,
                stage_id=stage_id,
                expected_run_epoch=session.run_epoch,
            )

        async def commit():
            try:
                return await commit_once()
            except Exception as first_error:
                try:
                    return await commit_once()
                except Exception as replay_error:
                    replay_error.add_note(
                        "Exact recovered model-completion promotion also failed after "
                        f"{type(first_error).__name__}: {first_error}"
                    )
                    raise replay_error from first_error

        task = asyncio.create_task(commit())
        outcome = await await_shielded_task_outcome(task)
        cancellation = outcome.cancellation
        error = outcome.error
        if isinstance(error, asyncio.CancelledError) and cancellation is None:
            error = unexpected_child_cancellation_error(
                error,
                operation="Recovered model-completion promotion",
            )
        if error is not None:
            if cancellation is not None:
                cancellation.add_note(
                    "Recovered model-completion promotion also failed: "
                    f"{type(error).__name__}: {error}"
                )
                raise cancellation from error
            raise error
        if outcome.result is None:
            result_error = RuntimeError(
                "Recovered model-completion promotion returned no acknowledgement."
            )
            if cancellation is not None:
                cancellation.add_note(str(result_error))
                raise cancellation from result_error
            raise result_error
        if cancellation is not None:
            raise cancellation
        return outcome.result.session

    async def recover_provider_operation(
        self,
        session: Session,
        stage: ModelCompletionStage,
        operation: RecoverableProviderOperation,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        invocation_context: InvocationContext | None = None,
        model_execution_selection: ModelExecutionSelection | None = None,
    ) -> ProviderOperationRecoveryResult:
        recovery_context = model_completion_recovery_context_from_stage(stage)
        publication_context = recovery_context or ModelCompletionRecoveryContext()
        publish = self._assistant_model_publication.recovery_publisher(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            publication_context=publication_context,
        )

        return await self._model_step_executor.recover_provider_operation(
            session=session,
            stage=stage,
            operation=operation,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            environment_name=_environment_name(registered_environment),
            recovery_context=recovery_context,
            model_completion_publisher=publish,
            invocation_context=invocation_context,
            model_execution_selection=model_execution_selection,
        )

    async def recover_provider_operation_start(
        self,
        session: Session,
        stage: ModelCompletionStage,
        start: RecoverableProviderOperationStart,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        invocation_context: InvocationContext | None = None,
        model_execution_selection: ModelExecutionSelection | None = None,
    ) -> ProviderOperationRecoveryResult:
        recovery_context = model_completion_recovery_context_from_stage(stage)
        publication_context = recovery_context or ModelCompletionRecoveryContext()
        publish = self._assistant_model_publication.recovery_publisher(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            publication_context=publication_context,
        )

        return await self._model_step_executor.recover_provider_operation_start(
            session=session,
            stage=stage,
            start=start,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            environment_name=_environment_name(registered_environment),
            model_completion_publisher=publish,
            invocation_context=invocation_context,
            model_execution_selection=model_execution_selection,
        )

    async def cancel_provider_operation(
        self,
        session: Session,
        stage: ModelCompletionStage,
        operation: RecoverableProviderOperation,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        invocation_context: InvocationContext | None = None,
        model_execution_selection: ModelExecutionSelection | None = None,
    ) -> ProviderOperationSnapshot | None:
        return await self._model_step_executor.cancel_provider_operation_for_interruption(
            session=session,
            stage=stage,
            operation=operation,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            environment_name=_environment_name(registered_environment),
            invocation_context=invocation_context,
            model_execution_selection=model_execution_selection,
        )
