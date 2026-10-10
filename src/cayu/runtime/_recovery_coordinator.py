"""Durable runtime continuation and crash-recovery ownership.

The coordinator owns paused-round continuation, recorded tool-round repair,
incomplete-session recovery, subagent reattachment, and abandoned-stream
finalization without importing or accepting :class:`CayuApp`. Public request
validation and registry ownership remain on the application façade; the
coordinator resolves registrations through narrow callbacks. Session execution,
interruption, turn accounting, and terminal hook orchestration are likewise
supplied through narrow callbacks by the composition root.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any, Literal
from uuid import uuid4

import cayu.sessions.pending_actions as pending_actions
from cayu._exception_groups import (
    _attach_exception_cause_preserving_graph,
    add_exception_note_safely,
    exception_cause,
    iter_exception_tree,
)
from cayu._task_wait import (
    await_shielded_task_outcome,
    restore_task_cancellation_requests,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_metadata,
    copy_durable_record,
    copy_json_value,
)
from cayu.approvals.review import (
    HumanReviewCall,
    HumanReviewConflict,
    HumanReviewContext,
    HumanReviewDenied,
    HumanReviewPolicy,
    HumanReviewReference,
    HumanReviewSource,
    HumanReviewView,
    build_review,
    require_current_review,
    require_review_authority,
)
from cayu.approvals.tools import (
    PendingToolApproval,
    PendingToolCallApproval,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
    ToolPolicyEvidence,
    expiry_resolution_actor,
    resolution_actor_payload,
)
from cayu.approvals.user_input import (
    PENDING_USER_INPUT_CHECKPOINT_KEY,
    USER_INPUT_SUPERSESSION_INTENT_KEY,
    PendingUserInput,
    UserInputPauseState,
    UserInputRecoveryRequest,
    UserInputResolutionIntent,
    UserInputResponse,
    UserInputSupersessionIntent,
    checkpoint_with_executing_user_input_resolution_intent,
    checkpoint_with_user_input_resolution_intent,
    checkpoint_without_exact_pending_user_input,
    event_with_pending_user_input_authority,
    event_with_user_input_supersession_authority,
    pending_user_input_digest,
    pending_user_input_identity,
    pending_user_input_interruption_payload,
    require_resolution_intent_matches_pending,
    user_input_answer_request_digest,
    user_input_lifecycle_authority_from_checkpoint,
    user_input_resolution_request_digest,
    user_input_supersession_intent_for,
)
from cayu.budgets._run_limit_accounting import (
    RunLimitAccountingContext,
    rebase_run_limit_accounting_context,
    resume_run_limit_accounting_context,
)
from cayu.budgets.base import (
    BudgetLimit,
    BudgetPolicy,
    copy_budget_policy,
    copy_request_budget_limits,
    request_budget_limits_for_session,
)
from cayu.budgets.run_limits import RunLimits, copy_run_limits, has_run_limits
from cayu.budgets.usage import session_usage_summary
from cayu.collaboration.access import CollaborationAccessContext
from cayu.context.structured_output import (
    StructuredOutputSpec,
    copy_structured_output_spec,
    require_secret_free_structured_output_spec,
)
from cayu.context.structured_output import (
    _require_native_structured_output_support as _require_provider_native_output_support,
)
from cayu.context.thinking import ThinkingConfig
from cayu.environments.factory import EnvironmentFactoryOperation
from cayu.events import (
    Event,
    EventType,
    copy_event,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
    event_with_execution_profile_authority,
)
from cayu.execution_units import (
    ModelAttemptIdentity,
    ToolRoundIdentity,
)
from cayu.failure_evidence import exception_evidence
from cayu.messages import Message, detach_message
from cayu.observability.hooks import RuntimeHookPhase
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime import _approval_publication as approval_publication
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _resume_ledger as resume_ledger
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_results as tool_results
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._approval_support import _pending_approval_and_round_for_atomic_claim
from cayu.runtime._auxiliary_invocation import AuxiliaryInvocationPolicy
from cayu.runtime._continuation_environment import ContinuationEnvironment
from cayu.runtime._continuation_task_failure import (
    ApprovalTaskFailureIdentity,
    approval_failure_event_id,
    approval_task_failure_payload,
    approval_task_failure_receipt_matches,
    approval_task_terminalization_idempotency_key,
    approval_task_terminalization_request,
    load_direct_task_failure_replay,
)
from cayu.runtime._delegated_event_stream import _close_delegated_event_stream
from cayu.runtime._diagnostics import (
    exception_diagnostic,
)
from cayu.runtime._durable_tool_round import (
    DeferredInteractionInput,
    DurableToolRound,
    _redactor_for_tool_calls,
)
from cayu.runtime._durable_tool_round import (
    _interrupted_tool_round_results as _interrupted_tool_round_results,
)
from cayu.runtime._environment_lifecycle import (
    EnvironmentLifecycle,
    exception_failure_payload,
)
from cayu.runtime._event_writer import (
    RuntimeEventWriter,
)
from cayu.runtime._execution_profile_continuation import ExecutionProfileContinuation
from cayu.runtime._foreground_child_wait import (
    ForegroundChildActionRequired,
    event_with_foreground_child_wait_authority,
)
from cayu.runtime._foreground_gate_continuation import ForegroundGatePolicyOwner, GateReplay
from cayu.runtime._foreground_subagent_recovery import ForegroundSubagentRecoveryRequired
from cayu.runtime._incomplete_session_recovery import (
    IncompleteSessionRecovery,
    _require_recovery_max_steps,
)
from cayu.runtime._interruption_coordinator import (
    _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY,
    _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY,
)
from cayu.runtime._invocation_lifecycle import (
    InvocationContext,
    reconstruct_invocation_context,
)
from cayu.runtime._manual_recovery_publication import (
    ManualRecoveryPublication,
    reconcile_manual_recovery_persistence,
)
from cayu.runtime._paused_tool_round import ApprovalRoundPause, PausedToolRound, UserInputRoundPause
from cayu.runtime._pending_tool_round_recovery import (
    PendingToolRoundRecovery,
    RegisteredAgentResolver,
    RegisteredEnvironmentResolver,
    _environment_name,
)
from cayu.runtime._provider_disposition_recovery import (
    ProviderDispositionRecovery,
)
from cayu.runtime._recovery_admission import (
    RecoveryAdmission,
    RecoveryMutationHook,
    _completed_tool_round_model_step,
    _continued_tool_exposure_profile_id,
    _RecoveryInvocationSemantics,
)
from cayu.runtime._recovery_claims import (
    _IncompleteRecoveryClaim,
    _IncompleteRecoveryClaimAuthority,
    _IncompleteRecoveryClaimLost,
    _require_live_incomplete_recovery_claim_acknowledgement,
)
from cayu.runtime._recovery_ownership import (
    RecoveryOwnership,
    _authoritative_recovery_ownership_failure,
    _checkpoint_with_rebased_session_run_operation,
    _checkpoint_without_active_incomplete_recovery_claim,
    _prepend_exception_cause,
    _recovery_abandonment_signal,
    _require_aware_datetime,
)
from cayu.runtime._recovery_requests import (
    RecoveryInterruptionRequest,
    RecoveryLimitStopRequest,
    RecoverySessionRunRequest,
    RecoveryTaskEventRequest,
    RecoveryTerminalEventRequest,
)
from cayu.runtime._run_limit_accounting import restore_run_limit_accounting_context
from cayu.runtime._run_limits import (
    RunLimitController,
    SessionUsageTracker,
)
from cayu.runtime._session_control import (
    SessionControl,
)
from cayu.runtime._session_engine import SessionEngine
from cayu.runtime._session_finalization import (
    SessionFinalization,
    _interaction_transition_replay_failures,
    _recovery_task_event,
)
from cayu.runtime._terminal_event_publication import TerminalEventPublication
from cayu.runtime._terminal_evidence_finalization import (
    _RECOVERY_RESUMABLE_SESSION_STATUSES,
    TerminalEvidenceFinalization,
    _terminal_finalization_failure_without_identity,
)
from cayu.runtime._terminal_evidence_reader import TerminalEvidenceReader
from cayu.runtime._terminal_finalization_lifetime import _terminal_finalization_process_control
from cayu.runtime._tool_effect_reconciliation import (
    ToolEffectReconciliationOwner,
    project_accepted_reconciliation,
    reconciliation_request_digest,
)
from cayu.runtime._tool_effect_state import (
    ToolEffectConflict,
    ToolEffectObservation,
    ToolEffectRecord,
    ToolEffectStateOwner,
    ToolEffectTerminal,
    _validate_selected_observation,
    _validate_selected_terminal,
)
from cayu.runtime._tool_invocation.admission import (
    policy_denial_payload_fields,
)
from cayu.runtime._tool_invocation.invocation import ToolInvocation
from cayu.runtime._user_input_recovery_evidence import UserInputRecoveryEvidence
from cayu.runtime.loop_policies import LoopPolicy
from cayu.runtime.provider_operations import (
    ProviderOperationEvidenceError,
    ProviderOperationPendingDisposition,
    ProviderOperationResolutionAction,
    ProviderOperationResolutionRequest,
    ProviderOperationResolutionResult,
    load_pending_provider_operation_disposition,
    prepare_provider_operation_resolution_request,
    provider_operation_resolution_outcome_event_id,
    resolve_provider_operation_stage,
    validate_provider_operation_resolution_outcome_event,
)
from cayu.runtime.tool_effects import (
    ToolEffectReconciliationRequest,
    ToolEffectReconciliationTarget,
    tool_effect_receipt_digest,
)
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions import _tool_call_evidence as tool_call_evidence
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_matches_session_epoch,
    checkpoint_with_active_invocation_execution_profile,
)
from cayu.sessions._foreground_child_checkpoint import ForegroundChildTerminal
from cayu.sessions._invocation_lifecycle import (
    InvocationLifecycleCommandConflict,
)
from cayu.sessions._terminal_evidence import (
    _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
    _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
    _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
    _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
    _SESSION_RUN_OPERATION_CHECKPOINT_KEY,
    _session_run_operation_from_checkpoint,
    _SessionRunOperation,
)
from cayu.sessions.base import (
    _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY,
    RuntimePublicationReceipt,
    SessionRunFenced,
    SessionRuntimePublicationConflict,
    SessionStatusConflict,
    SessionStore,
    _activate_owned_session_run_fence,
    _checkpoint_with_session_run_operation,
    _deactivate_session_interaction,
    _incomplete_recovery_claim_from_checkpoint,
    runtime_publication_checkpoint_value_digest,
)
from cayu.sessions.cleanup import (
    RecoveryCleanupStep,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
    INTERACTION_TERMINAL_EVENT_TYPES,
)
from cayu.sessions.records import (
    Session,
    SessionStatus,
)
from cayu.tasks._terminalization import _terminalize_claimed_task
from cayu.tasks.store import TaskStore
from cayu.tools import _argument_publication as tool_argument_publication
from cayu.tools.base import (
    _TOOL_POLICY_DENIAL_SOURCE,
    ToolEffect,
    ToolResult,
)
from cayu.tools.exposure import (
    ResolvedToolExposureAuthority,
    validate_resolved_tool_exposure_authority,
)
from cayu.tools.policy import ToolPolicyDecision
from cayu.tools.rounds import ToolRoundRecoveryRequest
from cayu.vaults.redaction import SecretRedactor

# Opaque store cursors require consuming each fetched page completely. When
# only one result slot remains, that can force one-candidate pages; cap the
# database round trips and continue through Cayu's outer cursor instead.

_TOOL_ROUND_RECOVERABLE_SESSION_STATUSES = {
    SessionStatus.RUNNING,
    SessionStatus.INTERRUPTING,
    SessionStatus.INTERRUPTED,
    SessionStatus.FAILED,
}


_MANUAL_RECOVERY_SECRET_SCOPE_UNAVAILABLE = (
    "Externally verified tool output is unavailable because the invocation "
    "secret scope could not be reconstructed."
)


def _public_manual_recovery_result(
    result: ToolResult,
    *,
    secret_resolution_scope: invocation_secrets.SecretResolutionScope,
) -> ToolResult:
    """Fail closed when recovery cannot positively prove a static secret scope."""

    if secret_resolution_scope == "static":
        return result.model_copy(deep=True)
    return ToolResult(
        content=_MANUAL_RECOVERY_SECRET_SCOPE_UNAVAILABLE,
        structured={
            "error": "invalid_tool_output",
            "manual_recovery": True,
            "outcome_unknown": False,
            "reason": "invocation_secret_scope_unavailable",
        },
        is_error=result.is_error,
    )


def _public_resolution_audit_fields(
    *,
    secret_resolution_scope: invocation_secrets.SecretResolutionScope,
    reason: str | None,
    metadata: dict[str, Any],
    redactor: SecretRedactor,
) -> dict[str, Any]:
    """Project operator audit text only with positive static-scope evidence."""

    if secret_resolution_scope != "static":
        return {"reason": None, "metadata": {}}
    return {
        "reason": reason,
        **approval_support.bounded_resolution_metadata_payload(
            metadata,
            redactor=redactor,
        ),
    }


def _optional_exception_type_name(
    error: BaseException | None,
    *,
    redactor: SecretRedactor,
) -> str:
    return (
        "Exception" if error is None else exception_diagnostic(error, redactor=redactor).error_type
    )


def _environment_factory_resolution_error_payload(
    error: BaseException,
    *,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    """Project one factory reconnect failure for durable recovery records."""

    return exception_diagnostic(
        error,
        empty_message="environment factory resolution failed",
        nonportable_message=(
            "Environment factory resolution failed with a non-portable diagnostic."
        ),
        redactor=redactor,
    ).payload_fields()


logger = logging.getLogger(__name__)

EffectiveRetryPolicy = Callable[[RetryPolicy | None], RetryPolicy]


def _checkpoint_with_legacy_approval_round(
    checkpoint: dict[str, Any] | None,
    *,
    approval: PendingToolApproval,
    redactor: SecretRedactor,
    runtime_session: Session | None = None,
) -> dict[str, Any] | None:
    """Atomically upgrade an approval-only checkpoint at its exact claim."""

    pending_round = pending_round_reader.pending_tool_round_from_checkpoint(
        checkpoint,
        redactor=redactor,
        runtime_session=runtime_session,
    )
    if pending_round is not None:
        return checkpoint
    current_approval = pending_approval_reader.pending_approval_from_checkpoint(
        checkpoint,
        redactor=redactor,
    )
    if current_approval != approval:
        raise RuntimeError("Pending tool approval changed before legacy round migration.")
    copied = {} if checkpoint is None else copy_durable_record(checkpoint, "checkpoint")
    copied[pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY] = (
        pending_approval_reader.planned_tool_round_from_pending_approval(approval).model_dump(
            mode="json"
        )
    )
    return copied


def _require_native_structured_output_support(
    structured_output: StructuredOutputSpec | None,
    *,
    registered_provider: runtime_records.RegisteredProvider,
) -> None:
    _require_provider_native_output_support(
        structured_output,
        provider_name=registered_provider.name,
        provider=registered_provider.provider,
    )


def _task_cancellation_count() -> int:
    """Return the current task's cancellation generation for boundary tracking."""
    task = asyncio.current_task()
    return 0 if task is None else task.cancelling()


def _authoritative_expired_recovery_claim_failure(
    operation_failure: BaseException | None,
    lease_failure: _IncompleteRecoveryClaimLost,
) -> BaseException:
    """Preserve fatal/cancellation authority while retaining expired-lease evidence."""

    return _authoritative_recovery_ownership_failure(operation_failure, lease_failure)


def _continuation_failure_payload(
    error: BaseException, *, session: Session, redactor: SecretRedactor
) -> dict[str, Any]:
    return {
        **exception_failure_payload(error, redactor=redactor),
        **(
            error.interruption_evidence()
            if isinstance(error, ForegroundSubagentRecoveryRequired)
            else {}
        ),
        "failure_evidence": exception_evidence(error)
        .model_copy(update={"session_id": session.id, "run_epoch": session.run_epoch})
        .model_dump(mode="json"),
    }


class _ManualRecoveryInterrupted(RuntimeError):
    """A durable interruption won before manual recovery could claim the session."""


class _ManualRecoveryCascadePending(RuntimeError):
    """Descendant interruption must finish before manual recovery can continue."""


@dataclass(frozen=True)
class _ManualRecoveryInterruptionFence:
    session: Session
    claim_id: str
    error: BaseException | None
    invocation_context: InvocationContext
    authority: _IncompleteRecoveryClaimAuthority

    def __post_init__(self) -> None:
        if (
            self.authority.session_id != self.session.id
            or self.authority.claim_id != self.claim_id
            or self.authority.run_epoch != self.session.run_epoch
        ):
            raise ValueError("Interrupted recovery authority does not match its session.")


@dataclass(frozen=True)
class _ManualRecoveryInterruptionReplay:
    event: Event


@dataclass(frozen=True)
class _ManualRecoveryEventDelivery:
    event: Event
    consumed: asyncio.Event


@dataclass(frozen=True)
class _ManualRecoveryStreamOutcome:
    error: BaseException | None
    interrupted_event: Event | None = None


@dataclass(frozen=True)
class _ManualRecoverySupervisorResult:
    error: BaseException | None
    cleanup_failure: BaseException | None


@dataclass(frozen=True)
class _ReconciledToolEffectReplay:
    record: ToolEffectRecord
    terminal_event: Event
    consumption_receipt: RuntimePublicationReceipt | None


@dataclass(frozen=True)
class _ToolEffectObservationReplay:
    record: ToolEffectRecord
    event: Event


class _FinalizedToolEffectRejection(ToolEffectConflict):
    """The receipt owner published interruption before rejecting its observation."""


RegisteredProviderResolver = Callable[[str], runtime_records.RegisteredProvider]
BudgetPolicyResolver = Callable[[], BudgetPolicy | None]


class RecoveryCoordinator:
    """Continue paused work and repair incomplete sessions from durable state."""

    def __init__(
        self,
        *,
        recovery_admission: RecoveryAdmission,
        provider_disposition: ProviderDispositionRecovery,
        incomplete_recovery: IncompleteSessionRecovery,
        session_store: SessionStore,
        foreground_gate_policy_owner: ForegroundGatePolicyOwner,
        require_participant_execution: Callable[
            [Session, CollaborationAccessContext | None], Awaitable[None]
        ],
        task_store: TaskStore | None,
        event_writer: RuntimeEventWriter,
        session_control: SessionControl[SessionUsageTracker],
        environment_lifecycle: EnvironmentLifecycle,
        run_limit_controller: RunLimitController,
        tool_invocation: ToolInvocation,
        pending_tool_round_recovery: PendingToolRoundRecovery,
        deferred_input: DeferredInteractionInput,
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
        effective_retry_policy: EffectiveRetryPolicy,
        engine: SessionEngine,
        terminal_event_publication: TerminalEventPublication,
        resolve_registered_agent: RegisteredAgentResolver,
        resolve_registered_provider: RegisteredProviderResolver,
        resolve_registered_environment: RegisteredEnvironmentResolver,
        resolve_budget_policy: BudgetPolicyResolver,
        execution_profile_continuation: ExecutionProfileContinuation,
        user_input_evidence: UserInputRecoveryEvidence,
        terminal_evidence: TerminalEvidenceReader,
        recovery_ownership: RecoveryOwnership,
        terminal_finalization: TerminalEvidenceFinalization,
        session_finalization: SessionFinalization,
        human_review_policy: HumanReviewPolicy | None = None,
        runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...] = (),
        loop_policies: tuple[LoopPolicy, ...] = (),
    ) -> None:
        self._human_review_policy = human_review_policy
        self._recovery_admission = recovery_admission
        self._provider_disposition = provider_disposition
        self._incomplete_recovery = incomplete_recovery
        self._session_store = session_store
        self._require_participant_execution = require_participant_execution
        self._foreground_gate_policy_owner = foreground_gate_policy_owner
        self._task_store = task_store
        self._event_writer = event_writer
        self._session_control = session_control
        self._environment_lifecycle = environment_lifecycle
        self._run_limit_controller = run_limit_controller
        self._pending_tool_round_recovery = pending_tool_round_recovery
        self._deferred_input = deferred_input
        self._tool_invocation = tool_invocation
        self._secret_redactor = secret_redactor
        self._clock = clock
        self._effective_retry_policy = effective_retry_policy
        self._engine = engine
        self._terminal_event_publication = terminal_event_publication
        self._resolve_registered_agent = resolve_registered_agent
        self._resolve_registered_provider = resolve_registered_provider
        self._resolve_registered_environment = resolve_registered_environment
        self._resolve_budget_policy = resolve_budget_policy
        self._execution_profile_continuation = execution_profile_continuation
        self._user_input_evidence = user_input_evidence
        self._terminal_evidence = terminal_evidence
        self._recovery_ownership = recovery_ownership
        self.terminal_finalization = terminal_finalization
        self._session_finalization = session_finalization
        self._runtime_hooks = runtime_hooks
        self._loop_policies = loop_policies
        self._effect_reconciliation_owner = ToolEffectReconciliationOwner()

    def _build_human_review(
        self,
        session: Session,
        checkpoint: dict[str, Any] | None,
        context: HumanReviewContext,
    ) -> HumanReviewView:
        policy = self._human_review_policy
        if policy is None:
            raise HumanReviewDenied()
        approval = pending_approval_reader.pending_approval_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
        )
        if approval is not None:
            approval, pending_round = _pending_approval_and_round_for_atomic_claim(
                checkpoint,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                gating_tool_call_id=approval.tool_call_id,
                redactor=self._secret_redactor,
                runtime_session=session,
            )
            pending = approval
            kind = "tool_approval"
            interaction_id = approval.approval_id
            scope = approval.secret_resolution_scope
            content = [approval.model_dump(mode="json"), pending_round.model_dump(mode="json")]
            resolution_intent = pending_approval_reader.approval_resolution_intent_from_checkpoint(
                checkpoint, redactor=self._secret_redactor
            )
            question, options = None, ()
            expires_at = approval.expires_at
        else:
            pending, resolution_intent = user_input_lifecycle_authority_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                current_run_epoch=session.run_epoch,
                runtime_session=session,
            )
            if (
                pending is None
                or pending.session_id != session.id
                or pending.session_instance_id != session.instance_id
            ):
                raise HumanReviewConflict()
            kind = "user_input"
            interaction_id = pending.input_id
            scope = (
                "unknown"
                if pending.assistant_publication is None
                else pending.assistant_publication.secret_resolution_scope
            )
            content = pending.model_dump(mode="json")
            question, options = pending.question, tuple(pending.options)
            expires_at = None
        if resolution_intent is not None:
            # Publication progress can change the checkpoint after a decision
            # was accepted. Bind a fresh recovery view to that exact progress
            # and claim; it must never become a new grant to execute the round.
            content = [content, resolution_intent.model_dump(mode="json")]
        if len(pending.tool_calls) > 128:
            raise HumanReviewConflict()
        calls = tuple(
            HumanReviewCall(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                on_grant=(
                    "withheld"
                    if pending_approval_reader.effective_tool_policy_evidence(call)
                    is not ToolPolicyEvidence.AUTHORITATIVE
                    else "denied"
                    if call.policy_decision == ToolPolicyDecision.DENY.value
                    else "eligible"
                    if call.policy_decision
                    in {ToolPolicyDecision.ALLOW.value, ToolPolicyDecision.REQUIRE_APPROVAL.value}
                    else "withheld"
                ),
            )
            for call in pending.tool_calls
        )
        executable = resolution_intent is None and all(
            pending_approval_reader.effective_tool_policy_evidence(call)
            is not ToolPolicyEvidence.AMBIGUOUS
            for call in pending.tool_calls
        )
        source = HumanReviewSource(
            kind=kind,
            interaction_id=interaction_id,
            tool_round_id=pending.tool_round_id,
            tool_call_id=pending.tool_call_id,
            secret_resolution_scope=scope,
            calls=calls,
            arguments_by_call={
                call.tool_call_id: copy_json_value(call.arguments, "review arguments")
                for call in pending.tool_calls
            },
            question=question,
            options=options,
            expires_at=expires_at,
            executable=executable,
        )
        return build_review(
            policy=policy,
            context=context,
            source=source,
            session_id=session.id,
            session_instance_id=session.instance_id,
            authoritative_content=content,
            redactor=self._secret_redactor,
            now=self._clock(),
        )

    async def inspect_human_review(
        self,
        session_id: str,
        context: HumanReviewContext,
    ) -> HumanReviewView:
        session = await self._session_store.load(session_id)
        if session is None:
            raise HumanReviewDenied()
        policy = require_review_authority(
            self._human_review_policy,
            context,
            session_id=session.id,
            session_metadata=session.metadata,
            action="inspect",
        )
        unavailable = HumanReviewView(
            status="unavailable",
            session_id=session_id,
            guidance="No current review is available; refresh pending interactions.",
        )
        try:
            checkpoint = await self._session_store.load_checkpoint(session_id)
            if session.status in {
                SessionStatus.COMPLETED,
                SessionStatus.INTERRUPTING,
            }:
                return unavailable
            approval_intent = pending_approval_reader.approval_resolution_intent_from_checkpoint(
                checkpoint, redactor=self._secret_redactor
            )
            _pending, intent = user_input_lifecycle_authority_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                current_run_epoch=session.run_epoch,
                runtime_session=session,
            )
            if (
                session.status is SessionStatus.FAILED
                and approval_intent is None
                and intent is None
            ):
                return unavailable
            result = self._build_human_review(session, checkpoint, context)
            # Never return a view read across incarnation, claim or checkpoint changes.
            current = await self._session_store.load(session_id)
            if (
                current != session
                or await self._session_store.load_checkpoint(session_id) != checkpoint
            ):
                return unavailable
            policy.audit(action="inspect", status=result.status)
            return result
        except Exception:
            return unavailable

    async def require_human_review_resolution_authority(
        self,
        session_id: str,
        reference: HumanReviewReference | None,
    ) -> None:
        if self._human_review_policy is None and reference is None:
            return
        if reference is None:
            raise HumanReviewDenied()
        session = await self._session_store.load(session_id)
        if session is None:
            raise HumanReviewDenied()
        require_review_authority(
            self._human_review_policy,
            reference.context,
            session_id=session_id,
            session_metadata=session.metadata,
            action="decide",
        )

    def _require_human_review_decision(
        self,
        reference: HumanReviewReference | None,
        session: Session,
        checkpoint: dict[str, Any] | None,
        *,
        denying: bool,
        accepted_request: bool = False,
    ) -> None:
        if reference is None and self._human_review_policy is None:
            return
        if reference is None:
            raise HumanReviewDenied()
        policy = require_review_authority(
            self._human_review_policy,
            reference.context,
            session_id=session.id,
            session_metadata=session.metadata,
            action="decide",
        )
        # Callers may reuse an accepted decision only after matching its exact
        # durable request digest inside this same atomic claim. Its original
        # view is not a new decision against mutable publication progress.
        if not accepted_request:
            current = self._build_human_review(session, checkpoint, reference.context)
            require_current_review(reference, current, denying=denying)
        try:
            policy.audit(action="decide", status="denied" if denying else "accepted")
        except Exception:
            raise HumanReviewDenied() from None

    async def _has_exact_persisted_user_input_manual_recovery(
        self,
        *,
        session: Session,
        pending: PendingUserInput,
        resolution_intent: UserInputResolutionIntent,
    ) -> bool:
        """Prove that a manual-recovery claim produced its exact terminal evidence."""

        if resolution_intent.resolution_stage != "manual-recovery":
            return False
        require_resolution_intent_matches_pending(resolution_intent, pending=pending)
        round_identity = ToolRoundIdentity(
            tool_round_id=pending.tool_round_id,
            model_step_id=pending.model_step_id,
            model_attempt_id=pending.model_attempt_id,
        )
        pending_call_ids = {call.tool_call_id for call in pending.tool_calls}
        matches: list[Event] = []
        for event in await self._session_store.load_events(session.id):
            tool_call_id = event.payload.get("tool_call_id")
            if (
                event.type not in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
                or event.session_id != session.id
                or event.payload.get("manual_recovery") is not True
                or event.payload.get("input_id") != pending.input_id
                or not round_identity.matches_payload(event.payload)
                or type(tool_call_id) is not str
                or tool_call_id not in pending_call_ids
                or event.payload.get("idempotency_key")
                != tool_execution.tool_idempotency_key(
                    session_id=session.id,
                    tool_round_id=pending.tool_round_id,
                    tool_call_id=tool_call_id,
                    pause_id=pending.input_id,
                )
                or event.payload.get("execution_profile_fingerprint")
                != pending.execution_profile_fingerprint
                or event.payload.get("resolution_request_digest")
                != resolution_intent.resolution_request_digest
            ):
                continue
            matches.append(event)
        if len(matches) > 1:
            raise SessionRuntimePublicationConflict(
                "User-input manual recovery has duplicate exact terminal evidence."
            )
        return len(matches) == 1

    async def _pause_gate_on_child(
        self,
        failure,
        *,
        session,
        registered_agent,
        registered_environment,
        execution_profile,
        invocation_context,
    ):
        session = await self._session_store.update_status(session.id, SessionStatus.INTERRUPTED)
        async for event in self._terminal_event_publication.publish_recovered(
            RecoveryTerminalEventRequest(
                event=event_with_foreground_child_wait_authority(
                    Event(
                        type=EventType.SESSION_INTERRUPTED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=_environment_name(registered_environment),
                        payload=failure.interruption_evidence(),
                    ),
                    failure.wait,
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            )
        ):
            yield event

    async def resume_foreground_gate(
        self,
        parent: Session,
        terminal: ForegroundChildTerminal,
        *,
        before_mutation: RecoveryMutationHook,
    ) -> bool:
        from cayu.runtime._foreground_gate_continuation import load_gate_replay

        replay = await load_gate_replay(self._session_store, parent=parent, terminal=terminal)
        if replay is None:
            return False
        policies = self._foreground_gate_policy_owner.resolve(replay.record)
        await before_mutation()
        if replay.record["kind"] == "approval":
            request = ToolApprovalRequest.model_validate(replay.record["request"]).model_copy(
                update={"loop_policies": policies}
            )
            checkpoint = await self._session_store.load_checkpoint(parent.id)
            approval = pending_approval_reader.pending_approval_from_checkpoint(checkpoint)
            if approval is None:
                raise SessionRunFenced("Foreground approval disappeared before continuation.")
            stream = self._resolve_tool_approval_owned(
                request,
                task_id=approval.task_id,
                before_mutation=before_mutation,
                foreground_gate=replay,
            )
        else:
            stream = self._resolve_user_input_owned(
                UserInputResponse.model_validate(replay.record["request"]).model_copy(
                    update={"loop_policies": policies}
                ),
                before_mutation=before_mutation,
                foreground_gate=replay,
            )
        with self._session_control.active_control_ownership(parent.id):
            async with _close_delegated_event_stream(stream) as owned:
                async for _ in owned:
                    pass
        return True

    async def _restore_closed_foreground_gate_policies(
        self,
        session: Session,
        request: ToolApprovalRequest | UserInputResponse | UserInputRecoveryRequest,
        receipt: RuntimePublicationReceipt,
    ) -> None:
        """Restore executable inputs, not execution authority, on exact receipt replay."""
        from cayu.runtime._foreground_child_continuation import (
            load_attached_foreground_continuation,
        )
        from cayu.runtime._foreground_gate_continuation import validate_gate_restoration_request

        if not request.loop_policies:
            return
        attached = await load_attached_foreground_continuation(session, store=self._session_store)
        if attached is None or attached.publication_id != receipt.publication_id:
            return
        effect = attached.terminal.wait.parent_effect

        async def interaction_finished() -> bool:
            return bool(
                await self._session_store.query_events(
                    EventQuery(
                        session_id=session.id,
                        interaction_id=effect.interaction_id,
                        event_types=tuple(INTERACTION_TERMINAL_EVENT_TYPES),
                        limit=1,
                    )
                )
            )

        # Completed/stopped work keeps its ordinary receipt-only replay contract.
        if await interaction_finished():
            return
        record = await self._foreground_gate_policy_owner.attached_record(
            self._session_store, parent=session, continuation=attached
        )
        validate_gate_restoration_request(record, request, redactor=self._secret_redactor)
        checkpoint = await self._session_store.load_checkpoint(session.id)
        semantics = attached.request
        snapshot = await self._execution_profile_continuation.validate(
            session=session,
            checkpoint=checkpoint,
            registered_agent=self._resolve_registered_agent(session.agent_name),
            registered_provider=self._resolve_registered_provider(session.provider_name),
            request_loop_policies=request.loop_policies,
            budget_policy=copy_budget_policy(self._resolve_budget_policy()),
            request_budget_limits=semantics.budget_limits,
            structured_output=semantics.structured_output,
            thinking=semantics.thinking,
            max_steps=semantics.max_steps,
            limits=semantics.limits,
            retry_policy=semantics.retry_policy,
            invocation_semantics_available=True,
            record_rejection=False,
        )
        if (
            snapshot.interaction_id != effect.interaction_id
            or snapshot.profile.fingerprint != effect.execution_profile_fingerprint
        ):
            raise SessionRunFenced("Policy restoration changed the original invocation authority.")
        current = await self._session_store.load(session.id)
        if (
            current is None
            or current.instance_id != session.instance_id
            or current.run_epoch != session.run_epoch
            or await load_attached_foreground_continuation(current, store=self._session_store)
            != attached
        ):
            raise SessionRunFenced("Attached continuation changed during policy restoration.")
        if not await interaction_finished():
            self._foreground_gate_policy_owner.retain(record, request.loop_policies)

    async def _prepare_gate_invocation(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        semantics: _RecoveryInvocationSemantics,
        structured_output: StructuredOutputSpec | None,
        request_loop_policies: tuple[LoopPolicy, ...],
        secret_resolution_scope: invocation_secrets.SecretResolutionScope,
    ) -> tuple[ActiveInvocationExecutionProfile, InvocationContext]:
        """Authenticate frozen runtime collaborators before a typed gate claim.

        Keep the validated checkpoint profile separate from the context's rebound
        profile. Callers retain their exact claim, receipt and resolution checks.
        """
        registered_agent = self._resolve_registered_agent(session.agent_name)
        registered_provider = self._resolve_registered_provider(session.provider_name)
        budget_policy_snapshot = copy_budget_policy(self._resolve_budget_policy())
        execution_profile_snapshot = await self._execution_profile_continuation.validate(
            session=session,
            checkpoint=checkpoint,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            request_loop_policies=request_loop_policies,
            budget_policy=budget_policy_snapshot,
            request_budget_limits=semantics.budget_limits,
            structured_output=semantics.structured_output,
            thinking=semantics.thinking,
            max_steps=semantics.max_steps,
            limits=semantics.limits,
            retry_policy=semantics.retry_policy,
            invocation_semantics_available=True,
        )
        _require_native_structured_output_support(
            structured_output, registered_provider=registered_provider
        )
        registered_environment = self._resolve_registered_environment(session.environment_name)
        invocation_secrets.require_continuation_secret_resolution_compatibility(
            secret_resolution_scope, registered_environment
        )
        invocation_context = reconstruct_invocation_context(
            runtime_hooks=self._runtime_hooks,
            loop_policies=self._loop_policies,
            session=session,
            execution_profile_snapshot=execution_profile_snapshot,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            budget_policy=budget_policy_snapshot,
            request_loop_policies=request_loop_policies,
        )
        return execution_profile_snapshot, invocation_context

    async def resolve_user_input(
        self,
        response: UserInputResponse | UserInputRecoveryRequest,
        *,
        before_mutation: RecoveryMutationHook | None = None,
        after_admission: RecoveryMutationHook | None = None,
        effect_reconciliation: ToolEffectReconciliationRequest | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        with self._session_control.active_control_ownership(response.session_id):
            async with _close_delegated_event_stream(
                self._resolve_user_input_owned(
                    response,
                    participant_context=participant_context,
                    before_mutation=before_mutation,
                    after_admission=after_admission,
                    effect_reconciliation=effect_reconciliation,
                )
            ) as stream:
                async for event in stream:
                    yield event

    async def _resolve_user_input_owned(
        self,
        response: UserInputResponse | UserInputRecoveryRequest,
        *,
        before_mutation: RecoveryMutationHook | None = None,
        after_admission: RecoveryMutationHook | None = None,
        effect_reconciliation: ToolEffectReconciliationRequest | None = None,
        foreground_gate: GateReplay | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Resume a session paused by ``ask_user`` with the user's answer.

        The answer becomes the ``ask_user`` tool result; any other tool calls in the same
        round (none ran before the pause) execute now, and the session continues.
        """
        loaded_session = await self._session_store.load(response.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {response.session_id}")

        answer_request_digest = (
            user_input_answer_request_digest(response)
            if foreground_gate is None
            else foreground_gate.answer_digest
        )
        resolution_request_digest = (
            user_input_resolution_request_digest(response)
            if foreground_gate is None
            else foreground_gate.resolution_digest
        )
        resolution_stage = (
            "answer" if foreground_gate is None else foreground_gate.input_resolution_stage
        )
        checkpoint = await self._session_store.load_checkpoint(loaded_session.id)
        close_receipt = await self._session_store.load_runtime_publication_receipt(
            loaded_session.id,
            f"user-input-close:{response.input_id}",
        )
        if close_receipt is not None:
            if (
                await self._user_input_evidence.classify_pause(
                    session=loaded_session,
                    checkpoint=checkpoint,
                    input_id=response.input_id,
                )
                is not UserInputPauseState.ANSWERED
            ):
                raise SessionRuntimePublicationConflict(
                    "User-input closure conflicts with durable lifecycle state."
                )
            closure_event = await self._user_input_evidence.exact_user_input_close_event(
                session=loaded_session,
                input_id=response.input_id,
                receipt=close_receipt,
                expected_resolution_request_digest=resolution_request_digest,
            )
            if before_mutation is not None:
                await before_mutation()
            await self._restore_closed_foreground_gate_policies(
                loaded_session, response, close_receipt
            )
            yield closure_event
            return

        pending, candidate_intent = user_input_lifecycle_authority_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            current_run_epoch=loaded_session.run_epoch,
            runtime_session=loaded_session,
        )
        if pending is None:
            pause_state = await self._user_input_evidence.classify_pause(
                session=loaded_session,
                checkpoint=checkpoint,
                input_id=response.input_id,
            )
            if pause_state is UserInputPauseState.SUPERSEDED:
                raise SessionRuntimePublicationConflict(
                    "User input was superseded by an external interruption."
                )
            if pause_state is UserInputPauseState.ANSWERED:
                raise SessionRuntimePublicationConflict(
                    "User input is answered but its exact closure cannot be replayed."
                )
            raise RuntimeError("Session has no pending user input.")
        if pending.input_id != response.input_id:
            raise ValueError(f"User input id does not match pending input: {response.input_id}")
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if (
            pending.session_id != loaded_session.id
            or pending.session_instance_id != loaded_session.instance_id
            or (
                active_profile is not None
                and pending.execution_profile_fingerprint != active_profile.profile.fingerprint
            )
        ):
            raise SessionRuntimePublicationConflict(
                "Pending user input has conflicting durable invocation authority."
            )
        await self._user_input_evidence.require_exact_user_input_open_receipt(
            session=loaded_session,
            pending=pending,
        )
        resume_after_manual_recovery = False
        if effect_reconciliation is not None and candidate_intent is None:
            raise ToolEffectConflict("Receipt recovery lost the original user-input answer intent.")
        if candidate_intent is not None:
            if candidate_intent.answer_request_digest != answer_request_digest:
                raise SessionRuntimePublicationConflict(
                    "User input was already claimed with a different resolution request."
                )
            if candidate_intent.resolution_stage == "answer":
                if candidate_intent.resolution_request_digest != resolution_request_digest:
                    raise SessionRuntimePublicationConflict(
                        "User input was already claimed with a different resolution request."
                    )
            else:
                resume_after_manual_recovery = (
                    loaded_session.status is SessionStatus.INTERRUPTED
                    and await self._has_exact_persisted_user_input_manual_recovery(
                        session=loaded_session,
                        pending=pending,
                        resolution_intent=candidate_intent,
                    )
                )
                if not resume_after_manual_recovery:
                    raise SessionRuntimePublicationConflict(
                        "User input was already claimed with a different resolution request."
                    )
        # The output-schema contract is fixed by the paused run's provider history; a resolver
        # cannot swap it (a spec matching or absent is fine; a differing one is rejected). Checked
        # before the status transition so it surfaces to the caller rather than being caught by the
        # resume's failure handler. All provider-dispatch semantics are then
        # compared with the frozen invocation profile before status changes.
        effective_structured_output = _effective_user_input_structured_output(
            structured_output=response.structured_output,
            pending=pending,
        )
        invocation_semantics = _effective_user_input_invocation_semantics(
            response=response,
            pending=pending,
            structured_output=effective_structured_output,
            effective_retry_policy=self._effective_retry_policy,
        )
        if effect_reconciliation is not None and (
            effect_reconciliation.max_steps is not None
            and effect_reconciliation.max_steps != invocation_semantics.max_steps
        ):
            raise ToolEffectConflict("Receipt continuation conflicts with the frozen step limit.")
        require_secret_free_structured_output_spec(
            effective_structured_output,
            redactor=self._secret_redactor,
            field_name="UserInputResponse.structured_output",
        )

        execution_profile_snapshot, invocation_context = await self._prepare_gate_invocation(
            session=loaded_session,
            checkpoint=checkpoint,
            semantics=invocation_semantics,
            structured_output=effective_structured_output,
            request_loop_policies=response.loop_policies,
            secret_resolution_scope="unknown"
            if pending.assistant_publication is None
            else pending.assistant_publication.secret_resolution_scope,
        )
        registered_agent = invocation_context.registered_agent
        registered_provider = invocation_context.registered_provider
        registered_environment = invocation_context.registered_environment
        budget_policy_snapshot = invocation_context.budget_policy
        claimed_intent: UserInputResolutionIntent | None = None

        def claim_exact_user_input(
            current_session: Session,
            current_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            nonlocal claimed_intent
            current_pending, current_intent = user_input_lifecycle_authority_from_checkpoint(
                current_checkpoint,
                redactor=self._secret_redactor,
                current_run_epoch=current_session.run_epoch,
                runtime_session=current_session,
            )
            if current_pending != pending or current_intent != candidate_intent:
                raise SessionRuntimePublicationConflict(
                    "Pending user-input authority changed before answer claim."
                )
            if effect_reconciliation is not None:
                self._require_human_review_decision(
                    effect_reconciliation.review_reference,
                    current_session,
                    current_checkpoint,
                    denying=True,
                    accepted_request=False,
                )
            self._require_human_review_decision(
                getattr(response, "review_reference", None),
                current_session,
                current_checkpoint,
                denying=False,
                accepted_request=current_intent is not None,
            )
            claimed_checkpoint, claimed_intent = checkpoint_with_user_input_resolution_intent(
                current_checkpoint,
                pending=pending,
                answer_request_digest=answer_request_digest,
                resolution_stage=resolution_stage,
                resolution_request_digest=resolution_request_digest,
                claim_run_epoch=current_session.run_epoch + 1,
                pause_resolved_at=self._clock(),
                redactor=self._secret_redactor,
                runtime_session=current_session,
                allow_manual_recovery_to_answer=(
                    resume_after_manual_recovery and foreground_gate is None
                ),
            )
            if foreground_gate is not None:
                claimed_checkpoint = foreground_gate.select(current_session, claimed_checkpoint)
            return claimed_checkpoint

        async def admit_exact_user_input_execution(claimed_session: Session) -> bool:
            nonlocal claimed_intent
            if claimed_intent is None:
                raise RuntimeError("User-input answer claim completed without durable intent.")
            try:
                claimed_intent = await self._admit_user_input_resolution_execution(
                    session=claimed_session,
                    pending=pending,
                    resolution_intent=claimed_intent,
                )
            except SessionRuntimePublicationConflict:
                return False
            return True

        (
            session,
            resumed_event,
        ) = await self._recovery_admission.transition_recovery_session_to_running(
            loaded_session,
            checkpoint=checkpoint,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            checkpoint_transform=claim_exact_user_input,
            execution_profile_snapshot=execution_profile_snapshot,
            before_mutation=before_mutation,
            before_resume=admit_exact_user_input_execution,
            after_admission=after_admission,
            invocation_context=invocation_context,
        )
        if resumed_event is not None:
            yield resumed_event
        if claimed_intent is None:
            raise RuntimeError("User-input answer claim completed without durable intent.")
        invocation_context = invocation_context.with_rebound_session(
            session,
            active_profile=execution_profile_snapshot.model_copy(
                update={"run_epoch": session.run_epoch}
            ),
        )

        continuation_stream = self.continue_user_input_resolution(
            response=response,
            participant_context=participant_context,
            session=session,
            pending=pending,
            resolution_intent=claimed_intent,
            resolution_stage=resolution_stage,
            closure_request_digest=resolution_request_digest,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            execution_profile_snapshot=execution_profile_snapshot,
            budget_policy=budget_policy_snapshot,
            invocation_context=invocation_context,
            effect_reconciliation=effect_reconciliation,
            foreground_gate=foreground_gate,
        )
        authoritative_failure: BaseException | None = None
        abandoned = False
        try:
            async for event in continuation_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            abandoned = _recovery_abandonment_signal(exc) is not None
            raise
        finally:
            await self._recovery_admission.cleanup_entrypoint_handoff(
                stream=continuation_stream,
                session_id=session.id,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=abandoned,
                release_run_fence=True,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=invocation_context,
            )

    async def recover_user_input_request(
        self,
        request: UserInputRecoveryRequest,
        *,
        before_mutation: RecoveryMutationHook | None = None,
        after_admission: RecoveryMutationHook | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Recover a user-input round stuck on `manual_recovery_required`.

        A tool in the paused round started on a prior resume but recorded no terminal event
        (a crash mid-tool), so it cannot be re-run automatically. The caller supplies the
        externally verified outcome for that `tool_call_id`; Cayu persists it as the tool's
        terminal result and continues the round (re-supplying `answer` in case the `ask_user`
        result was not recorded before the crash). Cayu does not infer the outcome itself.
        """
        loaded_session = await self._session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")

        answer_request_digest = user_input_answer_request_digest(request)
        resolution_request_digest = user_input_resolution_request_digest(request)
        checkpoint = await self._session_store.load_checkpoint(loaded_session.id)
        close_receipt = await self._session_store.load_runtime_publication_receipt(
            loaded_session.id,
            f"user-input-close:{request.input_id}",
        )
        if close_receipt is not None:
            if (
                await self._user_input_evidence.classify_pause(
                    session=loaded_session,
                    checkpoint=checkpoint,
                    input_id=request.input_id,
                )
                is not UserInputPauseState.ANSWERED
            ):
                raise SessionRuntimePublicationConflict(
                    "User-input closure conflicts with durable lifecycle state."
                )
            closure_event = await self._user_input_evidence.exact_user_input_close_event(
                session=loaded_session,
                input_id=request.input_id,
                receipt=close_receipt,
                expected_resolution_request_digest=resolution_request_digest,
            )
            if before_mutation is not None:
                await before_mutation()
            await self._restore_closed_foreground_gate_policies(
                loaded_session, request, close_receipt
            )
            yield closure_event
            return

        pending, candidate_intent = user_input_lifecycle_authority_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            current_run_epoch=loaded_session.run_epoch,
            runtime_session=loaded_session,
        )
        if pending is None:
            pause_state = await self._user_input_evidence.classify_pause(
                session=loaded_session,
                checkpoint=checkpoint,
                input_id=request.input_id,
            )
            if pause_state is UserInputPauseState.SUPERSEDED:
                raise SessionRuntimePublicationConflict(
                    "User input was superseded by an external interruption."
                )
            if pause_state is UserInputPauseState.ANSWERED:
                raise SessionRuntimePublicationConflict(
                    "User input is answered but its exact closure cannot be replayed."
                )
            raise SessionRuntimePublicationConflict(
                "User-input lifecycle is ambiguous; exact recovery authority is required."
            )
        if pending.input_id != request.input_id:
            raise ValueError(f"User input id does not match pending input: {request.input_id}")
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if (
            pending.session_id != loaded_session.id
            or pending.session_instance_id != loaded_session.instance_id
            or (
                active_profile is not None
                and pending.execution_profile_fingerprint != active_profile.profile.fingerprint
            )
        ):
            raise SessionRuntimePublicationConflict(
                "Pending user input has conflicting durable invocation authority."
            )
        await self._user_input_evidence.require_exact_user_input_open_receipt(
            session=loaded_session,
            pending=pending,
        )
        if candidate_intent is not None and (
            candidate_intent.answer_request_digest != answer_request_digest
            or (
                candidate_intent.resolution_stage == "manual-recovery"
                and candidate_intent.resolution_request_digest != resolution_request_digest
            )
            or (
                candidate_intent.resolution_stage == "answer"
                and loaded_session.status is not SessionStatus.INTERRUPTED
            )
        ):
            raise SessionRuntimePublicationConflict(
                "User input was already claimed with a different resolution request."
            )
        effective_structured_output = _effective_user_input_structured_output(
            structured_output=request.structured_output,
            pending=pending,
        )
        invocation_semantics = _effective_user_input_invocation_semantics(
            response=request,
            pending=pending,
            structured_output=effective_structured_output,
            effective_retry_policy=self._effective_retry_policy,
        )
        require_secret_free_structured_output_spec(
            effective_structured_output,
            redactor=self._secret_redactor,
            field_name="UserInputRecoveryRequest.structured_output",
        )

        pending_tool_call = approval_support.round_tool_call_for_recovery(
            pending_calls=pending.tool_calls,
            tool_call_id=request.tool_call_id,
        )
        approval_support.validate_round_recovery_target(
            events=await self._session_store.load_events(loaded_session.id),
            pending_calls=pending.tool_calls,
            tool_call_id=request.tool_call_id,
            input_id=pending.input_id,
            tool_round_identity=ToolRoundIdentity(
                tool_round_id=pending.tool_round_id,
                model_step_id=pending.model_step_id,
                model_attempt_id=pending.model_attempt_id,
            ),
        )
        execution_profile_snapshot, invocation_context = await self._prepare_gate_invocation(
            session=loaded_session,
            checkpoint=checkpoint,
            semantics=invocation_semantics,
            structured_output=effective_structured_output,
            request_loop_policies=request.loop_policies,
            secret_resolution_scope="unknown"
            if pending.assistant_publication is None
            else pending.assistant_publication.secret_resolution_scope,
        )
        registered_agent = invocation_context.registered_agent
        registered_provider = invocation_context.registered_provider
        registered_environment = invocation_context.registered_environment
        budget_policy_snapshot = invocation_context.budget_policy
        claimed_intent: UserInputResolutionIntent | None = None

        def claim_exact_user_input_recovery(
            current_session: Session,
            current_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            nonlocal claimed_intent
            current_pending, current_intent = user_input_lifecycle_authority_from_checkpoint(
                current_checkpoint,
                redactor=self._secret_redactor,
                current_run_epoch=current_session.run_epoch,
                runtime_session=current_session,
            )
            if current_pending != pending or current_intent != candidate_intent:
                raise SessionRuntimePublicationConflict(
                    "Pending user-input authority changed before recovery claim."
                )
            self._require_human_review_decision(
                request.review_reference,
                current_session,
                current_checkpoint,
                denying=current_intent is not None,
                accepted_request=(
                    current_intent is not None
                    and current_intent.resolution_stage == "manual-recovery"
                    and current_intent.resolution_request_digest == resolution_request_digest
                ),
            )
            claimed_checkpoint, claimed_intent = checkpoint_with_user_input_resolution_intent(
                current_checkpoint,
                pending=pending,
                answer_request_digest=answer_request_digest,
                resolution_stage="manual-recovery",
                resolution_request_digest=resolution_request_digest,
                claim_run_epoch=current_session.run_epoch + 1,
                pause_resolved_at=self._clock(),
                redactor=self._secret_redactor,
                runtime_session=current_session,
                allow_answer_to_manual_recovery=(
                    current_session.status is SessionStatus.INTERRUPTED
                ),
            )
            return claimed_checkpoint

        async def admit_exact_user_input_recovery_execution(
            claimed_session: Session,
        ) -> bool:
            nonlocal claimed_intent
            if claimed_intent is None:
                raise RuntimeError("User-input recovery claim completed without durable intent.")
            try:
                claimed_intent = await self._admit_user_input_resolution_execution(
                    session=claimed_session,
                    pending=pending,
                    resolution_intent=claimed_intent,
                )
            except SessionRuntimePublicationConflict:
                return False
            return True

        (
            session,
            resumed_event,
        ) = await self._recovery_admission.transition_recovery_session_to_running(
            loaded_session,
            checkpoint=checkpoint,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            checkpoint_transform=claim_exact_user_input_recovery,
            execution_profile_snapshot=execution_profile_snapshot,
            before_mutation=before_mutation,
            before_resume=admit_exact_user_input_recovery_execution,
            after_admission=after_admission,
            invocation_context=invocation_context,
        )
        if resumed_event is not None:
            yield resumed_event
        if claimed_intent is None:
            raise RuntimeError("User-input recovery claim completed without durable intent.")
        invocation_context = invocation_context.with_rebound_session(
            session,
            active_profile=execution_profile_snapshot.model_copy(
                update={"run_epoch": session.run_epoch}
            ),
        )
        recovery_stream = self.recover_user_input(
            request=request,
            participant_context=participant_context,
            loaded_session=loaded_session,
            session=session,
            pending=pending,
            resolution_intent=claimed_intent,
            closure_request_digest=resolution_request_digest,
            pending_tool_call=pending_tool_call,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            execution_profile_snapshot=execution_profile_snapshot,
            budget_policy=budget_policy_snapshot,
            invocation_context=invocation_context,
        )
        authoritative_failure: BaseException | None = None
        try:
            async for event in recovery_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            await self._recovery_admission.cleanup_entrypoint_handoff(
                stream=recovery_stream,
                session_id=session.id,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=False,
                release_run_fence=False,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=invocation_context,
            )

    async def resolve_tool_approval(
        self,
        request: ToolApprovalRequest,
        *,
        task_id: str | None = None,
        before_mutation: RecoveryMutationHook | None = None,
        after_admission: RecoveryMutationHook | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        with self._session_control.active_control_ownership(request.session_id):
            async with _close_delegated_event_stream(
                self._resolve_tool_approval_owned(
                    request,
                    participant_context=participant_context,
                    task_id=task_id,
                    before_mutation=before_mutation,
                    after_admission=after_admission,
                )
            ) as stream:
                async for event in stream:
                    yield event

    async def _resolve_tool_approval_owned(
        self,
        request: ToolApprovalRequest,
        *,
        task_id: str | None = None,
        before_mutation: RecoveryMutationHook | None = None,
        after_admission: RecoveryMutationHook | None = None,
        foreground_gate: GateReplay | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        loaded_session = await self._session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")

        close_receipt = await self._session_store.load_runtime_publication_receipt(
            loaded_session.id,
            f"approval-close:{request.approval_id}",
        )
        if close_receipt is not None:
            expected_identity = {
                "approval_id": request.approval_id,
                "tool_call_id": request.tool_call_id,
                "tool_round_id": request.tool_round_id,
                "requested_decision": request.decision.value,
                "resolution_request_digest": (
                    approval_support.approval_resolution_request_digest(request)
                ),
            }
            if close_receipt.kind != "approval-close" or any(
                close_receipt.intent.get(key) != value for key, value in expected_identity.items()
            ):
                raise RuntimeError(
                    "Tool approval was already closed with a conflicting identity or decision."
                )
            if len(close_receipt.appended_event_ids) != 1:
                raise SessionRuntimePublicationConflict(
                    "Tool approval closure receipt has invalid event evidence."
                )
            closure_records = await self._session_store.query_events(
                EventQuery(
                    session_id=loaded_session.id,
                    event_id=close_receipt.appended_event_ids[0],
                    limit=2,
                )
            )
            if (
                len(closure_records) != 1
                or closure_records[0].event.id != close_receipt.appended_event_ids[0]
            ):
                raise SessionRuntimePublicationConflict(
                    "Tool approval closure event is missing from durable history."
                )
            if before_mutation is not None:
                await before_mutation()
            closure_event = closure_records[0].event
            identity = ApprovalTaskFailureIdentity(
                approval_id=request.approval_id,
                tool_round_id=request.tool_round_id,
                tool_call_id=request.tool_call_id,
                resolution_request_digest=(
                    approval_support.approval_resolution_request_digest(request)
                ),
            )
            current_session = await self._recovery_ownership.require_session(loaded_session.id)
            task_failure_durable = bool(
                task_id is not None
                and (
                    (
                        request.task_worker_id is not None
                        and await self._approval_task_failure_receipt_is_durable(
                            task_id=task_id,
                            task_worker_id=request.task_worker_id,
                            task_handoff_id=request.task_handoff_id,
                            session=current_session,
                            identity=identity,
                        )
                    )
                    or (
                        request.task_worker_id is None
                        and await self._direct_approval_task_failure_is_durable(
                            task_id=task_id,
                            session=current_session,
                            identity=identity,
                        )
                    )
                )
            )
            if task_failure_durable:
                yield closure_event
                registered_agent = self._resolve_registered_agent(current_session.agent_name)
                registered_environment = self._resolve_registered_environment(
                    current_session.environment_name
                )
                registered_provider = self._resolve_registered_provider(
                    current_session.provider_name
                )
                checkpoint = await self._session_store.load_checkpoint(current_session.id)
                budget_policy_snapshot = copy_budget_policy(self._resolve_budget_policy())
                execution_profile_snapshot = await self._execution_profile_continuation.validate(
                    session=current_session,
                    checkpoint=checkpoint,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    budget_policy=budget_policy_snapshot,
                    require_open_interaction=False,
                    record_rejection=False,
                )
                invocation_context = reconstruct_invocation_context(
                    runtime_hooks=self._runtime_hooks,
                    loop_policies=self._loop_policies,
                    session=current_session,
                    execution_profile_snapshot=execution_profile_snapshot,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    budget_policy=budget_policy_snapshot,
                )
                async for event in self._finish_closed_approval_failure(
                    request=request,
                    task_id=task_id,
                    session=current_session,
                    closure_event=closure_event,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=invocation_context.profile,
                    invocation_context=invocation_context,
                ):
                    yield event
                return
            await self._restore_closed_foreground_gate_policies(
                current_session, request, close_receipt
            )
            await self._deferred_input.materialize_for_receipt(close_receipt)
            yield closure_event
            return

        checkpoint = await self._session_store.load_checkpoint(loaded_session.id)
        candidate_approval, candidate_round = _pending_approval_and_round_for_atomic_claim(
            checkpoint,
            approval_id=request.approval_id,
            tool_round_id=request.tool_round_id,
            gating_tool_call_id=request.tool_call_id,
            redactor=self._secret_redactor,
            runtime_session=loaded_session,
        )
        candidate_intent = pending_approval_reader.approval_resolution_intent_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
        )
        resolution_request_digest = (
            approval_support.approval_resolution_request_digest(request)
            if foreground_gate is None
            else foreground_gate.resolution_digest
        )
        can_create_resolution_intent = True
        try:
            candidate_events = await self._session_store.load_events(loaded_session.id)
            candidate_history = approval_support.approval_resolution_history(
                events=candidate_events,
                approval=candidate_approval,
            )
            candidate_decision = request.decision
            if (
                approval_support.pending_approval_expired(
                    candidate_approval,
                    self._clock(),
                )
                and not candidate_history.has_granted_activity
            ):
                candidate_decision = ToolApprovalDecision.DENY
            approval_support.validate_retry_decision(
                history=candidate_history,
                approval=candidate_approval,
                decision=candidate_decision,
            )
            candidate_outcomes = approval_support.recorded_tool_outcomes(
                events=candidate_events,
                approval=candidate_approval,
            )
            if candidate_history.has_resolution_activity or candidate_outcomes:
                # Durable resolution activity without an existing request digest
                # cannot prove which audit-bearing request authorized it. Never
                # infer that identity from redacted or bounded event payloads.
                can_create_resolution_intent = False
        except Exception:
            # The continuation repeats this validation inside its established
            # interruption-event boundary. Do not let a rejected legacy retry
            # become durable authority before that happens.
            can_create_resolution_intent = False
        effective_structured_output = _effective_approval_structured_output(
            structured_output=request.structured_output,
            pending_approval=candidate_approval,
        )
        invocation_semantics = _effective_approval_invocation_semantics(
            request=request,
            pending_approval=candidate_approval,
            structured_output=effective_structured_output,
            effective_retry_policy=self._effective_retry_policy,
        )
        require_secret_free_structured_output_spec(
            effective_structured_output,
            redactor=self._secret_redactor,
            field_name="ToolApprovalRequest.structured_output",
        )
        execution_profile_snapshot, invocation_context = await self._prepare_gate_invocation(
            session=loaded_session,
            checkpoint=checkpoint,
            semantics=invocation_semantics,
            structured_output=effective_structured_output,
            request_loop_policies=request.loop_policies,
            secret_resolution_scope=candidate_approval.secret_resolution_scope,
        )
        registered_agent = invocation_context.registered_agent
        registered_provider = invocation_context.registered_provider
        registered_environment = invocation_context.registered_environment
        budget_policy_snapshot = invocation_context.budget_policy
        pending_approval: PendingToolApproval | None = None
        pending_round: pending_rounds.PendingToolRound | None = None
        claimed_intent: pending_approval_reader.ApprovalResolutionIntent | None = None

        def claim_exact_approval(
            _current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            nonlocal claimed_intent, pending_approval, pending_round
            pending_approval, pending_round = _pending_approval_and_round_for_atomic_claim(
                checkpoint,
                approval_id=request.approval_id,
                tool_round_id=request.tool_round_id,
                gating_tool_call_id=request.tool_call_id,
                redactor=self._secret_redactor,
                runtime_session=_current_session,
            )
            if pending_approval != candidate_approval or pending_round != candidate_round:
                raise RuntimeError("Pending tool approval changed before it was claimed.")
            current_intent = pending_approval_reader.approval_resolution_intent_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
            )
            if current_intent != candidate_intent:
                raise RuntimeError(
                    "Approval resolution intent changed before the approval was claimed."
                )
            accepted_request = current_intent is not None and (
                current_intent.decision is request.decision
                and current_intent.resolution_request_digest == resolution_request_digest
                and (
                    self._human_review_policy is None
                    or current_intent.reviewed_approval_digest is not None
                )
            )
            if (
                current_intent is not None
                and not accepted_request
                and self._human_review_policy is not None
            ):
                raise HumanReviewConflict()
            self._require_human_review_decision(
                request.review_reference,
                _current_session,
                checkpoint,
                denying=request.decision is ToolApprovalDecision.DENY,
                accepted_request=accepted_request,
            )
            claimed_checkpoint = _checkpoint_with_legacy_approval_round(
                checkpoint,
                approval=pending_approval,
                redactor=self._secret_redactor,
                runtime_session=_current_session,
            )
            if current_intent is not None:
                claimed_intent = current_intent
                if foreground_gate is not None:
                    claimed_checkpoint = foreground_gate.select(
                        _current_session, claimed_checkpoint
                    )
                return claimed_checkpoint
            intent_decision = request.decision if can_create_resolution_intent else None
            if intent_decision is None:
                claimed_intent = None
                return claimed_checkpoint
            claimed_checkpoint = approval_support.checkpoint_with_approval_resolution_intent(
                claimed_checkpoint,
                approval=pending_approval,
                decision=intent_decision,
                pause_resolved_at=self._clock(),
                resolution_request_digest=resolution_request_digest,
                redactor=self._secret_redactor,
                reviewed_approval_digest=(
                    None
                    if request.review_reference is None
                    else runtime_publication_checkpoint_value_digest(
                        pending_approval.model_dump(mode="json")
                    )
                ),
                runtime_session=_current_session,
            )
            claimed_intent = pending_approval_reader.approval_resolution_intent_from_checkpoint(
                claimed_checkpoint,
                redactor=self._secret_redactor,
            )
            return claimed_checkpoint

        try:
            (
                session,
                resumed_event,
            ) = await self._recovery_admission.transition_recovery_session_to_running(
                loaded_session,
                checkpoint=checkpoint,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                checkpoint_transform=claim_exact_approval,
                execution_profile_snapshot=execution_profile_snapshot,
                before_mutation=before_mutation,
                after_admission=after_admission,
                invocation_context=invocation_context,
            )
        except (InvocationLifecycleCommandConflict, SessionRunFenced) as claim_conflict:
            # A competing rebind can advance durable invocation authority before
            # this request reaches its checkpoint transform. Revalidate the
            # externally supplied identity against the latest durable approval
            # before classifying an otherwise exact concurrent claim.
            latest_checkpoint = await self._session_store.load_checkpoint(loaded_session.id)
            _pending_approval_and_round_for_atomic_claim(
                latest_checkpoint,
                approval_id=request.approval_id,
                tool_round_id=request.tool_round_id,
                gating_tool_call_id=request.tool_call_id,
                redactor=self._secret_redactor,
                runtime_session=loaded_session,
            )
            raise SessionStatusConflict(
                "Tool approval was claimed by another invocation."
            ) from claim_conflict
        if pending_approval is None or pending_round is None:
            raise RuntimeError("Tool approval claim completed without approval state.")
        if task_id != pending_approval.task_id:
            raise RuntimeError("Tool approval changed its attached task identity.")
        if resumed_event is not None:
            yield resumed_event
        invocation_context = invocation_context.with_rebound_session(
            session,
            active_profile=execution_profile_snapshot.model_copy(
                update={"run_epoch": session.run_epoch}
            ),
        )
        continuation_stream = self.continue_tool_approval_resolution(
            request=request,
            participant_context=participant_context,
            session=session,
            pending_approval=pending_approval,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            execution_profile_snapshot=execution_profile_snapshot,
            budget_policy=budget_policy_snapshot,
            deferred_messages=pending_round.deferred_messages,
            claimed_resolution_intent=claimed_intent,
            invocation_context=invocation_context,
            foreground_gate=foreground_gate,
        )
        authoritative_failure: BaseException | None = None
        abandoned = False
        try:
            async for event in continuation_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            abandoned = _recovery_abandonment_signal(exc) is not None
            raise
        finally:
            await self._recovery_admission.cleanup_entrypoint_handoff(
                stream=continuation_stream,
                session_id=session.id,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=abandoned,
                release_run_fence=True,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=invocation_context,
            )

    async def resolve_provider_operation(
        self,
        request: ProviderOperationResolutionRequest,
        *,
        task_id: str | None = None,
        task_handoff_id: str | None = None,
        before_mutation: RecoveryMutationHook | None = None,
        after_admission: RecoveryMutationHook | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Accept one disposition and drive its durable effect to a recovery boundary."""

        if request.task_worker_id is None and task_handoff_id is not None:
            raise ValueError("Workerless provider resolution cannot carry a handoff identity.")
        if request.task_worker_id is not None and task_id is None:
            raise ValueError("Typed provider resolution requires an attached task identity.")

        request = prepare_provider_operation_resolution_request(
            request,
            redactor=self._secret_redactor,
        )

        async def prepare_resolution_mutation() -> None:
            if before_mutation is not None:
                await before_mutation()
            session = await self._session_store.load(request.session_id)
            stage = await self._session_store.load_model_completion_stage(
                request.session_id,
                request.stage_id,
            )
            if session is None or stage is None:
                raise ProviderOperationEvidenceError(
                    "Provider-operation resolution lost its model stage."
                )
            await (
                self._run_limit_controller.reconcile_borrowed_automatic_compaction_budget_authority(
                    session=session,
                    stage=stage,
                    allow_outcome_unknown=True,
                )
            )

        result = await resolve_provider_operation_stage(
            self._session_store,
            request,
            redactor=self._secret_redactor,
            before_resolution=prepare_resolution_mutation,
        )
        if not result.replayed:
            await self._event_writer.fan_out_persisted([result.event])
        yield result.event

        pending_resolution = await load_pending_provider_operation_disposition(
            self._session_store,
            request.session_id,
        )
        if pending_resolution is None:
            return
        pending, durable_result = pending_resolution
        if durable_result.record.request_digest != result.record.request_digest:
            raise ProviderOperationEvidenceError(
                "Pending provider-operation disposition changed after acceptance."
            )
        execution_started = await self._provider_operation_disposition_execution_started(
            pending=pending,
            result=durable_result,
        )
        if execution_started:
            for settlement_event in (
                await self._provider_disposition.settle_provider_operation_disposition_reservations(
                    pending=pending,
                    result=durable_result,
                )
            ):
                yield settlement_event
        if (
            pending.execution_claimed
            and request.task_worker_id is not None
            and (
                pending.execution_task_worker_id,
                pending.execution_task_handoff_id,
            )
            != (request.task_worker_id, task_handoff_id)
        ):
            if task_id is None:
                raise ProviderOperationEvidenceError(
                    "Provider-operation task continuation lost its attached task."
                )
            recovered = await self._incomplete_recovery.recover_incomplete_session_scoped(
                session=await self._recovery_ownership.require_session(pending.session_id),
                inactive_for_seconds=None,
                reason="elected attached-task provider continuation",
                metadata={},
                provider_disposition_task_id=task_id,
                provider_disposition_task_worker_id=request.task_worker_id,
                provider_disposition_task_handoff_id=task_handoff_id,
                provider_disposition_after_admission=after_admission,
                participant_context=participant_context,
            )
            for recovered_event in recovered.events:
                yield recovered_event
            return
        if execution_started:
            if pending.action is ProviderOperationResolutionAction.FAIL:
                # Failure continuation is deterministic after the interaction
                # transition: exact callers may race safely through terminal
                # event and hook reservation reconciliation.
                pass
            else:
                return
        disposition_stream = (
            self._provider_disposition.finish_pending_provider_operation_disposition(
                pending=pending,
                participant_context=participant_context,
                result=durable_result,
                task_id=task_id,
                task_worker_id=request.task_worker_id,
                task_handoff_id=task_handoff_id,
                after_admission=after_admission,
            )
        )
        try:
            try:
                async for event in disposition_stream:
                    yield event
            except ExceptionGroup as replay_failure:
                if _interaction_transition_replay_failures(
                    replay_failure
                ) is None or not await self._provider_operation_disposition_execution_started(
                    pending=pending,
                    result=durable_result,
                ):
                    raise
            except (SessionRunFenced, SessionStatusConflict):
                if not await self._provider_operation_disposition_execution_started(
                    pending=pending,
                    result=durable_result,
                ):
                    raise
        finally:
            await disposition_stream.aclose()

    async def _provider_operation_disposition_execution_started(
        self,
        *,
        pending: ProviderOperationPendingDisposition,
        result: ProviderOperationResolutionResult,
    ) -> bool:
        """Recognize exact durable progress owned by another disposition caller."""

        session = await self._session_store.load(pending.session_id)
        if session is None:
            raise KeyError(f"Session not found: {pending.session_id}")
        if pending.action is ProviderOperationResolutionAction.FALLBACK_RETRY:
            if session.status is not SessionStatus.RUNNING:
                return False
            checkpoint = await self._session_store.load_checkpoint(pending.session_id)
            active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            return bool(
                active_profile is not None
                and active_profile.session_id == pending.session_id
                and active_profile.interaction_id == result.event.interaction_id
                and active_profile.run_epoch == session.run_epoch
                and active_profile.profile.fingerprint == pending.execution_profile_fingerprint
            )

        if session.status is not SessionStatus.FAILED:
            return False
        interaction_event_id = provider_operation_resolution_outcome_event_id(
            result.record.resolution_id,
            "interaction_failed",
        )
        interaction_records = await self._session_store.query_events(
            EventQuery(
                session_id=pending.session_id,
                event_id=interaction_event_id,
                limit=2,
            )
        )
        if not interaction_records:
            return False
        if len(interaction_records) != 1:
            raise ProviderOperationEvidenceError(
                "Provider-operation failure has duplicate interaction evidence."
            )
        validate_provider_operation_resolution_outcome_event(
            interaction_records[0].event,
            resolution_event=result.event,
            outcome="interaction_failed",
            expected_execution_profile_fingerprint=pending.execution_profile_fingerprint,
        )
        return True

    async def inspect_tool_effect_target(
        self, *, session_id: str, tool_round_id: str, tool_call_id: str
    ) -> ToolEffectReconciliationTarget:
        """Read exact durable identity; this does not admit any recovery action."""
        session = await self._session_store.load(session_id)
        if session is None:
            raise ToolEffectConflict("The effect session was not found.")
        record = await ToolEffectStateOwner(self._session_store).resolve_call(
            session, tool_round_id=tool_round_id, tool_call_id=tool_call_id
        )
        if record is None:
            raise ToolEffectConflict("The durable external call was not found.")
        return ToolEffectReconciliationTarget(
            session_id=session.id,
            session_instance_id=session.instance_id,
            tool_round_id=record.intent.tool_round_id,
            tool_call_id=record.intent.tool_call_id,
            tool_name=record.intent.tool_name,
            idempotency_key=record.intent.idempotency_key,
            expected_run_epoch=session.run_epoch,
            expected_revision=record.revision,
        )

    async def recover_tool_approval_request(
        self,
        request: ToolApprovalRecoveryRequest | ToolEffectReconciliationRequest,
        *,
        before_mutation: RecoveryMutationHook | None = None,
        after_admission: RecoveryMutationHook | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        loaded_session = await self._session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")

        if type(request) is ToolEffectReconciliationRequest:
            effect = await ToolEffectStateOwner(self._session_store).resolve_call(
                loaded_session,
                tool_round_id=request.tool_round_id,
                tool_call_id=request.tool_call_id,
            )
            if effect is None or effect.intent.approval_id is None:
                raise ToolEffectConflict("Receipt recovery has no durable approval linkage.")
            approval_id = effect.intent.approval_id
            await self.require_human_review_resolution_authority(
                loaded_session.id, request.review_reference
            )
            replay = await self._preflight_tool_effect_reconciliation(
                session=loaded_session, request=request
            )
            if (
                isinstance(replay, _ToolEffectObservationReplay)
                and loaded_session.status is SessionStatus.INTERRUPTED
            ):
                yield copy_event(replay.event)
                assert replay.record.observation is not None
                if replay.record.observation.result.outcome == "conflict":
                    raise ToolEffectConflict("Application reconciliation rejected the receipt.")
                return
            if (
                isinstance(replay, _ReconciledToolEffectReplay)
                and replay.consumption_receipt is not None
            ):
                yield copy_event(replay.terminal_event)
                return
            if replay is None and request.expected_run_epoch != loaded_session.run_epoch:
                raise ToolEffectConflict("Receipt recovery has stale session authority.")
            request_loop_policies = ()
            requested_structured_output = None
        elif type(request) is ToolApprovalRecoveryRequest:
            approval_id = request.approval_id
            request_loop_policies = request.loop_policies
            requested_structured_output = request.structured_output
        else:
            raise TypeError("Approval recovery requires an exact recovery action.")
        checkpoint = await self._session_store.load_checkpoint(loaded_session.id)
        candidate_approval, candidate_round = _pending_approval_and_round_for_atomic_claim(
            checkpoint,
            approval_id=approval_id,
            tool_round_id=request.tool_round_id,
            recovery_tool_call_id=request.tool_call_id,
            redactor=self._secret_redactor,
            runtime_session=loaded_session,
        )
        candidate_intent = pending_approval_reader.approval_resolution_intent_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
        )
        effective_structured_output = _effective_approval_structured_output(
            structured_output=requested_structured_output,
            pending_approval=candidate_approval,
        )
        invocation_semantics = _effective_approval_invocation_semantics(
            request=request,
            pending_approval=candidate_approval,
            structured_output=effective_structured_output,
            effective_retry_policy=self._effective_retry_policy,
        )
        require_secret_free_structured_output_spec(
            effective_structured_output,
            redactor=self._secret_redactor,
            field_name="ToolApprovalRecoveryRequest.structured_output",
        )
        pending_tool_call = approval_support.pending_tool_call_for_recovery(
            approval=candidate_approval,
            tool_call_id=request.tool_call_id,
        )
        execution_profile_snapshot, invocation_context = await self._prepare_gate_invocation(
            session=loaded_session,
            checkpoint=checkpoint,
            semantics=invocation_semantics,
            structured_output=effective_structured_output,
            request_loop_policies=request_loop_policies,
            secret_resolution_scope=candidate_approval.secret_resolution_scope,
        )
        registered_agent = invocation_context.registered_agent
        registered_provider = invocation_context.registered_provider
        registered_environment = invocation_context.registered_environment
        budget_policy_snapshot = invocation_context.budget_policy
        pending_approval: PendingToolApproval | None = None
        pending_round: pending_rounds.PendingToolRound | None = None
        claimed_resolution_intent: pending_approval_reader.ApprovalResolutionIntent | None = None

        def claim_exact_approval(
            _current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            nonlocal claimed_resolution_intent, pending_approval, pending_round
            pending_approval, pending_round = _pending_approval_and_round_for_atomic_claim(
                checkpoint,
                approval_id=approval_id,
                tool_round_id=request.tool_round_id,
                recovery_tool_call_id=request.tool_call_id,
                redactor=self._secret_redactor,
                runtime_session=_current_session,
            )
            if pending_approval != candidate_approval or pending_round != candidate_round:
                raise RuntimeError("Pending tool approval changed before it was claimed.")
            self._require_human_review_decision(
                request.review_reference, _current_session, checkpoint, denying=True
            )
            current_intent = pending_approval_reader.approval_resolution_intent_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
            )
            if current_intent != candidate_intent:
                raise RuntimeError(
                    "Approval resolution intent changed before recovery was claimed."
                )
            if current_intent is not None:
                approval_support.require_resolution_intent_matches_approval(
                    current_intent,
                    approval=pending_approval,
                )
            claimed_resolution_intent = current_intent
            return _checkpoint_with_legacy_approval_round(
                checkpoint,
                approval=pending_approval,
                redactor=self._secret_redactor,
                runtime_session=_current_session,
            )

        (
            session,
            resumed_event,
        ) = await self._recovery_admission.transition_recovery_session_to_running(
            loaded_session,
            checkpoint=checkpoint,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            checkpoint_transform=claim_exact_approval,
            execution_profile_snapshot=execution_profile_snapshot,
            before_mutation=before_mutation,
            after_admission=after_admission,
            invocation_context=invocation_context,
        )
        if pending_approval is None or pending_round is None:
            raise RuntimeError("Tool approval recovery claim completed without approval state.")
        if resumed_event is not None:
            yield resumed_event
        invocation_context = invocation_context.with_rebound_session(
            session,
            active_profile=execution_profile_snapshot.model_copy(
                update={"run_epoch": session.run_epoch}
            ),
        )
        recovery_stream = self.recover_tool_approval(
            request=request,
            participant_context=participant_context,
            loaded_session=loaded_session,
            session=session,
            pending_approval=pending_approval,
            pending_tool_call=pending_tool_call,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            execution_profile_snapshot=execution_profile_snapshot,
            budget_policy=budget_policy_snapshot,
            deferred_messages=pending_round.deferred_messages,
            claimed_resolution_intent=claimed_resolution_intent,
            invocation_context=invocation_context,
        )
        authoritative_failure: BaseException | None = None
        try:
            async for event in recovery_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            await self._recovery_admission.cleanup_entrypoint_handoff(
                stream=recovery_stream,
                session_id=session.id,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=False,
                release_run_fence=False,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=invocation_context,
            )

    def _reject_approval_owned_tool_round_recovery(
        self,
        checkpoint: dict[str, Any] | None,
    ) -> None:
        pending_approval = pending_approval_reader.pending_approval_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
        )
        if pending_approval is not None:
            raise RuntimeError(
                "Pending approval-owned tool rounds must be recovered with "
                "ToolApprovalRecoveryRequest."
            )

    async def replay_consumed_tool_effect(
        self, request: ToolEffectReconciliationRequest
    ) -> Event | None:
        """Return exact consumed evidence without renewing task execution authority."""
        session = await self._session_store.load(request.session_id)
        if session is None:
            return None
        record = await ToolEffectStateOwner(self._session_store).resolve_call(
            session, tool_round_id=request.tool_round_id, tool_call_id=request.tool_call_id
        )
        if record is None or record.terminal is None:
            return None
        replay = await self._load_reconciled_tool_effect_replay(session=session, request=request)
        if (
            not isinstance(replay, _ReconciledToolEffectReplay)
            or replay.consumption_receipt is None
        ):
            return None
        if record.intent.approval_id is not None:
            await self.require_human_review_resolution_authority(
                session.id, request.review_reference
            )
        return copy_event(replay.terminal_event)

    async def _load_reconciled_tool_effect_replay(
        self,
        *,
        session: Session,
        request: ToolEffectReconciliationRequest,
    ) -> _ReconciledToolEffectReplay | _ToolEffectObservationReplay | None:
        """Read exact settlement and consumption proof without acquiring execution authority."""
        from cayu.sessions.base import runtime_publication_event_reference

        if request.session_instance_id != session.instance_id:
            raise ToolEffectConflict("Receipt replay has a different session incarnation.")
        record = await ToolEffectStateOwner(self._session_store).resolve_call(
            session,
            tool_round_id=request.tool_round_id,
            tool_call_id=request.tool_call_id,
        )
        if record is None:
            return None
        if record.terminal is None:
            if record.reconciliation_attempt is not None:
                # A newer admitted decision supersedes the prior observation's
                # replay authority even before its callback produces an outcome.
                return None
            observation = record.observation
            if observation is None or observation.request_digest != reconciliation_request_digest(
                request
            ):
                return None
            records = await self._session_store.query_events(
                EventQuery(session_id=session.id, event_id=observation.event_id, limit=1)
            )
            if len(records) != 1:
                raise ToolEffectConflict("Receipt replay lost its atomic observation event.")
            event = records[0].event
            _validate_selected_observation(record, (event,))
            return _ToolEffectObservationReplay(record, copy_event(event))
        if record.state not in {
            "reconciled_completed",
            "reconciled_failed",
        } or record.terminal.reconciliation_request_digest != reconciliation_request_digest(
            request
        ):
            raise ToolEffectConflict("Receipt replay differs from the selected reconciliation.")
        records = await self._session_store.query_events(
            EventQuery(session_id=session.id, event_id=record.terminal.event_id, limit=1)
        )
        if len(records) != 1:
            raise ToolEffectConflict("Receipt replay lost its atomically selected terminal event.")
        event = records[0].event
        _validate_selected_terminal(record, None, (event,))
        if event.interaction_id != record.intent.interaction_id:
            raise ToolEffectConflict("Receipt replay terminal has a different interaction.")
        if record.intent.approval_id is not None:
            consumption_kind = "approval-close"
            consumption_id = f"approval-close:{record.intent.approval_id}"
        elif record.intent.pause_id is not None:
            consumption_kind = "user-input-close"
            consumption_id = f"user-input-close:{record.intent.pause_id}"
        else:
            consumption_kind = "tool-round"
            consumption_id = f"tool-round:{record.intent.tool_round_id}"
        consumption = await self._session_store.load_runtime_publication_receipt(
            session.id, consumption_id
        )
        if consumption is not None:
            if record.intent.pause_id is not None:
                response = request.user_input_response
                if response is None or response.input_id != record.intent.pause_id:
                    raise ToolEffectConflict("Receipt replay lacks its exact user-input response.")
                await self._user_input_evidence.exact_user_input_close_event(
                    session=session,
                    input_id=record.intent.pause_id,
                    receipt=consumption,
                    expected_resolution_request_digest=user_input_resolution_request_digest(
                        response
                    ),
                )
            call_ids = consumption.intent.get("tool_call_ids")
            if (
                consumption.kind != consumption_kind
                or consumption.interaction_id != record.intent.interaction_id
                or any(
                    consumption.intent.get(name) != getattr(record.intent, name)
                    for name in ("tool_round_id", "model_step_id", "model_attempt_id")
                )
                or type(call_ids) is not list
                or record.intent.tool_call_id not in call_ids
                or runtime_publication_event_reference(event) not in consumption.referenced_events
            ):
                raise ToolEffectConflict("Receipt replay has conflicting consumption evidence.")
            if record.intent.approval_id is not None and (
                consumption.intent.get("approval_id") != record.intent.approval_id
                or consumption.intent.get("decision") != "approve"
                or consumption.intent.get("requested_decision") != "approve"
            ):
                raise ToolEffectConflict(
                    "Receipt replay has conflicting approval closure evidence."
                )
        return _ReconciledToolEffectReplay(record, copy_event(event), consumption)

    async def _preflight_tool_effect_reconciliation(
        self,
        *,
        session: Session,
        request: ToolEffectReconciliationRequest,
        source_run_epoch: int | None = None,
    ) -> _ReconciledToolEffectReplay | _ToolEffectObservationReplay | None:
        """Reject invalid effect authority before claims or extension setup."""
        replay = await self._load_reconciled_tool_effect_replay(session=session, request=request)
        if replay is not None:
            return replay
        epoch = session.run_epoch if source_run_epoch is None else source_run_epoch
        if request.expected_run_epoch != epoch:
            raise ToolEffectConflict("Receipt recovery has stale session authority.")
        record = await ToolEffectStateOwner(self._session_store).resolve_call(
            session, tool_round_id=request.tool_round_id, tool_call_id=request.tool_call_id
        )
        if record is None:
            raise ToolEffectConflict("Receipt recovery has no durable effect intent.")
        registered_agent = self._resolve_registered_agent(session.agent_name)
        tool = registered_agent.tools.get(record.intent.tool_name)
        if tool is None or tool.effect is not ToolEffect.EXTERNAL:
            raise ToolEffectConflict("Receipt recovery has no exact registered external tool.")
        self._effect_reconciliation_owner.prepare(
            request=request, record=record, run_epoch=epoch, registered=tool.effect_reconciler
        )
        return None

    async def recover_tool_round_request(
        self,
        request: ToolRoundRecoveryRequest | ToolEffectReconciliationRequest,
        *,
        before_mutation: RecoveryMutationHook | None = None,
        after_admission: RecoveryMutationHook | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Recover a crashed ordinary tool round with an operator-verified outcome.

        A tool call in a non-approval round started but recorded no terminal event
        (a crash mid-tool), so an automatic resume would close it as an
        unknown-outcome failure. The caller supplies the externally verified outcome
        for that `tool_call_id`; Cayu persists it as the call's terminal result and
        never re-runs the tool. One call per invocation: if other
        started-but-unresolved calls remain, the session returns to INTERRUPTED with
        `manual_recovery_required` naming the next call; otherwise the round closes
        from the recorded outcomes and the model loop continues. A crashed round can
        leave the session FAILED (an in-process persistence error) or in a stale live
        status (a process kill), so FAILED and RUNNING are accepted alongside
        INTERRUPTED. An existing INTERRUPTING transition wins rather than being
        reopened by recovery. The in-process claim registered while this recovery
        streams blocks duplicate work in this process, while a durable recovery
        claim serializes other workers and fences an expired owner. If this call
        fails after claiming a stale live session, the session closes to the
        resumable INTERRUPTED state. When the recovered terminal event is already
        durable, the evidence remains authoritative: do not retry the same
        `tool_call_id` — `resume(...)` finishes the round from the persisted outcome.
        """
        loaded_session = await self._session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")

        if type(request) is ToolEffectReconciliationRequest:
            effect = await ToolEffectStateOwner(self._session_store).resolve_call(
                loaded_session,
                tool_round_id=request.tool_round_id,
                tool_call_id=request.tool_call_id,
            )
            if request.user_input_response is not None and (
                effect is None or effect.intent.pause_id is None
            ):
                raise ToolEffectConflict("Receipt call is not owned by a user-input pause.")
            if effect is not None and effect.intent.approval_id is not None:
                async with _close_delegated_event_stream(
                    self.recover_tool_approval_request(
                        request,
                        before_mutation=before_mutation,
                        after_admission=after_admission,
                        participant_context=participant_context,
                    )
                ) as approval_stream:
                    async for event in approval_stream:
                        yield event
                return
            if effect is not None and effect.intent.pause_id is not None:
                response = request.user_input_response
                if response is None or (
                    response.input_id != effect.intent.pause_id
                    or response.session_id != request.session_id
                    or response.task_worker_id != request.task_worker_id
                    or response.task_handoff_id != request.task_handoff_id
                ):
                    raise ToolEffectConflict(
                        "Receipt recovery requires the exact user-input response."
                    )
                replay = await self._preflight_tool_effect_reconciliation(
                    session=loaded_session, request=request
                )
                if (
                    isinstance(replay, _ToolEffectObservationReplay)
                    and loaded_session.status is SessionStatus.INTERRUPTED
                ):
                    yield copy_event(replay.event)
                    assert replay.record.observation is not None
                    if replay.record.observation.result.outcome == "conflict":
                        raise ToolEffectConflict("Application reconciliation rejected the receipt.")
                    return
                if (
                    isinstance(replay, _ReconciledToolEffectReplay)
                    and replay.consumption_receipt is not None
                ):
                    yield copy_event(replay.terminal_event)
                    return
                async with _close_delegated_event_stream(
                    self.resolve_user_input(
                        response,
                        participant_context=participant_context,
                        before_mutation=before_mutation,
                        after_admission=after_admission,
                        effect_reconciliation=request,
                    )
                ) as input_stream:
                    async for event in input_stream:
                        yield event
                return
            replay = await self._preflight_tool_effect_reconciliation(
                session=loaded_session, request=request
            )
            if (
                isinstance(replay, _ToolEffectObservationReplay)
                and loaded_session.status is SessionStatus.INTERRUPTED
            ):
                yield copy_event(replay.event)
                assert replay.record.observation is not None
                if replay.record.observation.result.outcome == "conflict":
                    raise ToolEffectConflict("Application reconciliation rejected the receipt.")
                return
            if (
                isinstance(replay, _ReconciledToolEffectReplay)
                and replay.consumption_receipt is not None
            ):
                yield copy_event(replay.terminal_event)
                return
            if request.session_instance_id != loaded_session.instance_id or (
                replay is None and request.expected_run_epoch != loaded_session.run_epoch
            ):
                raise ToolEffectConflict("Receipt recovery has stale session authority.")
            requested_round_id = request.tool_round_id
            requested_structured_output = None
            request_loop_policies = ()
        elif type(request) is ToolRoundRecoveryRequest:
            requested_round_id = request.round_id
            requested_structured_output = request.structured_output
            request_loop_policies = request.loop_policies
        else:
            raise TypeError("Tool-round recovery requires an exact recovery action.")
        checkpoint, pending_round = await pending_round_reader.load_pending_tool_round(
            self._session_store,
            loaded_session.id,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            runtime_session=loaded_session,
        )
        if pending_round is None:
            raise RuntimeError("Session has no pending tool round.")
        self._reject_approval_owned_tool_round_recovery(checkpoint)
        if pending_round.tool_round_id != requested_round_id:
            raise ValueError("Tool round id does not match pending round.")
        effective_structured_output = _effective_tool_round_structured_output(
            structured_output=requested_structured_output,
            pending_round=pending_round,
        )
        invocation_semantics = _effective_tool_round_invocation_semantics(
            request=request,
            pending_round=pending_round,
            structured_output=effective_structured_output,
            effective_retry_policy=self._effective_retry_policy,
        )
        require_secret_free_structured_output_spec(
            effective_structured_output,
            redactor=self._secret_redactor,
            field_name="ToolRoundRecoveryRequest.structured_output",
        )

        pending_tool_call = approval_support.round_tool_call_for_recovery(
            pending_calls=pending_round.tool_calls,
            tool_call_id=request.tool_call_id,
        )
        registered_agent = self._resolve_registered_agent(loaded_session.agent_name)
        if pending_round.agent_name != registered_agent.spec.name:
            raise RuntimeError(
                f"Pending tool round belongs to a different agent: {pending_round.agent_name}."
            )
        registered_provider = self._resolve_registered_provider(loaded_session.provider_name)
        budget_policy_snapshot = copy_budget_policy(self._resolve_budget_policy())
        execution_profile_snapshot = await self._execution_profile_continuation.validate(
            session=loaded_session,
            checkpoint=checkpoint,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            request_loop_policies=request_loop_policies,
            budget_policy=budget_policy_snapshot,
            request_budget_limits=invocation_semantics.budget_limits,
            structured_output=invocation_semantics.structured_output,
            thinking=invocation_semantics.thinking,
            max_steps=invocation_semantics.max_steps,
            limits=invocation_semantics.limits,
            retry_policy=invocation_semantics.retry_policy,
            invocation_semantics_available=True,
        )
        _require_native_structured_output_support(
            effective_structured_output, registered_provider=registered_provider
        )
        registered_environment = self._resolve_registered_environment(
            loaded_session.environment_name
        )
        invocation_secrets.require_continuation_secret_resolution_compatibility(
            pending_approval_reader.tool_round_secret_resolution_scope(pending_round),
            registered_environment,
        )
        pending_operator_interruption = (
            checkpoint is not None and _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY in checkpoint
        )
        if (
            checkpoint is not None
            and _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY in checkpoint
            and not pending_operator_interruption
        ):
            raise _ManualRecoveryCascadePending(
                "Session has an incomplete background interruption cascade."
            )
        if before_mutation is not None:
            if self._session_control.has_active_tasks(loaded_session.id):
                raise RuntimeError(f"Session has active work in this process: {loaded_session.id}")
            interaction_records = await self._session_store.query_events(
                EventQuery(
                    session_id=loaded_session.id,
                    event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                )
            )
            if (
                not interaction_records
                or interaction_records[0].event.type in INTERACTION_TERMINAL_EVENT_TYPES
                or interaction_records[0].event.interaction_id is None
            ):
                raise RuntimeError(
                    "Pending tool recovery state has no open interaction. "
                    "Pre-interaction prerelease recovery state is unsupported."
                )
            await before_mutation()
        if (
            loaded_session.status in _RECOVERY_RESUMABLE_SESSION_STATUSES
            and not pending_operator_interruption
        ):
            original_pending_round = pending_round
            (
                loaded_session,
                checkpoint,
            ) = await self.terminal_finalization.reconcile_before_continuation(
                session=loaded_session,
                checkpoint=checkpoint,
            )
            pending_round = pending_round_reader.pending_tool_round_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
                runtime_session=loaded_session,
            )
            if pending_round is None or pending_round != original_pending_round:
                raise RuntimeError(
                    "Pending tool round changed while terminal evidence was reconciled."
                )
            pending_tool_call = approval_support.round_tool_call_for_recovery(
                pending_calls=pending_round.tool_calls,
                tool_call_id=request.tool_call_id,
            )
            execution_profile_snapshot = await self._execution_profile_continuation.validate(
                session=loaded_session,
                checkpoint=checkpoint,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                request_loop_policies=request_loop_policies,
                frozen_candidate_profile=execution_profile_snapshot.profile,
                budget_policy=budget_policy_snapshot,
                request_budget_limits=invocation_semantics.budget_limits,
                structured_output=invocation_semantics.structured_output,
                thinking=invocation_semantics.thinking,
                max_steps=invocation_semantics.max_steps,
                limits=invocation_semantics.limits,
                retry_policy=invocation_semantics.retry_policy,
                invocation_semantics_available=True,
            )
        if self._session_control.has_active_tasks(loaded_session.id):
            raise RuntimeError(f"Session has active work in this process: {loaded_session.id}")
        # Reserve the in-process slot before awaiting the durable transition. The
        # check and registration are await-free, so another local recovery cannot
        # advance the run epoch while this claimant is waiting on storage.
        current_task = asyncio.current_task()
        if current_task is not None:
            self._session_control.register_active_task(
                loaded_session.id,
                current_task,
                task_id=None,
                task_started=False,
                task_finished=False,
            )
        interaction_id: str | None = None
        recovery_stream: AsyncGenerator[Event, None] | None = None
        authoritative_failure: BaseException | None = None
        try:
            interaction_id = await self._recovery_admission.activate_latest_open_interaction(
                loaded_session.id
            )
            if interaction_id is None:
                raise RuntimeError(
                    "Pending tool recovery state has no open interaction. "
                    "Pre-interaction prerelease recovery state is unsupported."
                )
            recovery_stream = self.recover_tool_round(
                request=request,
                participant_context=participant_context,
                loaded_session=loaded_session,
                pending_round=pending_round,
                pending_tool_call=pending_tool_call,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                invocation_semantics=invocation_semantics,
                execution_profile_snapshot=execution_profile_snapshot,
                budget_policy=budget_policy_snapshot,
                after_admission=after_admission,
            )
            async for event in recovery_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            try:
                await self._recovery_admission.cleanup_entrypoint_handoff(
                    stream=recovery_stream,
                    session_id=loaded_session.id,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    authoritative_failure=authoritative_failure,
                    finalize_abandoned=False,
                    release_run_fence=False,
                    abort_environment_setup=False,
                    execution_profile=execution_profile_snapshot.profile,
                )
            finally:
                if interaction_id is not None:
                    _deactivate_session_interaction(loaded_session.id)
                if current_task is not None:
                    self._session_control.unregister_active_task(
                        loaded_session.id,
                        current_task,
                    )

    async def _admit_user_input_resolution_execution(
        self,
        *,
        session: Session,
        pending: PendingUserInput,
        resolution_intent: UserInputResolutionIntent,
    ) -> UserInputResolutionIntent:
        """Linearize exact continuation ownership before any governed work."""

        require_resolution_intent_matches_pending(resolution_intent, pending=pending)
        expected = resolution_intent.model_copy(update={"execution_state": "executing"})
        admitted: UserInputResolutionIntent | None = None

        def raise_process_control(
            *failures: BaseException | None,
        ) -> None:
            unique_failures: list[BaseException] = []
            seen_failure_ids: set[int] = set()
            for failure in failures:
                if failure is None or id(failure) in seen_failure_ids:
                    continue
                seen_failure_ids.add(id(failure))
                unique_failures.append(failure)
            process_control = next(
                (
                    candidate
                    for failure in unique_failures
                    if (candidate := _terminal_finalization_process_control(failure)) is not None
                ),
                None,
            )
            if process_control is None:
                return
            secondary_failures = [
                retained
                for failure in unique_failures
                if failure is not process_control
                if (
                    retained := _terminal_finalization_failure_without_identity(
                        failure,
                        process_control,
                    )
                )
                is not None
            ]
            if secondary_failures:
                secondary: BaseException
                if len(secondary_failures) == 1:
                    secondary = secondary_failures[0]
                else:
                    secondary = BaseExceptionGroup(
                        "User-input execution admission retained secondary failures.",
                        secondary_failures,
                    )
                if not _attach_exception_cause_preserving_graph(
                    process_control,
                    secondary,
                ):
                    raise BaseExceptionGroup(
                        "User-input execution admission retained process control and "
                        "secondary failures.",
                        [process_control, secondary],
                    ) from None
            raise process_control

        def admit(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            nonlocal admitted
            if (
                current_session.id != session.id
                or current_session.instance_id != session.instance_id
                or current_session.run_epoch != session.run_epoch
                or current_session.status is not SessionStatus.RUNNING
            ):
                raise SessionRuntimePublicationConflict(
                    "User-input resolution was superseded before execution admission."
                )
            try:
                updated, admitted = checkpoint_with_executing_user_input_resolution_intent(
                    checkpoint,
                    current_run_epoch=current_session.run_epoch,
                    runtime_session=current_session,
                    pending=pending,
                    intent=resolution_intent,
                    redactor=self._secret_redactor,
                )
            except RuntimeError as exc:
                raise SessionRuntimePublicationConflict(
                    "User-input resolution was superseded before execution admission."
                ) from exc
            return updated

        async def commit() -> UserInputResolutionIntent:
            await self._session_store.transform_checkpoint(session.id, admit)
            if admitted is None:
                raise RuntimeError(
                    "User-input execution admission completed without durable authority."
                )
            return admitted

        outcome = await await_shielded_task_outcome(asyncio.create_task(commit()))
        cancellation = outcome.cancellation
        cancellation_requests_consumed = outcome.cancellation_requests_consumed
        error = outcome.error
        if isinstance(error, asyncio.CancelledError) and cancellation is None:
            error = unexpected_child_cancellation_error(
                error,
                operation="User-input resolution execution admission",
            )
        if error is None:
            if outcome.result != expected:
                error = RuntimeError(
                    "User-input execution admission returned conflicting authority."
                )
            elif cancellation is not None:
                restore_task_cancellation_requests(
                    cancellation_requests_consumed,
                    cancellation=cancellation,
                )
                raise cancellation
            else:
                return expected

        reconciled: UserInputResolutionIntent | None = None

        def inspect(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> None:
            nonlocal reconciled
            if (
                current_session.id != session.id
                or current_session.instance_id != session.instance_id
                or current_session.run_epoch != session.run_epoch
                or current_session.status is not SessionStatus.RUNNING
            ):
                return None
            current_pending, current_intent = user_input_lifecycle_authority_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                current_run_epoch=current_session.run_epoch,
                runtime_session=current_session,
            )
            if current_pending == pending and current_intent == expected:
                reconciled = current_intent
            return None

        reconciliation = await await_shielded_task_outcome(
            asyncio.create_task(self._session_store.transform_checkpoint(session.id, inspect)),
            cancellation=cancellation,
        )
        cancellation = reconciliation.cancellation
        cancellation_requests_consumed += reconciliation.cancellation_requests_consumed
        reconciliation_error = reconciliation.error
        if isinstance(reconciliation_error, asyncio.CancelledError):
            reconciliation_error = unexpected_child_cancellation_error(
                reconciliation_error,
                operation="User-input execution admission reconciliation",
            )
        if reconciliation_error is not None:
            raise_process_control(error, reconciliation_error, cancellation)
            if cancellation is not None:
                cancellation.add_note(
                    "User-input execution admission and reconciliation failed during cancellation."
                )
                restore_task_cancellation_requests(
                    cancellation_requests_consumed,
                    cancellation=cancellation,
                )
                if reconciliation_error is error:
                    raise cancellation from error
                raise cancellation from BaseExceptionGroup(
                    "User-input execution admission and reconciliation failures.",
                    [error, reconciliation_error],
                )
            if reconciliation_error is error:
                raise error
            raise error from reconciliation_error
        if reconciled != expected:
            if cancellation is not None:
                cancellation.add_note(
                    "User-input execution admission did not commit before cancellation."
                )
                restore_task_cancellation_requests(
                    cancellation_requests_consumed,
                    cancellation=cancellation,
                )
                raise cancellation from error
            raise error
        raise_process_control(error, cancellation)
        if cancellation is not None:
            restore_task_cancellation_requests(
                cancellation_requests_consumed,
                cancellation=cancellation,
            )
            raise cancellation
        return expected

    def _continue_closed_tool_round(
        self,
        *,
        session: Session,
        environment: ContinuationEnvironment,
        registered_provider: runtime_records.RegisteredProvider,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
        budget_policy: BudgetPolicy | None,
        transcript: list[Message],
        semantics: _RecoveryInvocationSemantics,
        resolution: ToolApprovalRequest | UserInputResponse,
        task_id: str | None,
        model_step: int | None,
        tool_exposure: ResolvedToolExposureAuthority | None,
        run_limit_accounting: RunLimitAccountingContext | None,
        participant_context: CollaborationAccessContext | None,
    ) -> AsyncGenerator[Event, None]:
        """Hand an exactly closed human-gated round back to the admitted engine.

        Callers commit their typed closure and propagate its cancellation before
        entering here. They retain ownership of the returned stream and fence.
        """
        session_stream = self._engine.continue_run(
            RecoverySessionRunRequest(
                session=session,
                messages=transcript,
                messages_to_append=[],
                max_steps=semantics.max_steps,
                limits=semantics.limits,
                budget_limits=semantics.budget_limits,
                retry_policy=semantics.retry_policy,
                structured_output=semantics.structured_output,
                thinking=semantics.thinking,
                request_metadata=resolution.metadata,
                participant_context=participant_context,
                task_id=task_id,
                task_worker_id=resolution.task_worker_id,
                task_handoff_id=resolution.task_handoff_id,
                start_event_type=None,
                start_event_payload={},
                start_task_on_enter=False,
                release_run_fence_on_exit=False,
                run_limit_accounting=run_limit_accounting,
                completed_tool_round_model_step=_completed_tool_round_model_step(
                    model_step, max_steps=semantics.max_steps
                ),
                previous_tool_exposure_profile_id=_continued_tool_exposure_profile_id(
                    tool_exposure
                ),
                invocation_context=(
                    environment.invocation_context
                    if environment.invocation_context is not None
                    else reconstruct_invocation_context(
                        runtime_hooks=self._runtime_hooks,
                        loop_policies=self._loop_policies,
                        session=session,
                        execution_profile_snapshot=execution_profile_snapshot,
                        registered_agent=environment.agent,
                        registered_provider=registered_provider,
                        registered_environment=environment.registered_environment,
                        budget_policy=copy_budget_policy(budget_policy),
                        request_loop_policies=resolution.loop_policies,
                    )
                ),
            )
        )
        return self._session_control.stream_with_out_of_band_events(session.id, session_stream)

    async def continue_user_input_resolution(
        self,
        *,
        response: UserInputResponse,
        session: Session,
        pending: PendingUserInput,
        resolution_intent: UserInputResolutionIntent,
        resolution_stage: Literal["answer", "manual-recovery"],
        closure_request_digest: str,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
        budget_policy: BudgetPolicy | None,
        invocation_context: InvocationContext | None = None,
        emit_resume_event: bool = True,
        effect_reconciliation: ToolEffectReconciliationRequest | None = None,
        foreground_gate: GateReplay | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        await self._require_participant_execution(session, participant_context)
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_provider is not registered_provider
            or invocation_context.registered_environment is not registered_environment
            or invocation_context.profile is not execution_profile_snapshot.profile
            or invocation_context.budget_policy is not budget_policy
        ):
            raise RuntimeError("User-input recovery lost frozen invocation authority.")
        answer_request_digest = (
            user_input_answer_request_digest(response)
            if foreground_gate is None
            else foreground_gate.answer_digest
        )
        if closure_request_digest != resolution_intent.resolution_request_digest:
            raise RuntimeError("User-input continuation closure digest conflicts with its request.")
        require_resolution_intent_matches_pending(
            resolution_intent,
            pending=pending,
            answer_request_digest=answer_request_digest,
            resolution_stage=resolution_stage,
            resolution_request_digest=closure_request_digest,
        )
        tool_round_identity = ToolRoundIdentity(
            tool_round_id=pending.tool_round_id,
            model_step_id=pending.model_step_id,
            model_attempt_id=pending.model_attempt_id,
        )
        if pending.tool_exposure is not None:
            validate_resolved_tool_exposure_authority(
                pending.tool_exposure,
                registered_agent.tool_capabilities,
                catalogue_revision=registered_agent.tool_catalogue.revision,
            )
        pending_cleared = False
        tool_outcomes: list[runtime_records.ToolCallOutcome] = []
        round_owner: DurableToolRound | None = None
        # Restore the original run's config persisted on the pending input. Explicit
        # values have already been checked against the frozen invocation profile.
        invocation_semantics = _effective_user_input_invocation_semantics(
            response=response,
            pending=pending,
            structured_output=_effective_user_input_structured_output(
                structured_output=response.structured_output,
                pending=pending,
            ),
            effective_retry_policy=self._effective_retry_policy,
        )
        effective_max_steps = invocation_semantics.max_steps
        effective_limits = invocation_semantics.limits
        effective_budget_limits = invocation_semantics.budget_limits
        effective_retry_policy = invocation_semantics.retry_policy
        continued_run_limit_accounting = (
            None
            if pending.run_limit_accounting is None
            else resume_run_limit_accounting_context(
                pending.run_limit_accounting,
                resolved_at=resolution_intent.pause_resolved_at,
            )
        )
        effect_settlement: _ReconciledToolEffectReplay | _ToolEffectObservationReplay | None = None
        environment = ContinuationEnvironment(
            lifecycle=self._environment_lifecycle,
            session=session,
            agent=registered_agent,
            profile=execution_profile_snapshot.profile,
            registered_environment=registered_environment,
            invocation_context=invocation_context,
        )
        try:
            resolution_intent = await self._admit_user_input_resolution_execution(
                session=session,
                pending=pending,
                resolution_intent=resolution_intent,
            )
            if foreground_gate is None and any(
                (tool := registered_agent.tools.get(call.tool_name)) is not None
                and tool.child_session_recovery is not None
                for call in pending.tool_calls
            ):
                from cayu.runtime._foreground_gate_continuation import retain_gate_request

                await retain_gate_request(
                    self._session_store,
                    session=session,
                    request=response,
                    redactor=self._secret_redactor,
                    policy_owner=self._foreground_gate_policy_owner,
                )
            transcript_snapshot = await self._session_store.load_transcript_snapshot(session.id)
            try:
                transcript = [
                    detach_message(record.message) for record in transcript_snapshot.records
                ]
                user_input_transcript_cursor = transcript_snapshot.cursor
            finally:
                del transcript_snapshot
            resume_events = await self._session_store.load_events(session.id)
            if continued_run_limit_accounting is not None:
                continued_run_limit_accounting = rebase_run_limit_accounting_context(
                    continued_run_limit_accounting,
                    session_id=session.id,
                    limits=effective_limits,
                    budget_limits=request_budget_limits_for_session(
                        limits=effective_budget_limits,
                        agent_name=registered_agent.spec.name,
                        causal_budget_id=session.causal_budget_id,
                    ),
                    events=resume_events,
                    # Profile admission permits only an exact restatement of the
                    # frozen invocation semantics. It must not reset the durable
                    # accounting origin merely because the caller restated it.
                    reset_run_limits=False,
                    reset_budgets=False,
                    now=self._clock(),
                )
            async with contextlib.aclosing(environment.reconnect()) as preparation:
                async for event in preparation:
                    yield event

            if environment.error is not None:
                raise environment.error
            if effect_reconciliation is not None:
                if environment.invocation_context is None:
                    raise RuntimeError("Receipt recovery has no user-input invocation authority.")
                async with contextlib.aclosing(
                    self._settle_tool_effect_reconciliation(
                        request=effect_reconciliation,
                        source_run_epoch=effect_reconciliation.expected_run_epoch,
                        session=session,
                        invocation_context=environment.invocation_context,
                    )
                ) as settlement_stream:
                    async for item in settlement_stream:
                        if isinstance(item, Event):
                            yield copy_event(item)
                        else:
                            effect_settlement = item
                if effect_settlement is None:
                    raise RuntimeError("User-input receipt recovery returned no settlement.")
                if isinstance(effect_settlement, _ToolEffectObservationReplay):
                    await self._event_writer.fan_out_persisted([effect_settlement.event])
                    yield copy_event(effect_settlement.event)
                    async for event in self._interrupt_unresolved_tool_effect(
                        record=effect_settlement.record,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=environment.registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                        invocation_context=environment.invocation_context,
                    ):
                        yield event
                    return
                await self._event_writer.fan_out_persisted([effect_settlement.terminal_event])
                yield copy_event(effect_settlement.terminal_event)
                recovered_call = approval_support.round_tool_call_for_recovery(
                    pending_calls=pending.tool_calls,
                    tool_call_id=effect_reconciliation.tool_call_id,
                )
                async for event, _modified in self._tool_invocation.hooks.after_call(
                    session=session,
                    tool_event=effect_settlement.terminal_event,
                    registered_agent=registered_agent,
                    registered_environment=environment.registered_environment,
                    tool_call=approval_support.tool_call_request_from_pending(
                        recovered_call, arguments={}
                    ),
                    result=ToolResult.model_validate(
                        effect_settlement.terminal_event.payload["result"]
                    ),
                    task_id=pending.task_id,
                    execution_profile=execution_profile_snapshot.profile,
                    invocation_context=environment.invocation_context,
                    redactor=self._secret_redactor,
                    output_redactor=self._secret_redactor,
                    allow_modification=False,
                ):
                    yield event
                resume_events = await self._session_store.load_events(session.id)
            if emit_resume_event:
                yield await self._event_writer.emit(
                    event_with_execution_profile_authority(
                        event_with_runtime_payload_authority(
                            Event(
                                type=EventType.SESSION_RESUMED,
                                session_id=session.id,
                                agent_name=registered_agent.spec.name,
                                environment_name=environment.name,
                                payload={
                                    **tool_round_identity.payload(),
                                    "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                                    "input_id": pending.input_id,
                                    "tool_call_id": pending.tool_call_id,
                                    "resolved_by": resolution_actor_payload(response.resolved_by),
                                },
                            ),
                            "model_step_id",
                            "model_attempt_id",
                            "tool_round_id",
                            "input_id",
                        ),
                        execution_profile_snapshot.profile,
                    )
                )
            async with contextlib.aclosing(environment.bind()) as preparation:
                async for event in preparation:
                    yield event
            if environment.error is not None:
                raise environment.error

            round_tool_calls = [
                approval_support.tool_call_request_from_pending(pending_call)
                for pending_call in pending.tool_calls
            ]
            base_round_redactor = _redactor_for_tool_calls(
                self._secret_redactor,
                registered_agent=registered_agent,
                tool_calls=round_tool_calls,
            )
            legacy_publication_scope = (
                pending.assistant_message_state == "quarantined"
                and pending.assistant_publication is None
            )
            persisted_secret_resolution_scope = (
                "unknown"
                if pending.assistant_publication is None
                else pending.assistant_publication.secret_resolution_scope
            )
            pause_secret_resolution_scope = invocation_secrets.continuation_secret_resolution_scope(
                persisted_secret_resolution_scope,
                environment.registered_environment,
            )
            paused_round = await PausedToolRound.prepare(
                session_store=self._session_store,
                event_writer=self._event_writer,
                invocation=self._tool_invocation,
                clock=self._clock,
                session=session,
                agent=registered_agent,
                environment=environment.registered_environment,
                environment_name=environment.name,
                profile=execution_profile_snapshot.profile,
                invocation_context=environment.invocation_context,
                identity=tool_round_identity,
                task_id=pending.task_id,
                tool_calls=round_tool_calls,
                tool_exposure=pending.tool_exposure,
                redactor=base_round_redactor,
                secret_resolution_scope=pause_secret_resolution_scope,
                secret_redactor=self._secret_redactor,
                pause=UserInputRoundPause(pending.input_id),
            )
            round_owner = paused_round.round
            await round_owner.admit()

            # Reuse any outcomes already recorded for this round — e.g. a prior resume attempt
            # that ran some tools before a mid-resume failure — so a retry never re-executes a
            # side-effecting tool. The round was already projected against limits at pause time;
            # its remaining tools run on resume without a fresh budget projection (so the user's
            # answer is never discarded by a limit check here).
            native_events = await self._recover_terminal_foreground_effects_at_human_gate(
                session=session,
                pending=pending,
                registered_agent=registered_agent,
                registered_environment=environment.registered_environment,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=environment.invocation_context,
            )
            resume_events.extend(native_events)
            for native_event in native_events:
                yield native_event
            recorded_outcomes = approval_support.recorded_round_tool_outcomes(
                events=resume_events,
                pending_calls=pending.tool_calls,
                input_id=pending.input_id,
                tool_round_identity=tool_round_identity,
                staged_terminals=pending.staged_terminals,
            )
            restarted_staged_ids = await round_owner.fence_restarted_continuation(
                recorded_ids=set(recorded_outcomes),
                resume_undispatched_siblings=(
                    foreground_gate is not None and pause_secret_resolution_scope == "static"
                ),
            )
            pending_by_id = {call.tool_call_id: call for call in pending.tool_calls}

            # Build the round's outcomes in model order: a call already recorded (retry) is
            # reused; the answered ask_user call gets the injected answer; every other allowed
            # call executes now (none ran before the pause); a denied call is blocked.
            for tool_call in round_tool_calls:
                recorded_outcome = recorded_outcomes.get(tool_call.id)
                if recorded_outcome is not None:
                    # A terminal record from an earlier process proves the
                    # call is complete, but an additive pre-field checkpoint
                    # does not prove which invocation secrets it resolved.
                    await round_owner.record_continuation_scope(
                        tool_call.id,
                        execution_scope_unknown=legacy_publication_scope,
                    )
                    tool_outcomes.append(recorded_outcome)
                    continue
                if tool_call.id in restarted_staged_ids:
                    continue

                pending_call = pending_by_id[tool_call.id]
                registered_tool = registered_agent.executable_tool(tool_call.name)
                policy_evidence = pending_approval_reader.effective_tool_policy_evidence(
                    pending_call
                )
                policy_result = approval_support.policy_result_from_pending_tool_call(pending_call)
                if tool_call.id == pending.tool_call_id:
                    if policy_evidence is not ToolPolicyEvidence.AUTHORITATIVE:
                        raise RuntimeError(
                            "Pending user-input call has no authoritative policy decision."
                        )
                    for (
                        rejoined_event
                    ) in await self._tool_invocation.admission.rejoin_targeted_call(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=environment.registered_environment,
                        tool_call=tool_call,
                        task_id=pending.task_id,
                        invocation_context=environment.invocation_context,
                    ):
                        yield rejoined_event
                    idempotency_key = tool_execution.tool_idempotency_key(
                        session_id=session.id,
                        tool_round_id=tool_round_identity.tool_round_id,
                        tool_call_id=tool_call.id,
                        pause_id=pending.input_id,
                    )
                    result = ToolResult(
                        content=response.answer,
                        structured=response.structured,
                        artifacts=response.artifacts,
                        is_error=False,
                    )
                    await round_owner.record_continuation_scope(tool_call.id)
                    started_payload: dict[str, Any] = {
                        **tool_round_identity.payload(),
                        "tool_call_id": tool_call.id,
                        "idempotency_key": idempotency_key,
                        **tool_argument_publication.quarantined_argument_fields(),
                        "input_id": pending.input_id,
                    }
                    if registered_tool is not None:
                        started_payload["effect"] = registered_tool.effect.value
                    yield await self._event_writer.emit(
                        event_with_execution_profile_authority(
                            event_with_runtime_payload_authority(
                                Event(
                                    type=EventType.TOOL_CALL_STARTED,
                                    session_id=session.id,
                                    agent_name=registered_agent.spec.name,
                                    environment_name=environment.name,
                                    tool_name=tool_call.name,
                                    payload=started_payload,
                                ),
                                "model_step_id",
                                "model_attempt_id",
                                "tool_round_id",
                                "input_id",
                            ),
                            execution_profile_snapshot.profile,
                        )
                    )
                    async for (
                        event,
                        outcome,
                    ) in paused_round.publish_result(
                        event=Event(
                            type=EventType.TOOL_CALL_COMPLETED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment.name,
                            tool_name=tool_call.name,
                            payload={
                                **tool_round_identity.payload(),
                                "tool_call_id": tool_call.id,
                                "idempotency_key": idempotency_key,
                                "input_id": pending.input_id,
                                "resolved_by": resolution_actor_payload(response.resolved_by),
                                "result": result.model_dump(),
                            },
                        ),
                        tool_call=tool_call,
                        result=result,
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                call_taint_labels = approval_support.taint_labels_from_pending_tool_call(
                    pending_call
                )
                # `ToolInvocation.execute(check_policy=False)` does not re-enforce
                # the decision, so a DENY must be blocked here explicitly (mirroring the approval
                # resume) — otherwise a policy-denied sibling would execute. REQUIRE_APPROVAL
                # cannot occur: it would have preempted the ask_user pause with an approval pause.
                if (
                    policy_evidence is ToolPolicyEvidence.AUTHORITATIVE
                    and policy_result is not None
                    and policy_result.decision == ToolPolicyDecision.DENY
                ):
                    await round_owner.record_continuation_scope(tool_call.id)
                    public_policy_result = approval_support.public_policy_denial_result(
                        secret_resolution_scope=pause_secret_resolution_scope,
                        policy_result=policy_result,
                        publish_arguments=(
                            registered_tool is not None and registered_tool.publish_arguments
                        ),
                    )
                    reason = tool_execution.policy_denial_reason(public_policy_result)
                    blocked_result = tool_execution.blocked_tool_result(
                        public_policy_result,
                        reason=reason,
                    )
                    idempotency_key = tool_execution.tool_idempotency_key(
                        session_id=session.id,
                        tool_round_id=tool_round_identity.tool_round_id,
                        tool_call_id=tool_call.id,
                        pause_id=pending.input_id,
                    )
                    async for (
                        event,
                        outcome,
                    ) in paused_round.publish_result(
                        event=Event(
                            type=EventType.TOOL_CALL_BLOCKED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment.name,
                            tool_name=tool_call.name,
                            payload={
                                **tool_round_identity.payload(),
                                "tool_call_id": tool_call.id,
                                "idempotency_key": idempotency_key,
                                "input_id": pending.input_id,
                                **policy_denial_payload_fields(
                                    tool_name=tool_call.name,
                                    denied_by=_TOOL_POLICY_DENIAL_SOURCE,
                                    decision=public_policy_result.decision.value,
                                    reason=reason,
                                    metadata=public_policy_result.metadata,
                                ),
                                "result": blocked_result.model_dump(),
                            },
                        ),
                        tool_call=tool_call,
                        result=blocked_result,
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                if policy_evidence in {
                    ToolPolicyEvidence.AMBIGUOUS,
                    ToolPolicyEvidence.UNREGISTERED,
                    ToolPolicyEvidence.UNEXPOSED,
                }:
                    await round_owner.record_continuation_scope(tool_call.id)
                    async for (
                        event,
                        outcome,
                    ) in paused_round.reject_policy(
                        tool_call=tool_call,
                        policy_evidence=policy_evidence,
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                if policy_evidence is not ToolPolicyEvidence.AUTHORITATIVE:
                    raise RuntimeError(
                        "Pending user-input sibling has no executable policy authority."
                    )

                async for event, outcome in paused_round.execute(
                    tool_call=tool_call,
                    request_metadata=response.metadata,
                    budget_limits=pending.budget_limits or (),
                    auxiliary_invocation_policy=AuxiliaryInvocationPolicy(
                        limits=effective_limits,
                        retry_policy=effective_retry_policy,
                        accounting=continued_run_limit_accounting,
                    ),
                    policy_result=policy_result,
                    model_step=pending.model_step,
                    taint_labels=call_taint_labels,
                ):
                    yield event
                    if outcome is not None:
                        tool_outcomes.append(outcome)

            if round_owner.defers_terminals:
                async with contextlib.aclosing(
                    round_owner.publish_continuation(
                        already_published_ids=set(recorded_outcomes),
                        restarted_staged_ids=restarted_staged_ids,
                    )
                ) as terminal_events:
                    async for event, outcome in terminal_events:
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                outcomes_by_id = {outcome.call.id: outcome for outcome in tool_outcomes}
                tool_outcomes = [outcomes_by_id[call.id] for call in round_tool_calls]

            # The resume executes the round's tools sequentially in model order, so the outcome
            # list already lines up with the assistant tool-call parts.
            source_checkpoint = await self._session_store.load_checkpoint(session.id)
            current_pending, current_intent = user_input_lifecycle_authority_from_checkpoint(
                source_checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
                current_run_epoch=session.run_epoch,
                runtime_session=session,
            )
            if (
                current_pending is None
                or pending_user_input_digest(current_pending) != pending_user_input_digest(pending)
                or current_intent != resolution_intent
            ):
                raise SessionRuntimePublicationConflict(
                    "Pending user-input authority changed before atomic closure."
                )
            durable_round = pending_actions.pending_action_evidence_round_from_checkpoint(
                source_checkpoint
            )
            if (
                durable_round is None
                or pending_rounds.pending_tool_round_identity(durable_round) != tool_round_identity
            ):
                raise RuntimeError("Pending user-input round changed before transcript closure.")
            tool_result_messages = transcript_helpers.tool_result_messages(
                tool_outcomes,
                tool_round_identity=tool_round_identity,
            )
            transcript_messages = list(tool_result_messages)
            if durable_round.assistant_message_state == "quarantined":
                transcript_messages.insert(
                    0,
                    transcript_helpers.assistant_message_with_projected_tool_arguments(
                        tool_round_recovery.ready_assistant_publication_message(durable_round),
                        tool_outcomes,
                    ),
                )
            final_events = await self._pending_tool_round_recovery.load_tool_round_lifecycle_events(
                session_id=session.id,
                pending_round=durable_round,
            )
            lifecycle_event_types = {
                EventType.TOOL_CALL_STARTED,
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_FAILED,
                EventType.TOOL_CALL_BLOCKED,
            }
            lifecycle_events = [
                event
                for event in final_events
                if event.type in lifecycle_event_types
                and event.payload.get("input_id") == pending.input_id
                and tool_round_identity.matches_payload(event.payload)
            ]
            target_checkpoint = checkpoint_without_exact_pending_user_input(
                source_checkpoint,
                pending=current_pending,
                intent=resolution_intent,
                redactor=self._secret_redactor,
                runtime_session=session,
            )
            from cayu.sessions._foreground_child_checkpoint import gate_close_continuation

            gate_continuation = gate_close_continuation(
                source_checkpoint,
                pending=durable_round,
                publication_id=f"user-input-close:{pending.input_id}",
                metadata=self._secret_redactor.redact_json_values(response.metadata),
            )
            if gate_continuation is not None:
                target_checkpoint.pop("foreground_child_wait", None)
                target_checkpoint.pop("foreground_child_terminal", None)
                target_checkpoint["foreground_parent_continuation"] = gate_continuation
            from cayu.sessions._foreground_child_checkpoint import (
                FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY,
                post_action_continuation_for_close,
            )

            marker = post_action_continuation_for_close(
                source_checkpoint,
                session=session,
                wait_checkpoint=(
                    await self._session_store.load_checkpoint(session.parent_session_id)
                    if session.parent_session_id is not None
                    else None
                ),
                close_publication_id=f"user-input-close:{pending.input_id}",
                request_metadata=self._secret_redactor.redact_json_values(response.metadata),
                completed_model_step=_completed_tool_round_model_step(
                    pending.model_step, max_steps=effective_max_steps
                ),
            )
            if marker is not None:
                target_checkpoint[FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY] = marker
            close_event = event_with_pending_user_input_authority(
                event_with_execution_profile_authority(
                    event_with_runtime_payload_authority(
                        Event(
                            type=EventType.SESSION_CHECKPOINTED,
                            session_id=session.id,
                            interaction_id=pending.source_interaction_id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment.name,
                            payload={
                                "checkpoint": PENDING_USER_INPUT_CHECKPOINT_KEY,
                                "transition": "answered",
                                **tool_round_identity.payload(),
                                "input_id": pending.input_id,
                                "tool_call_id": pending.tool_call_id,
                                "source_run_epoch": pending.source_run_epoch,
                                "pause_digest": pending_user_input_digest(pending),
                                "resolution_request_digest": closure_request_digest,
                            },
                        ),
                        "model_step_id",
                        "model_attempt_id",
                        "tool_round_id",
                        "input_id",
                        "tool_call_id",
                        "pause_digest",
                        "resolution_request_digest",
                    ),
                    execution_profile_snapshot.profile,
                ),
                pending,
            )
            prepared_close = approval_publication.prepare_pending_action_publication(
                session_id=session.id,
                publication_id=f"user-input-close:{pending.input_id}",
                kind="user-input-close",
                intent={
                    **pending_user_input_identity(pending),
                    "claim_run_epoch": resolution_intent.claim_run_epoch,
                    "answer_request_digest": resolution_intent.answer_request_digest,
                    "execution_state": resolution_intent.execution_state,
                    "resolution_request_digest": closure_request_digest,
                    **(
                        {"foreground_parent_continuation": gate_continuation}
                        if gate_continuation is not None
                        else {}
                    ),
                    **(
                        {
                            "post_action_continuation_digest": runtime_publication_checkpoint_value_digest(
                                marker
                            )
                        }
                        if marker is not None
                        else {}
                    ),
                    "tool_call_ids": [call.tool_call_id for call in current_pending.tool_calls],
                    "event_ids": [close_event.id],
                    "referenced_event_ids": [event.id for event in lifecycle_events],
                },
                source_checkpoint=source_checkpoint,
                target_checkpoint=target_checkpoint,
                transcript_messages=transcript_messages,
                events=[close_event],
                referenced_events=lifecycle_events,
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=session.run_epoch,
                expected_transcript_cursor=user_input_transcript_cursor,
            )
            prepared_events = prepared_close.request.events
            if len(prepared_events) != 1:
                raise AssertionError("User-input closure must publish one checkpoint event.")
            close_event = prepared_events[0]
            close_cancellation = await round_owner.commit_continuation_close(
                approval_publication.publish_pending_action_with_exact_replay(
                    prepared_close,
                    session_store=self._session_store,
                    event_writer=self._event_writer,
                    fan_out=False,
                )
            )
            pending_cleared = True
            transcript.extend(transcript_messages)
            await self._event_writer.fan_out_persisted([close_event])
            yield close_event
            if close_cancellation is not None:
                raise close_cancellation

            forwarded_stream = self._continue_closed_tool_round(
                session=session,
                environment=environment,
                registered_provider=registered_provider,
                execution_profile_snapshot=execution_profile_snapshot,
                budget_policy=budget_policy,
                transcript=transcript,
                semantics=invocation_semantics,
                resolution=response,
                task_id=pending.task_id,
                model_step=pending.model_step,
                tool_exposure=pending.tool_exposure,
                run_limit_accounting=continued_run_limit_accounting,
                participant_context=participant_context,
            )
            try:
                async for event in forwarded_stream:
                    yield event
            except GeneratorExit:
                await forwarded_stream.aclose()
                raise
            self._foreground_gate_policy_owner.release(
                session=session, kind="input", action_id=pending.input_id
            )
        except _FinalizedToolEffectRejection:
            raise
        except ForegroundChildActionRequired as exc:
            async for event in self._pause_gate_on_child(
                exc,
                session=session,
                registered_agent=registered_agent,
                registered_environment=environment.registered_environment,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=environment.invocation_context,
            ):
                yield event
        except Exception as exc:
            if not pending_cleared:
                # The pending_user_input checkpoint is still present, so restore the resumable
                # INTERRUPTED state and emit a terminal event for closure (a SESSION_RESUMED was
                # already emitted). The caller can retry resolve_user_input; recorded outcomes
                # prevent re-running a tool that already completed. A tool that started with no
                # terminal (a crash mid-tool) cannot be re-run safely — flag it as needing manual
                # recovery so the retry is not a silent double-execution.
                # Carry the failure so a caller can distinguish "your answer failed, retry" from a
                # fresh pause (whose interrupted event has no error fields).
                checkpoint_at_failure = await self._session_store.load_checkpoint(session.id)
                interrupt_payload = (
                    None
                    if checkpoint_at_failure is None
                    else checkpoint_at_failure.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
                )
                supersession_intent: UserInputSupersessionIntent | None = None
                if type(interrupt_payload) is dict and (
                    USER_INPUT_SUPERSESSION_INTENT_KEY in interrupt_payload
                ):
                    try:
                        supersession_intent = UserInputSupersessionIntent.model_validate(
                            interrupt_payload[USER_INPUT_SUPERSESSION_INTENT_KEY]
                        )
                    except (TypeError, ValueError) as marker_error:
                        raise SessionRuntimePublicationConflict(
                            "External user-input supersession evidence is malformed."
                        ) from marker_error
                    expected_supersession = user_input_supersession_intent_for(
                        pending,
                        resolution_intent=resolution_intent,
                    )
                    if supersession_intent != expected_supersession:
                        raise SessionRuntimePublicationConflict(
                            "External interruption superseded a different user-input answer."
                        ) from exc
                    payload = copy_json_value(
                        interrupt_payload,
                        "pending_session_interrupt",
                    )
                else:
                    payload = {
                        **_continuation_failure_payload(
                            exc,
                            session=session,
                            redactor=self._secret_redactor,
                        ),
                        **tool_round_identity.payload(),
                        "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                        **pending_user_input_interruption_payload(pending),
                    }
                    if isinstance(exc, approval_support.RoundToolManualRecoveryRequired):
                        payload["manual_recovery_required"] = True
                        payload["tool_call_id"] = exc.tool_call_id
                        payload["tool_name"] = exc.tool_name
                    if isinstance(exc, resume_ledger.ToolCallEvidenceConflict):
                        payload[tool_call_evidence.TOOL_EVIDENCE_CONFLICT_PAYLOAD_KEY] = True
                session = await self._session_store.update_status(
                    session.id, SessionStatus.INTERRUPTED
                )
                interrupted_event = Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment.name,
                    payload=payload,
                )
                interrupted_event = event_with_execution_profile_authority(
                    interrupted_event,
                    execution_profile_snapshot.profile,
                )
                if supersession_intent is not None:
                    runtime_fields = tuple(
                        field_name
                        for field_name in (
                            "interruption_request_id",
                            "retry_request_id",
                            "attempt_id",
                        )
                        if type(payload.get(field_name)) is str
                    )
                    interrupted_event = event_with_runtime_payload_authority(
                        interrupted_event,
                        *runtime_fields,
                    )
                    interrupted_event = event_with_user_input_supersession_authority(
                        interrupted_event,
                        supersession_intent,
                    )
                async for event in self._terminal_event_publication.publish_recovered(
                    RecoveryTerminalEventRequest(
                        event=interrupted_event,
                        phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=environment.registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                        invocation_context=environment.invocation_context,
                    )
                ):
                    yield event
                return
            raise
        finally:
            if round_owner is not None:
                round_owner.finish_dispatch()
                round_owner.finish_continuation_timing()

    async def continue_tool_approval_resolution(
        self,
        *,
        request: ToolApprovalRequest,
        session: Session,
        pending_approval: PendingToolApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
        budget_policy: BudgetPolicy | None,
        deferred_messages: list[Message] | None = None,
        emit_resume_event: bool = True,
        enforce_expiry: bool = True,
        claimed_resolution_intent: pending_approval_reader.ApprovalResolutionIntent | None = None,
        recovery_closure_only: bool = False,
        invocation_context: InvocationContext | None = None,
        foreground_gate: GateReplay | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        # Even closure-only approval recovery reconnects its environment before
        # publishing the round, so it must not borrow historical participant authority.
        await self._require_participant_execution(session, participant_context)
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_provider is not registered_provider
            or invocation_context.registered_environment is not registered_environment
            or invocation_context.profile is not execution_profile_snapshot.profile
            or invocation_context.budget_policy is not budget_policy
        ):
            raise RuntimeError("Tool-approval recovery lost frozen invocation authority.")
        tool_round_identity = ToolRoundIdentity(
            tool_round_id=pending_approval.tool_round_id,
            model_step_id=pending_approval.model_step_id,
            model_attempt_id=pending_approval.model_attempt_id,
        )
        pending_approval_cleared = False
        clear_event: Event | None = None
        tool_outcomes: list[runtime_records.ToolCallOutcome] = []
        round_owner: DurableToolRound | None = None
        expired = False
        original_resolution_decision = request.decision
        resolution_request_digest = (
            approval_support.approval_resolution_request_digest(request)
            if foreground_gate is None
            else foreground_gate.resolution_digest
        )
        deferred_messages = (
            []
            if deferred_messages is None
            else [detach_message(message) for message in deferred_messages]
        )
        # Restore the original run's config persisted on the pending approval.
        # Explicit values have already been checked against the frozen profile.
        invocation_semantics = _effective_approval_invocation_semantics(
            request=request,
            pending_approval=pending_approval,
            structured_output=_effective_approval_structured_output(
                structured_output=request.structured_output,
                pending_approval=pending_approval,
            ),
            effective_retry_policy=self._effective_retry_policy,
        )
        effective_max_steps = invocation_semantics.max_steps
        effective_limits = invocation_semantics.limits
        effective_budget_limits = invocation_semantics.budget_limits
        effective_retry_policy = invocation_semantics.retry_policy
        continued_run_limit_accounting = (
            None
            if pending_approval.run_limit_accounting is None or claimed_resolution_intent is None
            else resume_run_limit_accounting_context(
                pending_approval.run_limit_accounting,
                resolved_at=claimed_resolution_intent.pause_resolved_at,
            )
        )
        environment = ContinuationEnvironment(
            lifecycle=self._environment_lifecycle,
            session=session,
            agent=registered_agent,
            profile=execution_profile_snapshot.profile,
            registered_environment=registered_environment,
            invocation_context=invocation_context,
        )
        try:
            transcript_snapshot = await self._session_store.load_transcript_snapshot(session.id)
            try:
                transcript = [
                    detach_message(record.message) for record in transcript_snapshot.records
                ]
                approval_transcript_cursor = transcript_snapshot.cursor
            finally:
                del transcript_snapshot
            approval_events = await self._session_store.load_events(session.id)
            resolved_budget_limits = request_budget_limits_for_session(
                limits=effective_budget_limits,
                agent_name=registered_agent.spec.name,
                causal_budget_id=session.causal_budget_id,
            )
            if continued_run_limit_accounting is not None:
                continued_run_limit_accounting = rebase_run_limit_accounting_context(
                    continued_run_limit_accounting,
                    session_id=session.id,
                    limits=effective_limits,
                    budget_limits=resolved_budget_limits,
                    events=approval_events,
                    # Profile admission permits only an exact restatement of the
                    # frozen invocation semantics. It must not reset the durable
                    # accounting origin merely because the caller restated it.
                    reset_run_limits=False,
                    reset_budgets=False,
                    now=self._clock(),
                )
            history = approval_support.approval_resolution_history(
                events=approval_events,
                approval=pending_approval,
            )
            # Expiry gates the FIRST grant only: a retry of an approval that
            # already has granted or executed activity was authorized
            # in-window before a crash, so coercing it to a denial would
            # contradict the recorded grant (and trip validate_retry_decision).
            if (
                enforce_expiry
                and approval_support.pending_approval_expired(pending_approval, self._clock())
                and not history.has_granted_activity
            ):
                expired = True
                # Captured before the coercion below replaces them on the request.
                requested_decision = request.decision
                triggered_by = request.resolved_by
                assert pending_approval.expires_at is not None
                expired_at_iso = pending_approval.expires_at.isoformat()
                request = ToolApprovalRequest(
                    session_id=request.session_id,
                    task_worker_id=request.task_worker_id,
                    approval_id=request.approval_id,
                    tool_round_id=request.tool_round_id,
                    tool_call_id=request.tool_call_id,
                    decision=ToolApprovalDecision.DENY,
                    reason=f"Tool approval expired at {expired_at_iso}.",
                    metadata=copy_durable_metadata(request.metadata),
                    resolved_by=expiry_resolution_actor(),
                    max_steps=request.max_steps,
                    limits=request.limits,
                    budget_limits=request.budget_limits,
                    retry_policy=request.retry_policy,
                    structured_output=request.structured_output,
                    thinking=request.thinking,
                    loop_policies=request.loop_policies,
                )
            approval_support.validate_retry_decision(
                history=history,
                approval=pending_approval,
                decision=request.decision,
            )
            resolved_by_payload = resolution_actor_payload(request.resolved_by)
            (
                current_checkpoint,
                publication_round,
            ) = await pending_round_reader.load_pending_tool_round(
                self._session_store,
                session.id,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
                runtime_session=session,
            )
            if (
                publication_round is None
                or pending_rounds.pending_tool_round_identity(publication_round)
                != tool_round_identity
            ):
                raise RuntimeError(
                    "Pending approval round changed before publication-scope recovery."
                )
            if publication_round.tool_exposure is not None:
                validate_resolved_tool_exposure_authority(
                    publication_round.tool_exposure,
                    registered_agent.tool_capabilities,
                    catalogue_revision=registered_agent.tool_catalogue.revision,
                )
            # Validate existing evidence before selecting any recovered native
            # result. A missing claim must not hide a contradictory descriptor.
            try:
                recorded_outcomes = approval_support.recorded_tool_outcomes(
                    events=approval_events,
                    approval=pending_approval,
                    staged_terminals=publication_round.staged_terminals,
                )
            except approval_support.ToolApprovalManualRecoveryRequired:
                # An exact native child may settle this missing outcome below.
                # Scope and staged-evidence conflicts must still fail above it.
                recorded_outcomes = {}
            if claimed_resolution_intent is None:
                if history.has_resolution_activity or recorded_outcomes:
                    raise RuntimeError(
                        "Tool approval cannot be retried automatically because prior durable "
                        "resolution activity has no exact resolution request identity."
                    )
                raise RuntimeError(
                    "Tool approval resolution request identity was not durably claimed."
                )
            current_intent = pending_approval_reader.approval_resolution_intent_from_checkpoint(
                current_checkpoint,
                redactor=self._secret_redactor,
            )
            if current_intent != claimed_resolution_intent:
                raise RuntimeError(
                    "Approval resolution intent changed after the approval was claimed."
                )
            if claimed_resolution_intent.decision is not original_resolution_decision:
                raise RuntimeError(
                    "Tool approval was already claimed with a different resolution decision."
                )
            if claimed_resolution_intent.resolution_request_digest is None:
                raise RuntimeError(
                    "Tool approval cannot be retried automatically because its durable "
                    "resolution intent predates exact resolution request identity."
                )
            if (
                not recovery_closure_only
                and claimed_resolution_intent.resolution_request_digest != resolution_request_digest
            ):
                raise RuntimeError(
                    "Tool approval was already claimed with a different resolution request."
                )
            if (
                not recovery_closure_only
                and foreground_gate is None
                and any(
                    (tool := registered_agent.tools.get(call.tool_name)) is not None
                    and tool.child_session_recovery is not None
                    for call in approval_support.pending_round_tool_calls(pending_approval)
                )
            ):
                from cayu.runtime._foreground_gate_continuation import retain_gate_request

                # An exact public retry may restore application-versioned live
                # policies after restart. Retain them before native reattachment
                # can defer to terminal delivery, which requires this owner to
                # select the child outcome. Profile and request identity have
                # already been validated; selection remains delivery-owned.
                await retain_gate_request(
                    self._session_store,
                    session=session,
                    request=request,
                    redactor=self._secret_redactor,
                    policy_owner=self._foreground_gate_policy_owner,
                )
            native_events = await self._recover_terminal_foreground_effects_at_human_gate(
                session=session,
                pending=pending_approval,
                registered_agent=registered_agent,
                registered_environment=environment.registered_environment,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=environment.invocation_context,
            )
            approval_events.extend(native_events)
            for native_event in native_events:
                yield native_event
            recorded_outcomes = approval_support.recorded_tool_outcomes(
                events=approval_events,
                approval=pending_approval,
                staged_terminals=publication_round.staged_terminals,
            )
            terminal_outcomes_cover_round = set(recorded_outcomes) == {
                pending_call.tool_call_id
                for pending_call in approval_support.pending_round_tool_calls(pending_approval)
            }
            if recovery_closure_only:
                if claimed_resolution_intent.decision is not ToolApprovalDecision.APPROVE:
                    raise RuntimeError(
                        "Manual tool approval recovery has no durable approval grant."
                    )
                if not terminal_outcomes_cover_round:
                    raise RuntimeError(
                        "Manual tool approval recovery cannot authorize pending sibling "
                        "execution; retry the exact original approval request."
                    )
                resolution_request_digest = claimed_resolution_intent.resolution_request_digest
            async with contextlib.aclosing(environment.reconnect()) as preparation:
                async for event in preparation:
                    yield event
            if environment.error is not None:
                raise environment.error
            if emit_resume_event:
                yield await self._event_writer.emit(
                    approval_support.resumed_event(
                        session=session,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment.name,
                        approval=pending_approval,
                        decision=request.decision,
                        resolved_by=request.resolved_by,
                        expired=expired,
                    )
                )
            if expired:
                yield await self._event_writer.emit(
                    event_with_runtime_payload_authority(
                        Event(
                            type=EventType.TOOL_CALL_APPROVAL_EXPIRED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment.name,
                            tool_name=pending_approval.tool_name,
                            payload={
                                **tool_round_identity.payload(),
                                "approval_id": pending_approval.approval_id,
                                "tool_call_id": pending_approval.tool_call_id,
                                **(
                                    {
                                        "execution_profile_fingerprint": (
                                            pending_approval.execution_profile_fingerprint
                                        )
                                    }
                                    if pending_approval.execution_profile_fingerprint is not None
                                    else {}
                                ),
                                "expires_at": expired_at_iso,
                                "requested_decision": requested_decision.value,
                                "resolved_by": resolved_by_payload,
                                "triggered_by": resolution_actor_payload(triggered_by),
                            },
                        ),
                        "model_step_id",
                        "model_attempt_id",
                        "tool_round_id",
                        "approval_id",
                        *(
                            ("execution_profile_fingerprint",)
                            if pending_approval.execution_profile_fingerprint is not None
                            else ()
                        ),
                    )
                )

            if request.decision not in {
                ToolApprovalDecision.APPROVE,
                ToolApprovalDecision.DENY,
            }:
                raise ValueError(f"Unsupported tool approval decision: {request.decision}")

            async with contextlib.aclosing(environment.bind()) as preparation:
                async for event in preparation:
                    yield event
            if environment.error is not None:
                raise environment.error

            if request.decision == ToolApprovalDecision.APPROVE:
                run_started_at = time.monotonic()
                limits = copy_run_limits(effective_limits)
                budget_limits = resolved_budget_limits
                if continued_run_limit_accounting is not None:
                    run_started_at, run_baseline, run_budget_authorities = (
                        restore_run_limit_accounting_context(
                            continued_run_limit_accounting,
                            session_id=session.id,
                            budget_limits=budget_limits,
                            now=self._clock(),
                        )
                    )
                else:
                    run_budget_authorities = None
                    run_baseline = (
                        session_usage_summary(session.id, approval_events)
                        if limits.scope == "run" and has_run_limits(limits)
                        else None
                    )
                budget_baseline_events = (
                    approval_events if _has_run_budget_limit(budget_limits) else []
                )
                request_budget_notify_events: list[Event] = []
                recorded_tool_outcomes = list(recorded_outcomes.values())
                pending_tool_calls: list[runtime_records.ToolCallRequest] = []
                executable_pending_tool_calls = 0
                for pending_tool_call in approval_support.pending_round_tool_calls(
                    pending_approval
                ):
                    if pending_tool_call.tool_call_id in recorded_outcomes:
                        continue
                    tool_call = approval_support.tool_call_request_from_pending(pending_tool_call)
                    pending_tool_calls.append(tool_call)
                    policy_evidence = pending_approval_reader.effective_tool_policy_evidence(
                        pending_tool_call
                    )
                    policy_result = approval_support.policy_result_from_pending_tool_call(
                        pending_tool_call
                    )
                    if (
                        policy_evidence is not ToolPolicyEvidence.AUTHORITATIVE
                        or policy_result is None
                        or policy_result.decision == ToolPolicyDecision.DENY
                    ):
                        continue
                    executable_pending_tool_calls += 1
                limit_evaluation = await self._run_limit_controller.evaluate_request_limits(
                    session=session,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment.name,
                    limits=limits,
                    budget_limits=budget_limits,
                    run_started_at=run_started_at,
                    run_baseline=run_baseline,
                    budget_baseline_events=budget_baseline_events,
                    run_budget_authorities=run_budget_authorities,
                    pending_tool_calls=executable_pending_tool_calls,
                    budget_notify_events=request_budget_notify_events,
                    pricing_provider_name=(
                        registered_provider.provider.billing_provider_name
                        or registered_provider.name
                    ),
                    execution_identity=ModelAttemptIdentity(
                        model_step_id=tool_round_identity.model_step_id,
                        model_attempt_id=tool_round_identity.model_attempt_id,
                    ),
                    execution_profile_fingerprint=(execution_profile_snapshot.profile.fingerprint),
                )
                for event in limit_evaluation.events:
                    yield event
                if limit_evaluation.decision is not None:
                    async for event in self._session_finalization.stop_recovered_session_for_limit(
                        RecoveryLimitStopRequest(
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=environment.registered_environment,
                            environment_name=environment.name,
                            decision=limit_evaluation.decision,
                            usage_summary=limit_evaluation.usage_summary,
                            cost_summary=limit_evaluation.cost_summary,
                            messages=transcript,
                            tool_calls=pending_tool_calls,
                            completed_tool_outcomes=recorded_tool_outcomes,
                            pending_approval_to_clear=pending_approval,
                            deferred_messages=deferred_messages,
                            requested_approval_decision=original_resolution_decision,
                            approval_resolution_request_digest=resolution_request_digest,
                            execution_profile=execution_profile_snapshot.profile,
                            invocation_context=environment.invocation_context,
                        )
                    ):
                        yield event
                    return

            pending_round_tool_calls = approval_support.pending_round_tool_calls(pending_approval)
            round_tool_calls = [
                approval_support.tool_call_request_from_pending(pending_tool_call)
                for pending_tool_call in pending_round_tool_calls
            ]
            base_round_redactor = _redactor_for_tool_calls(
                self._secret_redactor,
                registered_agent=registered_agent,
                tool_calls=round_tool_calls,
            )
            legacy_publication_scope = (
                publication_round.assistant_message_state == "quarantined"
                and publication_round.assistant_publication is None
            )
            pause_secret_resolution_scope = invocation_secrets.continuation_secret_resolution_scope(
                pending_approval.secret_resolution_scope,
                environment.registered_environment,
            )
            paused_round = await PausedToolRound.prepare(
                session_store=self._session_store,
                event_writer=self._event_writer,
                invocation=self._tool_invocation,
                clock=self._clock,
                session=session,
                agent=registered_agent,
                environment=environment.registered_environment,
                environment_name=environment.name,
                profile=execution_profile_snapshot.profile,
                invocation_context=environment.invocation_context,
                identity=tool_round_identity,
                task_id=pending_approval.task_id,
                tool_calls=round_tool_calls,
                tool_exposure=publication_round.tool_exposure,
                redactor=base_round_redactor,
                secret_resolution_scope=pause_secret_resolution_scope,
                secret_redactor=self._secret_redactor,
                pause=ApprovalRoundPause(pending_approval.approval_id),
            )
            round_owner = paused_round.round
            await round_owner.admit()

            restarted_staged_ids = await round_owner.fence_restarted_continuation(
                recorded_ids=set(recorded_outcomes),
                resume_undispatched_siblings=(
                    foreground_gate is not None and pause_secret_resolution_scope == "static"
                ),
            )

            for pending_tool_call, tool_call in zip(
                pending_round_tool_calls,
                round_tool_calls,
                strict=True,
            ):
                registered_tool = registered_agent.executable_tool(tool_call.name)
                policy_result = approval_support.policy_result_from_pending_tool_call(
                    pending_tool_call
                )
                policy_evidence = pending_approval_reader.effective_tool_policy_evidence(
                    pending_tool_call
                )
                call_taint_labels = approval_support.taint_labels_from_pending_tool_call(
                    pending_tool_call
                )
                recorded_outcome = recorded_outcomes.get(tool_call.id)
                if recorded_outcome is not None:
                    await round_owner.record_continuation_scope(
                        tool_call.id,
                        execution_scope_unknown=legacy_publication_scope,
                    )
                    tool_outcomes.append(recorded_outcome)
                    continue
                if tool_call.id in restarted_staged_ids:
                    continue

                if policy_evidence is ToolPolicyEvidence.UNEXPOSED:
                    await round_owner.record_continuation_scope(tool_call.id)
                    async for event, outcome in paused_round.reject_policy(
                        tool_call=tool_call,
                        policy_evidence=policy_evidence,
                        requested_decision=request.decision,
                        resolved_by_payload=resolved_by_payload,
                        resolution_reason=request.reason,
                        resolution_metadata=request.metadata,
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                if (
                    policy_evidence is ToolPolicyEvidence.AUTHORITATIVE
                    and policy_result is not None
                    and policy_result.decision == ToolPolicyDecision.DENY
                ):
                    await round_owner.record_continuation_scope(tool_call.id)
                    public_policy_result = approval_support.public_policy_denial_result(
                        secret_resolution_scope=pause_secret_resolution_scope,
                        policy_result=policy_result,
                        publish_arguments=(
                            registered_tool is not None and registered_tool.publish_arguments
                        ),
                    )
                    reason = tool_execution.policy_denial_reason(public_policy_result)
                    result = tool_execution.blocked_tool_result(
                        public_policy_result,
                        reason=reason,
                    )
                    idempotency_key = tool_execution.tool_idempotency_key(
                        session_id=session.id,
                        tool_round_id=tool_round_identity.tool_round_id,
                        tool_call_id=tool_call.id,
                        approval_id=pending_approval.approval_id,
                    )
                    async for (
                        event,
                        outcome,
                    ) in paused_round.publish_result(
                        event=Event(
                            type=EventType.TOOL_CALL_BLOCKED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment.name,
                            tool_name=tool_call.name,
                            payload={
                                **tool_round_identity.payload(),
                                "approval_id": pending_approval.approval_id,
                                "tool_call_id": tool_call.id,
                                "idempotency_key": idempotency_key,
                                **policy_denial_payload_fields(
                                    tool_name=tool_call.name,
                                    denied_by=_TOOL_POLICY_DENIAL_SOURCE,
                                    decision=public_policy_result.decision.value,
                                    reason=reason,
                                    metadata=public_policy_result.metadata,
                                ),
                                "result": result.model_dump(),
                            },
                        ),
                        tool_call=tool_call,
                        result=result,
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                if (
                    policy_evidence is ToolPolicyEvidence.AUTHORITATIVE
                    and policy_result is not None
                    and policy_result.decision == ToolPolicyDecision.REQUIRE_APPROVAL
                    and request.decision == ToolApprovalDecision.APPROVE
                ):
                    yield await self._publish_tool_approval_granted_once(
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment.name,
                        pending_approval=pending_approval,
                        tool_call=tool_call,
                        tool_round_identity=tool_round_identity,
                        reason=request.reason,
                        metadata=request.metadata,
                        resolved_by_payload=resolved_by_payload,
                        durable_events=approval_events,
                    )

                if request.decision == ToolApprovalDecision.DENY:
                    await round_owner.record_continuation_scope(tool_call.id)
                    approval_required = (
                        policy_evidence is ToolPolicyEvidence.AUTHORITATIVE
                        and policy_result is not None
                        and policy_result.decision == ToolPolicyDecision.REQUIRE_APPROVAL
                    ) or (
                        policy_evidence is ToolPolicyEvidence.AMBIGUOUS
                        and tool_call.id == pending_approval.tool_call_id
                    )
                    result = approval_support.approval_denied_tool_result(
                        request,
                        approval=pending_approval,
                        tool_call=tool_call,
                        approval_required=approval_required,
                    )
                    idempotency_key = tool_execution.tool_idempotency_key(
                        session_id=session.id,
                        tool_round_id=tool_round_identity.tool_round_id,
                        tool_call_id=tool_call.id,
                        approval_id=pending_approval.approval_id,
                    )
                    async for (
                        event,
                        outcome,
                    ) in paused_round.publish_result(
                        event=Event(
                            type=EventType.TOOL_CALL_APPROVAL_DENIED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment.name,
                            tool_name=tool_call.name,
                            payload={
                                **tool_round_identity.payload(),
                                "approval_id": pending_approval.approval_id,
                                "tool_call_id": tool_call.id,
                                "idempotency_key": idempotency_key,
                                "approval_required": approval_required,
                                "reason": request.reason,
                                **approval_support.bounded_resolution_metadata_payload(
                                    request.metadata,
                                    redactor=self._secret_redactor,
                                ),
                                "resolved_by": resolved_by_payload,
                                "expired": expired,
                                "result": result.model_dump(),
                            },
                        ),
                        tool_call=tool_call,
                        result=result,
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                if request.decision == ToolApprovalDecision.APPROVE and policy_evidence in {
                    ToolPolicyEvidence.AMBIGUOUS,
                    ToolPolicyEvidence.UNREGISTERED,
                }:
                    await round_owner.record_continuation_scope(tool_call.id)
                    async for (
                        event,
                        outcome,
                    ) in paused_round.reject_policy(
                        tool_call=tool_call,
                        policy_evidence=policy_evidence,
                        requested_decision=request.decision,
                        resolved_by_payload=resolved_by_payload,
                        resolution_reason=request.reason,
                        resolution_metadata=request.metadata,
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                if policy_evidence is not ToolPolicyEvidence.AUTHORITATIVE:
                    raise RuntimeError("Pending tool call has no executable policy authority.")

                async for event, outcome in paused_round.execute(
                    tool_call=tool_call,
                    request_metadata=request.metadata,
                    budget_limits=pending_approval.budget_limits or (),
                    auxiliary_invocation_policy=AuxiliaryInvocationPolicy(
                        limits=effective_limits,
                        retry_policy=effective_retry_policy,
                        accounting=continued_run_limit_accounting,
                    ),
                    model_step=publication_round.model_step,
                    taint_labels=call_taint_labels,
                ):
                    yield event
                    if outcome is not None:
                        tool_outcomes.append(outcome)

            if round_owner.defers_terminals:
                async with contextlib.aclosing(
                    round_owner.publish_continuation(
                        already_published_ids=set(recorded_outcomes),
                        restarted_staged_ids=restarted_staged_ids,
                    )
                ) as terminal_events:
                    async for event, outcome in terminal_events:
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)

            source_checkpoint, durable_round = await pending_round_reader.load_pending_tool_round(
                self._session_store,
                session.id,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
                runtime_session=session,
            )
            if durable_round is None or (
                durable_round.tool_round_id,
                durable_round.model_step_id,
                durable_round.model_attempt_id,
            ) != (
                pending_approval.tool_round_id,
                pending_approval.model_step_id,
                pending_approval.model_attempt_id,
            ):
                raise RuntimeError("Pending approval round changed before atomic closure.")
            final_events = await self._session_store.load_events(session.id)
            final_outcomes = approval_support.recorded_tool_outcomes(
                events=final_events,
                approval=pending_approval,
            )
            lifecycle_event_types = {
                EventType.TOOL_CALL_STARTED,
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_FAILED,
                EventType.TOOL_CALL_BLOCKED,
                EventType.TOOL_CALL_APPROVAL_DENIED,
            }
            lifecycle_events = [
                event
                for event in final_events
                if event.type in lifecycle_event_types
                and event.payload.get("approval_id") == pending_approval.approval_id
                and tool_round_identity.matches_payload(event.payload)
            ]
            ordered_outcomes = [
                final_outcomes[call.tool_call_id] for call in durable_round.tool_calls
            ]
            tool_result_messages = transcript_helpers.tool_result_messages(
                ordered_outcomes,
                tool_round_identity=tool_round_identity,
            )
            transcript_messages = list(tool_result_messages)
            if durable_round.assistant_message_state == "quarantined":
                transcript_messages.insert(
                    0,
                    transcript_helpers.assistant_message_with_projected_tool_arguments(
                        tool_round_recovery.ready_assistant_publication_message(durable_round),
                        ordered_outcomes,
                    ),
                )
            target_checkpoint = approval_support.checkpoint_without_exact_pending_approval_round(
                source_checkpoint,
                approval=pending_approval,
                redactor=self._secret_redactor,
                runtime_session=session,
            )
            from cayu.sessions._foreground_child_checkpoint import gate_close_continuation

            gate_continuation = gate_close_continuation(
                source_checkpoint,
                pending=durable_round,
                publication_id=f"approval-close:{pending_approval.approval_id}",
                metadata=self._secret_redactor.redact_json_values(request.metadata),
            )
            if gate_continuation is not None:
                target_checkpoint["foreground_parent_continuation"] = gate_continuation
            from cayu.sessions._foreground_child_checkpoint import (
                FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY,
                post_action_continuation_for_close,
            )

            marker = post_action_continuation_for_close(
                source_checkpoint,
                session=session,
                wait_checkpoint=(
                    await self._session_store.load_checkpoint(session.parent_session_id)
                    if session.parent_session_id is not None
                    else None
                ),
                close_publication_id=f"approval-close:{pending_approval.approval_id}",
                request_metadata=self._secret_redactor.redact_json_values(request.metadata),
                completed_model_step=_completed_tool_round_model_step(
                    durable_round.model_step, max_steps=effective_max_steps
                ),
            )
            if marker is not None:
                target_checkpoint[FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY] = marker
            clear_event = approval_support.cleared_event(
                session=session,
                agent_name=registered_agent.spec.name,
                environment_name=environment.name,
                approval=pending_approval,
            )
            prepared_close = approval_publication.prepare_approval_publication(
                session_id=session.id,
                publication_id=f"approval-close:{pending_approval.approval_id}",
                kind="approval-close",
                intent={
                    "schema_version": 1,
                    "approval_id": pending_approval.approval_id,
                    "tool_call_id": pending_approval.tool_call_id,
                    **tool_round_identity.payload(),
                    "decision": request.decision.value,
                    "requested_decision": original_resolution_decision.value,
                    "resolution_request_digest": resolution_request_digest,
                    **(
                        {"foreground_parent_continuation": gate_continuation}
                        if gate_continuation is not None
                        else {}
                    ),
                    **(
                        {
                            "post_action_continuation_digest": runtime_publication_checkpoint_value_digest(
                                marker
                            )
                        }
                        if marker is not None
                        else {}
                    ),
                    "tool_call_ids": [call.tool_call_id for call in durable_round.tool_calls],
                    "approval_digest": runtime_publication_checkpoint_value_digest(
                        pending_approval.model_dump(mode="json")
                    ),
                    "pending_round_digest": runtime_publication_checkpoint_value_digest(
                        durable_round.model_dump(mode="json")
                    ),
                    "event_ids": [clear_event.id],
                    "referenced_event_ids": [event.id for event in lifecycle_events],
                },
                source_checkpoint=source_checkpoint,
                target_checkpoint=target_checkpoint,
                transcript_messages=transcript_messages,
                events=[clear_event],
                referenced_events=lifecycle_events,
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=session.run_epoch,
                expected_transcript_cursor=approval_transcript_cursor,
            )
            prepared_events = prepared_close.request.events
            if len(prepared_events) != 1:
                raise AssertionError("Approval closure must publish one checkpoint event.")
            clear_event = prepared_events[0]
            close_cancellation = await round_owner.commit_continuation_close(
                approval_publication.publish_approval_with_exact_replay(
                    prepared_close,
                    session_store=self._session_store,
                    event_writer=self._event_writer,
                    fan_out=False,
                )
            )
            pending_approval_cleared = True
            materialized = await self._deferred_input.materialize_expected(
                session.id,
                deferred_messages,
                cancellation=close_cancellation,
            )
            transcript = materialized.messages
            close_cancellation = materialized.cancellation
            await self._event_writer.fan_out_persisted([clear_event])
            yield clear_event
            if close_cancellation is not None:
                raise close_cancellation

            forwarded_stream = self._continue_closed_tool_round(
                session=session,
                environment=environment,
                registered_provider=registered_provider,
                execution_profile_snapshot=execution_profile_snapshot,
                budget_policy=budget_policy,
                transcript=transcript,
                semantics=invocation_semantics,
                resolution=request,
                task_id=pending_approval.task_id,
                model_step=durable_round.model_step,
                tool_exposure=durable_round.tool_exposure,
                run_limit_accounting=continued_run_limit_accounting,
                participant_context=participant_context,
            )
            try:
                async for event in forwarded_stream:
                    yield event
            except GeneratorExit:
                await forwarded_stream.aclose()
                raise
            self._foreground_gate_policy_owner.release(
                session=session, kind="approval", action_id=pending_approval.approval_id
            )
        except GeneratorExit:
            await self._session_finalization.finalize_abandoned_session_by_id(
                session.id,
                registered_agent=registered_agent,
                registered_environment=environment.registered_environment,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=environment.invocation_context,
            )
            raise
        except ForegroundChildActionRequired as exc:
            async for event in self._pause_gate_on_child(
                exc,
                session=session,
                registered_agent=registered_agent,
                registered_environment=environment.registered_environment,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=environment.invocation_context,
            ):
                yield event
        except Exception as exc:
            if isinstance(exc, approval_support.ToolApprovalManualRecoveryRequired):
                session = await self._session_store.update_status(
                    session.id,
                    SessionStatus.INTERRUPTED,
                )
                async for event in self._terminal_event_publication.publish_recovered(
                    RecoveryTerminalEventRequest(
                        event=Event(
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment.name,
                            payload={
                                **_continuation_failure_payload(
                                    exc,
                                    session=session,
                                    redactor=self._secret_redactor,
                                ),
                                **tool_round_identity.payload(),
                                "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                                **approval_support.bounded_pending_approval_event_payload(
                                    pending_approval,
                                    redactor=self._secret_redactor,
                                ),
                                "approval_id": pending_approval.approval_id,
                                "tool_call_id": exc.tool_call_id,
                                "tool_name": exc.tool_name,
                                "manual_recovery_required": True,
                            },
                        ),
                        phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=environment.registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                        invocation_context=environment.invocation_context,
                    )
                ):
                    yield event
                return

            if not pending_approval_cleared:
                try:
                    clear_event = await self._load_exact_approval_close_event(
                        session_id=session.id,
                        approval=pending_approval,
                        requested_decision=original_resolution_decision,
                        resolution_request_digest=resolution_request_digest,
                    )
                    pending_approval_cleared = clear_event is not None
                except Exception as receipt_error:
                    exc.add_note(
                        "Exact approval-close receipt reconciliation failed; the approval "
                        "remains fail-closed: "
                        f"{type(receipt_error).__name__}: {receipt_error}"
                    )

            if not pending_approval_cleared:
                session = await self._session_store.update_status(
                    session.id,
                    SessionStatus.INTERRUPTED,
                )
                async for event in self._terminal_event_publication.publish_recovered(
                    RecoveryTerminalEventRequest(
                        event=Event(
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment.name,
                            payload={
                                **_continuation_failure_payload(
                                    exc,
                                    session=session,
                                    redactor=self._secret_redactor,
                                ),
                                **tool_round_identity.payload(),
                                "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                                **approval_support.bounded_pending_approval_event_payload(
                                    pending_approval,
                                    redactor=self._secret_redactor,
                                ),
                                **(
                                    {
                                        tool_call_evidence.TOOL_EVIDENCE_CONFLICT_PAYLOAD_KEY: True,
                                    }
                                    if isinstance(exc, resume_ledger.ToolCallEvidenceConflict)
                                    else {}
                                ),
                            },
                        ),
                        phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=environment.registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                        invocation_context=environment.invocation_context,
                    )
                ):
                    yield event
                return

            if clear_event is None:
                raise RuntimeError(
                    "Closed approval failure lost its deterministic closure event."
                ) from exc
            async for event in self._finish_closed_approval_failure(
                request=request,
                task_id=pending_approval.task_id,
                session=session,
                closure_event=clear_event,
                registered_agent=registered_agent,
                registered_environment=environment.registered_environment,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=environment.invocation_context,
            ):
                yield event
        finally:
            if round_owner is not None:
                round_owner.finish_dispatch()
                round_owner.finish_continuation_timing()

    async def _load_exact_approval_close_event(
        self,
        *,
        session_id: str,
        approval: PendingToolApproval,
        requested_decision: ToolApprovalDecision,
        resolution_request_digest: str,
    ) -> Event | None:
        """Load the exact closure event after proving its atomic receipt."""

        receipt = await self._session_store.load_runtime_publication_receipt(
            session_id,
            f"approval-close:{approval.approval_id}",
        )
        if receipt is None:
            return None
        expected_identity = {
            "approval_id": approval.approval_id,
            "tool_call_id": approval.tool_call_id,
            "tool_round_id": approval.tool_round_id,
            "requested_decision": requested_decision.value,
            "resolution_request_digest": resolution_request_digest,
        }
        if receipt.kind != "approval-close" or any(
            receipt.intent.get(key) != value for key, value in expected_identity.items()
        ):
            raise SessionRuntimePublicationConflict(
                "Approval-close receipt conflicts with the claimed resolution request."
            )
        if len(receipt.appended_event_ids) != 1 or receipt.intent.get("event_ids") != list(
            receipt.appended_event_ids
        ):
            raise SessionRuntimePublicationConflict(
                "Approval-close receipt has invalid event evidence."
            )
        event_id = receipt.appended_event_ids[0]
        records = await self._session_store.query_events(
            EventQuery(session_id=session_id, event_id=event_id, limit=2)
        )
        if len(records) != 1 or records[0].event.id != event_id:
            raise SessionRuntimePublicationConflict(
                "Approval-close receipt is missing its exact durable event."
            )
        event = records[0].event
        expected_payload = {
            "model_step_id": approval.model_step_id,
            "model_attempt_id": approval.model_attempt_id,
            "tool_round_id": approval.tool_round_id,
            "checkpoint": pending_approval_reader.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY,
            "approval_id": approval.approval_id,
            "tool_call_id": approval.tool_call_id,
            "cleared": True,
        }
        if (
            event.type is not EventType.SESSION_CHECKPOINTED
            or event.session_id != session_id
            or any(event.payload.get(key) != value for key, value in expected_payload.items())
        ):
            raise SessionRuntimePublicationConflict(
                "Approval-close event conflicts with its durable receipt."
            )
        return copy_event(event)

    async def _approval_task_failure_receipt_is_durable(
        self,
        *,
        task_id: str,
        task_worker_id: str,
        task_handoff_id: str | None,
        session: Session,
        identity: ApprovalTaskFailureIdentity,
    ) -> bool:
        if self._task_store is None or not self._task_store.supports_idempotent_terminalization:
            return False
        receipt = await self._task_store.load_task_terminalization_receipt(
            task_id,
            approval_task_terminalization_idempotency_key(
                task_id=task_id,
                session_id=session.id,
                identity=identity,
            ),
        )
        task = await self._task_store.load_task(task_id)
        return approval_task_failure_receipt_matches(
            receipt=receipt,
            task=task,
            task_id=task_id,
            task_worker_id=task_worker_id,
            task_handoff_id=task_handoff_id,
            session_id=session.id,
            session_instance_id=session.instance_id,
            identity=identity,
        )

    async def _direct_approval_task_failure_is_durable(
        self,
        *,
        task_id: str,
        session: Session,
        identity: ApprovalTaskFailureIdentity,
    ) -> bool:
        if self._task_store is None:
            return False
        task = await load_direct_task_failure_replay(
            self._task_store,
            task_id=task_id,
            session_id=session.id,
            session_instance_id=session.instance_id,
            expected_error=approval_task_failure_payload(
                session_id=session.id,
                identity=identity,
            ),
            claimed_terminalization_idempotency_key=(
                approval_task_terminalization_idempotency_key(
                    task_id=task_id,
                    session_id=session.id,
                    identity=identity,
                )
            ),
        )
        return task is not None

    async def _approval_failure_effect_is_durable(
        self,
        *,
        session: Session,
        identity: ApprovalTaskFailureIdentity,
    ) -> Event | None:
        records = await self._session_store.query_events(
            EventQuery(
                session_id=session.id,
                event_id=approval_failure_event_id(identity, "session_failed"),
                limit=2,
            )
        )
        if not records:
            return None
        if len(records) != 1:
            raise SessionRuntimePublicationConflict(
                "Approval failure has duplicate terminal session evidence."
            )
        event = records[0].event
        expected_payload = approval_task_failure_payload(
            session_id=session.id,
            identity=identity,
        )
        if (
            event.type is not EventType.SESSION_FAILED
            or event.session_id != session.id
            or event.interaction_id is not None
            or event.payload.get("error") != expected_payload["message"]
            or event.payload.get("error_type") != expected_payload["type"]
            or any(
                event.payload.get(field_name) != expected_payload[field_name]
                for field_name in (
                    "approval_id",
                    "tool_round_id",
                    "tool_call_id",
                    "resolution_request_digest",
                )
            )
        ):
            raise SessionRuntimePublicationConflict(
                "Approval failure terminal evidence conflicts with its resolution identity."
            )
        current = await self._session_store.load(session.id)
        if current is None or current.status is not SessionStatus.FAILED:
            raise SessionRuntimePublicationConflict(
                "Approval failure event exists without terminal session state."
            )
        return copy_event(event)

    async def _finish_closed_approval_failure(
        self,
        *,
        request: ToolApprovalRequest,
        task_id: str | None,
        session: Session,
        closure_event: Event,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity,
        invocation_context: InvocationContext | None,
    ) -> AsyncGenerator[Event, None]:
        """Finish one post-close approval failure through exact durable evidence."""

        identity = ApprovalTaskFailureIdentity(
            approval_id=request.approval_id,
            tool_round_id=request.tool_round_id,
            tool_call_id=request.tool_call_id,
            resolution_request_digest=(
                approval_support.approval_resolution_request_digest(request)
            ),
        )
        durable_terminal_event = await self._approval_failure_effect_is_durable(
            session=session,
            identity=identity,
        )
        if durable_terminal_event is not None:
            async for event in self._terminal_event_publication.publish_recovered(
                RecoveryTerminalEventRequest(
                    event=durable_terminal_event,
                    phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                    terminal_event_already_durable=True,
                    yield_durable_terminal_event=False,
                )
            ):
                yield event
            return
        failure_payload = approval_task_failure_payload(
            session_id=session.id,
            identity=identity,
        )
        if task_id is not None:
            if self._task_store is None:
                raise RuntimeError("Attached approval failure requires a task store.")
            if request.task_worker_id is None:
                task = await load_direct_task_failure_replay(
                    self._task_store,
                    task_id=task_id,
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    expected_error=failure_payload,
                    claimed_terminalization_idempotency_key=(
                        approval_task_terminalization_idempotency_key(
                            task_id=task_id,
                            session_id=session.id,
                            identity=identity,
                        )
                    ),
                )
                if task is None:
                    task = await self._task_store.fail_task(
                        task_id,
                        failure_payload,
                        worker_id=None,
                    )
            else:
                task = await _terminalize_claimed_task(
                    self._task_store,
                    approval_task_terminalization_request(
                        task_id=task_id,
                        task_worker_id=request.task_worker_id,
                        task_handoff_id=request.task_handoff_id,
                        session_id=session.id,
                        identity=identity,
                    ),
                )
            task_failed_template = _recovery_task_event(
                RecoveryTaskEventRequest(
                    event_type=EventType.TASK_FAILED,
                    task=task,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                )
            )
            task_failed = task_failed_template.model_copy(
                update={
                    "id": approval_failure_event_id(identity, "task_failed"),
                    "interaction_id": closure_event.interaction_id,
                    "timestamp": closure_event.timestamp,
                    "payload": {
                        **task_failed_template.payload,
                        "approval_id": identity.approval_id,
                        "tool_round_id": identity.tool_round_id,
                        "tool_call_id": identity.tool_call_id,
                        "resolution_request_digest": identity.resolution_request_digest,
                        "failure_type": failure_payload["type"],
                    },
                }
            )
            task_failed = event_with_runtime_generated_id(
                event_with_execution_profile_authority(
                    event_with_runtime_payload_authority(
                        task_failed,
                        "approval_id",
                        "tool_round_id",
                        "tool_call_id",
                    ),
                    execution_profile,
                )
            )
            persisted_task_failed = await self._event_writer.persist_exact_replay(task_failed)
            yield (await self._event_writer.fan_out_persisted([persisted_task_failed]))[0]

        session = await self._session_store.update_status(session.id, SessionStatus.FAILED)
        session_failed = event_with_runtime_generated_id(
            event_with_runtime_payload_authority(
                Event(
                    id=approval_failure_event_id(identity, "session_failed"),
                    type=EventType.SESSION_FAILED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=_environment_name(registered_environment),
                    timestamp=closure_event.timestamp,
                    payload={
                        "error": failure_payload["message"],
                        "error_type": failure_payload["type"],
                        "approval_id": identity.approval_id,
                        "tool_round_id": identity.tool_round_id,
                        "tool_call_id": identity.tool_call_id,
                        "resolution_request_digest": identity.resolution_request_digest,
                    },
                ),
                "approval_id",
                "tool_round_id",
                "tool_call_id",
            )
        )
        async for event in self._terminal_event_publication.publish_recovered(
            RecoveryTerminalEventRequest(
                event=session_failed,
                phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            )
        ):
            yield event

    async def _publish_tool_approval_granted_once(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        pending_approval: PendingToolApproval,
        tool_call: runtime_records.ToolCallRequest,
        tool_round_identity: ToolRoundIdentity,
        reason: str | None,
        metadata: dict[str, Any],
        resolved_by_payload: dict[str, Any] | None,
        durable_events: list[Event],
    ) -> Event:
        existing = [
            event
            for event in durable_events
            if event.type == EventType.TOOL_CALL_APPROVED
            and event.payload.get("approval_id") == pending_approval.approval_id
            and event.payload.get("tool_call_id") == tool_call.id
        ]
        if len(existing) > 1:
            raise resume_ledger.ToolCallEvidenceConflict(
                "Tool approval history contains duplicate approval-granted events."
            )
        if existing:
            # The continuation has already matched the retry against the
            # private immutable request digest, and approval_resolution_history
            # has validated this event's complete approval/round/call
            # descriptor. Reuse that durable grant without reconstructing
            # identity from public, bounded audit fields.
            persisted = existing[0]
        else:
            intended = self._event_writer.prepare(
                event_with_runtime_payload_authority(
                    Event(
                        type=EventType.TOOL_CALL_APPROVED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        tool_name=tool_call.name,
                        payload={
                            **tool_round_identity.payload(),
                            "approval_id": pending_approval.approval_id,
                            "tool_call_id": tool_call.id,
                            **(
                                {
                                    "execution_profile_fingerprint": (
                                        pending_approval.execution_profile_fingerprint
                                    )
                                }
                                if pending_approval.execution_profile_fingerprint is not None
                                else {}
                            ),
                            **_public_resolution_audit_fields(
                                secret_resolution_scope=(pending_approval.secret_resolution_scope),
                                reason=reason,
                                metadata=metadata,
                                redactor=self._secret_redactor,
                            ),
                            "resolved_by": resolved_by_payload,
                        },
                    ),
                    "model_step_id",
                    "model_attempt_id",
                    "tool_round_id",
                    "approval_id",
                    *(
                        ("execution_profile_fingerprint",)
                        if pending_approval.execution_profile_fingerprint is not None
                        else ()
                    ),
                )
            )
            persisted = await self._event_writer.persist_exact_replay(intended)
        await self._event_writer.fan_out_persisted([persisted])
        return persisted

    async def _interrupt_for_resumable_manual_recovery(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity,
        invocation_context: InvocationContext | None = None,
        payload: dict[str, Any],
    ) -> AsyncGenerator[Event, None]:
        """Close a durable or acknowledgement-ambiguous recovery to resumable state."""
        try:
            interrupted = await self._session_store.transition_status(
                session.id,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.INTERRUPTED,
            )
        except SessionStatusConflict:
            current = await self._recovery_ownership.require_session(session.id)
            if current.status not in {SessionStatus.INTERRUPTING, SessionStatus.INTERRUPTED}:
                raise
            # An operator interruption won the status transition. Finalize its
            # durable request so its identity, reason, and cascade are preserved.
            async for event in self._session_finalization.interrupt_recovery(
                RecoveryInterruptionRequest(
                    session=current,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=_environment_name(registered_environment),
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                )
            ):
                yield event
            return
        async for event in self._terminal_event_publication.publish_recovered(
            RecoveryTerminalEventRequest(
                event=event_with_execution_profile_authority(
                    Event(
                        type=EventType.SESSION_INTERRUPTED,
                        session_id=interrupted.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=_environment_name(registered_environment),
                        payload=payload,
                    ),
                    execution_profile,
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                session=interrupted,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            )
        ):
            yield event

    async def recover_user_input(
        self,
        *,
        request: UserInputRecoveryRequest,
        loaded_session: Session,
        session: Session,
        pending: PendingUserInput,
        resolution_intent: UserInputResolutionIntent,
        closure_request_digest: str,
        pending_tool_call: PendingToolCallApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
        budget_policy: BudgetPolicy | None,
        invocation_context: InvocationContext | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_provider is not registered_provider
            or invocation_context.registered_environment is not registered_environment
            or invocation_context.profile is not execution_profile_snapshot.profile
            or invocation_context.budget_policy is not budget_policy
        ):
            raise RuntimeError("User-input manual recovery lost frozen invocation authority.")
        answer_request_digest = user_input_answer_request_digest(request)
        resolution_request_digest = user_input_resolution_request_digest(request)
        if closure_request_digest != resolution_request_digest:
            raise RuntimeError(
                "User-input manual-recovery closure digest conflicts with its request."
            )
        require_resolution_intent_matches_pending(
            resolution_intent,
            pending=pending,
            answer_request_digest=answer_request_digest,
            resolution_stage="manual-recovery",
            resolution_request_digest=resolution_request_digest,
        )
        tool_round_identity = ToolRoundIdentity(
            tool_round_id=pending.tool_round_id,
            model_step_id=pending.model_step_id,
            model_attempt_id=pending.model_attempt_id,
        )
        recovery_prepared = False
        publication = ManualRecoveryPublication(self._event_writer)
        cancellation_baseline = _task_cancellation_count()
        authoritative_failure: BaseException | None = None
        abandoned = False
        environment = ContinuationEnvironment(
            lifecycle=self._environment_lifecycle,
            session=session,
            agent=registered_agent,
            profile=execution_profile_snapshot.profile,
            registered_environment=registered_environment,
            invocation_context=invocation_context,
        )
        try:
            await ToolEffectStateOwner(self._session_store).require_unverified_recovery_allowed(
                session,
                tool_round_id=pending.tool_round_id,
                tool_call_id=pending_tool_call.tool_call_id,
            )
            resolution_intent = await self._admit_user_input_resolution_execution(
                session=session,
                pending=pending,
                resolution_intent=resolution_intent,
            )
            recovered_result = ToolResult(
                content=request.message,
                structured=request.structured,
                artifacts=request.artifacts,
                is_error=request.outcome == ToolApprovalRecoveryOutcome.FAILED,
            )
            recovery_secret_resolution_scope = (
                "unknown"
                if pending.assistant_publication is None
                else pending.assistant_publication.secret_resolution_scope
            )
            public_recovered_result = _public_manual_recovery_result(
                recovered_result,
                secret_resolution_scope=recovery_secret_resolution_scope,
            )
            event_type = (
                EventType.TOOL_CALL_FAILED
                if recovered_result.is_error
                else EventType.TOOL_CALL_COMPLETED
            )
            events = await self._session_store.load_events(session.id)
            approval_support.validate_round_recovery_target(
                events=events,
                pending_calls=pending.tool_calls,
                tool_call_id=request.tool_call_id,
                input_id=pending.input_id,
                tool_round_identity=tool_round_identity,
            )
            async with contextlib.aclosing(environment.reconnect()) as preparation:
                async for event in preparation:
                    yield event
            if environment.error is not None:
                session = await self._session_store.update_status(
                    session.id,
                    SessionStatus.INTERRUPTED,
                )
                async for event in self._terminal_event_publication.publish_recovered(
                    RecoveryTerminalEventRequest(
                        event=event_with_execution_profile_authority(
                            Event(
                                type=EventType.SESSION_INTERRUPTED,
                                session_id=session.id,
                                agent_name=registered_agent.spec.name,
                                environment_name=environment.name,
                                payload={
                                    **tool_round_identity.payload(),
                                    "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                                    **pending_user_input_interruption_payload(pending),
                                    **_environment_factory_resolution_error_payload(
                                        environment.error,
                                        redactor=self._secret_redactor,
                                    ),
                                },
                            ),
                            execution_profile_snapshot.profile,
                        ),
                        phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=environment.registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                        invocation_context=environment.invocation_context,
                    )
                ):
                    yield event
                return
            recovery_tool_event, public_recovered_result = tool_results.redact_tool_result_event(
                event=Event(
                    type=event_type,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment.name,
                    tool_name=pending_tool_call.tool_name,
                    payload={
                        **tool_round_identity.payload(),
                        "tool_call_id": pending_tool_call.tool_call_id,
                        "idempotency_key": tool_execution.tool_idempotency_key(
                            session_id=session.id,
                            tool_round_id=tool_round_identity.tool_round_id,
                            tool_call_id=pending_tool_call.tool_call_id,
                            pause_id=pending.input_id,
                        ),
                        "input_id": pending.input_id,
                        "manual_recovery": True,
                        "resolution_request_digest": resolution_request_digest,
                        **tool_argument_publication.unavailable_argument_projection().payload_fields(),
                        **_public_resolution_audit_fields(
                            secret_resolution_scope=recovery_secret_resolution_scope,
                            reason=request.reason,
                            metadata=request.metadata,
                            redactor=self._secret_redactor,
                        ),
                        "resolved_by": resolution_actor_payload(request.resolved_by),
                        "result": public_recovered_result.model_dump(),
                    },
                ),
                result=public_recovered_result,
                redactor=self._secret_redactor,
            )
            recovery_tool_event = event_with_execution_profile_authority(
                recovery_tool_event,
                execution_profile_snapshot.profile,
            )
            recovery_tool_event = event_with_runtime_payload_authority(
                recovery_tool_event,
                "resolution_request_digest",
            )
            publication.event = recovery_tool_event
            recovery_events = [
                event_with_execution_profile_authority(
                    Event(
                        type=EventType.SESSION_RESUMED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment.name,
                        payload={
                            **tool_round_identity.payload(),
                            "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                            "input_id": pending.input_id,
                            "tool_call_id": pending.tool_call_id,
                            "resolved_by": resolution_actor_payload(request.resolved_by),
                        },
                    ),
                    execution_profile_snapshot.profile,
                ),
                recovery_tool_event,
            ]
            emitted_recovery_events = await publication.persist(session.id, recovery_events)
            await self._event_writer.fan_out_persisted(emitted_recovery_events)
            for event in emitted_recovery_events:
                yield event
            tool_call = approval_support.tool_call_request_from_pending(
                pending_tool_call,
                arguments={},
            )
            tool_event = emitted_recovery_events[-1]
            # Manual recovery persists the operator-supplied result before hooks run, so
            # after_tool_call is observe-only here (v1): the threaded modification is ignored.
            async for event, _modified in self._tool_invocation.hooks.after_call(
                session=session,
                tool_event=tool_event,
                registered_agent=registered_agent,
                registered_environment=environment.registered_environment,
                tool_call=tool_call,
                result=public_recovered_result,
                task_id=pending.task_id,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=environment.invocation_context,
                redactor=self._secret_redactor,
                output_redactor=self._secret_redactor,
                allow_modification=False,
            ):
                yield event
            recovery_prepared = True
        except (GeneratorExit, asyncio.CancelledError) as exc:
            authoritative_failure = exc
            abandoned = True
            raise
        except Exception as exc:
            authoritative_failure = exc
            try:
                reconciliation = await publication.reconcile()
            except BaseException as reconciliation_failure:
                authoritative_failure = reconciliation_failure
                abandoned = (
                    _recovery_abandonment_signal(
                        reconciliation_failure,
                        cancellation_baseline=cancellation_baseline,
                    )
                    is not None
                )
                raise
            if reconciliation.cancellation is not None:
                reconciliation.cancellation.add_note(
                    "Manual user-input recovery append failed while persistence "
                    "reconciliation was running."
                )
                authoritative_failure = reconciliation.cancellation
                abandoned = True
                raise reconciliation.cancellation from exc
            publication.persisted = reconciliation.persisted is True
            persistence_payload = reconciliation.failure_payload(redactor=self._secret_redactor)
            if persistence_payload is not None:
                diagnostic = exception_diagnostic(
                    exc,
                    redactor=self._secret_redactor,
                )
                try:
                    async for event in self._interrupt_for_resumable_manual_recovery(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=environment.registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                        invocation_context=environment.invocation_context,
                        payload={
                            **tool_round_identity.payload(),
                            "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                            **pending_user_input_interruption_payload(pending),
                            "input_id": pending.input_id,
                            "tool_call_id": pending_tool_call.tool_call_id,
                            **persistence_payload,
                            **diagnostic.payload_fields(),
                        },
                    ):
                        yield event
                except BaseException as interruption_failure:
                    authoritative_failure = interruption_failure
                    abandoned = (
                        _recovery_abandonment_signal(
                            interruption_failure,
                            cancellation_baseline=cancellation_baseline,
                        )
                        is not None
                    )
                    raise
                # The original failure is now represented by durable interrupted
                # state. It must not suppress a later fence-release failure.
                authoritative_failure = None
                return
            current_session = await self._recovery_ownership.require_session(session.id)
            if current_session.status in {
                SessionStatus.INTERRUPTING,
                SessionStatus.INTERRUPTED,
            }:
                if current_session.status is SessionStatus.INTERRUPTING:
                    current_session = await self._session_store.update_status(
                        session.id,
                        SessionStatus.INTERRUPTED,
                    )
                async for event in self._session_finalization.interrupt_recovery(
                    RecoveryInterruptionRequest(
                        session=current_session,
                        registered_agent=registered_agent,
                        registered_environment=environment.registered_environment,
                        environment_name=_environment_name(environment.registered_environment),
                        execution_profile=execution_profile_snapshot.profile,
                        invocation_context=environment.invocation_context,
                    )
                ):
                    yield event
                authoritative_failure = None
                return
            await self._session_store.update_status(session.id, loaded_session.status)
            raise
        except BaseExceptionGroup as exc:
            authoritative_failure = exc
            abandoned = (
                _recovery_abandonment_signal(
                    exc,
                    cancellation_baseline=cancellation_baseline,
                )
                is not None
            )
            if publication.persisted and not abandoned:
                async for event in self._session_finalization.interrupt_recovery(
                    RecoveryInterruptionRequest(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=environment.registered_environment,
                        environment_name=_environment_name(environment.registered_environment),
                        execution_profile=execution_profile_snapshot.profile,
                        invocation_context=environment.invocation_context,
                    )
                ):
                    yield event
            raise
        finally:
            if not recovery_prepared:
                await self._recovery_admission.cleanup_recovery_handoff(
                    stream=None,
                    session_id=session.id,
                    registered_agent=registered_agent,
                    registered_environment=environment.registered_environment,
                    authoritative_failure=authoritative_failure,
                    finalize_abandoned=abandoned,
                    release_run_fence=True,
                    execution_profile=execution_profile_snapshot.profile,
                    invocation_context=environment.invocation_context,
                )

        continuation_stream: AsyncGenerator[Event, None] | None = None
        authoritative_failure = None
        abandoned = False
        try:
            from cayu.runtime._foreground_gate_continuation import gate_input_response_from_recovery

            response = gate_input_response_from_recovery(request)
            continuation_stream = self.continue_user_input_resolution(
                response=response,
                participant_context=participant_context,
                session=session,
                pending=pending,
                resolution_intent=resolution_intent,
                resolution_stage="manual-recovery",
                closure_request_digest=closure_request_digest,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=environment.registered_environment,
                execution_profile_snapshot=execution_profile_snapshot,
                budget_policy=budget_policy,
                invocation_context=environment.invocation_context,
                emit_resume_event=False,
            )
            async for event in continuation_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            abandoned = (
                _recovery_abandonment_signal(
                    exc,
                    cancellation_baseline=cancellation_baseline,
                )
                is not None
            )
            raise
        finally:
            await self._recovery_admission.cleanup_recovery_handoff(
                stream=continuation_stream,
                session_id=session.id,
                registered_agent=registered_agent,
                registered_environment=environment.registered_environment,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=abandoned,
                release_run_fence=True,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=environment.invocation_context,
            )

    async def recover_tool_approval(
        self,
        *,
        request: ToolApprovalRecoveryRequest | ToolEffectReconciliationRequest,
        loaded_session: Session,
        session: Session,
        pending_approval: PendingToolApproval,
        pending_tool_call: PendingToolCallApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
        budget_policy: BudgetPolicy | None,
        deferred_messages: list[Message],
        claimed_resolution_intent: pending_approval_reader.ApprovalResolutionIntent | None,
        invocation_context: InvocationContext | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_provider is not registered_provider
            or invocation_context.registered_environment is not registered_environment
            or invocation_context.profile is not execution_profile_snapshot.profile
            or invocation_context.budget_policy is not budget_policy
        ):
            raise RuntimeError("Tool-approval manual recovery lost frozen invocation authority.")
        tool_round_identity = ToolRoundIdentity(
            tool_round_id=pending_approval.tool_round_id,
            model_step_id=pending_approval.model_step_id,
            model_attempt_id=pending_approval.model_attempt_id,
        )
        recovery_prepared = False
        publication = ManualRecoveryPublication(self._event_writer)
        cancellation_baseline = _task_cancellation_count()
        authoritative_failure: BaseException | None = None
        abandoned = False
        environment = ContinuationEnvironment(
            lifecycle=self._environment_lifecycle,
            session=session,
            agent=registered_agent,
            profile=execution_profile_snapshot.profile,
            registered_environment=registered_environment,
            invocation_context=invocation_context,
        )
        try:
            if type(request) is ToolApprovalRecoveryRequest:
                await ToolEffectStateOwner(self._session_store).require_unverified_recovery_allowed(
                    session,
                    tool_round_id=pending_approval.tool_round_id,
                    tool_call_id=pending_tool_call.tool_call_id,
                )
                recovered_result = approval_support.recovered_tool_result(
                    request=request,
                )
                recovery_secret_resolution_scope = pending_approval.secret_resolution_scope
                public_recovered_result = _public_manual_recovery_result(
                    recovered_result,
                    secret_resolution_scope=recovery_secret_resolution_scope,
                )
                event_type = (
                    EventType.TOOL_CALL_FAILED
                    if recovered_result.is_error
                    else EventType.TOOL_CALL_COMPLETED
                )
            # Recovery reconciles an externally executed side effect that was
            # authorized before the crash, so an expired window does not block it
            # (an expired-never-approved approval has no started tool to recover).
            # The out-of-window reconciliation is still stamped for the audit trail.
            recovered_after_expiry = approval_support.pending_approval_expired(
                pending_approval, self._clock()
            )
            events = await self._session_store.load_events(session.id)
            selected_reconciliation = (
                await self._preflight_tool_effect_reconciliation(
                    session=session, request=request, source_run_epoch=loaded_session.run_epoch
                )
                if type(request) is ToolEffectReconciliationRequest
                else None
            )
            if not isinstance(selected_reconciliation, _ReconciledToolEffectReplay):
                approval_support.validate_recovery_target(
                    events=events,
                    approval=pending_approval,
                    tool_call_id=request.tool_call_id,
                )
            async with contextlib.aclosing(environment.reconnect()) as preparation:
                async for event in preparation:
                    yield event
            if environment.error is not None:
                session = await self._session_store.update_status(
                    session.id,
                    SessionStatus.INTERRUPTED,
                )
                async for event in self._terminal_event_publication.publish_recovered(
                    RecoveryTerminalEventRequest(
                        event=Event(
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment.name,
                            payload={
                                **tool_round_identity.payload(),
                                "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                                **approval_support.bounded_pending_approval_event_payload(
                                    pending_approval,
                                    redactor=self._secret_redactor,
                                ),
                                **_environment_factory_resolution_error_payload(
                                    environment.error,
                                    redactor=self._secret_redactor,
                                ),
                                "approval_id": pending_approval.approval_id,
                            },
                        ),
                        phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=environment.registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                        invocation_context=environment.invocation_context,
                    )
                ):
                    yield event
                return
            if type(request) is ToolEffectReconciliationRequest:
                if environment.invocation_context is None:
                    raise RuntimeError(
                        "Approval receipt recovery has no claimed invocation authority."
                    )
                settlement = None
                settlement_stream = self._settle_tool_effect_reconciliation(
                    request=request,
                    source_run_epoch=loaded_session.run_epoch,
                    session=session,
                    invocation_context=environment.invocation_context,
                )
                async with contextlib.aclosing(settlement_stream) as owned_settlement:
                    async for item in owned_settlement:
                        if isinstance(item, Event):
                            yield copy_event(item)
                        else:
                            settlement = item
                if settlement is None:
                    raise RuntimeError("Receipt recovery returned no settlement.")
                if isinstance(settlement, _ToolEffectObservationReplay):
                    publication.event = settlement.event
                    publication.persisted = True
                    await self._event_writer.fan_out_persisted([settlement.event])
                    yield copy_event(settlement.event)
                    async for event in self._interrupt_unresolved_tool_effect(
                        record=settlement.record,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=environment.registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                        invocation_context=environment.invocation_context,
                    ):
                        yield event
                    return
                publication.event = settlement.terminal_event
                publication.persisted = True
                emitted_recovery_events = [settlement.terminal_event]
                public_recovered_result = ToolResult.model_validate(
                    settlement.terminal_event.payload["result"]
                )
            else:
                assert isinstance(request, ToolApprovalRecoveryRequest)
                recovery_tool_event, public_recovered_result = (
                    tool_results.redact_tool_result_event(
                        event=Event(
                            type=event_type,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment.name,
                            tool_name=pending_tool_call.tool_name,
                            payload={
                                **tool_round_identity.payload(),
                                "approval_id": pending_approval.approval_id,
                                "tool_call_id": pending_tool_call.tool_call_id,
                                "idempotency_key": tool_execution.tool_idempotency_key(
                                    session_id=session.id,
                                    tool_round_id=tool_round_identity.tool_round_id,
                                    tool_call_id=pending_tool_call.tool_call_id,
                                    approval_id=pending_approval.approval_id,
                                ),
                                "manual_recovery": True,
                                **tool_argument_publication.unavailable_argument_projection().payload_fields(),
                                **_public_resolution_audit_fields(
                                    secret_resolution_scope=recovery_secret_resolution_scope,
                                    reason=request.reason,
                                    metadata=request.metadata,
                                    redactor=self._secret_redactor,
                                ),
                                "resolved_by": resolution_actor_payload(request.resolved_by),
                                "expired": recovered_after_expiry,
                                "result": public_recovered_result.model_dump(),
                            },
                        ),
                        result=public_recovered_result,
                        redactor=self._secret_redactor,
                    )
                )
                recovery_tool_event = event_with_execution_profile_authority(
                    recovery_tool_event,
                    execution_profile_snapshot.profile,
                )
                publication.event = recovery_tool_event
                recovery_events = [
                    approval_support.resumed_event(
                        session=session,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment.name,
                        approval=pending_approval,
                        decision=ToolApprovalDecision.APPROVE,
                        resolved_by=request.resolved_by,
                        expired=recovered_after_expiry,
                    ),
                    recovery_tool_event,
                ]
                emitted_recovery_events = await publication.persist(session.id, recovery_events)
            await self._event_writer.fan_out_persisted(emitted_recovery_events)
            for event in emitted_recovery_events:
                yield event
            tool_call = approval_support.tool_call_request_from_pending(
                pending_tool_call,
                arguments={},
            )
            tool_event = emitted_recovery_events[-1]
            # Manual recovery persists the operator-supplied result before hooks run, so
            # after_tool_call is observe-only here (v1): the threaded modification is ignored.
            async for event, _modified in self._tool_invocation.hooks.after_call(
                session=session,
                tool_event=tool_event,
                registered_agent=registered_agent,
                registered_environment=environment.registered_environment,
                tool_call=tool_call,
                result=public_recovered_result,
                task_id=pending_approval.task_id,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=environment.invocation_context,
                redactor=self._secret_redactor,
                output_redactor=self._secret_redactor,
                allow_modification=False,
            ):
                yield event
            recovery_prepared = True
        except (GeneratorExit, asyncio.CancelledError) as exc:
            authoritative_failure = exc
            abandoned = True
            raise
        except Exception as exc:
            authoritative_failure = exc
            if type(request) is ToolEffectReconciliationRequest:
                # Finish the claimed recovery through its existing cleanup owner.
                abandoned = True
                raise
            try:
                reconciliation = await publication.reconcile()
            except BaseException as reconciliation_failure:
                authoritative_failure = reconciliation_failure
                abandoned = (
                    _recovery_abandonment_signal(
                        reconciliation_failure,
                        cancellation_baseline=cancellation_baseline,
                    )
                    is not None
                )
                raise
            if reconciliation.cancellation is not None:
                reconciliation.cancellation.add_note(
                    "Manual tool-approval recovery append failed while persistence "
                    "reconciliation was running."
                )
                authoritative_failure = reconciliation.cancellation
                abandoned = True
                raise reconciliation.cancellation from exc
            publication.persisted = reconciliation.persisted is True
            persistence_payload = reconciliation.failure_payload(redactor=self._secret_redactor)
            if persistence_payload is not None:
                diagnostic = exception_diagnostic(
                    exc,
                    redactor=self._secret_redactor,
                )
                try:
                    async for event in self._interrupt_for_resumable_manual_recovery(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=environment.registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                        invocation_context=environment.invocation_context,
                        payload={
                            **tool_round_identity.payload(),
                            "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                            **approval_support.bounded_pending_approval_event_payload(
                                pending_approval,
                                redactor=self._secret_redactor,
                            ),
                            "approval_id": pending_approval.approval_id,
                            "tool_call_id": pending_tool_call.tool_call_id,
                            **persistence_payload,
                            **diagnostic.payload_fields(),
                        },
                    ):
                        yield event
                except BaseException as interruption_failure:
                    authoritative_failure = interruption_failure
                    abandoned = (
                        _recovery_abandonment_signal(
                            interruption_failure,
                            cancellation_baseline=cancellation_baseline,
                        )
                        is not None
                    )
                    raise
                # The original failure is now represented by durable interrupted
                # state. It must not suppress a later fence-release failure.
                authoritative_failure = None
                return
            await self._session_store.update_status(session.id, loaded_session.status)
            raise
        except BaseExceptionGroup as exc:
            authoritative_failure = exc
            abandoned = (
                _recovery_abandonment_signal(
                    exc,
                    cancellation_baseline=cancellation_baseline,
                )
                is not None
            )
            if publication.persisted and not abandoned:
                async for event in self._session_finalization.interrupt_recovery(
                    RecoveryInterruptionRequest(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=environment.registered_environment,
                        environment_name=_environment_name(environment.registered_environment),
                        execution_profile=execution_profile_snapshot.profile,
                        invocation_context=environment.invocation_context,
                    )
                ):
                    yield event
            raise
        finally:
            if not recovery_prepared:
                await self._recovery_admission.cleanup_recovery_handoff(
                    stream=None,
                    session_id=session.id,
                    registered_agent=registered_agent,
                    registered_environment=environment.registered_environment,
                    authoritative_failure=authoritative_failure,
                    finalize_abandoned=abandoned,
                    release_run_fence=True,
                    execution_profile=execution_profile_snapshot.profile,
                    invocation_context=environment.invocation_context,
                )

        continuation_stream: AsyncGenerator[Event, None] | None = None
        authoritative_failure = None
        abandoned = False
        try:
            if type(request) is ToolEffectReconciliationRequest:
                approval_request = ToolApprovalRequest(
                    session_id=request.session_id,
                    task_worker_id=request.task_worker_id,
                    task_handoff_id=request.task_handoff_id,
                    approval_id=pending_approval.approval_id,
                    tool_round_id=request.tool_round_id,
                    tool_call_id=request.tool_call_id,
                    decision=ToolApprovalDecision.APPROVE,
                    max_steps=request.max_steps,
                )
            else:
                assert isinstance(request, ToolApprovalRecoveryRequest)
                approval_request = ToolApprovalRequest(
                    session_id=request.session_id,
                    task_worker_id=request.task_worker_id,
                    task_handoff_id=request.task_handoff_id,
                    approval_id=pending_approval.approval_id,
                    tool_round_id=request.tool_round_id,
                    tool_call_id=request.tool_call_id,
                    decision=ToolApprovalDecision.APPROVE,
                    reason=request.reason,
                    metadata=request.metadata,
                    resolved_by=request.resolved_by,
                    max_steps=request.max_steps,
                    limits=request.limits,
                    budget_limits=request.budget_limits,
                    retry_policy=request.retry_policy,
                    structured_output=request.structured_output,
                    thinking=request.thinking,
                    loop_policies=request.loop_policies,
                )
            continuation_stream = self.continue_tool_approval_resolution(
                request=approval_request,
                participant_context=participant_context,
                session=session,
                pending_approval=pending_approval,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=environment.registered_environment,
                execution_profile_snapshot=execution_profile_snapshot,
                budget_policy=budget_policy,
                invocation_context=environment.invocation_context,
                deferred_messages=deferred_messages,
                emit_resume_event=False,
                enforce_expiry=False,
                claimed_resolution_intent=claimed_resolution_intent,
                recovery_closure_only=True,
            )
            async for event in continuation_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            abandoned = (
                type(request) is ToolEffectReconciliationRequest
                or _recovery_abandonment_signal(
                    exc,
                    cancellation_baseline=cancellation_baseline,
                )
                is not None
            )
            raise
        finally:
            await self._recovery_admission.cleanup_recovery_handoff(
                stream=continuation_stream,
                session_id=session.id,
                registered_agent=registered_agent,
                registered_environment=environment.registered_environment,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=abandoned,
                release_run_fence=True,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=environment.invocation_context,
            )

    async def _claim_manual_tool_round_recovery(
        self,
        *,
        session: Session,
        pending_round: pending_rounds.PendingToolRound,
        pending_tool_call: PendingToolCallApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
        budget_policy: BudgetPolicy | None,
        request_loop_policies: tuple[LoopPolicy, ...],
        after_admission: RecoveryMutationHook | None = None,
    ) -> (
        _IncompleteRecoveryClaim
        | _ManualRecoveryInterruptionFence
        | _ManualRecoveryInterruptionReplay
    ):
        """Claim recovery or fence an operator interruption that won the race."""
        claim_id = str(uuid4())
        run_operation_id = str(uuid4())
        claim_expires_at: datetime | None = None
        claim_run_epoch: int | None = None
        claimed_run_operation_id: str | None = None
        session_before_fence: Session | None = None

        async def reconstruct_claim_invocation_context(
            claimed_session: Session,
            authority: _IncompleteRecoveryClaimAuthority,
        ) -> InvocationContext:
            try:
                return reconstruct_invocation_context(
                    runtime_hooks=self._runtime_hooks,
                    loop_policies=self._loop_policies,
                    session=claimed_session,
                    execution_profile_snapshot=execution_profile_snapshot,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    budget_policy=budget_policy,
                    request_loop_policies=request_loop_policies,
                    recovery_claim_id=authority.claim_id,
                )
            except BaseException as reconstruction_failure:
                failure = reconstruction_failure
                # Claiming is durable before invocation reconstruction. If
                # corrupt metadata or a runtime mismatch prevents rebuilding
                # the context, explicitly abandon or transfer that ownership
                # instead of leaving the caller's exact epoch stranded.
                await self._recovery_ownership.run_cleanup_steps(
                    authoritative_failure=failure,
                    steps=(
                        (
                            "abandoned manual recovery reconstruction finalization",
                            lambda: self._session_finalization.finalize_abandoned_session_by_id(
                                claimed_session.id,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                execution_profile=execution_profile_snapshot.profile,
                                run_terminal_hooks=False,
                            ),
                        ),
                        (
                            "manual recovery reconstruction claim cleanup",
                            lambda: self._recovery_ownership.cleanup_claim(
                                authority=authority,
                                authoritative_failure=failure,
                                execution_profile=execution_profile_snapshot.profile,
                                claim_has_not_dispatched_work=True,
                            ),
                        ),
                    ),
                )
                raise

        def require_matching_pending_call(checkpoint: dict[str, Any] | None) -> None:
            self._reject_approval_owned_tool_round_recovery(checkpoint)
            current_round = pending_round_reader.pending_tool_round_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
                runtime_session=session,
            )
            if current_round is None:
                raise RuntimeError("Session has no pending tool round.")
            if current_round.tool_round_id != pending_round.tool_round_id:
                raise RuntimeError("Pending tool round changed before recovery claimed it.")
            if current_round != pending_round:
                raise RuntimeError("Pending tool round changed before recovery claimed it.")
            current_tool_call = approval_support.round_tool_call_for_recovery(
                pending_calls=current_round.tool_calls,
                tool_call_id=pending_tool_call.tool_call_id,
            )
            if current_tool_call != pending_tool_call:
                raise RuntimeError("Pending tool call changed before recovery claimed it.")

        def claim_checkpoint(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            claimed_at: datetime,
        ) -> dict[str, Any]:
            nonlocal claim_expires_at, claim_run_epoch
            nonlocal claimed_run_operation_id, session_before_fence
            _require_aware_datetime(claimed_at, "manual recovery claim clock")
            pending_operator_interruption = (
                checkpoint is not None and _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY in checkpoint
            )
            interruption_advanced = (
                pending_operator_interruption
                or current_session.status == SessionStatus.INTERRUPTING
                or (
                    current_session.status == SessionStatus.INTERRUPTED
                    and (
                        session.status != SessionStatus.INTERRUPTED
                        or current_session.run_epoch != session.run_epoch
                    )
                )
            )
            if interruption_advanced:
                raise _ManualRecoveryInterrupted(
                    "Session interruption became durable before manual recovery claimed it."
                )
            if (
                checkpoint is not None
                and _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY in checkpoint
            ):
                raise _ManualRecoveryCascadePending(
                    "Session has an incomplete background interruption cascade."
                )
            checkpoint = _checkpoint_without_active_incomplete_recovery_claim(
                checkpoint,
                now=claimed_at,
            )
            require_matching_pending_call(checkpoint)
            claim_expires_at = claimed_at + self._recovery_ownership.claim_lease_duration
            claim_run_epoch = current_session.run_epoch + 1
            session_before_fence = current_session.model_copy(deep=True)
            updated = {} if checkpoint is None else copy_durable_record(checkpoint, "checkpoint")
            updated = checkpoint_with_active_invocation_execution_profile(
                updated,
                session_id=current_session.id,
                interaction_id=execution_profile_snapshot.interaction_id,
                run_epoch=current_session.run_epoch + 1,
                profile=execution_profile_snapshot.profile,
                expected=execution_profile_snapshot,
            )
            updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = {
                "version": 1,
                "claim_id": claim_id,
                "claimed_at": claimed_at.isoformat(),
                "claim_expires_at": claim_expires_at.isoformat(),
                "operation": "manual_tool_round_recovery",
                **pending_rounds.pending_tool_round_identity(pending_round).payload(),
                "tool_call_id": pending_tool_call.tool_call_id,
            }
            existing_operation = _session_run_operation_from_checkpoint(updated)
            if current_session.status == SessionStatus.RUNNING and existing_operation is not None:
                claimed_run_operation_id = existing_operation.operation_id
                return _checkpoint_with_rebased_session_run_operation(
                    updated,
                    previous_run_epoch=current_session.run_epoch,
                    run_epoch=claim_run_epoch,
                )
            claimed_run_operation_id = run_operation_id
            return _checkpoint_with_session_run_operation(
                checkpoint=updated,
                current_session=current_session,
                operation_id=run_operation_id,
            )

        transition_started = time.monotonic()
        transition_task = asyncio.create_task(
            self._recovery_ownership.reserve_and_fence_incomplete_recovery(
                session.id,
                statuses=_TOOL_ROUND_RECOVERABLE_SESSION_STATUSES,
                inactive_for_seconds=None,
                target_status=SessionStatus.RUNNING,
                checkpoint_transform=claim_checkpoint,
            )
        )
        outcome = await await_shielded_task_outcome(transition_task)
        claim_error = outcome.error
        if isinstance(claim_error, SessionRunFenced):
            # The lifecycle command authenticates the complete source snapshot
            # before entering ``claim_checkpoint``. A concurrent operator stop
            # can therefore trip that exact-state fence before the recovery-
            # specific callback observes and classifies the durable signal.
            # Re-read only to recover that positive classification; all other
            # state changes retain the authoritative SessionRunFenced result.
            try:
                current_session = await self._recovery_ownership.require_session(session.id)
                current_checkpoint = await self._session_store.load_checkpoint(session.id)
            except Exception as inspection_failure:
                add_exception_note_safely(
                    claim_error,
                    "Manual recovery fence classification also failed: "
                    f"{type(inspection_failure).__name__}.",
                )
            else:
                if (
                    (
                        current_checkpoint is not None
                        and _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY in current_checkpoint
                    )
                    or current_session.status is SessionStatus.INTERRUPTING
                    or (
                        current_session.status is SessionStatus.INTERRUPTED
                        and (
                            session.status is not SessionStatus.INTERRUPTED
                            or current_session.run_epoch != session.run_epoch
                        )
                    )
                ):
                    claim_error = _ManualRecoveryInterrupted(
                        "Session interruption became durable before manual recovery claimed it."
                    )
                elif (
                    current_checkpoint is not None
                    and _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY in current_checkpoint
                ):
                    claim_error = _ManualRecoveryCascadePending(
                        "Session has an incomplete background interruption cascade."
                    )

        if isinstance(claim_error, _ManualRecoveryInterrupted):
            current_session = await self._recovery_ownership.require_session(session.id)
            current_checkpoint = await self._session_store.load_checkpoint(session.id)
            current_profile = active_invocation_execution_profile_from_checkpoint(
                current_checkpoint
            )
            pending_interrupt = (
                None
                if current_checkpoint is None
                else current_checkpoint.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
            )
            if (
                current_session.status is SessionStatus.INTERRUPTED
                and pending_interrupt is None
                and current_profile is not None
                and current_profile.interaction_id == execution_profile_snapshot.interaction_id
                and current_profile.profile == execution_profile_snapshot.profile
                and active_invocation_execution_profile_matches_session_epoch(
                    current_profile,
                    session_id=current_session.id,
                    run_epoch=current_session.run_epoch,
                )
            ):
                require_matching_pending_call(current_checkpoint)
                # Typed release may retain the profile at the current epoch.
                # Replay completed terminal evidence without taking a new fence:
                # doing so would invent a second interruption for this stop.
                inspection = await self._terminal_evidence.inspect(
                    session=current_session,
                    checkpoint=current_checkpoint,
                )
                terminal_event = inspection.event
                if (
                    terminal_event is not None
                    and inspection.run_operation is None
                    and _incomplete_recovery_claim_from_checkpoint(current_checkpoint) is None
                    and terminal_event.payload.get("interruption_type")
                    == _INTERRUPTION_TYPE_OPERATOR_REQUESTED
                ):
                    return _ManualRecoveryInterruptionReplay(event=copy_event(terminal_event))

        if claim_error is not None:
            if isinstance(claim_error, _ManualRecoveryCascadePending):
                if outcome.cancellation is not None:
                    outcome.cancellation.add_note(
                        "Manual tool-round recovery was blocked by an incomplete "
                        "background interruption cascade."
                    )
                    raise outcome.cancellation from claim_error
                raise claim_error
            if isinstance(claim_error, _ManualRecoveryInterrupted):

                def fence_interruption(
                    _current_session: Session,
                    checkpoint: dict[str, Any] | None,
                    claimed_at: datetime,
                ) -> dict[str, Any]:
                    nonlocal claim_expires_at, claim_run_epoch
                    require_matching_pending_call(checkpoint)
                    _require_aware_datetime(claimed_at, "manual recovery fence clock")
                    claim_expires_at = claimed_at + self._recovery_ownership.claim_lease_duration
                    claim_run_epoch = _current_session.run_epoch + 1
                    updated = (
                        {} if checkpoint is None else copy_durable_record(checkpoint, "checkpoint")
                    )
                    current_profile = active_invocation_execution_profile_from_checkpoint(updated)
                    if (
                        current_profile is None
                        or current_profile.interaction_id
                        != execution_profile_snapshot.interaction_id
                        or current_profile.profile != execution_profile_snapshot.profile
                        or not active_invocation_execution_profile_matches_session_epoch(
                            current_profile,
                            session_id=_current_session.id,
                            run_epoch=_current_session.run_epoch,
                        )
                    ):
                        raise _ManualRecoveryInterrupted(
                            "Active invocation profile changed before the interrupted "
                            "manual recovery was fenced."
                        )
                    updated = checkpoint_with_active_invocation_execution_profile(
                        updated,
                        session_id=_current_session.id,
                        interaction_id=current_profile.interaction_id,
                        run_epoch=claim_run_epoch,
                        profile=current_profile.profile,
                        expected=current_profile,
                    )
                    updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = {
                        "version": 1,
                        "claim_id": claim_id,
                        "claimed_at": claimed_at.isoformat(),
                        "claim_expires_at": claim_expires_at.isoformat(),
                        "operation": "manual_tool_round_interruption_fence",
                        **pending_rounds.pending_tool_round_identity(pending_round).payload(),
                        "tool_call_id": pending_tool_call.tool_call_id,
                    }
                    return _checkpoint_with_rebased_session_run_operation(
                        updated,
                        previous_run_epoch=_current_session.run_epoch,
                        run_epoch=claim_run_epoch,
                    )

                interruption_fence_started = time.monotonic()
                fence_outcome = await await_shielded_task_outcome(
                    asyncio.create_task(
                        self._recovery_ownership.reserve_and_fence_incomplete_recovery(
                            session.id,
                            statuses={
                                SessionStatus.INTERRUPTING,
                                SessionStatus.INTERRUPTED,
                            },
                            inactive_for_seconds=None,
                            checkpoint_transform=fence_interruption,
                        )
                    ),
                    cancellation=outcome.cancellation,
                )
                interruption_error: BaseException | None = fence_outcome.cancellation
                if fence_outcome.error is not None:
                    interruption_error = fence_outcome.cancellation or fence_outcome.error
                    reconciliation_outcome = await await_shielded_task_outcome(
                        asyncio.create_task(
                            self._recovery_ownership.load_owned_incomplete_recovery_claim(
                                session.id,
                                claim_id,
                                expected_run_epoch=claim_run_epoch,
                            )
                        ),
                        cancellation=fence_outcome.cancellation,
                    )
                    reconciliation_cancellation = reconciliation_outcome.cancellation
                    reconciliation_failure = reconciliation_outcome.error
                    if reconciliation_failure is not None:
                        if not isinstance(
                            reconciliation_failure,
                            Exception | asyncio.CancelledError,
                        ):
                            raise reconciliation_failure from fence_outcome.error
                        interruption_error.add_note(
                            "Could not reconcile whether the interrupted manual recovery "
                            "fence committed: "
                            f"{type(reconciliation_failure).__name__}."
                        )
                        if reconciliation_cancellation is not None:
                            reconciliation_cancellation.add_note(
                                "Interrupted manual recovery fence transition also failed: "
                                f"{type(fence_outcome.error).__name__}."
                            )
                            raise reconciliation_cancellation from fence_outcome.error
                        raise fence_outcome.error
                    elif reconciliation_outcome.result is not None:
                        fenced_session = reconciliation_outcome.result
                        if reconciliation_cancellation is not None:
                            interruption_error = reconciliation_cancellation
                            interruption_error.add_note(
                                "Interrupted manual recovery fence transition also failed: "
                                f"{type(fence_outcome.error).__name__}."
                            )
                    else:
                        if reconciliation_cancellation is not None:
                            reconciliation_cancellation.add_note(
                                "Interrupted manual recovery fence transition also failed: "
                                f"{type(fence_outcome.error).__name__}."
                            )
                            raise reconciliation_cancellation from fence_outcome.error
                        raise fence_outcome.error
                else:
                    fenced_session = fence_outcome.result
                if fenced_session is None:
                    raise RuntimeError("Interrupted manual recovery fence returned no session.")
                if claim_expires_at is None or claim_run_epoch is None:
                    raise RuntimeError("Interrupted manual recovery fence persisted no claim.")
                if fenced_session.run_epoch != claim_run_epoch:
                    raise RuntimeError(
                        "Interrupted manual recovery fence returned an unexpected run epoch."
                    )
                run_fence = _activate_owned_session_run_fence(fenced_session)
                authority = _IncompleteRecoveryClaimAuthority(
                    session_id=fenced_session.id,
                    claim_id=claim_id,
                    run_fence=run_fence,
                )
                invocation_context = await reconstruct_claim_invocation_context(
                    fenced_session,
                    authority,
                )
                local_lease_deadline = (
                    interruption_fence_started
                    + self._recovery_ownership.claim_lease_duration.total_seconds()
                )
                try:
                    _require_live_incomplete_recovery_claim_acknowledgement(
                        session_id=fenced_session.id,
                        local_lease_deadline=local_lease_deadline,
                    )
                except _IncompleteRecoveryClaimLost as lease_failure:
                    authoritative_failure = _authoritative_expired_recovery_claim_failure(
                        interruption_error,
                        lease_failure,
                    )
                    await self._recovery_ownership.cleanup_claim(
                        authority=authority,
                        authoritative_failure=authoritative_failure,
                        execution_profile=execution_profile_snapshot.profile,
                        invocation_context=invocation_context,
                        claim_has_not_dispatched_work=True,
                    )
                    if authoritative_failure is not lease_failure:
                        authoritative_failure.add_note(
                            "The interrupted manual recovery fence acknowledgement "
                            "also consumed its complete lease."
                        )
                        raise authoritative_failure from exception_cause(authoritative_failure)
                    raise
                return _ManualRecoveryInterruptionFence(
                    session=fenced_session,
                    claim_id=claim_id,
                    error=interruption_error,
                    invocation_context=invocation_context,
                    authority=authority,
                )

            reconciliation_outcome = await await_shielded_task_outcome(
                asyncio.create_task(
                    self._recovery_ownership.load_owned_incomplete_recovery_claim(
                        session.id,
                        claim_id,
                        expected_run_epoch=claim_run_epoch,
                    )
                ),
                cancellation=outcome.cancellation,
            )
            reconciliation_cancellation = reconciliation_outcome.cancellation
            reconciliation_failure = reconciliation_outcome.error
            if reconciliation_cancellation is None and isinstance(
                reconciliation_failure,
                asyncio.CancelledError,
            ):
                reconciliation_cancellation = reconciliation_failure
            authoritative_failure = reconciliation_cancellation or claim_error
            if reconciliation_failure is not None:
                if not isinstance(reconciliation_failure, Exception | asyncio.CancelledError):
                    raise reconciliation_failure from claim_error
                authoritative_failure.add_note(
                    "Could not reconcile whether the manual tool-round recovery claim "
                    f"committed: {type(reconciliation_failure).__name__}."
                )
            elif reconciliation_outcome.result is not None:
                reconciled_session = reconciliation_outcome.result
                run_fence = _activate_owned_session_run_fence(reconciled_session)
                authority = _IncompleteRecoveryClaimAuthority(
                    session_id=reconciled_session.id,
                    claim_id=claim_id,
                    run_fence=run_fence,
                )
                invocation_context = await reconstruct_claim_invocation_context(
                    reconciled_session,
                    authority,
                )
                await self._recovery_ownership.run_cleanup_steps(
                    authoritative_failure=authoritative_failure,
                    steps=(
                        (
                            "ambiguous manual recovery claim finalization",
                            lambda: self._session_finalization.finalize_abandoned_session_by_id(
                                reconciled_session.id,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                execution_profile=execution_profile_snapshot.profile,
                                invocation_context=invocation_context,
                            ),
                        ),
                        (
                            "ambiguous manual recovery claim cleanup",
                            lambda: self._recovery_ownership.cleanup_claim(
                                authority=authority,
                                authoritative_failure=authoritative_failure,
                                execution_profile=execution_profile_snapshot.profile,
                                invocation_context=invocation_context,
                                claim_has_not_dispatched_work=True,
                            ),
                        ),
                    ),
                )
            if reconciliation_cancellation is not None:
                reconciliation_cancellation.add_note(
                    "Manual tool-round recovery claim transition also failed: "
                    f"{type(claim_error).__name__}."
                )
                raise reconciliation_cancellation from claim_error
            raise claim_error
        claimed_session = outcome.result
        if claimed_session is None:
            raise RuntimeError("Manual tool-round recovery claim returned no session.")

        # The durable transition ran in a shielded child task. Bind its epoch to
        # the caller that will perform recovery writes and eventual cleanup.
        run_fence = _activate_owned_session_run_fence(claimed_session)
        await self._session_control.execution_presence.ensure(claimed_session)
        authority = _IncompleteRecoveryClaimAuthority(
            session_id=claimed_session.id,
            claim_id=claim_id,
            run_fence=run_fence,
        )
        invocation_context = await reconstruct_claim_invocation_context(
            claimed_session,
            authority,
        )
        if (
            claim_expires_at is None
            or claim_run_epoch is None
            or claimed_run_operation_id is None
            or session_before_fence is None
            or claimed_session.run_epoch != claim_run_epoch
        ):
            invariant_failure = RuntimeError(
                "Manual tool-round recovery transition did not persist its claim."
            )
            await self._recovery_ownership.run_cleanup_steps(
                authoritative_failure=invariant_failure,
                steps=(
                    (
                        "abandoned manual recovery finalization",
                        lambda: self._session_finalization.finalize_abandoned_session_by_id(
                            claimed_session.id,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            execution_profile=execution_profile_snapshot.profile,
                            invocation_context=invocation_context,
                        ),
                    ),
                    (
                        "manual recovery claim cleanup",
                        lambda: self._recovery_ownership.cleanup_claim(
                            authority=authority,
                            authoritative_failure=invariant_failure,
                            execution_profile=execution_profile_snapshot.profile,
                            invocation_context=invocation_context,
                            claim_has_not_dispatched_work=True,
                        ),
                    ),
                ),
            )
            raise invariant_failure

        claim = _IncompleteRecoveryClaim(
            claim_id=claim_id,
            claim_expires_at=claim_expires_at,
            local_lease_deadline=(
                transition_started + self._recovery_ownership.claim_lease_duration.total_seconds()
            ),
            session_before_fence=session_before_fence,
            session=claimed_session,
            run_operation=_SessionRunOperation(
                operation_id=claimed_run_operation_id,
                run_epoch=claimed_session.run_epoch,
            ),
            invocation_context=invocation_context,
            authority=authority,
        )
        try:
            _require_live_incomplete_recovery_claim_acknowledgement(
                session_id=claimed_session.id,
                local_lease_deadline=claim.local_lease_deadline,
            )
        except _IncompleteRecoveryClaimLost as lease_failure:
            authoritative_failure = _authoritative_expired_recovery_claim_failure(
                outcome.cancellation,
                lease_failure,
            )
            await self._recovery_ownership.cleanup_claim(
                authority=claim.require_authority(),
                authoritative_failure=authoritative_failure,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=invocation_context,
                claim_has_not_dispatched_work=True,
            )
            if outcome.cancellation is not None:
                outcome.cancellation.add_note(
                    "The manual recovery claim acknowledgement also consumed its complete lease."
                )
                raise outcome.cancellation from exception_cause(outcome.cancellation)
            raise
        if after_admission is not None:
            try:
                await after_admission()
            except BaseException as authority_failure:
                failure = authority_failure
                await self._recovery_ownership.run_cleanup_steps(
                    authoritative_failure=failure,
                    steps=(
                        (
                            "abandoned manual recovery finalization",
                            lambda: self._session_finalization.finalize_abandoned_session_by_id(
                                claimed_session.id,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                execution_profile=execution_profile_snapshot.profile,
                                invocation_context=invocation_context,
                                run_terminal_hooks=False,
                            ),
                        ),
                        (
                            "manual recovery claim cleanup",
                            lambda: self._recovery_ownership.cleanup_claim(
                                authority=authority,
                                authoritative_failure=failure,
                                execution_profile=execution_profile_snapshot.profile,
                                invocation_context=invocation_context,
                                claim_has_not_dispatched_work=True,
                            ),
                        ),
                    ),
                )
                raise
        if outcome.cancellation is None:
            return claim

        await self._recovery_ownership.run_cleanup_steps(
            authoritative_failure=outcome.cancellation,
            steps=(
                (
                    "abandoned manual recovery finalization",
                    lambda: self._session_finalization.finalize_abandoned_session_by_id(
                        claimed_session.id,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                        invocation_context=invocation_context,
                    ),
                ),
                (
                    "manual recovery claim cleanup",
                    lambda: self._recovery_ownership.cleanup_claim(
                        authority=authority,
                        authoritative_failure=outcome.cancellation,
                        execution_profile=execution_profile_snapshot.profile,
                        invocation_context=invocation_context,
                        claim_has_not_dispatched_work=True,
                    ),
                ),
            ),
        )
        raise outcome.cancellation

    async def recover_tool_round(
        self,
        *,
        request: ToolRoundRecoveryRequest | ToolEffectReconciliationRequest,
        loaded_session: Session,
        pending_round: pending_rounds.PendingToolRound,
        pending_tool_call: PendingToolCallApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        invocation_semantics: _RecoveryInvocationSemantics,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
        budget_policy: BudgetPolicy | None,
        after_admission: RecoveryMutationHook | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Claim one manual recovery durably and stream its owned continuation."""
        caller_runtime_task = asyncio.current_task()
        interrupted_baseline = await self._session_control.latest_interrupted_event(
            loaded_session.id
        )
        interrupted_baseline_id = None if interrupted_baseline is None else interrupted_baseline.id
        claim = await self._claim_manual_tool_round_recovery(
            session=loaded_session,
            pending_round=pending_round,
            pending_tool_call=pending_tool_call,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            execution_profile_snapshot=execution_profile_snapshot,
            budget_policy=budget_policy,
            request_loop_policies=(
                request.loop_policies if type(request) is ToolRoundRecoveryRequest else ()
            ),
            after_admission=after_admission,
        )
        if isinstance(claim, _ManualRecoveryInterruptionReplay):
            yield copy_event(claim.event)
            return
        invocation_context = claim.invocation_context
        if invocation_context is None:
            raise RuntimeError("Manual recovery claim lost its invocation context.")
        if isinstance(claim, _ManualRecoveryInterruptionFence):
            authoritative_failure = claim.error
            interruption_request = RecoveryInterruptionRequest(
                session=claim.session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=_environment_name(registered_environment),
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=invocation_context,
            )

            async def finalize_owned_interruption() -> None:
                async for _ in self._session_finalization.interrupt_recovery(interruption_request):
                    pass

            try:
                if claim.error is not None:
                    # Reconciliation proved this exact fence committed. Finish
                    # its operator interruption directly even when the session
                    # was already INTERRUPTED; the generic abandoned-session
                    # finalizer deliberately ignores terminal statuses.
                    await self._recovery_ownership.run_cleanup_steps(
                        authoritative_failure=claim.error,
                        steps=(
                            (
                                "interrupted manual recovery finalization",
                                finalize_owned_interruption,
                            ),
                        ),
                    )
                    raise claim.error
                async for event in self._session_finalization.interrupt_recovery(
                    interruption_request
                ):
                    yield event
            except BaseException as exc:
                authoritative_failure = exc
                raise
            finally:
                await self._recovery_ownership.run_cleanup_steps(
                    authoritative_failure=authoritative_failure,
                    steps=(
                        (
                            "interrupted manual recovery claim release",
                            lambda: self._recovery_ownership.cleanup_claim(
                                authority=claim.authority,
                                authoritative_failure=authoritative_failure,
                                execution_profile=execution_profile_snapshot.profile,
                                invocation_context=invocation_context,
                                claim_has_not_dispatched_work=True,
                            ),
                        ),
                    ),
                )
            return
        stop_heartbeat = asyncio.Event()
        heartbeat_task = asyncio.create_task(
            self._recovery_ownership.heartbeat(
                session_id=claim.session.id,
                claim_id=claim.claim_id,
                local_lease_deadline=claim.local_lease_deadline,
                stop=stop_heartbeat,
            )
        )
        stop_interruption_watch = asyncio.Event()
        interruption_watch_task = asyncio.create_task(
            self._recovery_ownership.watch_manual_recovery_interruption(
                session_id=claim.session.id,
                interrupted_baseline_id=interrupted_baseline_id,
                stop=stop_interruption_watch,
            )
        )
        recover_claimed = (
            self._reconcile_tool_effect_claimed
            if type(request) is ToolEffectReconciliationRequest
            else self._recover_tool_round_claimed
        )
        recovery_stream = recover_claimed(
            request=request,
            participant_context=participant_context,
            loaded_session=claim.session_before_fence,
            session=claim.session,
            run_operation=claim.run_operation,
            pending_round=pending_round,
            pending_tool_call=pending_tool_call,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            invocation_semantics=invocation_semantics,
            execution_profile_snapshot=execution_profile_snapshot,
            budget_policy=budget_policy,
            invocation_context=invocation_context,
        )
        deliveries: asyncio.Queue[_ManualRecoveryEventDelivery | _ManualRecoveryStreamOutcome] = (
            asyncio.Queue(maxsize=2)
        )
        consumer_stopped = asyncio.Event()
        supervisor_started = asyncio.Event()
        consumer_stop_failure: BaseException | None = None
        forwarded_interrupted_event_ids: set[str] = set()

        def heartbeat_failure() -> BaseException | None:
            if not heartbeat_task.done():
                return None
            if heartbeat_task.cancelled():
                return _IncompleteRecoveryClaimLost(
                    "Manual tool-round recovery claim heartbeat was cancelled unexpectedly."
                )
            failure = heartbeat_task.exception()
            if failure is not None:
                return failure
            return _IncompleteRecoveryClaimLost(
                "Manual tool-round recovery claim heartbeat stopped unexpectedly."
            )

        def interruption_watch_failure() -> BaseException | None:
            if not interruption_watch_task.done():
                return None
            if interruption_watch_task.cancelled():
                return RuntimeError(
                    "Manual tool-round recovery interruption watcher was cancelled unexpectedly."
                )
            failure = interruption_watch_task.exception()
            if failure is not None:
                return failure
            if interruption_watch_task.result():
                return asyncio.CancelledError(
                    "Manual tool-round recovery was interrupted by a durable request."
                )
            return RuntimeError(
                "Manual tool-round recovery interruption watcher stopped unexpectedly."
            )

        async def stop_claim_heartbeat() -> None:
            stop_heartbeat.set()
            await heartbeat_task

        async def stop_interruption_watcher() -> None:
            stop_interruption_watch.set()
            await interruption_watch_task

        async def forward_recovery_events() -> None:
            async for event in recovery_stream:
                delivery = _ManualRecoveryEventDelivery(
                    event=event,
                    consumed=asyncio.Event(),
                )
                await deliveries.put(delivery)
                if event.type == EventType.SESSION_INTERRUPTED:
                    forwarded_interrupted_event_ids.add(event.id)
                await delivery.consumed.wait()
                if consumer_stopped.is_set():
                    raise asyncio.CancelledError

        async def supervise_recovery() -> _ManualRecoverySupervisorResult:
            recovery_task = asyncio.create_task(forward_recovery_events())
            supervisor_runtime_task = asyncio.current_task()
            authoritative_failure: BaseException | None = None
            cleanup_failure: BaseException | None = None
            durable_interruption_observed = False
            recovery_transition_fenced = False
            recovery_worker_quiescent = False
            recovery_handoff_quiescent = False
            if supervisor_runtime_task is not None:
                self._session_control.register_active_control_task(
                    claim.session.id,
                    supervisor_runtime_task,
                )
            supervisor_started.set()

            async def stop_recovery_worker() -> None:
                nonlocal recovery_transition_fenced, recovery_worker_quiescent
                try:
                    if not recovery_task.done() and not recovery_task.cancelling():
                        recovery_task.cancel()
                    await asyncio.gather(recovery_task, return_exceptions=True)
                finally:
                    recovery_worker_quiescent = recovery_task.done()
                if recovery_task.cancelled():
                    try:
                        recovery_task.result()
                    except asyncio.CancelledError as child_cancellation:
                        child_failure = exception_cause(child_cancellation)
                        if child_failure is None:
                            return
                        recovery_transition_fenced = any(
                            isinstance(candidate, SessionRunFenced)
                            for candidate in iter_exception_tree(child_failure)
                        )
                        raise child_failure from None
                child_failure = recovery_task.exception()
                if child_failure is not None:
                    recovery_transition_fenced = any(
                        isinstance(candidate, SessionRunFenced)
                        for candidate in iter_exception_tree(child_failure)
                    )
                    raise child_failure

            async def settle_recovery_handoff() -> None:
                nonlocal recovery_handoff_quiescent
                await self._recovery_admission.cleanup_recovery_handoff(
                    stream=recovery_stream,
                    session_id=claim.session.id,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    authoritative_failure=authoritative_failure,
                    finalize_abandoned=(
                        authoritative_failure is not None and not recovery_transition_fenced
                    ),
                    release_run_fence=False,
                    execution_profile=execution_profile_snapshot.profile,
                    invocation_context=invocation_context,
                )
                recovery_handoff_quiescent = True

            try:
                done, _pending = await asyncio.wait(
                    {recovery_task, heartbeat_task, interruption_watch_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if interruption_watch_task in done:
                    failure = interruption_watch_failure()
                    if failure is None:  # pragma: no cover - task completion is exhaustive.
                        raise AssertionError(
                            "Recovery interruption watcher completed without an outcome."
                        )
                    durable_interruption_observed = (
                        not interruption_watch_task.cancelled()
                        and interruption_watch_task.exception() is None
                        and interruption_watch_task.result()
                    )
                    raise failure
                if heartbeat_task in done:
                    failure = heartbeat_failure()
                    if failure is None:  # pragma: no cover - task completion is exhaustive.
                        raise AssertionError("Recovery heartbeat completed without an outcome.")
                    raise failure
                recovery_task.result()
            except BaseException as exc:
                authoritative_failure = (
                    consumer_stop_failure
                    if (
                        consumer_stopped.is_set()
                        and consumer_stop_failure is not None
                        and isinstance(exc, asyncio.CancelledError)
                    )
                    else exc
                )

            try:
                try:
                    cleanup_failures = await self._recovery_ownership.run_cleanup_steps(
                        authoritative_failure=authoritative_failure,
                        steps=(
                            ("manual tool-round recovery event worker stop", stop_recovery_worker),
                            RecoveryCleanupStep(
                                "manual tool-round recovery interruption watcher stop",
                                stop_interruption_watcher,
                                independent_with_previous=True,
                            ),
                            (
                                "manual tool-round recovery handoff cleanup",
                                settle_recovery_handoff,
                            ),
                            ("manual tool-round recovery heartbeat stop", stop_claim_heartbeat),
                            (
                                "manual tool-round recovery claim release",
                                lambda: self._recovery_ownership.cleanup_claim(
                                    authority=claim.require_authority(),
                                    authoritative_failure=authoritative_failure,
                                    execution_profile=execution_profile_snapshot.profile,
                                    invocation_context=invocation_context,
                                    recovery_work_quiescent=(
                                        recovery_worker_quiescent and recovery_handoff_quiescent
                                    ),
                                ),
                            ),
                        ),
                    )
                    if cleanup_failures:
                        cleanup_failure = BaseExceptionGroup(
                            "Manual recovery cleanup failed",
                            [failure for _operation, failure in cleanup_failures],
                        )
                except BaseException as cleanup_error:
                    cleanup_failure = cleanup_error
                    authoritative_failure = cleanup_error
                interrupted_event: Event | None = None
                if durable_interruption_observed and cleanup_failure is None:
                    try:
                        candidate = await self._session_control.wait_for_interrupted_event(
                            claim.session.id
                        )
                    except BaseException as lookup_failure:
                        if authoritative_failure is not None:
                            authoritative_failure.add_note(
                                "The durable operator interruption was finalized, but its "
                                "terminal event could not be reconstructed."
                            )
                            _prepend_exception_cause(authoritative_failure, lookup_failure)
                        else:  # pragma: no cover - the watcher always supplies a failure.
                            authoritative_failure = lookup_failure
                    else:
                        if (
                            candidate is not None
                            and candidate.id != interrupted_baseline_id
                            and candidate.payload.get("interruption_type")
                            == _INTERRUPTION_TYPE_OPERATOR_REQUESTED
                        ):
                            if candidate.id not in forwarded_interrupted_event_ids:
                                interrupted_event = candidate
                            authoritative_failure = None
                await deliveries.put(
                    _ManualRecoveryStreamOutcome(
                        error=authoritative_failure,
                        interrupted_event=interrupted_event,
                    )
                )
            finally:
                if supervisor_runtime_task is not None:
                    self._session_control.unregister_active_control_task(
                        claim.session.id,
                        supervisor_runtime_task,
                    )
            return _ManualRecoverySupervisorResult(
                error=authoritative_failure,
                cleanup_failure=cleanup_failure,
            )

        supervisor_task = asyncio.create_task(supervise_recovery())
        supervisor_start_outcome = await await_shielded_task_outcome(
            asyncio.create_task(supervisor_started.wait())
        )
        if caller_runtime_task is not None:
            # CayuApp reserves the caller task before the durable claim. Once
            # the supervisor is live, transfer process-local ownership so an
            # operator interrupt targets one recovery layer rather than both.
            self._session_control.unregister_active_task(
                claim.session.id,
                caller_runtime_task,
            )
        pending_delivery: _ManualRecoveryEventDelivery | None = None
        authoritative_failure: BaseException | None = None

        async def stop_supervisor() -> None:
            nonlocal consumer_stop_failure
            consumer_stop_failure = authoritative_failure
            consumer_stopped.set()
            if pending_delivery is not None:
                pending_delivery.consumed.set()
            if not supervisor_task.done():
                supervisor_task.cancel()
            await asyncio.gather(supervisor_task, return_exceptions=True)
            if isinstance(consumer_stop_failure, GeneratorExit):
                supervisor_result = supervisor_task.result()
                if supervisor_result.cleanup_failure is not None:
                    raise supervisor_result.cleanup_failure

        try:
            if supervisor_start_outcome.error is not None:
                raise supervisor_start_outcome.error
            if supervisor_start_outcome.cancellation is not None:
                raise supervisor_start_outcome.cancellation
            while True:
                item = await deliveries.get()
                if isinstance(item, _ManualRecoveryStreamOutcome):
                    if item.error is not None:
                        raise item.error
                    if item.interrupted_event is not None:
                        yield item.interrupted_event
                    return
                pending_delivery = item
                yield item.event
                item.consumed.set()
                pending_delivery = None
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            await self._recovery_ownership.run_cleanup_steps(
                authoritative_failure=authoritative_failure,
                steps=(("manual tool-round recovery supervisor stop", stop_supervisor),),
            )

    async def _recover_tool_round_claimed(
        self,
        *,
        request: ToolRoundRecoveryRequest | ToolEffectReconciliationRequest,
        loaded_session: Session,
        session: Session,
        run_operation: _SessionRunOperation | None,
        pending_round: pending_rounds.PendingToolRound,
        pending_tool_call: PendingToolCallApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        invocation_semantics: _RecoveryInvocationSemantics,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
        budget_policy: BudgetPolicy | None,
        invocation_context: InvocationContext | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Persist one operator-verified ordinary tool outcome and continue safely."""
        if type(request) is not ToolRoundRecoveryRequest:
            raise TypeError("Manual recovery requires an exact operator result request.")
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_provider is not registered_provider
            or invocation_context.registered_environment is not registered_environment
            or invocation_context.profile is not execution_profile_snapshot.profile
            or invocation_context.budget_policy is not budget_policy
        ):
            raise RuntimeError("Manual tool-round recovery lost frozen invocation authority.")
        if run_operation is None:
            raise RuntimeError("Manual tool-round recovery has no durable run operation.")
        await ToolEffectStateOwner(self._session_store).require_unverified_recovery_allowed(
            session,
            tool_round_id=pending_round.tool_round_id,
            tool_call_id=pending_tool_call.tool_call_id,
        )
        recovered_result = ToolResult(
            content=request.message,
            structured=request.structured,
            artifacts=request.artifacts,
            is_error=request.outcome == ToolApprovalRecoveryOutcome.FAILED,
        )
        recovery_secret_resolution_scope = (
            pending_approval_reader.tool_round_secret_resolution_scope(pending_round)
        )
        public_recovered_result = _public_manual_recovery_result(
            recovered_result,
            secret_resolution_scope=recovery_secret_resolution_scope,
        )
        event_type = (
            EventType.TOOL_CALL_FAILED
            if recovered_result.is_error
            else EventType.TOOL_CALL_COMPLETED
        )
        environment_name = _environment_name(registered_environment)
        recovery_persisted = False
        cancellation_baseline = _task_cancellation_count()
        recovery_event_to_reconcile: Event | None = None

        try:
            events = await self._session_store.load_events(session.id)
            (
                isolated_dispatched_ids,
                isolated_call_ids,
            ) = await self._pending_tool_round_recovery.isolated_tool_dispatch_ids(
                session=session,
                pending_round=pending_round,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
            tool_round_recovery.validate_tool_round_recovery_target(
                events=events,
                pending_round=pending_round,
                tool_call_id=request.tool_call_id,
                execution_started=(
                    request.tool_call_id in isolated_dispatched_ids
                    if request.tool_call_id in isolated_call_ids
                    else None
                ),
            )
            factory_started_event = await self._environment_lifecycle.emit_factory_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=invocation_context,
            )
            if factory_started_event is not None:
                yield factory_started_event
            factory_resolution = await self._environment_lifecycle.resolve_factory(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=factory_started_event,
                operation=EnvironmentFactoryOperation.RECONNECT,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=invocation_context,
            )
            registered_environment = factory_resolution.registered_environment
            if registered_environment is not None and invocation_context is not None:
                invocation_context = invocation_context.with_registered_environment(
                    registered_environment,
                    validated_profile=execution_profile_snapshot.profile,
                )
            environment_name = _environment_name(registered_environment)
            for event in factory_resolution.events:
                yield event
            if factory_resolution.error is not None:
                async for event in self._interrupt_for_resumable_manual_recovery(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile_snapshot.profile,
                    invocation_context=invocation_context,
                    payload={
                        "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                        **pending_rounds.pending_tool_round_identity(pending_round).payload(),
                        **_environment_factory_resolution_error_payload(
                            factory_resolution.error,
                            redactor=self._secret_redactor,
                        ),
                    },
                ):
                    yield event
                return
            recovery_tool_event, public_recovered_result = tool_results.redact_tool_result_event(
                event=Event(
                    type=event_type,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=pending_tool_call.tool_name,
                    payload={
                        **pending_rounds.pending_tool_round_identity(pending_round).payload(),
                        "tool_call_id": pending_tool_call.tool_call_id,
                        "idempotency_key": tool_execution.tool_idempotency_key(
                            session_id=session.id,
                            tool_round_id=pending_round.tool_round_id,
                            tool_call_id=pending_tool_call.tool_call_id,
                        ),
                        "manual_recovery": True,
                        **tool_argument_publication.unavailable_argument_projection().payload_fields(),
                        **_public_resolution_audit_fields(
                            secret_resolution_scope=recovery_secret_resolution_scope,
                            reason=request.reason,
                            metadata=request.metadata,
                            redactor=self._secret_redactor,
                        ),
                        "resolved_by": resolution_actor_payload(request.resolved_by),
                        "result": public_recovered_result.model_dump(),
                    },
                ),
                result=public_recovered_result,
                redactor=self._secret_redactor,
            )
            recovery_tool_event = event_with_execution_profile_authority(
                recovery_tool_event,
                execution_profile_snapshot.profile,
            )
            recovery_event_to_reconcile = recovery_tool_event
            recovery_events = [
                event_with_execution_profile_authority(
                    Event(
                        type=EventType.SESSION_RESUMED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload={
                            "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                            **pending_rounds.pending_tool_round_identity(pending_round).payload(),
                            "tool_call_id": pending_tool_call.tool_call_id,
                            "resolved_by": resolution_actor_payload(request.resolved_by),
                        },
                    ),
                    execution_profile_snapshot.profile,
                ),
                recovery_tool_event,
            ]
            emitted_recovery_events = await self._event_writer.persist_many(
                session.id, recovery_events
            )
            recovery_persisted = True
            await self._event_writer.fan_out_persisted(emitted_recovery_events)
            for event in emitted_recovery_events:
                yield event
            tool_call = approval_support.tool_call_request_from_pending(
                pending_tool_call,
                arguments={},
            )
            tool_event = emitted_recovery_events[-1]
            # The operator outcome is durable before hooks run. Recovery hooks are
            # observe-only so they cannot rewrite externally verified evidence.
            async for event, _modified in self._tool_invocation.hooks.after_call(
                session=session,
                tool_event=tool_event,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                result=public_recovered_result,
                task_id=pending_round.task_id,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=invocation_context,
                redactor=self._secret_redactor,
                output_redactor=self._secret_redactor,
                allow_modification=False,
            ):
                yield event

            events = await self._session_store.load_events(session.id)
            recorded_outcomes, started_ids = tool_round_recovery.recorded_tool_outcomes(
                events=events,
                pending_round=pending_round,
            )
            (
                isolated_dispatched_ids,
                isolated_call_ids,
            ) = await self._pending_tool_round_recovery.isolated_tool_dispatch_ids(
                session=session,
                pending_round=pending_round,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
            effective_started_ids = (started_ids - isolated_call_ids) | isolated_dispatched_ids
            remaining_ids = effective_started_ids - set(recorded_outcomes)
            if remaining_ids:
                next_call = next(
                    call for call in pending_round.tool_calls if call.tool_call_id in remaining_ids
                )
                async for event in self._interrupt_for_resumable_manual_recovery(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile_snapshot.profile,
                    invocation_context=invocation_context,
                    payload={
                        "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                        "manual_recovery_required": True,
                        **pending_rounds.pending_tool_round_identity(pending_round).payload(),
                        "tool_call_id": next_call.tool_call_id,
                        "tool_name": next_call.tool_name,
                    },
                ):
                    yield event
                return
        except (GeneratorExit, asyncio.CancelledError) as abandonment:
            await self._recovery_ownership.run_cleanup_steps(
                authoritative_failure=abandonment,
                steps=(
                    (
                        "abandoned session finalization",
                        lambda: self._session_finalization.finalize_abandoned_session_by_id(
                            session.id,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            execution_profile=execution_profile_snapshot.profile,
                            invocation_context=invocation_context,
                        ),
                    ),
                ),
            )
            raise
        except Exception as exc:
            reconciliation_error: Exception | None = None
            if not recovery_persisted and recovery_event_to_reconcile is not None:
                try:
                    reconciliation = await reconcile_manual_recovery_persistence(
                        self._event_writer, recovery_event_to_reconcile
                    )
                except BaseException as reconciliation_failure:
                    if (
                        _recovery_abandonment_signal(
                            reconciliation_failure,
                            cancellation_baseline=cancellation_baseline,
                        )
                        is not None
                    ):
                        await self._recovery_ownership.run_cleanup_steps(
                            authoritative_failure=reconciliation_failure,
                            steps=(
                                (
                                    "abandoned session finalization",
                                    lambda: (
                                        self._session_finalization.finalize_abandoned_session_by_id(
                                            session.id,
                                            registered_agent=registered_agent,
                                            registered_environment=registered_environment,
                                            execution_profile=execution_profile_snapshot.profile,
                                            invocation_context=invocation_context,
                                        )
                                    ),
                                ),
                            ),
                        )
                    raise
                if reconciliation.cancellation is not None:
                    reconciliation.cancellation.add_note(
                        "Manual tool-round recovery append failed while persistence "
                        "reconciliation was running."
                    )
                    await self._recovery_ownership.run_cleanup_steps(
                        authoritative_failure=reconciliation.cancellation,
                        steps=(
                            (
                                "abandoned session finalization",
                                lambda: self._session_finalization.finalize_abandoned_session_by_id(
                                    session.id,
                                    registered_agent=registered_agent,
                                    registered_environment=registered_environment,
                                    execution_profile=execution_profile_snapshot.profile,
                                    invocation_context=invocation_context,
                                ),
                            ),
                        ),
                    )
                    raise reconciliation.cancellation from exc
                recovery_persisted = reconciliation.persisted is True
                reconciliation_error = reconciliation.error
            if not recovery_persisted and reconciliation_error is None:
                if isinstance(exc, SessionRunFenced):
                    raise
                if loaded_session.status in {
                    SessionStatus.RUNNING,
                    SessionStatus.INTERRUPTING,
                }:
                    if loaded_session.status == SessionStatus.INTERRUPTING:
                        session = await self._session_store.transition_status(
                            session.id,
                            from_statuses={SessionStatus.RUNNING},
                            to_status=SessionStatus.INTERRUPTING,
                        )
                    diagnostic = exception_diagnostic(
                        exc,
                        redactor=self._secret_redactor,
                    )
                    async for event in self._interrupt_for_resumable_manual_recovery(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                        invocation_context=invocation_context,
                        payload={
                            "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                            **pending_rounds.pending_tool_round_identity(pending_round).payload(),
                            "tool_call_id": pending_tool_call.tool_call_id,
                            "manual_recovery_stale_live_failure": True,
                            **diagnostic.payload_fields(),
                            "resolved_by": resolution_actor_payload(request.resolved_by),
                        },
                    ):
                        yield event
                    return
                try:

                    def restore_checkpoint(
                        _current_session: Session,
                        checkpoint: dict[str, Any] | None,
                    ) -> dict[str, Any] | None:
                        current_operation = _session_run_operation_from_checkpoint(checkpoint)
                        if checkpoint is None or current_operation != run_operation:
                            raise RuntimeError(
                                "Manual recovery run operation changed before rollback."
                            )
                        updated = copy_durable_record(checkpoint, "checkpoint")
                        updated.pop(_SESSION_RUN_OPERATION_CHECKPOINT_KEY)
                        return updated

                    await self._session_store.transition_status_and_checkpoint(
                        session.id,
                        from_statuses={SessionStatus.RUNNING},
                        to_status=loaded_session.status,
                        checkpoint_transform=restore_checkpoint,
                    )
                except SessionStatusConflict:
                    current = await self._recovery_ownership.require_session(session.id)
                    if current.status not in {
                        SessionStatus.INTERRUPTING,
                        SessionStatus.INTERRUPTED,
                    }:
                        raise
                    async for event in self._session_finalization.interrupt_recovery(
                        RecoveryInterruptionRequest(
                            session=current,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            environment_name=_environment_name(registered_environment),
                            execution_profile=execution_profile_snapshot.profile,
                            invocation_context=invocation_context,
                        )
                    ):
                        yield event
                    return
                raise
            persistence_payload = (
                {"manual_recovery_persisted": True}
                if recovery_persisted
                else {
                    "manual_recovery_persistence_unknown": True,
                    "persistence_reconciliation_error_type": (
                        _optional_exception_type_name(
                            reconciliation_error,
                            redactor=self._secret_redactor,
                        )
                    ),
                }
            )
            diagnostic = exception_diagnostic(
                exc,
                redactor=self._secret_redactor,
            )
            async for event in self._interrupt_for_resumable_manual_recovery(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=invocation_context,
                payload={
                    "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                    **pending_rounds.pending_tool_round_identity(pending_round).payload(),
                    "tool_call_id": pending_tool_call.tool_call_id,
                    **persistence_payload,
                    **diagnostic.payload_fields(),
                    "resolved_by": resolution_actor_payload(request.resolved_by),
                },
            ):
                yield event
            return
        except BaseExceptionGroup as exc:
            abandonment = _recovery_abandonment_signal(
                exc,
                cancellation_baseline=cancellation_baseline,
            )
            if recovery_persisted and abandonment is None:
                async for event in self._session_finalization.interrupt_recovery(
                    RecoveryInterruptionRequest(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                        execution_profile=execution_profile_snapshot.profile,
                        invocation_context=invocation_context,
                    )
                ):
                    yield event
            if abandonment is not None:
                await self._recovery_ownership.run_cleanup_steps(
                    authoritative_failure=exc,
                    steps=(
                        (
                            "abandoned session finalization",
                            lambda: self._session_finalization.finalize_abandoned_session_by_id(
                                session.id,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                execution_profile=execution_profile_snapshot.profile,
                                invocation_context=invocation_context,
                            ),
                        ),
                    ),
                )
            raise

        session_stream: AsyncGenerator[Event, None] | None = None
        authoritative_failure: BaseException | None = None
        try:
            continuation = await self._recovery_admission.prepare_recovered_tool_round_continuation(
                session=session,
                participant_context=participant_context,
                pending_round=pending_round,
                invocation_semantics=invocation_semantics,
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
                        request_loop_policies=request.loop_policies,
                    )
                ),
                request_metadata=request.metadata,
                task_worker_id=request.task_worker_id,
                task_handoff_id=request.task_handoff_id,
            )
            session_stream = self._engine.continue_run(continuation)
            async for event in session_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            await self._recovery_admission.cleanup_recovery_handoff(
                stream=session_stream,
                session_id=session.id,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                authoritative_failure=authoritative_failure,
                # The manual-recovery supervisor drains this worker before
                # finalizing the session and releasing its claim. This nested
                # continuation owns stream closure, not a second finalization.
                finalize_abandoned=False,
                release_run_fence=False,
                abort_environment_setup=False,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=invocation_context,
            )

    async def _interrupt_unresolved_tool_effect(
        self,
        *,
        record: ToolEffectRecord,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity,
        invocation_context: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        """Finish an accepted unresolved decision through the existing interruption owner."""
        assert record.observation is not None
        result = record.observation.result
        async for event in self._interrupt_for_resumable_manual_recovery(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
            payload={
                "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                **{
                    name: getattr(record.intent, name)
                    for name in (
                        "model_step_id",
                        "model_attempt_id",
                        "tool_round_id",
                        "tool_call_id",
                    )
                },
                "tool_effect_reconciliation": {
                    "schema_version": 1,
                    "outcome": result.outcome,
                    "observation": result.observation,
                    "retryable_lookup": result.retryable,
                },
            },
        ):
            yield event
        if result.outcome == "conflict":
            raise _FinalizedToolEffectRejection("Application reconciliation rejected the receipt.")

    async def _settle_tool_effect_reconciliation(
        self,
        *,
        request: ToolEffectReconciliationRequest,
        source_run_epoch: int,
        session: Session,
        invocation_context: InvocationContext,
    ) -> AsyncGenerator[Event | _ReconciledToolEffectReplay | _ToolEffectObservationReplay, None]:
        """Select evidence under an existing recovery claim; never own continuation."""
        invocation_context._validate()
        invocation_context.with_admitted_session(session)
        owner = ToolEffectStateOwner(self._session_store)
        record = await owner.resolve_call(
            session, tool_round_id=request.tool_round_id, tool_call_id=request.tool_call_id
        )
        if record is None:
            raise ToolEffectConflict("Receipt recovery has no durable effect intent.")
        registered_agent = invocation_context.registered_agent
        registered_tool = registered_agent.tools.get(record.intent.tool_name)
        if registered_tool is None or registered_tool.effect is not ToolEffect.EXTERNAL:
            raise ToolEffectConflict("Receipt recovery has no exact registered external tool.")
        if record.intent.execution_profile_fingerprint != invocation_context.profile.fingerprint:
            raise ToolEffectConflict("Receipt recovery conflicts with the original profile.")
        replay = await self._load_reconciled_tool_effect_replay(session=session, request=request)
        if replay is not None:
            yield replay
            return
        prepared = self._effect_reconciliation_owner.prepare(
            request=request,
            record=record,
            run_epoch=source_run_epoch,
            registered=registered_tool.effect_reconciler,
        )
        started = self._event_writer.prepare(
            event_with_execution_profile_authority(
                Event(
                    type=EventType.TOOL_EFFECT_RECONCILIATION_STARTED,
                    session_id=session.id,
                    interaction_id=record.intent.interaction_id,
                    agent_name=record.intent.agent_name,
                    environment_name=record.intent.environment_name,
                    tool_name=record.intent.tool_name,
                    payload={
                        "schema_version": 1,
                        "request_digest": prepared.request_digest,
                        "intent_digest": prepared.context.intent_digest,
                        "dispatch_id": record.dispatch_id,
                        "expected_revision": request.expected_revision,
                        "expected_run_epoch": request.expected_run_epoch,
                        "lookup": request.lookup,
                        **{
                            name: getattr(record.intent, name)
                            for name in (
                                "model_step_id",
                                "model_attempt_id",
                                "tool_round_id",
                                "tool_call_id",
                                "approval_id",
                            )
                        },
                    },
                ),
                invocation_context.profile,
            )
        )
        admitted = await owner.start_reconciliation(
            record,
            source_run_epoch=source_run_epoch,
            run_epoch=session.run_epoch,
            request_digest=prepared.request_digest,
            lookup=request.lookup,
            event=started,
        )
        for delivered in await self._event_writer.fan_out_persisted([started]):
            yield delivered
        accepted = await self._effect_reconciliation_owner.reconcile(
            request=request,
            record=record,
            run_epoch=source_run_epoch,
            registered=registered_tool.effect_reconciler,
        )
        record = admitted
        result = project_accepted_reconciliation(
            accepted,
            registered=registered_tool.effect_reconciler,
            redactor=_redactor_for_tool_calls(
                self._secret_redactor,
                registered_agent=registered_agent,
                tool_calls=[
                    runtime_records.ToolCallRequest(
                        id=record.intent.tool_call_id,
                        name=record.intent.tool_name,
                        arguments={},
                        arguments_state="unavailable",
                    )
                ],
            ),
        )
        identity = {
            name: getattr(record.intent, name)
            for name in (
                "model_step_id",
                "model_attempt_id",
                "tool_round_id",
                "tool_call_id",
                "idempotency_key",
            )
        }
        if record.intent.approval_id is not None:
            identity["approval_id"] = record.intent.approval_id
        if record.intent.pause_id is not None:
            identity["input_id"] = record.intent.pause_id
        if result.receipt is None:
            event = Event(
                type=(
                    EventType.TOOL_EFFECT_RECONCILIATION_CONFLICT
                    if result.outcome == "conflict"
                    else EventType.TOOL_EFFECT_RECONCILIATION_OBSERVED
                ),
                session_id=session.id,
                interaction_id=record.intent.interaction_id,
                agent_name=registered_agent.spec.name,
                environment_name=_environment_name(invocation_context.registered_environment),
                tool_name=record.intent.tool_name,
                payload={
                    "schema_version": 1,
                    **({"kind": "validator_rejected"} if result.outcome == "conflict" else {}),
                    **identity,
                    "request_digest": accepted.request_digest,
                    "result": result.model_dump(mode="json"),
                    "resource_versions": {**record.resource_versions, **result.resource_versions},
                },
            )
        else:
            receipt = result.receipt
            receipt_evidence = {
                "schema_version": 1,
                "receipt_id": receipt.receipt_id,
                "receipt_schema": receipt.receipt_schema,
                "receipt_schema_version": receipt.receipt_schema_version,
                "outcome": receipt.outcome,
                "source": receipt.source,
                "observed_at": receipt.observed_at.isoformat(),
                "receipt_digest": tool_effect_receipt_digest(receipt),
                "integrity": dict(receipt.integrity),
                "resource_versions": dict(receipt.resource_versions),
            }
            validation_event = self._event_writer.prepare(
                event_with_execution_profile_authority(
                    Event(
                        type=EventType.TOOL_EFFECT_RECEIPT_VALIDATED,
                        session_id=session.id,
                        interaction_id=record.intent.interaction_id,
                        agent_name=record.intent.agent_name,
                        environment_name=record.intent.environment_name,
                        tool_name=record.intent.tool_name,
                        payload={
                            **{
                                key: value
                                for key, value in identity.items()
                                if key != "idempotency_key"
                            },
                            "schema_version": 1,
                            "request_digest": accepted.request_digest,
                            "receipt_evidence": receipt_evidence,
                        },
                    ),
                    invocation_context.profile,
                )
            )
            terminal_result = ToolResult(
                content=receipt.message,
                structured=receipt.structured,
                artifacts=receipt.artifacts,
                is_error=receipt.outcome == "failed",
            )
            event = Event(
                type=EventType.TOOL_CALL_FAILED
                if terminal_result.is_error
                else EventType.TOOL_CALL_COMPLETED,
                session_id=session.id,
                interaction_id=record.intent.interaction_id,
                agent_name=registered_agent.spec.name,
                environment_name=_environment_name(invocation_context.registered_environment),
                tool_name=record.intent.tool_name,
                payload={
                    **identity,
                    **tool_argument_publication.unavailable_argument_projection().payload_fields(),
                    "effect_reconciled": True,
                    "reconciliation_state": "reconciled",
                    "receipt_id": receipt.receipt_id,
                    "receipt_evidence": receipt_evidence,
                    "result": terminal_result.model_dump(mode="json"),
                },
            )
        event = self._event_writer.prepare(
            event_with_execution_profile_authority(event, invocation_context.profile)
        )
        if result.receipt is None:
            observed = await owner.transition(
                record,
                state="outcome_unknown",
                run_epoch=session.run_epoch,
                observation=ToolEffectObservation(
                    event_id=event.id,
                    request_digest=accepted.request_digest,
                    result=result,
                ),
                events=(event,),
            )
            yield _ToolEffectObservationReplay(observed, event)
            return
        settled = await owner.transition(
            record,
            state="reconciled_failed"
            if result.receipt.outcome == "failed"
            else "reconciled_completed",
            run_epoch=session.run_epoch,
            terminal=ToolEffectTerminal(
                event_id=event.id,
                result_digest=sha256(
                    canonical_durable_json_bytes(event.payload["result"], "effect_terminal_result")
                ).hexdigest(),
                receipt=result.receipt,
                reconciliation_request_digest=accepted.request_digest,
            ),
            events=(validation_event, event),
        )
        for delivered in await self._event_writer.fan_out_persisted([validation_event]):
            yield delivered
        yield _ReconciledToolEffectReplay(settled, event, None)

    async def _reconcile_tool_effect_claimed(
        self,
        *,
        request: ToolRoundRecoveryRequest | ToolEffectReconciliationRequest,
        loaded_session: Session,
        session: Session,
        run_operation: _SessionRunOperation | None,
        pending_round: pending_rounds.PendingToolRound,
        pending_tool_call: PendingToolCallApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        invocation_semantics: _RecoveryInvocationSemantics,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
        budget_policy: BudgetPolicy | None,
        invocation_context: InvocationContext | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Validate and settle a receipt under the existing recovery supervisor."""
        if type(request) is not ToolEffectReconciliationRequest:
            raise TypeError("Receipt recovery requires an exact reconciliation request.")
        if invocation_context is None or run_operation is None:
            raise RuntimeError("Receipt recovery has no claimed invocation authority.")
        invocation_context._validate()
        invocation_context.with_admitted_session(session)
        if (
            invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_provider is not registered_provider
            or invocation_context.registered_environment is not registered_environment
            or invocation_context.profile is not execution_profile_snapshot.profile
            or invocation_context.budget_policy is not budget_policy
        ):
            raise RuntimeError("Receipt recovery substituted frozen invocation authority.")
        stream: AsyncGenerator[Event, None] | None = None
        failure: BaseException | None = None
        try:
            await self._preflight_tool_effect_reconciliation(
                session=session, request=request, source_run_epoch=loaded_session.run_epoch
            )
            factory_started_event = await self._environment_lifecycle.emit_factory_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=invocation_context,
            )
            if factory_started_event is not None:
                yield factory_started_event
            factory_resolution = await self._environment_lifecycle.resolve_factory(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=factory_started_event,
                operation=EnvironmentFactoryOperation.RECONNECT,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=invocation_context,
            )
            registered_environment = factory_resolution.registered_environment
            if registered_environment is not None:
                invocation_context = invocation_context.with_registered_environment(
                    registered_environment,
                    validated_profile=execution_profile_snapshot.profile,
                )
            for event in factory_resolution.events:
                yield event
            if factory_resolution.error is not None:
                async for event in self._interrupt_for_resumable_manual_recovery(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile_snapshot.profile,
                    invocation_context=invocation_context,
                    payload={
                        "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                        **pending_rounds.pending_tool_round_identity(pending_round).payload(),
                        **_environment_factory_resolution_error_payload(
                            factory_resolution.error,
                            redactor=self._secret_redactor,
                        ),
                    },
                ):
                    yield event
                return
            settlement = None
            settlement_stream = self._settle_tool_effect_reconciliation(
                request=request,
                source_run_epoch=loaded_session.run_epoch,
                session=session,
                invocation_context=invocation_context,
            )
            async with contextlib.aclosing(settlement_stream) as owned_settlement:
                async for item in owned_settlement:
                    if isinstance(item, Event):
                        yield copy_event(item)
                    else:
                        settlement = item
            if settlement is None:
                raise RuntimeError("Receipt recovery returned no settlement.")
            if isinstance(settlement, _ToolEffectObservationReplay):
                await self._event_writer.fan_out_persisted([settlement.event])
                yield copy_event(settlement.event)
                async for event in self._interrupt_unresolved_tool_effect(
                    record=settlement.record,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile_snapshot.profile,
                    invocation_context=invocation_context,
                ):
                    yield event
                return
            await self._event_writer.fan_out_persisted([settlement.terminal_event])
            yield copy_event(settlement.terminal_event)
            if settlement.consumption_receipt is not None:
                return
            # Receipt selection is already durable. Hooks observe the selected
            # result through the existing hook owner and cannot replace it.
            async for event, _modified in self._tool_invocation.hooks.after_call(
                session=session,
                tool_event=settlement.terminal_event,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=approval_support.tool_call_request_from_pending(
                    pending_tool_call, arguments={}
                ),
                result=ToolResult.model_validate(settlement.terminal_event.payload["result"]),
                task_id=pending_round.task_id,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=invocation_context,
                redactor=self._secret_redactor,
                output_redactor=self._secret_redactor,
                allow_modification=False,
            ):
                yield event
            continuation = await self._recovery_admission.prepare_recovered_tool_round_continuation(
                session=session,
                participant_context=participant_context,
                pending_round=pending_round,
                invocation_semantics=invocation_semantics,
                invocation_context=invocation_context,
                request_metadata={},
                task_worker_id=request.task_worker_id,
                task_handoff_id=request.task_handoff_id,
            )
            stream = self._engine.continue_run(continuation)
            async for event in stream:
                yield event
        except BaseException as exc:
            failure = exc
            raise
        finally:
            await self._recovery_admission.cleanup_recovery_handoff(
                stream=stream,
                session_id=session.id,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                authoritative_failure=failure,
                # Same supervisor ownership as manual tool-round recovery.
                finalize_abandoned=False,
                release_run_fence=False,
                abort_environment_setup=False,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=invocation_context,
            )

    def detached_recovery_work(self) -> set[asyncio.Future[Any]]:
        """Tool-effect reconciliation lookups whose caller stopped waiting."""

        return {
            *self._effect_reconciliation_owner.running(),
        }

    async def _recover_terminal_foreground_effects_at_human_gate(
        self,
        *,
        session: Session,
        pending: PendingToolApproval | PendingUserInput,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity,
        invocation_context: InvocationContext | None,
    ) -> list[Event]:
        """Restore native outcomes under an admitted gate resolution, without consuming it."""
        checkpoint = await self._session_store.load_checkpoint(session.id)
        children = await self._pending_tool_round_recovery.subagent_children_by_idempotency_key(
            session.id
        )
        emitted: list[Event] = []
        for call in pending.tool_calls:
            registered_tool = registered_agent.executable_tool(call.tool_name)
            if registered_tool is None or registered_tool.child_session_recovery is None:
                continue
            record = await ToolEffectStateOwner(self._session_store).resolve_call(
                session, tool_round_id=pending.tool_round_id, tool_call_id=call.tool_call_id
            )
            if record is None or record.state not in {"executing", "outcome_unknown"}:
                continue
            intent = record.intent
            approval_id = pending.approval_id if isinstance(pending, PendingToolApproval) else None
            input_id = pending.input_id if isinstance(pending, PendingUserInput) else None
            key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_round_id=pending.tool_round_id,
                tool_call_id=call.tool_call_id,
                approval_id=approval_id,
                pause_id=input_id,
            )
            if (
                intent.tool_name != call.tool_name
                or intent.idempotency_key != key
                or intent.approval_id != approval_id
                or intent.pause_id != input_id
                or intent.execution_profile_fingerprint != execution_profile.fingerprint
            ):
                raise RuntimeError("Foreground recovery conflicts with the admitted gate identity.")
            child = children.get(key)
            metadata = None if child is None else child.metadata.get("subagent")
            if type(metadata) is not dict or metadata.get("mode") != "foreground":
                continue
            arguments = await self._pending_tool_round_recovery.subagent_recovery_arguments(
                checkpoint=checkpoint,
                parent_session=session,
                tool_name=call.tool_name,
                tool_round_id=pending.tool_round_id,
                tool_call_id=call.tool_call_id,
                idempotency_key=key,
                fallback=call.arguments,
            )
            result = await self._pending_tool_round_recovery.reattached_subagent_result(
                children,
                key,
                parent_checkpoint=checkpoint,
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                tool_round_id=pending.tool_round_id,
                arguments=arguments,
                parent_session=session,
                registered_agent=registered_agent,
            )
            if result is None:
                continue
            tool_call = runtime_records.ToolCallRequest(
                id=call.tool_call_id, name=call.tool_name, arguments=arguments
            )
            event = Event(
                type=EventType.TOOL_CALL_FAILED
                if result.is_error
                else EventType.TOOL_CALL_COMPLETED,
                session_id=session.id,
                interaction_id=intent.interaction_id,
                agent_name=registered_agent.spec.name,
                environment_name=_environment_name(registered_environment),
                tool_name=call.tool_name,
                payload={
                    "model_step_id": intent.model_step_id,
                    "model_attempt_id": intent.model_attempt_id,
                    "tool_round_id": intent.tool_round_id,
                    "tool_call_id": intent.tool_call_id,
                    "idempotency_key": key,
                    "recovered": True,
                    **(
                        {"approval_id": approval_id}
                        if approval_id is not None
                        else {"input_id": input_id}
                    ),
                    **tool_argument_publication.unavailable_argument_projection().payload_fields(),
                },
            )

            async def publish(event: Event, selected: ToolEffectRecord = record) -> Event:
                return await self._pending_tool_round_recovery.emit_confirmed_native_tool_terminal(
                    session=session, record=selected, event=event
                )

            async for (
                published,
                _outcome,
            ) in self._tool_invocation.terminals.publish_result(
                event=event,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                result=result,
                task_id=pending.task_id,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                allow_modification=False,
                hooks_already_completed=True,
                terminal_event_emitter=publish,
                redactor=_redactor_for_tool_calls(
                    self._secret_redactor, registered_agent=registered_agent, tool_calls=[tool_call]
                ),
            ):
                emitted.append(published)
        return emitted


def _effective_tool_round_structured_output(
    *,
    structured_output: StructuredOutputSpec | None,
    pending_round: pending_rounds.PendingToolRound,
) -> StructuredOutputSpec | None:
    if type(pending_round) is not pending_rounds.PendingToolRound:
        raise TypeError("Pending tool round must be a PendingToolRound.")
    if structured_output is None:
        return copy_structured_output_spec(pending_round.structured_output)
    if pending_round.structured_output is None:
        return copy_structured_output_spec(structured_output)
    if structured_output.model_dump(mode="json") != pending_round.structured_output.model_dump(
        mode="json"
    ):
        raise ValueError("structured_output does not match the crashed run contract.")
    return copy_structured_output_spec(pending_round.structured_output)


def _effective_tool_round_invocation_semantics(
    *,
    request: ToolRoundRecoveryRequest | ToolEffectReconciliationRequest,
    pending_round: pending_rounds.PendingToolRound,
    structured_output: StructuredOutputSpec | None,
    effective_retry_policy: EffectiveRetryPolicy,
) -> _RecoveryInvocationSemantics:
    if type(request) is ToolEffectReconciliationRequest:
        recorded_max_steps = _require_recovery_max_steps(pending_round.max_steps)
        if request.max_steps is not None and request.max_steps > recorded_max_steps:
            raise ValueError("Receipt continuation cannot increase the recorded step limit.")
        return _RecoveryInvocationSemantics(
            max_steps=recorded_max_steps if request.max_steps is None else request.max_steps,
            limits=copy_run_limits(pending_round.limits or RunLimits()),
            budget_limits=copy_request_budget_limits(pending_round.budget_limits or ()),
            retry_policy=effective_retry_policy(pending_round.retry_policy),
            structured_output=copy_structured_output_spec(structured_output),
            thinking=pending_round.thinking,
        )
    if type(request) is not ToolRoundRecoveryRequest:
        raise TypeError("Tool-round recovery requires a ToolRoundRecoveryRequest.")
    if type(pending_round) is not pending_rounds.PendingToolRound:
        raise TypeError("Pending tool round must be a PendingToolRound.")
    _require_recovery_max_steps(pending_round.max_steps)
    return _RecoveryInvocationSemantics(
        max_steps=(
            request.max_steps
            if request.max_steps is not None
            else _require_recovery_max_steps(pending_round.max_steps)
        ),
        limits=copy_run_limits(
            request.limits if request.limits is not None else pending_round.limits or RunLimits()
        ),
        budget_limits=copy_request_budget_limits(
            request.budget_limits
            if request.budget_limits is not None
            else pending_round.budget_limits or ()
        ),
        retry_policy=effective_retry_policy(
            request.retry_policy if request.retry_policy is not None else pending_round.retry_policy
        ),
        structured_output=copy_structured_output_spec(structured_output),
        thinking=request.thinking if request.thinking is not None else pending_round.thinking,
    )


def _effective_user_input_max_steps(
    *,
    max_steps: int | None,
    pending: PendingUserInput,
) -> int:
    # Restore recorded invocation settings; profile admission validates overrides.
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    recorded = _require_recovery_max_steps(pending.max_steps)
    return recorded if max_steps is None else max_steps


def _effective_user_input_run_limits(
    *,
    limits: RunLimits | None,
    pending: PendingUserInput,
) -> RunLimits:
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    if limits is not None:
        return copy_run_limits(limits)
    if pending.limits is not None:
        return copy_run_limits(pending.limits)
    return RunLimits()


def _effective_user_input_budget_limits(
    *,
    budget_limits: tuple[BudgetLimit, ...] | None,
    pending: PendingUserInput,
) -> tuple[BudgetLimit, ...]:
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    if budget_limits is not None:
        return copy_request_budget_limits(budget_limits)
    if pending.budget_limits is not None:
        return copy_request_budget_limits(pending.budget_limits)
    return ()


def _effective_user_input_retry_policy(
    *,
    retry_policy: RetryPolicy | None,
    pending: PendingUserInput,
) -> RetryPolicy | None:
    # RetryPolicy is frozen, so the persisted reference is safe to reuse.
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    if retry_policy is not None:
        return retry_policy
    return pending.retry_policy


def _effective_user_input_structured_output(
    *,
    structured_output: StructuredOutputSpec | None,
    pending: PendingUserInput,
) -> StructuredOutputSpec | None:
    # Mirror _effective_approval_structured_output: inherit the paused run's spec when
    # the resolver supplies none; profile admission owns the final equality decision.
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    if structured_output is None:
        return copy_structured_output_spec(pending.structured_output)
    if pending.structured_output is None:
        return copy_structured_output_spec(structured_output)
    if not _structured_output_specs_equal(structured_output, pending.structured_output):
        raise ValueError("structured_output does not match the paused run contract.")
    return copy_structured_output_spec(pending.structured_output)


def _effective_user_input_invocation_semantics(
    *,
    response: UserInputResponse | UserInputRecoveryRequest,
    pending: PendingUserInput,
    structured_output: StructuredOutputSpec | None,
    effective_retry_policy: EffectiveRetryPolicy,
) -> _RecoveryInvocationSemantics:
    if type(response) not in (UserInputResponse, UserInputRecoveryRequest):
        raise TypeError("User-input continuation requires a validated response.")
    return _RecoveryInvocationSemantics(
        max_steps=_effective_user_input_max_steps(max_steps=response.max_steps, pending=pending),
        limits=_effective_user_input_run_limits(limits=response.limits, pending=pending),
        budget_limits=_effective_user_input_budget_limits(
            budget_limits=response.budget_limits,
            pending=pending,
        ),
        retry_policy=effective_retry_policy(
            _effective_user_input_retry_policy(
                retry_policy=response.retry_policy,
                pending=pending,
            )
        ),
        structured_output=copy_structured_output_spec(structured_output),
        thinking=response.thinking if response.thinking is not None else pending.thinking,
    )


def _effective_approval_thinking(
    *,
    thinking: ThinkingConfig | None,
    pending_approval: PendingToolApproval,
) -> ThinkingConfig | None:
    # Restore the original run's thinking config on an approval continuation. Profile
    # admission decides whether an explicit value preserves the frozen invocation.
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if thinking is not None:
        return thinking
    return pending_approval.thinking


def _effective_approval_max_steps(
    *,
    max_steps: int | None,
    pending_approval: PendingToolApproval,
) -> int:
    # Restore recorded invocation settings; profile admission validates overrides.
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    recorded = _require_recovery_max_steps(pending_approval.max_steps)
    return recorded if max_steps is None else max_steps


def _effective_approval_run_limits(
    *,
    limits: RunLimits | None,
    pending_approval: PendingToolApproval,
) -> RunLimits:
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if limits is not None:
        return copy_run_limits(limits)
    if pending_approval.limits is not None:
        return copy_run_limits(pending_approval.limits)
    return RunLimits()


def _effective_approval_budget_limits(
    *,
    budget_limits: tuple[BudgetLimit, ...] | None,
    pending_approval: PendingToolApproval,
) -> tuple[BudgetLimit, ...]:
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if budget_limits is not None:
        return copy_request_budget_limits(budget_limits)
    if pending_approval.budget_limits is not None:
        return copy_request_budget_limits(pending_approval.budget_limits)
    return ()


def _effective_approval_retry_policy(
    *,
    retry_policy: RetryPolicy | None,
    pending_approval: PendingToolApproval,
) -> RetryPolicy | None:
    # RetryPolicy is frozen, so the persisted reference is safe to reuse.
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if retry_policy is not None:
        return retry_policy
    return pending_approval.retry_policy


def _effective_approval_structured_output(
    *,
    structured_output: StructuredOutputSpec | None,
    pending_approval: PendingToolApproval,
) -> StructuredOutputSpec | None:
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if structured_output is None:
        return copy_structured_output_spec(pending_approval.structured_output)
    if pending_approval.structured_output is None:
        return copy_structured_output_spec(structured_output)
    if not _structured_output_specs_equal(
        structured_output,
        pending_approval.structured_output,
    ):
        raise ValueError("Tool approval structured_output does not match the pending run contract.")
    return copy_structured_output_spec(pending_approval.structured_output)


def _effective_approval_invocation_semantics(
    *,
    request: ToolApprovalRequest | ToolApprovalRecoveryRequest | ToolEffectReconciliationRequest,
    pending_approval: PendingToolApproval,
    structured_output: StructuredOutputSpec | None,
    effective_retry_policy: EffectiveRetryPolicy,
) -> _RecoveryInvocationSemantics:
    if type(request) is ToolEffectReconciliationRequest:
        return _RecoveryInvocationSemantics(
            max_steps=_effective_approval_max_steps(
                max_steps=request.max_steps, pending_approval=pending_approval
            ),
            limits=_effective_approval_run_limits(limits=None, pending_approval=pending_approval),
            budget_limits=_effective_approval_budget_limits(
                budget_limits=None, pending_approval=pending_approval
            ),
            retry_policy=effective_retry_policy(
                _effective_approval_retry_policy(
                    retry_policy=None, pending_approval=pending_approval
                )
            ),
            structured_output=copy_structured_output_spec(structured_output),
            thinking=_effective_approval_thinking(thinking=None, pending_approval=pending_approval),
        )
    if type(request) not in (ToolApprovalRequest, ToolApprovalRecoveryRequest):
        raise TypeError("Tool-approval continuation requires a validated request.")
    assert isinstance(request, (ToolApprovalRequest, ToolApprovalRecoveryRequest))
    return _RecoveryInvocationSemantics(
        max_steps=_effective_approval_max_steps(
            max_steps=request.max_steps,
            pending_approval=pending_approval,
        ),
        limits=_effective_approval_run_limits(
            limits=request.limits,
            pending_approval=pending_approval,
        ),
        budget_limits=_effective_approval_budget_limits(
            budget_limits=request.budget_limits,
            pending_approval=pending_approval,
        ),
        retry_policy=effective_retry_policy(
            _effective_approval_retry_policy(
                retry_policy=request.retry_policy,
                pending_approval=pending_approval,
            )
        ),
        structured_output=copy_structured_output_spec(structured_output),
        thinking=_effective_approval_thinking(
            thinking=request.thinking,
            pending_approval=pending_approval,
        ),
    )


def _structured_output_specs_equal(
    left: StructuredOutputSpec,
    right: StructuredOutputSpec,
) -> bool:
    if type(left) is not StructuredOutputSpec or type(right) is not StructuredOutputSpec:
        raise TypeError("Structured output comparison requires StructuredOutputSpec values.")
    return left.model_dump(mode="json") == right.model_dump(mode="json")


def _has_run_budget_limit(limits: tuple[BudgetLimit, ...]) -> bool:
    return any(limit.scope == "run" for limit in limits)
