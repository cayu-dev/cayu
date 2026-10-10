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
import base64
import binascii
import contextlib
import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4, uuid5

from cayu.runtime._continuation_environment import ContinuationEnvironment
from cayu.runtime._durable_tool_round import (
    DeferredInteractionInput,
    DurableToolRound,
)
from cayu.runtime._durable_tool_round import (
    _interrupted_tool_round_results as _interrupted_tool_round_results,
)
from cayu.runtime._execution_profile_continuation import ExecutionProfileContinuation
from cayu.runtime._invocation_lifecycle import reconstruct_invocation_context
from cayu.runtime._manual_recovery_publication import (
    ManualRecoveryPublication,
    reconcile_manual_recovery_persistence,
)
from cayu.runtime._model_completion_contracts import (
    ModelCompletionBoundaryReconciliation,
    ModelCompletionManualRecoveryRequired,
)
from cayu.runtime._model_completion_recovery import ModelCompletionRecovery
from cayu.runtime._paused_tool_round import ApprovalRoundPause, PausedToolRound, UserInputRoundPause
from cayu.runtime._pending_tool_round_recovery import (
    PendingToolRoundRecovery,
    RegisteredAgentResolver,
    RegisteredEnvironmentResolver,
    _environment_name,
    _matches_recoverable_subagent_child,
)
from cayu.runtime._recovery_ownership import (
    RecoveryOwnership,
    _authoritative_recovery_ownership_failure,
    _checkpoint_with_rebased_session_run_operation,
    _prepend_exception_cause,
    _recovery_abandonment_signal,
    _require_aware_datetime,
)
from cayu.runtime._recovery_requests import (
    ProviderOperationFailureRequest,
    RecoveryAbandonedTurnRequest,
    RecoveryInterruptionRequest,
    RecoveryLimitStopRequest,
    RecoverySessionRunRequest,
    RecoveryTaskEventRequest,
    RecoveryTerminalEventRequest,
)
from cayu.runtime._session_finalization import (
    SessionFinalization,
    _checkpoint_with_pending_session_interrupt,
)
from cayu.runtime._structured_output_tool_round import has_recoverable_structured_output_round
from cayu.runtime._terminal_event_publication import TerminalEventPublication
from cayu.runtime._terminal_evidence_finalization import _RECOVERY_RESUMABLE_SESSION_STATUSES
from cayu.runtime._terminal_evidence_reader import TerminalEvidenceReader
from cayu.runtime._user_input_recovery_evidence import UserInputRecoveryEvidence
from cayu.runtime._workspace_observation_recovery import (
    WorkspaceObservationRecovery,
)
from cayu.sessions import _completion_finalization as completion_finalization
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions import _staged_tool_terminal_reader as staged_terminal_reader

if TYPE_CHECKING:
    from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait

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
    require_clean_nonblank,
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
    ambiguous_pending_user_input_from_checkpoint,
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
    ModelStepIdentity,
    ToolRoundIdentity,
)
from cayu.failure_evidence import exception_evidence
from cayu.messages import Message, detach_message
from cayu.observability.hooks import RuntimeHookPhase
from cayu.providers.retry_policy import RetryPolicy
from cayu.resource_access import ResourceAccessPolicy, resource_recovery
from cayu.runtime import _approval_publication as approval_publication
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _resume_ledger as resume_ledger
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_call_replay as tool_call_replay
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_results as tool_results
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._approval_support import _pending_approval_and_round_for_atomic_claim
from cayu.runtime._auxiliary_invocation import AuxiliaryInvocationPolicy
from cayu.runtime._continuation_task_failure import (
    ApprovalTaskFailureIdentity,
    approval_failure_event_id,
    approval_task_failure_payload,
    approval_task_failure_receipt_matches,
    approval_task_terminalization_idempotency_key,
    approval_task_terminalization_request,
    load_direct_task_failure_replay,
    provider_operation_task_failure_payload,
    runtime_task_terminalization_idempotency_key,
)
from cayu.runtime._delegated_event_stream import _close_delegated_event_stream
from cayu.runtime._diagnostics import (
    ExceptionDiagnostic,
    bound_diagnostic_text,
    exception_diagnostic,
    task_failure_payload_from_diagnostic,
)
from cayu.runtime._durable_subagents import (
    durable_subagent_submission_from_checkpoint,
    durable_subagent_submission_receipt_from_checkpoint,
    durable_subagent_submission_seed_from_checkpoint,
    durable_subagent_submissions_from_checkpoint,
    require_durable_subagent_intent_matches_seed,
    require_durable_subagent_receipt_matches_intent,
    require_durable_subagent_receipt_matches_seed,
)
from cayu.runtime._environment_lifecycle import (
    EnvironmentLifecycle,
    exception_failure_payload,
)
from cayu.runtime._event_writer import (
    RuntimeEventWriter,
)
from cayu.runtime._foreground_child_wait import (
    ForegroundChildActionRequired,
    event_with_foreground_child_wait_authority,
)
from cayu.runtime._foreground_gate_continuation import ForegroundGatePolicyOwner, GateReplay
from cayu.runtime._foreground_subagent_recovery import ForegroundSubagentRecoveryRequired
from cayu.runtime._interruption_coordinator import (
    _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY,
    _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY,
)
from cayu.runtime._invocation_lifecycle import (
    InvocationContext,
)
from cayu.runtime._model_completion_contracts import (
    ModelCompletionRecoveryContext,
    model_completion_recovery_context_from_stage,
)
from cayu.runtime._model_errors import (
    _FallbackBillingCancellationStateCheckFailed,
    detach_billing_identity_cancellation_group,
)
from cayu.runtime._recovery_claims import (
    _IncompleteRecoveryClaim,
    _IncompleteRecoveryClaimAuthority,
    _IncompleteRecoveryClaimLost,
    _require_live_incomplete_recovery_claim_acknowledgement,
)
from cayu.runtime._run_limit_accounting import restore_run_limit_accounting_context
from cayu.runtime._run_limits import (
    RunLimitController,
    SessionUsageTracker,
)
from cayu.runtime._session_control import (
    SessionControl,
)
from cayu.runtime._terminal_evidence_finalization import (
    TerminalEvidenceFinalization,
    _terminal_finalization_failure_without_identity,
)
from cayu.runtime._terminal_evidence_reader import _provider_cancellation_interrupt_payload
from cayu.runtime._terminal_finalization_lifetime import _terminal_finalization_process_control
from cayu.runtime._tool_completion import (
    load_recorded_tool_completion_policy,
    pending_round_has_completion_success,
    recorded_terminal_tool_completion_payload,
    recorded_tool_completion_result,
    tool_completion_requires_execution,
)
from cayu.runtime._tool_effect_preparation_recovery import (
    settle_prepared_tool_effects,
)
from cayu.runtime._tool_effect_reconciliation import (
    ToolEffectReconciliationOwner,
    project_accepted_reconciliation,
    reconciliation_request_digest,
)
from cayu.runtime._tool_effect_state import (
    ToolEffectConflict,
    ToolEffectObservation,
    ToolEffectReconciliationRequired,
    ToolEffectRecord,
    ToolEffectStateOwner,
    ToolEffectTerminal,
    _validate_selected_observation,
    _validate_selected_terminal,
)
from cayu.runtime._tool_invocation.admission import (
    ToolApprovalRequired,
    policy_denial_payload_fields,
)
from cayu.runtime._tool_round_executor import (
    ToolRoundExecutor,
)
from cayu.runtime._work_attempt_invocation import WorkAttemptInvocationAuthority
from cayu.runtime._work_attempt_session_mutation import record_work_attempt_execution_stop
from cayu.runtime.loop_policies import LoopPolicy
from cayu.runtime.provider_operations import (
    ProviderOperationEvidenceError,
    ProviderOperationPendingDisposition,
    ProviderOperationResolutionAction,
    ProviderOperationResolutionConflict,
    ProviderOperationResolutionRequest,
    ProviderOperationResolutionResult,
    ProviderOperationUnavailableReason,
    checkpoint_with_provider_operation_disposition_execution_owner,
    clear_pending_provider_operation_disposition,
    load_pending_provider_operation_disposition,
    prepare_provider_operation_resolution_request,
    provider_operation_duplicate_request_risk,
    provider_operation_resolution_outcome_event_id,
    resolve_provider_operation_stage,
    validate_provider_operation_resolution_outcome_event,
)
from cayu.runtime.tool_effects import (
    ToolEffectReconciliationRequest,
    ToolEffectReconciliationTarget,
    tool_effect_receipt_digest,
)
from cayu.sessions import _model_completion_publication as model_completion_publication
from cayu.sessions import _tool_call_evidence as tool_call_evidence
from cayu.sessions._durable_operation_ownership import DurableOperationOwnership
from cayu.sessions._execution_profile_checkpoint import (
    EXECUTION_PROFILE_METADATA_KEY,
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
    active_invocation_execution_profile_matches_session_epoch,
    checkpoint_with_active_invocation_execution_profile,
    execution_profile_from_session_metadata,
)
from cayu.sessions._foreground_child_checkpoint import ForegroundChildTerminal
from cayu.sessions._invocation_lifecycle import (
    InvocationLifecycleCommandConflict,
    invocation_lifecycle_receipt_history_present,
)
from cayu.sessions._invocation_terminal_decision import (
    settled_invocation_terminal_decision_from_checkpoint,
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
    CheckpointTransform,
    RuntimePublicationReceipt,
    SessionRunFenced,
    SessionRuntimePublicationConflict,
    SessionStatusConflict,
    SessionStore,
    _activate_owned_session_run_fence,
    _activate_session_interaction,
    _activate_session_run_fence,
    _checkpoint_with_session_run_operation,
    _deactivate_session_interaction,
    _deactivate_session_run_fence,
    _incomplete_recovery_claim_from_checkpoint,
    runtime_publication_checkpoint_value_digest,
)
from cayu.sessions.cleanup import (
    RecoveryCleanup,
    RecoveryCleanupStep,
    RecoveryCleanupSupervisor,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
    INTERACTION_TERMINAL_EVENT_TYPES,
    InteractionStatus,
    InteractionSummaryEvidence,
)
from cayu.sessions.invocation import (
    SessionExecutionSource,
    SessionInvocationBinding,
    inherited_session_invocation,
)
from cayu.sessions.queries import MAX_SESSION_LIST_CURSOR_BYTES, SessionOrder, SessionQuery
from cayu.sessions.records import (
    Session,
    SessionStatus,
    _queued_dispatch_session_instance_fingerprint,
)
from cayu.sessions.recovery import (
    MAX_INCOMPLETE_SESSIONS_RECOVERY_CURSOR_BYTES,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionRecoveryResult,
    IncompleteSessionsRecoveryPage,
    IncompleteSessionsRecoveryRequest,
)
from cayu.tasks._terminalization import _terminalize_claimed_task
from cayu.tasks.dispatch import (
    _new_prepared_subagent_dispatch_envelope,
    _require_dispatch_task_authority,
    _task_matches_queued_dispatch,
)
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import Task, TaskStatus, copy_task
from cayu.tasks.store import TaskStore
from cayu.tasks.terminalization import TaskTerminalizationRequest, TaskTerminalKind
from cayu.tools import _argument_publication as tool_argument_publication
from cayu.tools.base import (
    _TOOL_POLICY_DENIAL_SOURCE,
    ToolEffect,
    ToolResult,
)
from cayu.tools.exposure import (
    ALL_REGISTERED_TOOLS_PROFILE_ID,
    ResolvedToolExposureAuthority,
    tool_capability_ceiling_from_session_metadata,
    validate_resolved_tool_exposure_authority,
)
from cayu.tools.policy import ToolPolicyDecision
from cayu.tools.rounds import ToolRoundRecoveryRequest
from cayu.vaults.redaction import SecretRedactor
from cayu.workspaces.observation_recovery import (
    await_workspace_observation_store_read,
    workspace_observations_from_checkpoint,
)

_ABANDONED_UNREPLAYABLE_TOOL_ROUND_CHECKPOINT_KEY = "abandoned_unreplayable_tool_round"
_COMPLETION_FINALIZATION_TASK_EVENT_NAMESPACE = UUID("ae86f400-31e6-4cd2-95f3-7d6f115c21a1")
_PROVIDER_OPERATION_UNAVAILABLE_INTERRUPT_NAMESPACE = UUID("c7b311fa-d36b-4ecb-a93a-c96e4c047f01")
_INCOMPLETE_RECOVERY_CURSOR_VERSION = 1
# Opaque store cursors require consuming each fetched page completely. When
# only one result slot remains, that can force one-candidate pages; cap the
# database round trips and continue through Cayu's outer cursor instead.
_INCOMPLETE_RECOVERY_MAX_STORE_PAGES = 10
_INCOMPLETE_RECOVERY_STATUS_ORDER = (
    # Process the target status before states recovery can move into it, so a
    # continuation page cannot rediscover a session this sweep interrupted.
    SessionStatus.INTERRUPTED,
    SessionStatus.INTERRUPTING,
    SessionStatus.RUNNING,
    SessionStatus.PENDING,
    SessionStatus.FAILED,
    SessionStatus.COMPLETED,
)

_TOOL_ROUND_RECOVERABLE_SESSION_STATUSES = {
    SessionStatus.RUNNING,
    SessionStatus.INTERRUPTING,
    SessionStatus.INTERRUPTED,
    SessionStatus.FAILED,
}
_UNREPLAYABLE_TOOL_ROUND_ARCHIVE_SESSION_STATUSES = frozenset(
    {
        SessionStatus.INTERRUPTING,
        SessionStatus.INTERRUPTED,
        SessionStatus.FAILED,
    }
)


def _retain_abandoned_unreplayable_tool_round(
    checkpoint: dict[str, Any],
    durable_round: dict[str, Any],
) -> dict[str, Any]:
    """Retain every opaque round while making repeated archival idempotent."""

    copied = copy_durable_record(checkpoint, "checkpoint")
    abandoned = copied.get(_ABANDONED_UNREPLAYABLE_TOOL_ROUND_CHECKPOINT_KEY)
    if abandoned is None:
        copied[_ABANDONED_UNREPLAYABLE_TOOL_ROUND_CHECKPOINT_KEY] = {
            "schema_version": 1,
            "reason": "opaque_provider_state",
            "tool_round": durable_round,
        }
        return copied
    if type(abandoned) is not dict or type(abandoned.get("tool_round")) is not dict:
        raise RuntimeError("Session retains malformed abandoned tool-round evidence.")
    if abandoned["tool_round"] == durable_round:
        return copied
    prior = abandoned.get("prior_tool_rounds", [])
    if type(prior) is not list or any(type(item) is not dict for item in prior):
        raise RuntimeError("Session retains malformed abandoned tool-round history.")
    abandoned["prior_tool_rounds"] = [*prior, abandoned["tool_round"]]
    abandoned["tool_round"] = durable_round
    return copied


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

CheckpointTransformFactory = Callable[[dict[str, Any]], CheckpointTransform]
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


def _completed_tool_round_model_step(step: int | None, *, max_steps: int) -> int:
    """Retain consumed allowance from the authenticated pending round."""
    if type(step) is not int or not 1 <= step <= max_steps:
        raise RuntimeError("Tool-round continuation has no valid consumed model-step evidence.")
    return step


def _continued_tool_exposure_profile_id(
    authority: ResolvedToolExposureAuthority | None,
) -> str:
    """Return the profile preceding a post-tool continuation model step."""

    if authority is None:
        # Pending rounds written before compact exposure authority was added used
        # the byte-compatible expose-all policy. Execution-profile validation
        # guarantees that the registered catalog and policy did not drift.
        return ALL_REGISTERED_TOOLS_PROFILE_ID
    if type(authority) is not ResolvedToolExposureAuthority:
        raise TypeError("authority must be a ResolvedToolExposureAuthority or None.")
    return authority.profile_id


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


class _RecoveryPreflightMutationRequired(RuntimeError):
    """Internal sentinel proving that recovery reached its first write boundary."""


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
class _RecoveryInvocationSemantics:
    """Exact provider-dispatch semantics reconstructed for one continuation."""

    max_steps: int
    limits: RunLimits
    budget_limits: tuple[BudgetLimit, ...]
    retry_policy: RetryPolicy
    structured_output: StructuredOutputSpec | None
    thinking: ThinkingConfig | None


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


RunSession = Callable[[RecoverySessionRunRequest], AsyncGenerator[Event, None]]
ProviderOperationFailureStream = Callable[[ProviderOperationFailureRequest], AsyncIterator[Event]]
LimitStopEventStream = Callable[[RecoveryLimitStopRequest], AsyncIterator[Event]]
TaskEventFactory = Callable[[RecoveryTaskEventRequest], Event]
RegisteredProviderResolver = Callable[[str], runtime_records.RegisteredProvider]
BudgetPolicyResolver = Callable[[], BudgetPolicy | None]


RecoveryInterruptionStream = Callable[[RecoveryInterruptionRequest], AsyncIterator[Event]]
PendingSessionInterruptCheckpoint = Callable[[dict[str, Any], datetime], CheckpointTransform]
AbandonedTurnCompleted = Callable[[RecoveryAbandonedTurnRequest], Awaitable[Session]]
IncompleteRecoveryScopeHook = Callable[[str], Awaitable[None]]
RecoveryMutationHook = Callable[[], Awaitable[None]]
RecoveryExecutionAdmissionHook = Callable[[Session], Awaitable[bool]]
IncompleteRecoveryResultHook = Callable[
    [IncompleteSessionRecoveryResult, InvocationContext | None],
    Awaitable[IncompleteSessionRecoveryResult],
]
CommittedRuntimeTaskFailureRecovery = Callable[
    [Session, dict[str, Any] | None, SessionStatus, RecoveryMutationHook],
    Awaitable[IncompleteSessionRecoveryResult | None],
]
MaterializeDeferredInteractionInput = Callable[[str], Awaitable[bool]]
ResumeInteraction = Callable[
    [
        Session,
        runtime_records.RegisteredAgentState,
        runtime_records.RegisteredEnvironment | None,
    ],
    Awaitable[Event | None],
]
InteractionTransitionReplayFailures = Callable[
    [BaseException],
    tuple[Exception, ...] | None,
]


class RecoveryCoordinator:
    """Continue paused work and repair incomplete sessions from durable state."""

    def __init__(
        self,
        *,
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
        tool_round_executor: ToolRoundExecutor,
        pending_tool_round_recovery: PendingToolRoundRecovery,
        deferred_input: DeferredInteractionInput,
        workspace_observation_recovery: WorkspaceObservationRecovery,
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
        checkpoint_transform: CheckpointTransformFactory,
        effective_retry_policy: EffectiveRetryPolicy,
        run_session: RunSession,
        terminal_event_publication: TerminalEventPublication,
        task_event: TaskEventFactory,
        resolve_registered_agent: RegisteredAgentResolver,
        resolve_registered_provider: RegisteredProviderResolver,
        resolve_registered_environment: RegisteredEnvironmentResolver,
        resolve_budget_policy: BudgetPolicyResolver,
        execution_profile_continuation: ExecutionProfileContinuation,
        model_completion_recovery: ModelCompletionRecovery,
        user_input_evidence: UserInputRecoveryEvidence,
        terminal_evidence: TerminalEvidenceReader,
        recovery_ownership: RecoveryOwnership,
        terminal_finalization: TerminalEvidenceFinalization,
        session_finalization: SessionFinalization,
        interaction_transition_replay_failures: InteractionTransitionReplayFailures,
        recovery_cleanup_supervisor: RecoveryCleanupSupervisor,
        resource_access_policy: ResourceAccessPolicy | None = None,
        human_review_policy: HumanReviewPolicy | None = None,
        runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...] = (),
        loop_policies: tuple[LoopPolicy, ...] = (),
    ) -> None:
        self._resource_access_policy = resource_access_policy
        self._human_review_policy = human_review_policy
        self._session_store = session_store
        self._require_participant_execution = require_participant_execution
        self._foreground_gate_policy_owner = foreground_gate_policy_owner
        self._task_store = task_store
        self._event_writer = event_writer
        self._session_control = session_control
        self._environment_lifecycle = environment_lifecycle
        self._run_limit_controller = run_limit_controller
        self._tool_round_executor = tool_round_executor
        self._pending_tool_round_recovery = pending_tool_round_recovery
        self._deferred_input = deferred_input
        self._workspace_observation_recovery = workspace_observation_recovery
        self._tool_invocation = tool_round_executor.invocation
        self._secret_redactor = secret_redactor
        self._clock = clock
        self._checkpoint_transform = checkpoint_transform
        self._effective_retry_policy = effective_retry_policy
        self._run_session = run_session
        self._terminal_event_publication = terminal_event_publication
        self._task_event = task_event
        self._resolve_registered_agent = resolve_registered_agent
        self._resolve_registered_provider = resolve_registered_provider
        self._resolve_registered_environment = resolve_registered_environment
        self._resolve_budget_policy = resolve_budget_policy
        self._execution_profile_continuation = execution_profile_continuation
        self._model_completion_recovery = model_completion_recovery
        self._user_input_evidence = user_input_evidence
        self._terminal_evidence = terminal_evidence
        self._recovery_ownership = recovery_ownership
        self.terminal_finalization = terminal_finalization
        self._session_finalization = session_finalization
        self._interaction_transition_replay_failures = interaction_transition_replay_failures
        if type(recovery_cleanup_supervisor) is not RecoveryCleanupSupervisor:
            raise TypeError("recovery_cleanup_supervisor must be a RecoveryCleanupSupervisor.")
        self._recovery_cleanup_supervisor = recovery_cleanup_supervisor
        self._runtime_hooks = runtime_hooks
        self._loop_policies = loop_policies
        self._committed_runtime_task_failure_recovery: (
            CommittedRuntimeTaskFailureRecovery | None
        ) = None
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

    def bind_committed_runtime_task_failure_recovery(
        self,
        recovery: CommittedRuntimeTaskFailureRecovery,
    ) -> None:
        """Bind the session owner that can finish one committed terminal winner."""

        if not callable(recovery):
            raise TypeError("recovery must be callable.")
        if self._committed_runtime_task_failure_recovery is not None:
            raise RuntimeError("Committed runtime task failure recovery is already bound.")
        self._committed_runtime_task_failure_recovery = recovery

    async def _cleanup_recovery_handoff(
        self,
        *,
        stream: AsyncGenerator[Event, None] | None,
        session_id: str,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        authoritative_failure: BaseException | None,
        finalize_abandoned: bool,
        release_run_fence: bool,
        abort_environment_setup: bool = True,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> None:
        if invocation_context is not None:
            # A nested continuation can atomically consume a queued message and
            # transfer the same run epoch from interaction A to interaction B.
            # Recovery entrypoints retain their original local A context, so
            # derive cleanup authority from the durable checkpoint before any
            # owner/fence cleanup observes it.  ``with_queued_interaction``
            # authenticates the complete stable profile and collaborator set;
            # incompatible durable state remains fail-closed below.
            current_session = await self._session_store.load(session_id)
            current_checkpoint = await self._session_store.load_checkpoint(session_id)
            current_profile = active_invocation_execution_profile_from_checkpoint(
                current_checkpoint
            )
            if (
                current_session is not None
                and current_profile is not None
                and current_profile.run_epoch == invocation_context.active_profile.run_epoch
                and current_profile.interaction_id
                != invocation_context.active_profile.interaction_id
            ):
                invocation_context = invocation_context.with_queued_interaction(
                    current_session,
                    active_profile=current_profile,
                )
        cleanup_steps: list[tuple[str, RecoveryCleanup]] = []
        if stream is not None:
            cleanup_steps.append(("nested stream close", stream.aclose))
        if finalize_abandoned:
            cleanup_steps.append(
                (
                    "abandoned session finalization",
                    lambda: self._session_finalization.finalize_abandoned_session_by_id(
                        session_id,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    ),
                )
            )
        if abort_environment_setup and authoritative_failure is not None:
            cleanup_steps.append(
                (
                    "environment setup abort",
                    lambda: self._environment_lifecycle.abort_environment_setup(
                        session_id=session_id,
                        original_error=authoritative_failure,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    ),
                )
            )
        if release_run_fence:
            cleanup_steps.append(
                (
                    "run fence release",
                    lambda: self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                        session_id=session_id,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    ),
                )
            )
        await self._recovery_ownership.run_cleanup_steps(
            authoritative_failure=authoritative_failure,
            steps=tuple(cleanup_steps),
        )

    async def _cleanup_entrypoint_handoff(
        self,
        *,
        stream: AsyncGenerator[Event, None] | None,
        session_id: str,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        authoritative_failure: BaseException | None,
        finalize_abandoned: bool,
        release_run_fence: bool,
        abort_environment_setup: bool = True,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> None:
        try:
            await self._cleanup_recovery_handoff(
                stream=stream,
                session_id=session_id,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=finalize_abandoned,
                release_run_fence=release_run_fence,
                abort_environment_setup=abort_environment_setup,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            )
        finally:
            # Cleanup can run in a shielded child task with copied context. The
            # public caller must never retain a stale run epoch after handoff.
            _deactivate_session_run_fence(session_id)
            _deactivate_session_interaction(session_id)

    async def _activate_latest_open_interaction(self, session_id: str) -> str | None:
        records = await self._session_store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not records or records[0].event.type in INTERACTION_TERMINAL_EVENT_TYPES:
            return None
        interaction_id = records[0].event.interaction_id
        if interaction_id is None:
            raise RuntimeError("Interaction lifecycle event has no interaction identity.")
        _activate_session_interaction(session_id, interaction_id)
        return interaction_id

    async def _transition_recovery_session_to_running(
        self,
        loaded_session: Session,
        *,
        checkpoint: dict[str, Any] | None,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        from_statuses: set[SessionStatus] | None = None,
        checkpoint_transform: CheckpointTransform | None = None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile | None = None,
        preserve_open_interaction_on_failure: bool = False,
        before_mutation: RecoveryMutationHook | None = None,
        before_resume: RecoveryExecutionAdmissionHook | None = None,
        after_admission: RecoveryMutationHook | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> tuple[Session, Event | None]:
        """Claim a paused session without leaving cancellation outcome-uncertain.

        Session stores activate run fences in task-local context after the durable
        transition commits. Keeping the transition in a shielded child task lets it
        reach a definite result even if the caller is cancelled at that boundary.
        A successful claim is then activated in the caller's context. If cancellation
        arrived, the claim is finalized and released before that cancellation is
        propagated.
        """
        if type(preserve_open_interaction_on_failure) is not bool:
            raise TypeError("preserve_open_interaction_on_failure must be a bool.")
        if invocation_context is not None and (
            invocation_context.binding.session_id != loaded_session.id
            or invocation_context.binding.session_instance_id != loaded_session.instance_id
            or invocation_context.binding.run_epoch != loaded_session.run_epoch
            or registered_agent is not invocation_context.registered_agent
            or registered_environment is not invocation_context.registered_environment
            or execution_profile_snapshot is None
            or invocation_context.profile is not execution_profile_snapshot.profile
        ):
            raise RuntimeError("Recovery admission substituted frozen invocation authority.")
        expected_statuses = (
            {SessionStatus.INTERRUPTED} if from_statuses is None else set(from_statuses)
        )
        if before_mutation is not None:
            preflight_events = await self._session_store.query_events(
                EventQuery(
                    session_id=loaded_session.id,
                    event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                )
            )
            if (
                not preflight_events
                or preflight_events[0].event.type in INTERACTION_TERMINAL_EVENT_TYPES
            ):
                raise RuntimeError(
                    "Pending recovery state has no open interaction. "
                    "Pre-interaction prerelease recovery state is unsupported."
                )
            if preflight_events[0].event.interaction_id is None:
                raise RuntimeError("Interaction lifecycle event has no interaction identity.")
            await before_mutation()
        if loaded_session.status in expected_statuses:
            (
                loaded_session,
                checkpoint,
            ) = await self.terminal_finalization.reconcile_before_continuation(
                session=loaded_session,
                checkpoint=checkpoint,
            )
            if execution_profile_snapshot is not None:
                reconciled_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
                if (
                    reconciled_profile is None
                    or reconciled_profile.session_id != execution_profile_snapshot.session_id
                    or reconciled_profile.interaction_id
                    != execution_profile_snapshot.interaction_id
                    or reconciled_profile.profile != execution_profile_snapshot.profile
                ):
                    raise RuntimeError(
                        "Active invocation profile changed during terminal-evidence recovery."
                    )
                execution_profile_snapshot = execution_profile_snapshot.model_copy(
                    update={"run_epoch": reconciled_profile.run_epoch}
                )
        session_id = loaded_session.id
        latest_events = await self._session_store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not latest_events or latest_events[0].event.type in INTERACTION_TERMINAL_EVENT_TYPES:
            raise RuntimeError(
                "Pending recovery state has no open interaction. "
                "Pre-interaction prerelease recovery state is unsupported."
            )
        interaction_id = latest_events[0].event.interaction_id
        if interaction_id is None:
            raise RuntimeError("Interaction lifecycle event has no interaction identity.")
        if (
            execution_profile_snapshot is not None
            and execution_profile_snapshot.interaction_id != interaction_id
        ):
            raise RuntimeError(
                "Active invocation execution profile belongs to another interaction."
            )
        run_operation_id = str(uuid4())

        def reject_active_incomplete_recovery(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            # Expired recovery ownership is reconciled through the store-time
            # claim path above.  This final typed rebind must reject any marker
            # that appeared after that reconciliation instead of interpreting
            # its lease with a worker clock.
            if _incomplete_recovery_claim_from_checkpoint(checkpoint) is not None:
                raise RuntimeError("Session has an active incomplete-session recovery operation.")
            if checkpoint_transform is not None:
                checkpoint = checkpoint_transform(current_session, checkpoint)
            if execution_profile_snapshot is not None:
                current_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
                if current_profile is None or current_profile != execution_profile_snapshot:
                    raise SessionRunFenced(
                        "Active invocation profile changed before continuation claimed it."
                    )
                checkpoint = checkpoint_with_active_invocation_execution_profile(
                    checkpoint,
                    session_id=current_session.id,
                    interaction_id=interaction_id,
                    run_epoch=current_session.run_epoch + 1,
                    profile=execution_profile_snapshot.profile,
                    expected=execution_profile_snapshot,
                )
            return _checkpoint_with_session_run_operation(
                checkpoint=checkpoint,
                current_session=current_session,
                operation_id=run_operation_id,
            )

        transition_task = asyncio.create_task(
            self._recovery_ownership.fence_or_rebind_active_invocation(
                session_id,
                statuses=expected_statuses,
                target_status=SessionStatus.RUNNING,
                checkpoint_transform=reject_active_incomplete_recovery,
            )
        )
        cancellation: asyncio.CancelledError | None = None
        transition_failure: BaseException | None = None
        while not transition_task.done():
            try:
                await asyncio.shield(transition_task)
            except asyncio.CancelledError as exc:
                if transition_task.cancelled():
                    transition_failure = exc
                    break
                if cancellation is None:
                    cancellation = exc
            except BaseException as exc:
                transition_failure = exc
                break

        session: Session | None = None
        if transition_failure is None:
            try:
                session = transition_task.result()
            except BaseException as exc:
                transition_failure = exc

        if session is None:
            if cancellation is not None:
                if transition_failure is not None:
                    cancellation.add_note(
                        "Continuation recovery transition also failed after cancellation: "
                        f"{type(transition_failure).__name__}."
                    )
                raise cancellation from transition_failure
            if transition_failure is None:
                raise RuntimeError("Continuation recovery transition completed without a session.")
            raise transition_failure

        # The transition activated this epoch only in the child task's copied
        # context. The caller owns all subsequent writes and cleanup.
        _activate_session_run_fence(session)
        _activate_session_interaction(session.id, interaction_id)
        rebound_invocation_context = (
            None
            if invocation_context is None or execution_profile_snapshot is None
            else invocation_context.with_rebound_session(
                session,
                active_profile=execution_profile_snapshot.model_copy(
                    update={"run_epoch": session.run_epoch}
                ),
            )
        )
        post_admission_authority_confirmed = after_admission is None
        try:
            if cancellation is not None:
                raise cancellation
            if after_admission is not None:
                await after_admission()
                post_admission_authority_confirmed = True
            # Continuations can execute accepted tools before entering the model
            # loop. Publish this epoch's owner before dispatching any such work.
            await self._session_control.execution_presence.ensure(session)
            if before_resume is not None and not await before_resume(session):
                return session, None
            resumed_event = await self._session_finalization.resume_recovery_interaction(
                session,
                registered_agent,
                registered_environment,
            )
        except BaseException as exc:
            try:
                cleanup_steps: list[tuple[str, RecoveryCleanup]] = []
                if not preserve_open_interaction_on_failure:
                    cleanup_steps.append(
                        (
                            "abandoned session finalization",
                            lambda: self._session_finalization.finalize_abandoned_session_by_id(
                                session.id,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                execution_profile=(
                                    None
                                    if execution_profile_snapshot is None
                                    else execution_profile_snapshot.profile
                                ),
                                invocation_context=rebound_invocation_context,
                                run_terminal_hooks=post_admission_authority_confirmed,
                            ),
                        )
                    )
                cleanup_steps.append(
                    (
                        "run fence release",
                        lambda: (
                            self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                                session_id=session.id,
                                execution_profile=(
                                    None
                                    if execution_profile_snapshot is None
                                    else execution_profile_snapshot.profile
                                ),
                                invocation_context=rebound_invocation_context,
                            )
                        ),
                    )
                )
                await self._recovery_ownership.run_cleanup_steps(
                    authoritative_failure=exc,
                    steps=tuple(cleanup_steps),
                )
            finally:
                _deactivate_session_run_fence(session.id)
                _deactivate_session_interaction(session.id)
            raise
        if cancellation is None:
            return session, resumed_event

        try:
            cleanup_steps = []
            if not preserve_open_interaction_on_failure:
                cleanup_steps.append(
                    (
                        "abandoned session finalization",
                        lambda: self._session_finalization.finalize_abandoned_session_by_id(
                            session.id,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            execution_profile=(
                                None
                                if execution_profile_snapshot is None
                                else execution_profile_snapshot.profile
                            ),
                            invocation_context=rebound_invocation_context,
                        ),
                    )
                )
            cleanup_steps.append(
                (
                    "run fence release",
                    lambda: self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                        session_id=session.id,
                        execution_profile=(
                            None
                            if execution_profile_snapshot is None
                            else execution_profile_snapshot.profile
                        ),
                        invocation_context=rebound_invocation_context,
                    ),
                )
            )
            await self._recovery_ownership.run_cleanup_steps(
                authoritative_failure=cancellation,
                steps=tuple(cleanup_steps),
            )
        finally:
            # Shielded cleanup runs in a copied context. Never leave the caller's
            # task-local epoch active if it catches and handles the cancellation.
            _deactivate_session_run_fence(session.id)
            _deactivate_session_interaction(session.id)
        raise cancellation

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

        session, resumed_event = await self._transition_recovery_session_to_running(
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
            await self._cleanup_entrypoint_handoff(
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

        session, resumed_event = await self._transition_recovery_session_to_running(
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
            await self._cleanup_entrypoint_handoff(
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
            session, resumed_event = await self._transition_recovery_session_to_running(
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
            await self._cleanup_entrypoint_handoff(
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
            for settlement_event in await self._settle_provider_operation_disposition_reservations(
                pending=pending,
                result=durable_result,
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
            recovered = await self._recover_incomplete_session_scoped(
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
        disposition_stream = self._finish_pending_provider_operation_disposition(
            pending=pending,
            participant_context=participant_context,
            result=durable_result,
            task_id=task_id,
            task_worker_id=request.task_worker_id,
            task_handoff_id=task_handoff_id,
            after_admission=after_admission,
        )
        try:
            try:
                async for event in disposition_stream:
                    yield event
            except ExceptionGroup as replay_failure:
                if self._interaction_transition_replay_failures(
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

    async def _provider_operation_disposition_effect_is_durable(
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

    async def _retire_completed_provider_operation_disposition(
        self,
        *,
        pending: ProviderOperationPendingDisposition,
        result: ProviderOperationResolutionResult,
        terminal_hook_authority: RecoveryTerminalEventRequest | None = None,
    ) -> bool:
        await self._settle_provider_operation_disposition_reservations(
            pending=pending,
            result=result,
        )
        if not await self._provider_operation_disposition_effect_is_durable(
            pending=pending,
            result=result,
            terminal_hook_authority=terminal_hook_authority,
        ):
            return False
        await clear_pending_provider_operation_disposition(self._session_store, pending)
        return True

    async def _settle_provider_operation_disposition_reservations(
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

    async def _finish_pending_provider_operation_disposition(
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

        for settlement_event in await self._settle_provider_operation_disposition_reservations(
            pending=pending,
            result=result,
        ):
            yield settlement_event

        if await self._retire_completed_provider_operation_disposition(
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
            if not await self._retire_completed_provider_operation_disposition(
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

            session, resumed_event = await self._transition_recovery_session_to_running(
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

            fallback_stream = self._run_pending_provider_operation_fallback(
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
                await self._cleanup_entrypoint_handoff(
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

    async def _automatic_provider_disposition_task_context(
        self,
        pending: ProviderOperationPendingDisposition,
    ) -> tuple[str | None, bool]:
        """Resolve direct task authority or require an elected typed continuation.

        Generic session recovery carries no task-worker credential. It may finish
        an ordinary workerless attachment, but it must never consume or bypass an
        interrupted-handoff generation owned by an elected worker. A terminal
        task likewise requires its exact receipt-authenticated typed replay.
        """

        stage = await self._session_store.load_model_completion_stage(
            pending.session_id,
            pending.stage_id,
        )
        if stage is None:
            raise RuntimeError("Resolved provider-operation stage is missing.")
        recovery_context = model_completion_recovery_context_from_stage(stage)
        if recovery_context is None:
            raise RuntimeError(
                "Provider-operation disposition requires durable model-completion context."
            )
        task_id = recovery_context.task_id
        if task_id is None:
            return None, False
        if self._task_store is None:
            raise RuntimeError("Attached provider disposition requires a task store.")
        task = await self._task_store.load_task(task_id)
        session = await self._session_store.load(pending.session_id)
        if task is None or session is None:
            raise RuntimeError("Attached provider disposition lost its task or session.")
        if task.session_id != session.id or task.session_instance_id != session.instance_id:
            raise RuntimeError("Attached provider disposition changed task-session identity.")
        if task.status is TaskStatus.FAILED:
            direct_failure = await load_direct_task_failure_replay(
                self._task_store,
                task_id=task_id,
                session_id=session.id,
                session_instance_id=session.instance_id,
                expected_error=provider_operation_task_failure_payload(session_id=session.id),
                claimed_terminalization_idempotency_key=(
                    runtime_task_terminalization_idempotency_key(
                        task_id=task_id,
                        session_id=session.id,
                        kind=TaskTerminalKind.FAILED,
                    )
                ),
            )
            return task_id, direct_failure is None
        if task.status is not TaskStatus.RUNNING or task.worker_id is not None:
            return task_id, True
        try:
            await self._task_store.load_direct_attached_task_resume(
                task_id,
                session_id=session.id,
                session_instance_id=session.instance_id,
            )
        except (KeyError, NotImplementedError, ValueError):
            return task_id, True
        return task_id, False

    async def _run_pending_provider_operation_fallback(
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
        session_stream = self._run_session(
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
                    if not await self._retire_completed_provider_operation_disposition(
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
            if not await self._retire_completed_provider_operation_disposition(
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
                if not await self._retire_completed_provider_operation_disposition(
                    pending=pending,
                    result=result,
                ):
                    raise RuntimeError(
                        "Provider-operation fallback limit has no durable terminal outcome."
                    )
            cleanup = (
                self._cleanup_entrypoint_handoff
                if release_run_fence_on_cleanup
                else self._cleanup_recovery_handoff
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

        session, resumed_event = await self._transition_recovery_session_to_running(
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
            await self._cleanup_entrypoint_handoff(
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
            interaction_id = await self._activate_latest_open_interaction(loaded_session.id)
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
                await self._cleanup_entrypoint_handoff(
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
        session_stream = self._run_session(
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
            base_round_redactor = self._tool_round_executor.redactor_for_tool_calls(
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
            base_round_redactor = self._tool_round_executor.redactor_for_tool_calls(
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
            task_failed_template = self._task_event(
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
                await self._cleanup_recovery_handoff(
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
            await self._cleanup_recovery_handoff(
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
                await self._cleanup_recovery_handoff(
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
            await self._cleanup_recovery_handoff(
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
                await self._cleanup_recovery_handoff(
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
            continuation = await self._prepare_recovered_tool_round_continuation(
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
            session_stream = self._run_session(continuation)
            async for event in session_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            await self._cleanup_recovery_handoff(
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
            redactor=self._tool_round_executor.redactor_for_tool_calls(
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
            continuation = await self._prepare_recovered_tool_round_continuation(
                session=session,
                participant_context=participant_context,
                pending_round=pending_round,
                invocation_semantics=invocation_semantics,
                invocation_context=invocation_context,
                request_metadata={},
                task_worker_id=request.task_worker_id,
                task_handoff_id=request.task_handoff_id,
            )
            stream = self._run_session(continuation)
            async for event in stream:
                yield event
        except BaseException as exc:
            failure = exc
            raise
        finally:
            await self._cleanup_recovery_handoff(
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

    async def _prepare_recovered_tool_round_continuation(
        self,
        *,
        session: Session,
        pending_round: pending_rounds.PendingToolRound,
        invocation_semantics: _RecoveryInvocationSemantics,
        invocation_context: InvocationContext,
        request_metadata: dict[str, Any],
        task_worker_id: str | None,
        task_handoff_id: str | None,
        completed_model_step: int | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> RecoverySessionRunRequest:
        """Restore a claimed round without deciding its outcome or acquiring a new owner.

        Receipt settlement and operator recovery share continuation preparation.
        The caller retains the recovery claim, stream supervision, and cleanup.
        """
        if type(invocation_context) is not InvocationContext:
            raise TypeError("Recovered continuation requires authenticated invocation authority.")
        invocation_context._validate()
        invocation_context.with_admitted_session(session)
        if pending_round.agent_name != invocation_context.registered_agent.spec.name:
            raise RuntimeError("Recovered continuation belongs to another agent.")
        transcript = await self._session_store.load_transcript(session.id)
        continued_run_limit_accounting = pending_round.run_limit_accounting
        if continued_run_limit_accounting is not None:
            recovery_events = await self._session_store.load_events(session.id)
            continued_run_limit_accounting = rebase_run_limit_accounting_context(
                continued_run_limit_accounting,
                session_id=session.id,
                limits=invocation_semantics.limits,
                budget_limits=request_budget_limits_for_session(
                    limits=invocation_semantics.budget_limits,
                    agent_name=invocation_context.registered_agent.spec.name,
                    causal_budget_id=session.causal_budget_id,
                ),
                events=recovery_events,
                reset_run_limits=False,
                reset_budgets=False,
                now=self._clock(),
            )
        return RecoverySessionRunRequest(
            session=session,
            participant_context=participant_context,
            invocation_context=invocation_context,
            messages=transcript,
            messages_to_append=[],
            max_steps=invocation_semantics.max_steps,
            limits=invocation_semantics.limits,
            budget_limits=invocation_semantics.budget_limits,
            retry_policy=invocation_semantics.retry_policy,
            structured_output=invocation_semantics.structured_output,
            thinking=invocation_semantics.thinking,
            request_metadata=copy_durable_metadata(request_metadata, "recovery.request_metadata"),
            task_id=pending_round.task_id,
            task_worker_id=task_worker_id,
            task_handoff_id=task_handoff_id,
            start_event_type=None,
            start_event_payload={},
            start_task_on_enter=False,
            release_run_fence_on_exit=False,
            run_limit_accounting=continued_run_limit_accounting,
            completed_tool_round_model_step=_completed_tool_round_model_step(
                pending_round.model_step if completed_model_step is None else completed_model_step,
                max_steps=invocation_semantics.max_steps,
            ),
            previous_tool_exposure_profile_id=(
                _continued_tool_exposure_profile_id(pending_round.tool_exposure)
            ),
        )

    async def recover_incomplete_session(
        self,
        request: IncompleteSessionRecoveryRequest,
        *,
        before_mutation: RecoveryMutationHook | None = None,
        participant_context: CollaborationAccessContext | None = None,
        retain_open_interaction_invocation: bool = False,
        retain_invocation_context: Callable[[InvocationContext], None] | None = None,
        preserve_interaction_id: str | None = None,
        _work_attempt: WorkAttemptInvocationAuthority | None = None,
        execution_to_wait: _ExternalExecutionToWait | None = None,
    ) -> IncompleteSessionRecoveryResult:
        """Repair incomplete state, including an exact post-close child continuation.

        Closed tools are never replayed. A retained foreground action-close
        marker can authorize the next model step only after profile admission
        and a fenced recovery claim; uncertain model work retains its own gate.
        """
        if _work_attempt is not None:
            if type(_work_attempt) is not WorkAttemptInvocationAuthority:
                raise TypeError("Governed recovery requires authenticated work-attempt authority.")
            admission = _work_attempt.admission
            if (
                before_mutation is None
                or preserve_interaction_id != admission.interaction_id
                or request.session_id != admission.session_id
            ):
                raise ValueError("Governed recovery conflicts with its exact claim boundary.")
        if retain_invocation_context is not None and not retain_open_interaction_invocation:
            raise ValueError("Invocation context retention requires an open recovery invocation.")
        if preserve_interaction_id is not None:
            require_clean_nonblank(preserve_interaction_id, "preserve_interaction_id")
            if before_mutation is None or retain_open_interaction_invocation:
                raise ValueError(
                    "Execution-owner replacement requires an exact claim hook and full cleanup."
                )
        retained_invocation_context: InvocationContext | None = None

        def retain_context(context: InvocationContext) -> None:
            nonlocal retained_invocation_context
            retained_invocation_context = context
            if retain_invocation_context is not None:
                retain_invocation_context(context)

        session = await self._session_store.load(request.session_id)
        if session is None:
            raise KeyError(f"Session not found: {request.session_id}") from None
        if _work_attempt is not None:
            admitted = _work_attempt.admission
            profile = execution_profile_from_session_metadata(session.metadata)
            if (
                SessionInvocationBinding(
                    id=session.id,
                    session_instance_id=session.instance_id,
                    invocation=session.invocation,
                )
                != admitted.session_invocation
                or profile is None
                or profile.fingerprint != admitted.source_execution_profile_fingerprint
            ):
                raise ValueError("Governed recovery session authority changed before admission.")
        recovered = await self._recover_incomplete_session_scoped(
            execution_to_wait=execution_to_wait,
            participant_context=participant_context,
            _work_attempt=_work_attempt,
            preserve_interaction_id=preserve_interaction_id,
            session=session,
            inactive_for_seconds=request.inactive_for_seconds,
            reason=request.reason,
            metadata=request.metadata,
            before_mutation=before_mutation,
            retain_open_interaction_invocation=retain_open_interaction_invocation,
            retain_invocation_context=(
                retain_context if retain_open_interaction_invocation else None
            ),
        )
        return await self._finish_provider_operation_disposition_after_recovery(
            recovered,
            before_mutation=before_mutation,
            invocation_context=retained_invocation_context,
        )

    async def interrupt_incomplete_session_for_manual_tool_recovery(
        self,
        request: IncompleteSessionRecoveryRequest,
    ) -> IncompleteSessionRecoveryResult:
        """Fence a stale run and retain its pending tool effect for a typed decision."""

        session = await self._session_store.load(request.session_id)
        if session is None:
            raise KeyError(f"Session not found: {request.session_id}") from None
        return await self._recover_incomplete_session_scoped(
            session=session,
            inactive_for_seconds=request.inactive_for_seconds,
            reason=request.reason,
            metadata=request.metadata,
            interrupt_for_manual_tool_recovery=True,
        )

    async def _finish_provider_operation_disposition_after_recovery(
        self,
        recovered: IncompleteSessionRecoveryResult,
        *,
        before_mutation: RecoveryMutationHook | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> IncompleteSessionRecoveryResult:
        if IncompleteSessionRecoveryAction.PENDING_ALLOCATION_CLEANUP in recovered.actions:
            return recovered
        pending_resolution = await load_pending_provider_operation_disposition(
            self._session_store,
            recovered.session_id,
        )
        if pending_resolution is None:
            return recovered
        pending, result = pending_resolution
        (
            task_id,
            requires_typed_continuation,
        ) = await self._automatic_provider_disposition_task_context(pending)
        if requires_typed_continuation:
            return recovered
        if before_mutation is not None:
            await before_mutation()
        disposition_events = [
            event
            async for event in self._finish_pending_provider_operation_disposition(
                pending=pending,
                result=result,
                invocation_context=invocation_context,
                task_id=task_id,
            )
        ]
        current = await self._recovery_ownership.require_session(recovered.session_id)
        retained_actions = tuple(
            action
            for action in recovered.actions
            if action
            not in {
                IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,
                IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,
            }
        )
        return recovered.model_copy(
            update={
                "status": current.status,
                "actions": (
                    *retained_actions,
                    IncompleteSessionRecoveryAction.REPAIRED_PROVIDER_OPERATION_RESOLUTION,
                ),
                "events": (*recovered.events, *disposition_events),
                "message": "Finished the accepted provider-operation resolution.",
            }
        )

    async def _pending_durable_subagent_recovery_guard(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        previous_status: SessionStatus,
    ) -> IncompleteSessionRecoveryResult | None:
        """Keep one prepared child under its queue owner instead of abandoning it."""

        if session.status is not SessionStatus.PENDING or session.run_epoch != 0:
            return None
        intents = durable_subagent_submissions_from_checkpoint(checkpoint)
        subagent = session.metadata.get("subagent")
        if not intents:
            if type(subagent) is dict and subagent.get("mode") == "durable":
                raise RuntimeError("Pending durable subagent has no prepared execution intent.")
            return None
        if len(intents) != 1:
            raise RuntimeError("Pending durable subagent has ambiguous execution intents.")
        intent = intents[0]
        idempotency_key = intent.idempotency_key
        if (
            type(subagent) is not dict
            or subagent.get("mode") != "durable"
            or subagent.get("idempotency_key") != idempotency_key
        ):
            raise RuntimeError("Pending durable subagent has conflicting spawn metadata.")
        parent = await self._session_store.load(intent.parent_session_id)
        if parent is None:
            raise RuntimeError("Pending durable subagent has no durable parent session.")
        parent_checkpoint = await self._session_store.load_checkpoint(parent.id)
        parent_intent = durable_subagent_submission_from_checkpoint(
            parent_checkpoint,
            idempotency_key=idempotency_key,
        )
        parent_seed = durable_subagent_submission_seed_from_checkpoint(
            parent_checkpoint,
            idempotency_key=idempotency_key,
        )
        parent_receipt = durable_subagent_submission_receipt_from_checkpoint(
            parent_checkpoint,
            idempotency_key=idempotency_key,
        )
        if parent_intent is not None and parent_seed is not None and parent_receipt is None:
            require_durable_subagent_intent_matches_seed(parent_intent, parent_seed)
            parent_authority_matches = parent_intent == intent
        elif parent_intent is None and parent_receipt is not None:
            require_durable_subagent_receipt_matches_intent(parent_receipt, intent)
            if parent_seed is not None:
                require_durable_subagent_intent_matches_seed(intent, parent_seed)
                require_durable_subagent_receipt_matches_seed(parent_receipt, parent_seed)
            parent_authority_matches = True
        else:
            raise RuntimeError(
                "Pending durable subagent has incomplete parent submission authority."
            )
        if (
            not parent_authority_matches
            or intent.child_session_id != session.id
            or intent.parent_session_instance_fingerprint
            != _queued_dispatch_session_instance_fingerprint(parent)
            or session.parent_session_id != parent.id
            or session.causal_budget_id != intent.causal_budget_id
            or session.agent_name != intent.agent_name
            or session.provider_name != intent.child_provider_name
            or session.model != intent.child_model
            or session.runtime_name != intent.child_runtime_name
            or session.runtime_version != intent.child_runtime_version
            or session.environment_name != intent.environment_name
            or session.invocation
            != inherited_session_invocation(
                parent.invocation,
                source=SessionExecutionSource.SUBAGENT,
            )
            or execution_profile_from_session_metadata(session.metadata)
            != intent.child_execution_profile
            or session.metadata.get("subagent") != intent.request.metadata.get("subagent")
        ):
            raise RuntimeError(
                "Pending durable subagent conflicts with its prepared execution authority."
            )

        if self._task_store is None:
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                events=(),
                message=(
                    "Prepared durable subagent requires its task store for reconciliation; "
                    "generic abandonment recovery skipped."
                ),
            )
        task = await self._task_store.load_task(intent.queue_task_id)
        if task is None:
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                events=(),
                message=(
                    "Prepared durable subagent is awaiting recoverable parent queue "
                    "publication; generic abandonment recovery skipped."
                ),
            )
        envelope = _new_prepared_subagent_dispatch_envelope(
            intent=intent,
            session_instance_fingerprint=_queued_dispatch_session_instance_fingerprint(session),
        )
        if not _task_matches_queued_dispatch(
            task,
            task_type=intent.queue_task_type,
            parent_task_id=intent.parent_task_id,
            envelope=envelope,
        ):
            raise RuntimeError("Pending durable subagent queue task has conflicting authority.")
        parent_binding = await self._session_store.load_invocation_snapshot(parent.id)
        if parent_binding is None:
            raise RuntimeError("Pending durable subagent parent invocation is unavailable.")
        _require_dispatch_task_authority(
            task,
            envelope=envelope,
            session_binding=parent_binding,
            task_type=intent.queue_task_type,
        )
        if task.status is TaskStatus.COMPLETED:
            raise RuntimeError("Pending durable subagent conflicts with a terminal queue task.")
        if task.status in {TaskStatus.FAILED, TaskStatus.CANCELLED}:
            # Exact queue authority proves that no worker can admit this child.
            # Let ordinary claimed recovery durably close the pristine pending
            # session rather than preserving it as live work forever.
            return None
        return IncompleteSessionRecoveryResult(
            session_id=session.id,
            previous_status=previous_status,
            status=session.status,
            actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
            events=(),
            message=(
                "Prepared durable subagent remains owned by its exact nonterminal queue task; "
                "generic abandonment recovery skipped."
            ),
        )

    async def recover_incomplete_sessions(
        self,
        request: IncompleteSessionsRecoveryRequest,
        *,
        before_recovery: IncompleteRecoveryScopeHook | None = None,
        before_mutation: IncompleteRecoveryScopeHook | None = None,
        after_recovery: IncompleteRecoveryScopeHook | None = None,
        reconcile_result: IncompleteRecoveryResultHook | None = None,
    ) -> IncompleteSessionsRecoveryPage:
        """Fault-isolate one bounded, resumable recovery page."""
        requested_statuses = tuple(
            status for status in _INCOMPLETE_RECOVERY_STATUS_ORDER if status in request.statuses
        )
        start_index = 0
        initial_session_cursor: str | None = None
        if request.cursor is not None:
            cursor_status, initial_session_cursor = _decode_incomplete_recovery_cursor(
                request.cursor,
                request=request,
            )
            start_index = requested_statuses.index(cursor_status)

        results: list[IncompleteSessionRecoveryResult] = []
        result_session_ids: set[str] = set()
        inspected_session_count = 0
        store_page_count = 0

        def continuation_after_status(status_index: int) -> str | None:
            next_index = status_index + 1
            if next_index >= len(requested_statuses):
                return None
            return _encode_incomplete_recovery_cursor(
                status=requested_statuses[next_index],
                session_cursor=None,
                request=request,
            )

        for status_index in range(start_index, len(requested_statuses)):
            status = requested_statuses[status_index]
            terminal_status = status in _RECOVERY_RESUMABLE_SESSION_STATUSES
            cursor = initial_session_cursor if status_index == start_index else None
            seen_cursors = set() if cursor is None else {cursor}
            while (
                len(results) < request.limit and inspected_session_count < request.inspection_limit
            ):
                inspection_remaining = request.inspection_limit - inspected_session_count
                result_remaining = request.limit - len(results)
                # SessionStore cursors are opaque. Never stop partway through a
                # store page and try to synthesize a cursor from a Session:
                # custom stores may not use Cayu's built-in cursor encoding.
                # With at most one result per candidate and a page no larger
                # than the remaining result capacity, either bound can be
                # reached only on the final candidate in this store page.
                query_limit = min(1000, inspection_remaining, result_remaining)
                page = await self._session_store.list_sessions(
                    SessionQuery(
                        status=status,
                        inactive_for_seconds=request.inactive_for_seconds,
                        limit=query_limit,
                        cursor=cursor,
                        order_by=SessionOrder.UPDATED_AT_DESC,
                    )
                )
                store_page_count += 1
                if not page.sessions:
                    if page.next_cursor is not None:
                        raise RuntimeError(
                            "Session store returned an empty recovery page with a cursor."
                        )
                    if store_page_count >= _INCOMPLETE_RECOVERY_MAX_STORE_PAGES:
                        return IncompleteSessionsRecoveryPage(
                            results=tuple(results),
                            inspected_session_count=inspected_session_count,
                            next_cursor=continuation_after_status(status_index),
                        )
                    break
                if len(page.sessions) > query_limit:
                    raise RuntimeError(
                        "Session store returned more recovery candidates than requested."
                    )
                encoded_page_cursor: str | None = None
                if page.next_cursor is not None:
                    if page.next_cursor in seen_cursors:
                        raise RuntimeError(
                            "Session store returned a repeated cursor during "
                            "incomplete-session recovery."
                        )
                    encoded_page_cursor = _encode_incomplete_recovery_cursor(
                        status=status,
                        session_cursor=page.next_cursor,
                        request=request,
                    )

                for candidate_index, candidate in enumerate(page.sessions):
                    inspected_session_count += 1
                    if candidate.id in result_session_ids:
                        result = None
                    else:
                        result = await self._recover_incomplete_session_fault_isolated(
                            session=candidate,
                            request=request,
                            before_recovery=before_recovery,
                            before_mutation=before_mutation,
                            after_recovery=after_recovery,
                            reconcile_result=reconcile_result,
                        )
                    if result is not None and not (
                        terminal_status
                        and result.actions == (IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,)
                    ):
                        results.append(result)
                        result_session_ids.add(result.session_id)

                    result_limit_reached = len(results) >= request.limit
                    inspection_limit_reached = inspected_session_count >= request.inspection_limit
                    if not result_limit_reached and not inspection_limit_reached:
                        continue

                    if candidate_index + 1 < len(page.sessions):
                        raise RuntimeError(
                            "Incomplete-session recovery reached a page bound before "
                            "consuming the store page."
                        )
                    next_cursor = (
                        encoded_page_cursor
                        if encoded_page_cursor is not None
                        else continuation_after_status(status_index)
                    )
                    return IncompleteSessionsRecoveryPage(
                        results=tuple(results),
                        inspected_session_count=inspected_session_count,
                        next_cursor=next_cursor,
                    )

                if store_page_count >= _INCOMPLETE_RECOVERY_MAX_STORE_PAGES:
                    next_cursor = (
                        encoded_page_cursor
                        if encoded_page_cursor is not None
                        else continuation_after_status(status_index)
                    )
                    return IncompleteSessionsRecoveryPage(
                        results=tuple(results),
                        inspected_session_count=inspected_session_count,
                        next_cursor=next_cursor,
                    )
                if page.next_cursor is None:
                    break
                seen_cursors.add(page.next_cursor)
                cursor = page.next_cursor

            initial_session_cursor = None

        return IncompleteSessionsRecoveryPage(
            results=tuple(results),
            inspected_session_count=inspected_session_count,
            next_cursor=None,
        )

    async def terminalize_zero_work_interruption(
        self,
        *,
        session: Session,
        inactive_for_seconds: int | None,
        commit: bool = False,
        recovery_ownership: DurableOperationOwnership | None = None,
    ) -> IncompleteSessionRecoveryResult | None:
        from cayu.runtime._zero_work_interruption import ZeroWorkInterruptionRequest

        if session.status not in {SessionStatus.INTERRUPTING, SessionStatus.INTERRUPTED}:
            return None
        if self._session_control.has_active_tasks(session.id):
            return None
        if self._task_store is not None and await self._task_store.list_tasks(
            TaskQuery(session_id=session.id, limit=1)
        ):
            return None
        checkpoint = await self._session_store.load_checkpoint(session.id)
        request = ZeroWorkInterruptionRequest(
            session, checkpoint, inactive_for_seconds, commit, recovery_ownership
        )
        try:
            publication = await self._session_store._terminalize_zero_work_interruption(request)
        except Exception:
            if not commit:
                raise
            readback = await self._session_store._terminalize_zero_work_interruption(
                ZeroWorkInterruptionRequest(
                    session, checkpoint, inactive_for_seconds, False, recovery_ownership
                )
            )
            if readback is None or not readback.replayed:
                raise
            publication = readback
        if publication is None:
            return None
        return IncompleteSessionRecoveryResult(
            session_id=session.id,
            previous_status=session.status,
            status=publication.session.status,
            actions=(IncompleteSessionRecoveryAction.TERMINALIZED_ZERO_WORK,),
            events=publication.events,
            message="Terminalized proven zero work without reconstructing an executable profile.",
        )

    async def preflight_incomplete_session(
        self,
        *,
        session: Session,
        inactive_for_seconds: int | None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> IncompleteSessionRecoveryResult | None:
        """Validate one recovery path without acquiring claims or mutating state.

        ``None`` means the exact current state reached the coordinator's guarded
        mutation boundary. A returned result is a read-only disposition such as
        an active owner, a pending approval, or already-complete terminal state.
        The sentinel is raised only by the same ``before_mutation`` hook used by
        verifier-aware recovery admission, so registration and execution-profile
        incompatibilities are reported before an operator plan can authorize a
        write. Historical accounting defers this sentinel until continuation
        preflight; profile rejection diagnostics are not published by planning.
        """

        if type(session) is not Session:
            raise TypeError("session must be a Session.")

        async def prevent_mutation() -> None:
            raise _RecoveryPreflightMutationRequired

        try:
            return await self._recover_incomplete_session_scoped(
                session=session.model_copy(deep=True),
                inactive_for_seconds=inactive_for_seconds,
                reason="operator_recovery_plan_preflight",
                metadata={"source": "registered_application_recovery_plan"},
                before_mutation=prevent_mutation,
                record_profile_rejection=False,
                participant_context=participant_context,
            )
        except _RecoveryPreflightMutationRequired:
            return None

    async def _recover_incomplete_session_fault_isolated(
        self,
        *,
        session: Session,
        request: IncompleteSessionsRecoveryRequest,
        before_recovery: IncompleteRecoveryScopeHook | None,
        before_mutation: IncompleteRecoveryScopeHook | None,
        after_recovery: IncompleteRecoveryScopeHook | None,
        reconcile_result: IncompleteRecoveryResultHook | None,
    ) -> IncompleteSessionRecoveryResult:
        retained_invocation_context: InvocationContext | None = None

        def retain_invocation_context(context: InvocationContext) -> None:
            nonlocal retained_invocation_context
            if (
                retained_invocation_context is not None
                and retained_invocation_context is not context
            ):
                raise RuntimeError("Batch recovery produced conflicting live invocation authority.")
            retained_invocation_context = context

        try:
            if before_recovery is not None:
                await before_recovery(session.id)
            try:
                result = await self._recover_incomplete_session_scoped(
                    session=session,
                    inactive_for_seconds=request.inactive_for_seconds,
                    reason=request.reason,
                    metadata=request.metadata,
                    before_mutation=(
                        None if before_mutation is None else lambda: before_mutation(session.id)
                    ),
                    retain_open_interaction_invocation=(reconcile_result is not None),
                    retain_invocation_context=(
                        retain_invocation_context if reconcile_result is not None else None
                    ),
                )
                result = await self._finish_provider_operation_disposition_after_recovery(
                    result,
                    before_mutation=(
                        None if before_mutation is None else lambda: before_mutation(session.id)
                    ),
                    invocation_context=retained_invocation_context,
                )
                return (
                    result
                    if reconcile_result is None
                    else await reconcile_result(result, retained_invocation_context)
                )
            finally:
                if after_recovery is not None:
                    await after_recovery(session.id)
        except Exception as exc:
            diagnostic = exception_diagnostic(
                exc,
                empty_message="recovery failed",
                nonportable_message="Recovery failed with a non-portable diagnostic.",
                redactor=self._secret_redactor,
            )
            logger.warning(
                "Recovery failed for session %s (agent %s): error_type=%s error=%s",
                session.id,
                session.agent_name,
                diagnostic.error_type,
                diagnostic.message,
            )
            try:
                reloaded = await self._session_store.load(session.id)
            except Exception:
                reloaded = None
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=session.status,
                status=session.status if reloaded is None else reloaded.status,
                actions=(IncompleteSessionRecoveryAction.FAILED,),
                message=bound_diagnostic_text(
                    f"Recovery failed: {diagnostic.error_type}: {diagnostic.message}"
                ),
            )

    @resource_recovery
    async def _recover_incomplete_session_scoped(
        self,
        *,
        session: Session,
        inactive_for_seconds: int | None,
        reason: str,
        metadata: dict[str, Any],
        before_mutation: RecoveryMutationHook | None = None,
        record_profile_rejection: bool = True,
        retain_open_interaction_invocation: bool = False,
        retain_invocation_context: Callable[[InvocationContext], None] | None = None,
        provider_disposition_task_id: str | None = None,
        provider_disposition_task_worker_id: str | None = None,
        provider_disposition_task_handoff_id: str | None = None,
        provider_disposition_after_admission: RecoveryMutationHook | None = None,
        participant_context: CollaborationAccessContext | None = None,
        interrupt_for_manual_tool_recovery: bool = False,
        preserve_interaction_id: str | None = None,
        _work_attempt: WorkAttemptInvocationAuthority | None = None,
        execution_to_wait: _ExternalExecutionToWait | None = None,
    ) -> IncompleteSessionRecoveryResult:
        from cayu.runtime._abandoned_session_recovery import require_abandoned_execution_matches

        require_abandoned_execution_matches(session)
        reason = require_clean_nonblank(reason, "reason")
        metadata = copy_durable_metadata(metadata)
        previous_status = session.status

        if self._session_control.has_active_tasks(session.id):
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                events=(),
                message="Session has active work in this CayuApp process; recovery skipped.",
            )

        if inactive_for_seconds is not None and self._session_store.supports_session_execution:
            execution = await self._session_store.inspect_session_execution(session.id)
            if execution.state == "executing":
                if execution.lease_expires_at is None:
                    raise RuntimeError("Executing session presence lacks its lease expiry.")
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=session.status,
                    actions=(IncompleteSessionRecoveryAction.SKIPPED_EXECUTION_OWNER,),
                    execution_lease_expires_at=execution.lease_expires_at,
                    message=(
                        "Session execution owner lease is active until "
                        f"{execution.lease_expires_at.isoformat()}; retry recovery after expiry."
                    ),
                )

        return await self._recover_incomplete_session_owned(
            execution_to_wait=execution_to_wait,
            _work_attempt=_work_attempt,
            preserve_interaction_id=preserve_interaction_id,
            session=session,
            inactive_for_seconds=inactive_for_seconds,
            reason=reason,
            metadata=metadata,
            previous_status=previous_status,
            before_mutation=before_mutation,
            record_profile_rejection=record_profile_rejection,
            retain_open_interaction_invocation=retain_open_interaction_invocation,
            retain_invocation_context=retain_invocation_context,
            provider_disposition_task_id=provider_disposition_task_id,
            provider_disposition_task_worker_id=provider_disposition_task_worker_id,
            provider_disposition_task_handoff_id=provider_disposition_task_handoff_id,
            provider_disposition_after_admission=provider_disposition_after_admission,
            participant_context=participant_context,
            interrupt_for_manual_tool_recovery=interrupt_for_manual_tool_recovery,
        )

    async def _recover_incomplete_session_owned(
        self,
        *,
        session: Session,
        inactive_for_seconds: int | None,
        reason: str,
        metadata: dict[str, Any],
        previous_status: SessionStatus,
        before_mutation: RecoveryMutationHook | None,
        retain_open_interaction_invocation: bool,
        retain_invocation_context: Callable[[InvocationContext], None] | None,
        record_profile_rejection: bool = True,
        provider_disposition_task_id: str | None = None,
        provider_disposition_task_worker_id: str | None = None,
        provider_disposition_task_handoff_id: str | None = None,
        provider_disposition_after_admission: RecoveryMutationHook | None = None,
        participant_context: CollaborationAccessContext | None = None,
        interrupt_for_manual_tool_recovery: bool = False,
        preserve_interaction_id: str | None = None,
        _work_attempt: WorkAttemptInvocationAuthority | None = None,
        execution_to_wait: _ExternalExecutionToWait | None = None,
    ) -> IncompleteSessionRecoveryResult:

        external_recovery_record = None
        external_recovery_cleanup = False
        if execution_to_wait is not None:
            from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait

            if type(execution_to_wait) is not _ExternalExecutionToWait:
                raise PermissionError("External recovery requires its registered runtime owner.")
            (
                external_recovery_record,
                external_recovery_cleanup,
            ) = await execution_to_wait.inspect_recovery(session)

        if (provider_disposition_task_id is None) != (
            provider_disposition_task_worker_id is None
        ) or (
            provider_disposition_task_handoff_id is not None
            and provider_disposition_task_worker_id is None
        ):
            raise ValueError(
                "Typed provider recovery requires task and worker identities; handoff "
                "authority additionally requires both."
            )

        mutation_admitted = False

        async def admit_before_mutation() -> None:
            nonlocal mutation_admitted
            if mutation_admitted or before_mutation is None:
                return
            await before_mutation()
            mutation_admitted = True

        zero_work = await self.terminalize_zero_work_interruption(
            session=session,
            inactive_for_seconds=inactive_for_seconds,
        )
        if zero_work is not None:
            await admit_before_mutation()
            committed = await self.terminalize_zero_work_interruption(
                session=session,
                inactive_for_seconds=inactive_for_seconds,
                commit=True,
            )
            if committed is not None:
                return committed
            raise RuntimeError("Zero-work interruption authority changed before publication.")

        checkpoint = await self._session_store.load_checkpoint(session.id)
        if self._committed_runtime_task_failure_recovery is not None:
            recovered_runtime_failure = await self._committed_runtime_task_failure_recovery(
                session,
                checkpoint,
                previous_status,
                admit_before_mutation,
            )
            if recovered_runtime_failure is not None:
                return recovered_runtime_failure
        from cayu.runtime._producer_completion_replay import producer_completion_requires_execution

        if session.status is SessionStatus.RUNNING and producer_completion_requires_execution(
            checkpoint
        ):
            # Historical model publication permits neither policy execution nor
            # destructive fallback cleanup under an unauthenticated caller.
            # Refuse before claiming a replacement epoch; explicit stop/cleanup
            # retains its separate owner and does not require answer replay.
            await self._require_participant_execution(session, participant_context)
        pending_completion_finalization = (
            completion_finalization.pending_completion_finalization_from_checkpoint(checkpoint)
        )
        ambiguous_user_input = ambiguous_pending_user_input_from_checkpoint(checkpoint)
        if ambiguous_user_input is not None:
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.AMBIGUOUS_PENDING_USER_INPUT,),
                events=(),
                message=(
                    "Session has a historical user-input pause without exact authority; "
                    "explicitly interrupt the session before starting new work."
                ),
            )
        pending_provider_interrupt = _provider_cancellation_interrupt_payload(checkpoint)
        active_invocation_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if active_invocation_profile is None and invocation_lifecycle_receipt_history_present(
            checkpoint
        ):
            raise RuntimeError(
                "Incomplete-session recovery lost durable invocation profile authority."
            )
        pending_provider_disposition = await load_pending_provider_operation_disposition(
            self._session_store,
            session.id,
            checkpoint=checkpoint,
        )
        pending_provider_disposition_effect_is_durable = False
        if pending_provider_disposition is not None:
            # A durable disposition records intent, not current permission to
            # execute its fallback or terminal hooks after reconstruction.
            await self._require_participant_execution(session, participant_context)
            (
                pending_disposition_record,
                pending_disposition_result,
            ) = pending_provider_disposition
            pending_provider_disposition_effect_is_durable = (
                await self._provider_operation_disposition_effect_is_durable(
                    pending=pending_disposition_record,
                    result=pending_disposition_result,
                )
            )
        if active_invocation_profile is not None and not (
            active_invocation_execution_profile_matches_session_epoch(
                active_invocation_profile,
                session_id=session.id,
                run_epoch=session.run_epoch,
            )
        ):
            if external_recovery_record is not None:
                current = await self._recovery_ownership.require_session(session.id)
                if (
                    current.instance_id == session.instance_id
                    and current.run_epoch != session.run_epoch
                ):
                    # A competing owner advanced between the session and
                    # checkpoint reads. Do not reinterpret that newer profile
                    # or acquire another epoch from the stale wait snapshot.
                    # The adapter separately reconciles the exact native ticket.
                    return IncompleteSessionRecoveryResult(
                        session_id=session.id,
                        previous_status=previous_status,
                        status=current.status,
                        actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                        events=(),
                        message="Another owner advanced external wait recovery.",
                    )
            raise RuntimeError(
                "Active invocation execution profile does not match the recovery epoch."
            )
        pending_allocations = await self._environment_lifecycle.pending_allocation_names(session)
        unsettled_environment_lifecycle = await self._environment_lifecycle.has_unsettled_progress(
            session_id=session.id,
            interaction_id=(
                None
                if active_invocation_profile is None
                else active_invocation_profile.interaction_id
            ),
        )
        durable_child_guard = await self._pending_durable_subagent_recovery_guard(
            session=session,
            checkpoint=checkpoint,
            previous_status=previous_status,
        )
        if durable_child_guard is not None:
            return durable_child_guard
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
        from cayu.sessions._foreground_child_checkpoint import (
            FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY,
            post_action_continuation_from_checkpoint,
        )

        post_action_transfer: CheckpointTransform | None = None
        post_action_round: pending_rounds.PendingToolRound | None = None
        post_action = post_action_continuation_from_checkpoint(checkpoint)
        if post_action is not None:
            latest_model = await self._session_store.query_events(
                EventQuery(
                    session_id=session.id,
                    event_type=EventType.MODEL_STARTED,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                )
            )
            # Once a later model attempt is durable, its normal recovery owner
            # takes over. A retained close marker must never rewind that work.
            if (
                session.status is not SessionStatus.RUNNING
                or await self._session_store.load_active_model_completion_stage(session.id)
                is not None
                or (
                    latest_model
                    and latest_model[0].event.payload.get("model_step_id")
                    != post_action.pending_tool_round.get("model_step_id")
                )
            ):
                post_action = None
        if post_action is not None:
            if (
                post_action.wait.child_session_id != session.id
                or post_action.wait.child_session_instance_id != session.instance_id
            ):
                raise RuntimeError("Post-action continuation targets another child session.")
            parent = await self._session_store.load(post_action.wait.parent_effect.session_id)
            if (
                parent is None
                or parent.id != session.parent_session_id
                or parent.instance_id != post_action.wait.parent_effect.session_instance_id
            ):
                raise RuntimeError("Post-action continuation lost its parent incarnation.")
            parent_effect = await ToolEffectStateOwner(self._session_store).resolve_call(
                parent,
                tool_round_id=post_action.wait.parent_effect.tool_round_id,
                tool_call_id=post_action.wait.parent_effect.tool_call_id,
            )
            if parent_effect is None or parent_effect.intent != post_action.wait.parent_effect:
                raise RuntimeError("Post-action continuation lost its exact parent effect.")
            close_receipt = await self._session_store.load_runtime_publication_receipt(
                session.id, post_action.close_publication_id
            )
            expected_kind = (
                "approval-close"
                if post_action.action_kind == "tool_approval"
                else "user-input-close"
            )
            if (
                close_receipt is None
                or close_receipt.kind != expected_kind
                or close_receipt.session_id != session.id
                or close_receipt.publication_id != post_action.close_publication_id
                or close_receipt.intent.get("post_action_continuation_digest")
                != runtime_publication_checkpoint_value_digest(post_action.model_dump(mode="json"))
                or close_receipt.interaction_id != post_action.wait.child_interaction_id
                or any(
                    close_receipt.intent.get(field) != post_action.pending_tool_round.get(field)
                    for field in ("tool_round_id", "model_step_id", "model_attempt_id")
                )
                or post_action.continuation_revision != post_action.wait.revision
                or close_receipt.intent.get(
                    "approval_id" if expected_kind == "approval-close" else "input_id"
                )
                != post_action.action_id
            ):
                raise RuntimeError("Post-action continuation lacks exact close authority.")
            if pending_tool_round is not None:
                raise RuntimeError("Post-action continuation conflicts with a live tool round.")
            marker_value = post_action.model_dump(mode="json")
            restored = copy_json_value(post_action.pending_tool_round, "post_action.pending_round")

            def claim_post_action(
                current_session: Session, current: dict[str, Any] | None
            ) -> dict[str, Any] | None:
                if (
                    current_session.instance_id != session.instance_id
                    or current_session.run_epoch != session.run_epoch
                    or current is None
                    or current.get(FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY) != marker_value
                    or pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY in current
                ):
                    raise _IncompleteRecoveryClaimLost(
                        "Post-action continuation was already claimed or changed."
                    )
                updated = copy_json_value(current, "post_action.claimed_checkpoint")
                # Keep the marker across claim acknowledgement loss or process
                # death before the next model dispatch. The closed round is not
                # pending work and must never be republished or replayed.
                return updated

            # Inspect the completed round without publishing pending work.
            # Its exact marker is checked inside the admitted epoch claim below;
            # recovery planning must remain entirely read-only.
            post_action_transfer = claim_post_action
            post_action_round = pending_round_reader.pending_tool_round_from_checkpoint(
                {pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY: restored},
                redactor=self._secret_redactor,
                consume_on_rejection=True,
                runtime_session=session,
            )
            if (
                post_action_round is None
                or active_invocation_profile is None
                or post_action_round.execution_profile_fingerprint
                != active_invocation_profile.profile.fingerprint
                or post_action.wait.child_interaction_id != active_invocation_profile.interaction_id
            ):
                raise RuntimeError("Post-action continuation conflicts with its invocation.")
        workspace_observations = workspace_observations_from_checkpoint(checkpoint)
        self._workspace_observation_recovery.validate_workspace_observation_recovery_authority(
            session=session,
            observations=workspace_observations,
            pending_round=pending_tool_round,
            execution_profile_snapshot=active_invocation_profile,
        )
        deferred_input = await self._session_store.load_deferred_interaction_input(session.id)
        active_model_completion = await self._session_store.load_active_model_completion_stage(
            session.id
        )
        terminal_repair_required = False
        terminal_binding_finalization = None
        if session.status in _RECOVERY_RESUMABLE_SESSION_STATUSES:
            terminal_repair = await self.terminal_finalization.repair_required(
                session=session,
                checkpoint=checkpoint,
            )
            terminal_repair_required = terminal_repair
            has_pending_work = (
                execution_to_wait is not None
                or pending_approval is not None
                or pending_user_input is not None
                or pending_tool_round is not None
                or deferred_input is not None
                or active_model_completion is not None
                or bool(workspace_observations)
                or pending_provider_disposition is not None
                or pending_completion_finalization is not None
                or unsettled_environment_lifecycle
                or bool(pending_allocations)
            )
            if (
                not has_pending_work
                and active_invocation_profile is not None
                and not active_invocation_execution_profile_is_released(
                    active_invocation_profile,
                    session_id=session.id,
                    run_epoch=session.run_epoch,
                )
            ):
                terminal_binding_finalization = (
                    await self._environment_lifecycle.prepare_terminal_binding_finalization(
                        session=session,
                        execution_profile=active_invocation_profile.profile,
                        registered_environment=self._resolve_registered_environment(
                            session.environment_name
                        ),
                    )
                )
                if terminal_binding_finalization is not None:
                    pending_completion_finalization = terminal_binding_finalization.marker()
                    has_pending_work = True
            if not terminal_repair and not has_pending_work:
                if active_invocation_profile is not None and not (
                    active_invocation_execution_profile_is_released(
                        active_invocation_profile,
                        session_id=session.id,
                        run_epoch=session.run_epoch,
                    )
                ):
                    await admit_before_mutation()
                    return await self._settle_terminal_invocation_closure_owned(
                        session=session,
                        inactive_for_seconds=inactive_for_seconds,
                        previous_status=previous_status,
                        execution_profile_snapshot=active_invocation_profile,
                    )
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=session.status,
                    actions=(IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,),
                    events=(),
                    message="Session is terminal and has durable terminal evidence.",
                )
            if terminal_repair and not has_pending_work and active_invocation_profile is None:
                # A session that never acquired invocation authority can repair
                # its durable terminal evidence directly.  Once an invocation
                # profile exists, continue through registration/profile
                # validation and context reconstruction below so the matching
                # interaction settlement cannot be published from durable
                # session fields alone.
                await admit_before_mutation()
                return await self.terminal_finalization.repair_owned(
                    session=session,
                    inactive_for_seconds=inactive_for_seconds,
                    previous_status=previous_status,
                )

        try:
            registered_agent = self._resolve_registered_agent(session.agent_name)
        except KeyError:
            if terminal_repair_required:
                await admit_before_mutation()
                return await self.terminal_finalization.repair_owned(
                    session=session,
                    inactive_for_seconds=inactive_for_seconds,
                    previous_status=previous_status,
                )
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.SKIPPED_UNREGISTERED_AGENT,),
                events=(),
                message=(f"Agent not registered: {session.agent_name!r}; session left untouched."),
            )
        try:
            registered_environment = self._resolve_registered_environment(session.environment_name)
        except KeyError:
            if terminal_repair_required:
                await admit_before_mutation()
                return await self.terminal_finalization.repair_owned(
                    session=session,
                    inactive_for_seconds=inactive_for_seconds,
                    previous_status=previous_status,
                )
            raise
        registered_provider = self._resolve_registered_provider(session.provider_name)
        if (
            session.status is SessionStatus.RUNNING
            and _work_attempt is None
            and await tool_completion_requires_execution(
                self._session_store, session, checkpoint, registered_agent
            )
        ):
            await self._require_participant_execution(session, participant_context)
        if workspace_observations:
            # A static registration already provides the complete stable
            # environment authority. Reject a foreign lifecycle before profile
            # continuation or the recovery claim can mutate durable ownership.
            # Factory templates remain deliberately unmaterialized here.
            self._workspace_observation_recovery.validate_workspace_observation_recovery_authority(
                session=session,
                observations=workspace_observations,
                pending_round=pending_tool_round,
                execution_profile_snapshot=active_invocation_profile,
                registered_environment=registered_environment,
            )
        pre_admission_profiled_session = False
        if (
            session.status is SessionStatus.PENDING
            and session.run_epoch == 0
            and active_invocation_profile is None
            and pending_approval is None
            and pending_user_input is None
            and pending_tool_round is None
            and deferred_input is None
            and active_model_completion is None
            and not workspace_observations
            and pending_provider_disposition is None
            and not unsettled_environment_lifecycle
            and EXECUTION_PROFILE_METADATA_KEY in session.metadata
        ):
            interaction_records = await self._session_store.query_events(
                EventQuery(
                    session_id=session.id,
                    event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                )
            )
            if not interaction_records:
                # Presence of the reserved metadata key is not authority. A
                # pristine session is exempt only when its complete baseline
                # survives defensive reconstruction.
                execution_profile_from_session_metadata(session.metadata)
                pre_admission_profiled_session = True
        requires_execution_profile = (
            bool(pending_allocations)
            or pending_approval is not None
            or pending_user_input is not None
            or pending_tool_round is not None
            or deferred_input is not None
            or active_model_completion is not None
            or bool(workspace_observations)
            or pending_provider_disposition is not None
            or (
                pending_provider_interrupt is not None
                and session.status not in _RECOVERY_RESUMABLE_SESSION_STATUSES
            )
            or active_invocation_profile is not None
            or pending_completion_finalization is not None
            or (
                not pre_admission_profiled_session
                and session.status not in _RECOVERY_RESUMABLE_SESSION_STATUSES
                and EXECUTION_PROFILE_METADATA_KEY in session.metadata
            )
        )
        execution_profile_snapshot = None
        budget_policy_snapshot: BudgetPolicy | None = None
        if requires_execution_profile:
            published_model = model_completion_publication.model_step_publication_from_checkpoint(
                checkpoint
            )
            if published_model is not None and active_model_completion is None:
                # This accounts for already-published work under its original
                # reservation identity. It neither promotes a model stage nor
                # permits continuation under the current configuration.
                # Planning skips this accounting write but must still validate
                # continuation compatibility below. Its mutation gate remains
                # unadmitted, so subsequent writes still stop preflight.
                # Real recovery settles first.
                with contextlib.suppress(_RecoveryPreflightMutationRequired):
                    await self._model_completion_recovery.reconcile_published_model_budget(
                        session=session,
                        pointer=published_model,
                        before_mutation=admit_before_mutation,
                    )
            budget_policy_snapshot = copy_budget_policy(self._resolve_budget_policy())
            pending_disposition = (
                None if pending_provider_disposition is None else pending_provider_disposition[0]
            )
            if pending_completion_finalization is not None:
                # This recovery dispatches neither provider nor tools. Reuse
                # the frozen invocation identity across process-local object
                # replacement, then validate the current binding and target
                # against their dedicated durable recovery record.
                if active_invocation_profile is None:
                    raise RuntimeError(
                        "Pending completion finalization lost its frozen invocation profile."
                    )
                execution_profile_snapshot = active_invocation_profile
            else:
                execution_profile_snapshot = await self._execution_profile_continuation.validate(
                    session=session,
                    checkpoint=checkpoint,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    budget_policy=budget_policy_snapshot,
                    request_loop_policies=(
                        None
                        if execution_to_wait is None
                        else execution_to_wait.request_loop_policies
                    ),
                    require_open_interaction=not (
                        execution_to_wait is not None
                        or (
                            active_model_completion is not None
                            and active_model_completion.stage.purpose == "auxiliary-inference"
                            and session.status in _RECOVERY_RESUMABLE_SESSION_STATUSES
                        )
                        or (
                            (terminal_repair_required or bool(pending_allocations))
                            and session.status in _RECOVERY_RESUMABLE_SESSION_STATUSES
                        )
                        or pending_provider_disposition_effect_is_durable
                        or (
                            pending_disposition is not None
                            and pending_disposition.action is ProviderOperationResolutionAction.FAIL
                            and session.status is SessionStatus.FAILED
                        )
                    ),
                    additional_profile_fingerprints=(
                        ()
                        if pending_disposition is None
                        else (pending_disposition.execution_profile_fingerprint,)
                    ),
                    record_rejection=record_profile_rejection,
                )
            if pending_completion_finalization is not None:
                expected_terminal_status = (
                    SessionStatus.INTERRUPTED
                    if pending_completion_finalization["outcome"] == "interrupted"
                    else SessionStatus.FAILED
                )
                if session.status not in {SessionStatus.RUNNING, expected_terminal_status}:
                    raise RuntimeError(
                        "Pending finalization conflicts with the original terminal outcome."
                    )
                if registered_environment is None or (
                    registered_environment.spec.name
                    != pending_completion_finalization["environment_name"]
                ):
                    raise RuntimeError(
                        "Pending completion finalization resolved a different environment."
                    )
                if (
                    execution_profile_snapshot.profile.fingerprint
                    != pending_completion_finalization["execution_profile_fingerprint"]
                ):
                    raise RuntimeError("Pending completion finalization execution profile changed.")
        claim: _IncompleteRecoveryClaim | None = None
        invocation_context: InvocationContext | None = None
        authoritative_failure: BaseException | None = None
        provider_execution_transfer: CheckpointTransform | None = None
        command_recovery_context: InvocationContext | None = None
        if provider_disposition_task_id is not None:
            if pending_provider_disposition is None:
                raise ProviderOperationEvidenceError(
                    "Typed provider recovery lost its pending disposition."
                )
            expected_pending = pending_provider_disposition[0]
            if not expected_pending.execution_claimed or (
                expected_pending.execution_task_worker_id,
                expected_pending.execution_task_handoff_id,
            ) == (
                provider_disposition_task_worker_id,
                provider_disposition_task_handoff_id,
            ):
                raise ProviderOperationEvidenceError(
                    "Typed provider recovery has no predecessor execution owner to fence."
                )

            def transfer_provider_execution(
                _session: Session,
                current_checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any]:
                return checkpoint_with_provider_operation_disposition_execution_owner(
                    current_checkpoint,
                    expected=expected_pending,
                    task_worker_id=provider_disposition_task_worker_id,
                    task_handoff_id=provider_disposition_task_handoff_id,
                )

            provider_execution_transfer = transfer_provider_execution

        def transfer_recovery_checkpoint(
            current_session: Session, current: dict[str, Any] | None
        ) -> dict[str, Any] | None:
            if external_recovery_record is not None:
                from cayu.runtime._continuation_recovery import (
                    require_recovery_frontier,
                    require_resolved_native_pause,
                )

                require_recovery_frontier(external_recovery_record, current_session, current)
                # Recheck under the native claim transaction: the original turn
                # may have paused after the read-only recovery inspection.
                # A retained terminal control can still request ordinary cleanup.
                if not external_recovery_cleanup:
                    require_resolved_native_pause(current)
            if post_action_transfer is not None:
                current = post_action_transfer(current_session, current)
            if provider_execution_transfer is not None:
                current = provider_execution_transfer(current_session, current)
            return current

        try:
            await admit_before_mutation()
            claim = await self._recovery_ownership.claim(
                session=session,
                inactive_for_seconds=inactive_for_seconds,
                execution_profile_snapshot=execution_profile_snapshot,
                target_status=SessionStatus.RUNNING if execution_to_wait is not None else None,
                checkpoint_transform=(
                    transfer_recovery_checkpoint
                    if post_action_transfer is not None
                    or provider_execution_transfer is not None
                    or external_recovery_record is not None
                    else None
                ),
            )
            if claim is None:
                current = await self._recovery_ownership.require_session(session.id)
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=current.status,
                    actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                    events=(),
                    message="Session activity or recovery ownership changed; recovery skipped.",
                )
            if execution_profile_snapshot is not None:
                invocation_context = reconstruct_invocation_context(
                    runtime_hooks=self._runtime_hooks,
                    loop_policies=self._loop_policies,
                    session=claim.session,
                    execution_profile_snapshot=execution_profile_snapshot,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    budget_policy=budget_policy_snapshot,
                    recovery_claim_id=claim.claim_id,
                    work_attempt=_work_attempt,
                    request_loop_policies=(
                        () if execution_to_wait is None else execution_to_wait.request_loop_policies
                    ),
                )
            if provider_disposition_after_admission is not None:
                await provider_disposition_after_admission()

            def retain_cleanup_context(context: InvocationContext) -> None:
                nonlocal invocation_context
                invocation_context = context

            async def recover_claimed_session() -> IncompleteSessionRecoveryResult:
                nonlocal registered_environment, invocation_context, command_recovery_context
                assert claim is not None
                recovery_session = claim.session
                if terminal_binding_finalization is not None:
                    if invocation_context is None:
                        raise RuntimeError("Terminal binding cleanup lost invocation authority.")
                    await self._environment_lifecycle.checkpoint_terminal_binding_finalization(
                        session=claim.session,
                        invocation_context=invocation_context,
                        expected=terminal_binding_finalization,
                    )
                external_cleanup = False
                if execution_to_wait is not None:
                    if invocation_context is None:
                        raise RuntimeError("External wait recovery lost invocation authority.")
                    external_cleanup = await execution_to_wait.recovery_cleanup_requested(
                        invocation_context
                    )
                    if external_cleanup:
                        await execution_to_wait.attach_cleanup_writer(invocation_context)
                # Terminal external controls use ordinary interruption recovery,
                # including its effect/environment cleanup guards, not another
                # execution of the unfinished turn or its stop policies.
                if execution_to_wait is not None and not external_cleanup:
                    assert invocation_context is not None
                    boundary = (
                        await self._model_completion_recovery.reconcile_model_completion_boundary(
                            claim.session,
                            invocation_context=invocation_context,
                            registered_agent=registered_agent,
                            registered_provider=registered_provider,
                            registered_environment=registered_environment,
                        )
                    )
                    semantics = await execution_to_wait.require_recovered_result(
                        invocation_context, boundary
                    )
                    resumed = await self._event_writer.emit(
                        Event(
                            type=EventType.SESSION_RESUMED,
                            session_id=session.id,
                            agent_name=session.agent_name,
                            environment_name=_environment_name(registered_environment),
                            payload={
                                "recovered_external_wait": True,
                                "run_epoch": claim.session.run_epoch,
                            },
                        )
                    )
                    stream = self._run_session(
                        RecoverySessionRunRequest(
                            session=claim.session,
                            invocation_context=invocation_context,
                            messages=await self._session_store.load_transcript(session.id),
                            messages_to_append=[],
                            max_steps=semantics.max_steps,
                            limits=semantics.limits,
                            budget_limits=semantics.budget_limits,
                            retry_policy=semantics.retry_policy,
                            structured_output=semantics.structured_output,
                            tool_completion=semantics.tool_completion,
                            thinking=semantics.thinking,
                            request_metadata=semantics.request_metadata,
                            task_id=None,
                            task_worker_id=None,
                            task_handoff_id=None,
                            start_event_type=None,
                            start_event_payload={},
                            start_task_on_enter=False,
                            release_run_fence_on_exit=False,
                            run_limit_accounting=semantics.run_limit_accounting,
                            execution_to_wait=execution_to_wait,
                        )
                    )
                    events = [*boundary.recovery_events, resumed]
                    async with _close_delegated_event_stream(stream) as owned:
                        async for event in owned:
                            events.append(event)
                    current = await self._recovery_ownership.require_session(session.id)
                    return IncompleteSessionRecoveryResult(
                        session_id=session.id,
                        previous_status=previous_status,
                        status=current.status,
                        actions=(IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,),
                        events=tuple(events),
                        message="Recovered the external wait's committed whole-turn result.",
                    )
                if (
                    not external_cleanup
                    and post_action_round is not None
                    and post_action is not None
                ):
                    if invocation_context is None:
                        raise RuntimeError("Post-action continuation lost invocation authority.")
                    if post_action_round.limits is None or post_action_round.budget_limits is None:
                        raise RuntimeError("Post-action continuation lost its original limits.")
                    continuation = await self._prepare_recovered_tool_round_continuation(
                        session=claim.session,
                        pending_round=post_action_round,
                        invocation_semantics=_RecoveryInvocationSemantics(
                            max_steps=_require_recovery_max_steps(post_action_round.max_steps),
                            limits=post_action_round.limits,
                            budget_limits=post_action_round.budget_limits,
                            retry_policy=self._effective_retry_policy(
                                post_action_round.retry_policy
                            ),
                            structured_output=post_action_round.structured_output,
                            thinking=post_action_round.thinking,
                        ),
                        invocation_context=invocation_context,
                        request_metadata=post_action.request_metadata,
                        task_worker_id=None,
                        task_handoff_id=None,
                        completed_model_step=post_action.completed_model_step,
                    )
                    stream = self._run_session(continuation)
                    async with _close_delegated_event_stream(stream) as owned_stream:
                        events = [event async for event in owned_stream]

                    def retire_close_marker(
                        _session: Session, current: dict[str, Any] | None
                    ) -> dict[str, Any] | None:
                        if (
                            current is None
                            or current.get(FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY)
                            != marker_value
                        ):
                            return current
                        updated = copy_json_value(current, "post_action.completed_checkpoint")
                        updated.pop(FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY)
                        return updated

                    await self._session_store.transform_checkpoint(session.id, retire_close_marker)
                    current_session = await self._recovery_ownership.require_session(session.id)
                    return IncompleteSessionRecoveryResult(
                        session_id=session.id,
                        previous_status=previous_status,
                        status=current_session.status,
                        actions=(IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,),
                        events=tuple(events),
                        message="Continued the foreground child after its committed action close.",
                    )
                if pending_allocations:
                    if (
                        registered_environment is None
                        or pending_allocations != (registered_environment.spec.name,)
                        or execution_profile_snapshot is None
                        or invocation_context is None
                    ):
                        raise RuntimeError("Pending allocation lost exact recovery authority.")
                    try:
                        await self._environment_lifecycle.reap_pending_allocation(
                            session=claim.session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            execution_profile=execution_profile_snapshot.profile,
                            invocation_context=invocation_context,
                        )
                    except Exception as exc:
                        diagnostic = exception_diagnostic(exc, redactor=self._secret_redactor)
                        return IncompleteSessionRecoveryResult(
                            session_id=session.id,
                            previous_status=previous_status,
                            status=claim.session.status,
                            actions=(IncompleteSessionRecoveryAction.PENDING_ALLOCATION_CLEANUP,),
                            message=bound_diagnostic_text(
                                "Allocation cleanup remains durably owned; retry "
                                "recover_incomplete_session. " + diagnostic.message
                            ),
                        )
                lifecycle_events = list(
                    await self._environment_lifecycle.reconcile_orphaned_progress(
                        session_id=claim.session.id,
                        interaction_id=(
                            None
                            if active_invocation_profile is None
                            else active_invocation_profile.interaction_id
                        ),
                        observed_at=self._clock(),
                    )
                )
                # Factory registrations have no live runner/workspace. Reuse the
                # round owner's exact recorded/staged evidence before inspecting
                # missing command results. Completed sibling tools need not
                # implement command recovery; unknown effects remain fenced.
                pending = pending_round_reader.pending_tool_round_from_checkpoint(
                    await self._session_store.load_checkpoint(claim.session.id)
                )
                settled_call_ids: set[str] = set()
                if (
                    not interrupt_for_manual_tool_recovery
                    and registered_environment is not None
                    and registered_environment.factory_backed
                    and invocation_context is not None
                    and pending is not None
                    and pending.source_run_epoch is not None
                    and pending.tool_calls
                ):
                    lifecycle_events_for_round = (
                        await self._pending_tool_round_recovery.load_tool_round_lifecycle_events(
                            session_id=claim.session.id, pending_round=pending
                        )
                    )
                    recorded_outcomes, _ = tool_round_recovery.recorded_tool_outcomes(
                        events=lifecycle_events_for_round, pending_round=pending
                    )
                    terminal_events = [
                        *lifecycle_events_for_round,
                        *(
                            item.event
                            for item in staged_terminal_reader.staged_terminal_records(pending)
                        ),
                    ]
                    settled_call_ids = {
                        call_id
                        for call_id in recorded_outcomes
                        if self._pending_tool_round_recovery.tool_terminals_are_settled(
                            [
                                event
                                for event in terminal_events
                                if event.payload.get("tool_call_id") == call_id
                            ],
                            [call_id],
                        )
                    }
                if (
                    not interrupt_for_manual_tool_recovery
                    and registered_environment is not None
                    and registered_environment.factory_backed
                    and invocation_context is not None
                    and pending is not None
                    and pending.source_run_epoch is not None
                    and pending.tool_calls
                    # Fully settled rounds need only native evidence repair,
                    # not a runner reconnect or Docker binding authority.
                    and any(
                        call.tool_call_id not in settled_call_ids for call in pending.tool_calls
                    )
                    and all(
                        [
                            call.tool_call_id in settled_call_ids
                            or await self._pending_tool_round_recovery.has_recoverable_durable_tool_result(
                                session=recovery_session,
                                tool_round_id=pending.tool_round_id,
                                tool_call_id=call.tool_call_id,
                            )
                            for call in pending.tool_calls
                        ]
                    )
                ):
                    try:
                        command_binding_authority = (
                            await self._environment_lifecycle.authorize_command_binding_recovery(
                                session=claim.session,
                                invocation_context=invocation_context,
                                source_run_epoch=pending.source_run_epoch,
                            )
                        )
                        factory_started = await self._environment_lifecycle.emit_factory_started(
                            session=claim.session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            execution_profile=invocation_context.profile,
                            invocation_context=invocation_context,
                        )
                        if factory_started is not None:
                            lifecycle_events.append(factory_started)
                        resolution = await self._environment_lifecycle.resolve_factory(
                            session=claim.session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            started_event=factory_started,
                            operation=EnvironmentFactoryOperation.RECONNECT,
                            execution_profile=invocation_context.profile,
                            invocation_context=invocation_context,
                            command_recovery=command_binding_authority,
                        )
                        lifecycle_events.extend(resolution.events)
                        registered_environment = resolution.registered_environment
                        if registered_environment is not None:
                            invocation_context = invocation_context.with_registered_environment(
                                registered_environment, validated_profile=invocation_context.profile
                            )
                        if resolution.error is not None:
                            raise resolution.error
                        binding_started = await self._environment_lifecycle.emit_binding_started(
                            session=claim.session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            execution_profile=invocation_context.profile,
                            invocation_context=invocation_context,
                        )
                        if binding_started is not None:
                            lifecycle_events.append(binding_started)
                        bound = await self._environment_lifecycle.bind(
                            session=claim.session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            started_event=binding_started,
                            execution_profile=invocation_context.profile,
                            invocation_context=invocation_context,
                            command_recovery=command_binding_authority,
                        )
                        lifecycle_events.extend(bound.events)
                        registered_environment = bound.registered_environment
                        if registered_environment is not None:
                            invocation_context = invocation_context.with_registered_environment(
                                registered_environment, validated_profile=invocation_context.profile
                            )
                        if bound.error is not None:
                            raise bound.error
                        command_recovery_context = invocation_context
                    except BaseException as error:
                        await self._environment_lifecycle.abort_environment_setup(
                            session_id=claim.session.id,
                            original_error=error,
                            execution_profile=invocation_context.profile,
                            invocation_context=invocation_context,
                        )
                        raise
                recovered = await self._recover_incomplete_session(
                    retain_cleanup_context=retain_cleanup_context,
                    preserve_interaction_id=preserve_interaction_id,
                    participant_context=participant_context,
                    session=claim.session,
                    session_before_fence=claim.session_before_fence,
                    previous_status=previous_status,
                    inactive_for_seconds=inactive_for_seconds,
                    reason=reason,
                    metadata=metadata,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    invocation_context=invocation_context,
                    claim_id=claim.claim_id,
                    execution_profile_snapshot=execution_profile_snapshot,
                    budget_policy=budget_policy_snapshot,
                    provider_disposition_task_id=provider_disposition_task_id,
                    provider_disposition_task_worker_id=(provider_disposition_task_worker_id),
                    provider_disposition_task_handoff_id=(provider_disposition_task_handoff_id),
                    interrupt_for_manual_tool_recovery=(interrupt_for_manual_tool_recovery),
                )
                if (
                    registered_environment is not None
                    and not pending_allocations
                    and await self._environment_lifecycle.release_deferred_materialization(
                        session=claim.session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                    )
                ):
                    # A deferred container left by a crash holds nothing the
                    # runtime must publish; the session materializes anew.
                    recovered = recovered.model_copy(
                        update={
                            "actions": (
                                IncompleteSessionRecoveryAction.REAPED_ALLOCATION,
                                *recovered.actions,
                            ),
                        }
                    )
                if pending_allocations:
                    recovered = recovered.model_copy(
                        update={
                            "actions": (
                                IncompleteSessionRecoveryAction.REAPED_ALLOCATION,
                                *(
                                    action
                                    for action in recovered.actions
                                    if action
                                    is not IncompleteSessionRecoveryAction.SKIPPED_TERMINAL
                                ),
                            ),
                        }
                    )
                if not lifecycle_events:
                    return recovered
                retained_actions = tuple(
                    action
                    for action in recovered.actions
                    if action is not IncompleteSessionRecoveryAction.SKIPPED_TERMINAL
                )
                return recovered.model_copy(
                    update={
                        "actions": (
                            IncompleteSessionRecoveryAction.RECONCILED_ENVIRONMENT_LIFECYCLE,
                            *retained_actions,
                        ),
                        "events": (*lifecycle_events, *recovered.events),
                        "message": (
                            "Reconciled orphaned environment lifecycle progress. "
                            + recovered.message
                        ),
                    },
                    deep=True,
                )

            return await self._recovery_ownership.run_with_heartbeat(
                claim=claim,
                recovery=recover_claimed_session,
            )
        except BaseException as exc:
            authoritative_failure = exc
            if command_recovery_context is not None:
                try:
                    await self._environment_lifecycle.abort_environment_setup(
                        session_id=session.id,
                        original_error=exc,
                        execution_profile=command_recovery_context.profile,
                        invocation_context=command_recovery_context,
                    )
                except BaseException as cleanup_error:
                    authoritative_failure = cleanup_error
                    raise
            raise
        finally:
            if claim is not None:
                await self._recovery_ownership.cleanup_claim(
                    authority=claim.require_authority(),
                    authoritative_failure=authoritative_failure,
                    release_environment_cleanup=(
                        pending_provider_disposition is not None
                        and pending_provider_disposition[0].action
                        is ProviderOperationResolutionAction.FALLBACK_RETRY
                    ),
                    execution_profile=(
                        None
                        if execution_profile_snapshot is None
                        else execution_profile_snapshot.profile
                    ),
                    invocation_context=invocation_context,
                    retain_open_interaction_invocation=(retain_open_interaction_invocation),
                    retain_invocation_context=retain_invocation_context,
                )

    async def _settle_terminal_invocation_closure_owned(
        self,
        *,
        session: Session,
        inactive_for_seconds: int | None,
        previous_status: SessionStatus,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
    ) -> IncompleteSessionRecoveryResult:
        """Fence a dead terminal owner whose hooks never released its run epoch."""

        claim: _IncompleteRecoveryClaim | None = None
        authoritative_failure: BaseException | None = None
        try:
            claim = await self._recovery_ownership.claim(
                session=session,
                inactive_for_seconds=inactive_for_seconds,
                execution_profile_snapshot=execution_profile_snapshot,
            )
            if claim is None:
                current = await self._recovery_ownership.require_session(session.id)
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=current.status,
                    actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                    events=(),
                    message="Terminal invocation ownership is still active.",
                )
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=claim.session.status,
                actions=(IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_OWNERSHIP,),
                events=(),
                message="Recovered terminal invocation ownership after worker loss.",
            )
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            if claim is not None:
                await self._recovery_ownership.cleanup_claim(
                    authority=claim.require_authority(),
                    authoritative_failure=authoritative_failure,
                )

    def detached_recovery_work(self) -> set[asyncio.Future[Any]]:
        """Recovery work still running after its caller stopped waiting.

        Claim renewal writes that outlived their heartbeat, and workspace
        artifact reads and tool-effect reconciliation lookups whose caller was
        cancelled or timed out. All may still use stores or application code.
        """

        return {
            *self._recovery_ownership.detached_recovery_work(),
            *self._workspace_observation_recovery.detached_recovery_work,
            *self._effect_reconciliation_owner.running(),
        }

    async def _require_governed_completion_task(
        self,
        *,
        session: Session,
        marker: dict[str, Any],
        invocation_context: InvocationContext,
    ) -> Task:
        """Validate the exact task before governed workspace recovery effects."""
        if (
            type(invocation_context) is not InvocationContext
            or invocation_context.work_attempt is None
        ):
            raise TypeError("Governed completion recovery requires its invocation authority.")
        admission = invocation_context.work_attempt.admission
        if (
            admission.execution_entry is None
            or marker.get("task_id") != admission.task_id
            or session.id != admission.session_id
            or session.instance_id != admission.session_invocation.session_instance_id
            or invocation_context.binding.session_id != session.id
            or invocation_context.binding.session_instance_id != session.instance_id
            or invocation_context.profile.fingerprint
            != admission.source_execution_profile_fingerprint
        ):
            raise RuntimeError("Governed completion recovery has conflicting task authority.")
        if self._task_store is None:
            raise RuntimeError("Governed completion recovery requires its task store.")
        raw_task = await self._task_store.load_task(admission.task_id)
        if type(raw_task) is not Task:
            raise RuntimeError("Governed completion recovery lost its exact running task.")
        task = copy_task(raw_task)
        if (
            task.id != admission.task_id
            or task.work_contract != admission.contract
            or task.session_id != admission.session_id
            or task.session_instance_id != admission.session_invocation.session_instance_id
            or task.status is not TaskStatus.RUNNING
        ):
            raise RuntimeError("Governed completion recovery lost its exact running task.")
        return task

    async def _record_governed_completion_stop(self, context: InvocationContext) -> None:
        """Retain the stop before the workspace marker or its reply can disappear."""
        if context.work_attempt is None or self._task_store is None:
            raise RuntimeError("Governed completion stop has no attempt owner.")
        await record_work_attempt_execution_stop(
            self._task_store,
            admission=context.work_attempt.admission,
            reason="workspace_finalization_recovery",
            redactor=self._secret_redactor,
        )

    async def _settle_recovered_completion_task(
        self,
        *,
        session: Session,
        marker: dict[str, Any],
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment,
        invocation_context: InvocationContext | None = None,
    ) -> Event | None:
        """Settle ordinary tasks; governed tasks retain their verified owner."""

        if invocation_context is not None and invocation_context.work_attempt is not None:
            await self._require_governed_completion_task(
                session=session, marker=marker, invocation_context=invocation_context
            )
            return None

        marker_has_task_identity = "task_id" in marker
        marker_task_id = marker.get("task_id")
        if self._task_store is None:
            if marker_has_task_identity and marker_task_id is not None:
                raise RuntimeError(
                    "Completion finalization recovery requires its durable task store."
                )
            return None
        if not marker_has_task_identity:
            tasks = await self._task_store.list_tasks(
                TaskQuery(
                    status=TaskStatus.RUNNING,
                    session_id=session.id,
                    limit=2,
                )
            )
            if len(tasks) > 1:
                raise RuntimeError(
                    "Legacy completion finalization marker has multiple running attached tasks."
                )
            if not tasks:
                return None
            task = tasks[0]
        else:
            if marker_task_id is None:
                return None
            if type(marker_task_id) is not str:
                raise RuntimeError("Completion finalization marker has an invalid task identity.")
            task = await self._task_store.load_task(marker_task_id)
            if task is None:
                raise RuntimeError("Completion finalization task is missing from durable storage.")
        if task.session_instance_id != session.instance_id:
            raise RuntimeError(
                "Completion finalization task belongs to a different session incarnation."
            )
        if task.status is TaskStatus.CANCELLED:
            return None
        if task.status is TaskStatus.COMPLETED:
            raise RuntimeError(
                "Completion finalization task was published before workspace output committed."
            )

        failure_payload = task_failure_payload_from_diagnostic(
            ExceptionDiagnostic(
                message=(
                    "Workspace output committed during recovery after the original "
                    "completion owner became unavailable."
                ),
                error_type="WorkspaceCompletionFinalizationRecovered",
            ),
            session_id=session.id,
            additional_fields={
                "phase": "workspace_finalize_recovery",
                "workspace_output_committed": True,
            },
        )
        if task.status is TaskStatus.FAILED:
            if task.error != failure_payload:
                return None
        elif task.status is not TaskStatus.RUNNING:
            raise RuntimeError(
                "Completion finalization task is not running or terminal during recovery."
            )
        elif task.worker_id is None:
            replayed = await load_direct_task_failure_replay(
                self._task_store,
                task_id=task.id,
                session_id=session.id,
                session_instance_id=session.instance_id,
                expected_error=failure_payload,
                claimed_terminalization_idempotency_key=(
                    runtime_task_terminalization_idempotency_key(
                        task_id=task.id,
                        session_id=session.id,
                        kind=TaskTerminalKind.FAILED,
                    )
                ),
            )
            task = (
                replayed
                if replayed is not None
                else await self._task_store.fail_task(
                    task.id,
                    failure_payload,
                    worker_id=None,
                )
            )
        else:
            if task.lease_expires_at is None:
                raise RuntimeError(
                    "Claimed completion finalization task lost its lease generation."
                )
            if not self._task_store.supports_attached_task_recovery_terminalization:
                raise RuntimeError(
                    "Task store cannot atomically settle an expired attached task owner."
                )
            task = await self._task_store.recover_attached_task_failure(
                TaskTerminalizationRequest(
                    task_id=task.id,
                    worker_id=task.worker_id,
                    lease_expires_at=task.lease_expires_at,
                    handoff_id=task.interrupted_handoff_id,
                    kind=TaskTerminalKind.FAILED,
                    error=failure_payload,
                    idempotency_key=runtime_task_terminalization_idempotency_key(
                        task_id=task.id,
                        session_id=session.id,
                        kind=TaskTerminalKind.FAILED,
                    ),
                ),
                session_id=session.id,
                session_instance_id=session.instance_id,
            )

        event_id = str(
            uuid5(
                _COMPLETION_FINALIZATION_TASK_EVENT_NAMESPACE,
                f"{session.id}\0{session.instance_id}\0{task.id}\0failed",
            )
        )
        event_template = self._task_event(
            RecoveryTaskEventRequest(
                event_type=EventType.TASK_FAILED,
                task=task,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
        )
        intended = event_with_runtime_generated_id(
            event_template.model_copy(
                update={
                    "id": event_id,
                    "timestamp": task.completed_at,
                    "payload": {
                        **event_template.payload,
                        "failure_phase": "workspace_finalize_recovery",
                        "workspace_output_committed": True,
                    },
                },
                deep=True,
            )
        )
        persisted = await self._event_writer.persist_exact_replay(intended)
        return (await self._event_writer.fan_out_persisted([persisted]))[0]

    async def _recover_pending_completion_finalization(
        self,
        *,
        session: Session,
        session_before_fence: Session,
        previous_status: SessionStatus,
        claim_id: str,
        marker: dict[str, Any],
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment,
        execution_profile: ExecutionProfileIdentity,
        invocation_context: InvocationContext,
        retain_cleanup_context: Callable[[InvocationContext], None],
    ) -> IncompleteSessionRecoveryResult:
        """Reconnect and retry finalization without changing its terminal outcome."""

        completing = marker["outcome"] == "completed"
        terminal_status = (
            SessionStatus.INTERRUPTED
            if marker["outcome"] == "interrupted"
            else SessionStatus.FAILED
        )
        terminal_event_type = {
            "completed": EventType.SESSION_COMPLETED,
            "failed": EventType.SESSION_FAILED,
            "interrupted": EventType.SESSION_INTERRUPTED,
        }[marker["outcome"]]
        if completing and invocation_context.work_attempt is not None:
            await self._require_governed_completion_task(
                session=session, marker=marker, invocation_context=invocation_context
            )
            await self._record_governed_completion_stop(invocation_context)

        if session.status is SessionStatus.RUNNING:
            recovery_claim_id = invocation_context.recovery_claim_id
            if recovery_claim_id is None:
                raise RuntimeError(
                    "Running completion finalization recovery lost its durable claim."
                )

            def fail_pending_completion(
                current_session: Session,
                checkpoint: dict[str, Any] | None,
                store_now: datetime,
            ) -> dict[str, Any]:
                if (
                    current_session.instance_id != session.instance_id
                    or current_session.run_epoch != session.run_epoch
                ):
                    raise SessionRunFenced(
                        "Completion finalization recovery lost its session authority."
                    )
                claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
                if (
                    claim is None
                    or claim[0] != recovery_claim_id
                    or claim[1] <= store_now
                    or completion_finalization.pending_completion_finalization_from_checkpoint(
                        checkpoint
                    )
                    != marker
                ):
                    raise SessionRunFenced(
                        "Completion finalization recovery lost its exact durable marker."
                    )
                return copy_durable_record(checkpoint, "checkpoint")

            session = await self._session_store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.RUNNING},
                to_status=terminal_status,
                store_time_checkpoint_transform=fail_pending_completion,
            )
        elif session.status is not terminal_status:
            raise RuntimeError("Pending finalization conflicts with the session terminal outcome.")

        completion_recovery = await self._environment_lifecycle.authorize_completion_recovery(
            session=session, invocation_context=invocation_context, marker=marker
        )
        events: list[Event] = []
        resolved_environment = registered_environment
        resolved_context = invocation_context
        authoritative_error: BaseException | None = None
        try:
            disposal_recovered = await self._environment_lifecycle.recover_completion_disposal(
                session=session,
                registered_agent=registered_agent,
                registered_environment=resolved_environment,
                execution_profile=execution_profile,
                marker=marker,
                completion_recovery=completion_recovery,
                invocation_context=resolved_context,
            )
            if not disposal_recovered:
                factory_started = await self._environment_lifecycle.emit_factory_started(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=resolved_environment,
                    execution_profile=execution_profile,
                    invocation_context=resolved_context,
                )
                if factory_started is not None:
                    events.append(factory_started)
                factory_resolution = await self._environment_lifecycle.resolve_factory(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=resolved_environment,
                    started_event=factory_started,
                    operation=EnvironmentFactoryOperation.RECONNECT,
                    execution_profile=execution_profile,
                    invocation_context=resolved_context,
                    completion_recovery=completion_recovery,
                )
                events.extend(factory_resolution.events)
                resolved_environment = factory_resolution.registered_environment
                if resolved_environment is None:
                    raise RuntimeError("Completion finalization recovery resolved no environment.")
                resolved_context = resolved_context.with_registered_environment(
                    resolved_environment,
                    validated_profile=execution_profile,
                )
                retain_cleanup_context(resolved_context)
                if factory_resolution.error is not None:
                    raise factory_resolution.error
                binding_started = await self._environment_lifecycle.emit_binding_started(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=resolved_environment,
                    execution_profile=execution_profile,
                    invocation_context=resolved_context,
                )
                if binding_started is not None:
                    events.append(binding_started)
                binding_result = await self._environment_lifecycle.bind(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=resolved_environment,
                    started_event=binding_started,
                    execution_profile=execution_profile,
                    invocation_context=resolved_context,
                    completion_recovery=completion_recovery,
                )
                events.extend(binding_result.events)
                resolved_environment = binding_result.registered_environment
                if resolved_environment is None:
                    raise RuntimeError(
                        "Completion finalization recovery lost its bound environment."
                    )
                resolved_context = resolved_context.with_registered_environment(
                    resolved_environment,
                    validated_profile=execution_profile,
                )
                retain_cleanup_context(resolved_context)
                if binding_result.error is not None:
                    raise binding_result.error
                finalized = await self._environment_lifecycle.finalize_terminal_event(
                    event=Event(
                        type=terminal_event_type,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=resolved_environment.spec.name,
                        payload={
                            "completion_finalization_recovery": True,
                            **await recorded_terminal_tool_completion_payload(
                                self._session_store, session
                            ),
                        },
                    ),
                    session=session,
                    registered_environment=resolved_environment,
                    execution_profile=execution_profile,
                    invocation_context=resolved_context,
                )
                events.extend(finalized.events)
                finalize_error = finalized.event.payload.get("binding_finalize_error")
                if type(finalize_error) is dict:
                    return IncompleteSessionRecoveryResult(
                        session_id=session.id,
                        previous_status=previous_status,
                        status=session.status,
                        actions=(IncompleteSessionRecoveryAction.FAILED,),
                        events=tuple(events),
                        message=(
                            "Workspace finalization remains pending; the terminal "
                            "session was not re-executed."
                        ),
                    )
            if completing:
                task_failed_event = await self._settle_recovered_completion_task(
                    session=session,
                    marker=marker,
                    registered_agent=registered_agent,
                    registered_environment=resolved_environment,
                    invocation_context=resolved_context,
                )
                if task_failed_event is not None:
                    events.append(task_failed_event)
            terminal_repair = await self.terminal_finalization.repair(
                session=session,
                terminal_run_epoch=session_before_fence.run_epoch,
                terminal_timestamp=session_before_fence.updated_at,
                previous_status=previous_status,
                claim_id=claim_id,
            )
            events.extend(terminal_repair.events)
            await self._environment_lifecycle.clear_completion_finalization(
                session_id=session.id,
                expected_marker=marker,
            )
            await self._environment_lifecycle.abort_environment_setup(
                session_id=session.id,
                original_error=None,
                allow_deferred_settlement=True,
                execution_profile=execution_profile,
                invocation_context=resolved_context,
            )
        except BaseException as exc:
            authoritative_error = exc
            raise
        finally:
            if authoritative_error is not None:
                try:
                    await self._environment_lifecycle.abort_environment_setup(
                        session_id=session.id,
                        original_error=authoritative_error,
                        execution_profile=execution_profile,
                        invocation_context=resolved_context,
                    )
                except BaseException as cleanup_error:
                    if cleanup_error is not authoritative_error:
                        raise BaseExceptionGroup(
                            "Completion finalization recovery and cleanup failed.",
                            [authoritative_error, cleanup_error],
                        ) from cleanup_error
        return IncompleteSessionRecoveryResult(
            session_id=session.id,
            previous_status=previous_status,
            status=session.status,
            actions=(IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_FINALIZATION,),
            events=tuple(events),
            message=(
                "Recovered committed workspace output without re-running model or tool effects."
            ),
        )

    async def _recover_incomplete_session(
        self,
        *,
        retain_cleanup_context: Callable[[InvocationContext], None],
        session: Session,
        session_before_fence: Session,
        previous_status: SessionStatus,
        inactive_for_seconds: int | None,
        reason: str,
        metadata: dict[str, Any],
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        invocation_context: InvocationContext | None,
        claim_id: str,
        execution_profile_snapshot: ActiveInvocationExecutionProfile | None,
        budget_policy: BudgetPolicy | None,
        provider_disposition_task_id: str | None = None,
        provider_disposition_task_worker_id: str | None = None,
        provider_disposition_task_handoff_id: str | None = None,
        participant_context: CollaborationAccessContext | None = None,
        interrupt_for_manual_tool_recovery: bool = False,
        preserve_interaction_id: str | None = None,
    ) -> IncompleteSessionRecoveryResult:
        if (execution_profile_snapshot is None) != (invocation_context is None):
            raise RuntimeError(
                "Incomplete recovery requires profile authority and its context together."
            )
        if invocation_context is not None:
            if execution_profile_snapshot is None:
                raise RuntimeError("Incomplete recovery lost its execution-profile authority.")
            if (
                invocation_context.binding.session_id != session.id
                or invocation_context.binding.session_instance_id != session.instance_id
                or invocation_context.binding.run_epoch != session.run_epoch
                or invocation_context.profile is not execution_profile_snapshot.profile
            ):
                raise RuntimeError("Incomplete recovery lost its reconstructed invocation context.")
        actions: list[IncompleteSessionRecoveryAction] = []
        events: list[Event] = []
        checkpoint = await self._session_store.load_checkpoint(session.id)
        pending_completion_finalization = (
            completion_finalization.pending_completion_finalization_from_checkpoint(checkpoint)
        )
        provider_interrupt_payload = _provider_cancellation_interrupt_payload(checkpoint)
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
        environment_name = _environment_name(registered_environment)

        if pending_completion_finalization is not None:
            if (
                registered_environment is None
                or execution_profile_snapshot is None
                or invocation_context is None
            ):
                raise RuntimeError("Pending completion finalization lost recovery authority.")
            if any(
                item is not None
                for item in (
                    provider_interrupt_payload,
                    pending_approval,
                    pending_user_input,
                    pending_tool_round,
                )
            ):
                raise RuntimeError(
                    "Pending completion finalization conflicts with other recovery work."
                )
            return await self._recover_pending_completion_finalization(
                retain_cleanup_context=retain_cleanup_context,
                session=session,
                session_before_fence=session_before_fence,
                previous_status=previous_status,
                claim_id=claim_id,
                marker=pending_completion_finalization,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile_snapshot.profile,
                invocation_context=invocation_context,
            )

        if interrupt_for_manual_tool_recovery:
            if (
                pending_tool_round is None
                or pending_approval is not None
                or pending_user_input is not None
                or provider_interrupt_payload is not None
            ):
                raise RuntimeError(
                    "Manual tool-recovery handoff requires one pending ordinary tool round."
                )
            if session.status in {SessionStatus.PENDING, SessionStatus.RUNNING}:
                interrupt_payload = {
                    **pending_rounds.pending_tool_round_identity(pending_tool_round).payload(),
                    "reason": reason,
                    "metadata": metadata,
                    "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                    "recovered": True,
                    "manual_recovery_required": True,
                    "interruption_request_id": str(uuid4()),
                }
                session = await self._session_store.transition_status_and_checkpoint(
                    session.id,
                    from_statuses={session.status},
                    to_status=SessionStatus.INTERRUPTING,
                    checkpoint_transform=_checkpoint_with_pending_session_interrupt(
                        interrupt_payload, cascade_created_at=self._clock()
                    ),
                )
            if session.status is SessionStatus.INTERRUPTING:
                session = await self._finalize_interrupting_for_recovery(
                    recovery_claim_id=claim_id,
                    preserve_interaction_id=preserve_interaction_id,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    events=events,
                    execution_profile=(
                        None
                        if execution_profile_snapshot is None
                        else execution_profile_snapshot.profile
                    ),
                    invocation_context=invocation_context,
                )
            if session.status is not SessionStatus.INTERRUPTED:
                raise RuntimeError(
                    "Manual tool-recovery handoff did not reach an interrupted session."
                )
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,),
                events=tuple(events),
                message="Fenced the stale run and retained its pending manual tool recovery.",
            )

        if provider_interrupt_payload is not None:
            if session.status is SessionStatus.INTERRUPTED:
                return await self.terminal_finalization.repair(
                    session=session,
                    terminal_run_epoch=session_before_fence.run_epoch,
                    terminal_timestamp=session_before_fence.updated_at,
                    previous_status=previous_status,
                    claim_id=claim_id,
                )
            if session.status in {SessionStatus.PENDING, SessionStatus.RUNNING}:
                session = await self._session_store.transition_status_and_checkpoint(
                    session.id,
                    from_statuses={session.status},
                    to_status=SessionStatus.INTERRUPTING,
                    checkpoint_transform=_checkpoint_with_pending_session_interrupt(
                        provider_interrupt_payload, cascade_created_at=self._clock()
                    ),
                )
            elif session.status is not SessionStatus.INTERRUPTING:
                raise RuntimeError(
                    "Provider cancellation interruption marker conflicts with session status."
                )
            session = await self._finalize_interrupting_for_recovery(
                recovery_claim_id=claim_id,
                preserve_interaction_id=preserve_interaction_id,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                events=events,
                execution_profile=(
                    None
                    if execution_profile_snapshot is None
                    else execution_profile_snapshot.profile
                ),
                invocation_context=invocation_context,
            )
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,),
                events=tuple(events),
                message="Finalized a durable provider cancellation interruption.",
            )

        pending_provider_resolution = await load_pending_provider_operation_disposition(
            self._session_store,
            session.id,
            checkpoint=checkpoint,
        )
        if pending_provider_resolution is not None:
            pending_disposition, resolution_result = pending_provider_resolution
            if provider_disposition_task_id is not None:
                source_stage = await self._session_store.load_model_completion_stage(
                    session.id,
                    pending_disposition.stage_id,
                )
                recovery_context = (
                    None
                    if source_stage is None
                    else model_completion_recovery_context_from_stage(source_stage)
                )
                if (
                    recovery_context is None
                    or recovery_context.task_id != provider_disposition_task_id
                    or not pending_disposition.execution_claimed
                    or pending_disposition.execution_task_worker_id
                    != provider_disposition_task_worker_id
                    or pending_disposition.execution_task_handoff_id
                    != provider_disposition_task_handoff_id
                ):
                    raise ProviderOperationEvidenceError(
                        "Typed provider recovery conflicts with its transferred authority."
                    )
            if await self._retire_completed_provider_operation_disposition(
                pending=pending_disposition,
                result=resolution_result,
            ):
                actions.append(
                    IncompleteSessionRecoveryAction.REPAIRED_PROVIDER_OPERATION_RESOLUTION
                )
                checkpoint = await self._session_store.load_checkpoint(session.id)
            elif pending_disposition.action is ProviderOperationResolutionAction.FAIL:
                if provider_disposition_task_id is None:
                    (
                        task_id,
                        requires_typed_continuation,
                    ) = await self._automatic_provider_disposition_task_context(pending_disposition)
                    task_worker_id = None
                else:
                    task_id = provider_disposition_task_id
                    task_worker_id = provider_disposition_task_worker_id
                    requires_typed_continuation = False
                if requires_typed_continuation:
                    return IncompleteSessionRecoveryResult(
                        session_id=session.id,
                        previous_status=previous_status,
                        status=session.status,
                        actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                        events=tuple(events),
                        message=(
                            "Accepted provider-operation failure awaits its elected "
                            "attached-task continuation."
                        ),
                    )
                if execution_profile_snapshot is None:
                    raise RuntimeError(
                        "Provider-operation failure recovery has no execution profile."
                    )
                async for event in self._session_finalization.fail_recovered_provider_operation(
                    ProviderOperationFailureRequest(
                        resolution_event=resolution_result.event,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile_snapshot.profile,
                        task_id=task_id,
                        task_worker_id=task_worker_id,
                        task_handoff_id=provider_disposition_task_handoff_id,
                        legacy_resolution_without_profile=(
                            resolution_result.record.execution_profile_fingerprint is None
                        ),
                        invocation_context=invocation_context,
                    )
                ):
                    events.append(event)
                failed_session = await self._recovery_ownership.require_session(
                    pending_disposition.session_id
                )
                terminal_event_id = provider_operation_resolution_outcome_event_id(
                    resolution_result.record.resolution_id,
                    "session_failed",
                )
                terminal_records = await self._session_store.query_events(
                    EventQuery(
                        session_id=pending_disposition.session_id,
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
                if not await self._retire_completed_provider_operation_disposition(
                    pending=pending_disposition,
                    result=resolution_result,
                    terminal_hook_authority=terminal_hook_authority,
                ):
                    raise RuntimeError(
                        "Recovered provider-operation failure has no terminal outcome."
                    )
                current = await self._recovery_ownership.require_session(session.id)
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=current.status,
                    actions=(
                        IncompleteSessionRecoveryAction.REPAIRED_PROVIDER_OPERATION_RESOLUTION,
                    ),
                    events=tuple(events),
                    message="Finished the accepted provider-operation failure.",
                )
            elif session.status is SessionStatus.INTERRUPTED:
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=session.status,
                    actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                    events=tuple(events),
                    message=(
                        "Accepted provider-operation fallback is ready for fenced continuation."
                    ),
                )
            elif session.status is SessionStatus.RUNNING:
                if provider_disposition_task_id is None:
                    (
                        _task_id,
                        requires_typed_continuation,
                    ) = await self._automatic_provider_disposition_task_context(pending_disposition)
                    task_worker_id = None
                else:
                    _task_id = provider_disposition_task_id
                    task_worker_id = provider_disposition_task_worker_id
                    requires_typed_continuation = False
                if requires_typed_continuation:
                    return IncompleteSessionRecoveryResult(
                        session_id=session.id,
                        previous_status=previous_status,
                        status=session.status,
                        actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                        events=tuple(events),
                        message=(
                            "Accepted provider-operation fallback awaits its elected "
                            "attached-task continuation."
                        ),
                    )
                if execution_profile_snapshot is None:
                    raise RuntimeError(
                        "Provider-operation fallback recovery has no execution profile."
                    )
                source_stage = await self._session_store.load_model_completion_stage(
                    session.id,
                    pending_disposition.stage_id,
                )
                if source_stage is None:
                    raise RuntimeError("Resolved provider-operation stage is missing.")
                recovery_context = model_completion_recovery_context_from_stage(source_stage)
                if recovery_context is None:
                    raise RuntimeError(
                        "Provider-operation fallback requires durable model-completion context."
                    )
                interaction_id = await self._activate_latest_open_interaction(session.id)
                if interaction_id is None:
                    raise RuntimeError(
                        "Provider-operation fallback recovery has no open interaction."
                    )
                async for event in self._run_pending_provider_operation_fallback(
                    pending=pending_disposition,
                    participant_context=participant_context,
                    result=resolution_result,
                    session=session,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    execution_profile_snapshot=execution_profile_snapshot,
                    recovery_context=recovery_context,
                    budget_policy=budget_policy,
                    release_run_fence_on_cleanup=False,
                    task_worker_id=task_worker_id,
                    task_handoff_id=provider_disposition_task_handoff_id,
                    invocation_context=invocation_context,
                ):
                    events.append(event)
                current = await self._recovery_ownership.require_session(session.id)
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=current.status,
                    actions=(
                        IncompleteSessionRecoveryAction.REPAIRED_PROVIDER_OPERATION_RESOLUTION,
                    ),
                    events=tuple(events),
                    message="Finished the accepted provider-operation fallback.",
                )

        if session.status in _RECOVERY_RESUMABLE_SESSION_STATUSES and (
            await self.terminal_finalization.repair_required(
                session=session,
                checkpoint=checkpoint,
            )
        ):
            repaired = await self.terminal_finalization.repair(
                session=session,
                terminal_run_epoch=session_before_fence.run_epoch,
                terminal_timestamp=session_before_fence.updated_at,
                previous_status=previous_status,
                claim_id=claim_id,
            )
            actions.extend(repaired.actions)
            events.extend(repaired.events)
            session = await self._recovery_ownership.require_session(session.id)
            checkpoint = await self._session_store.load_checkpoint(session.id)

            if settled_invocation_terminal_decision_from_checkpoint(checkpoint) is not None:
                # Terminal repair already authenticated and published the exact
                # winner. Its retired human gate is not missing executable work:
                # do not re-enter model reconciliation after repairing closure.
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=session.status,
                    actions=tuple(actions),
                    events=tuple(events),
                    message="Recovered the committed terminal decision without redispatch.",
                )

        if inactive_for_seconds is not None:
            events.append(
                await self._event_writer.emit(
                    Event(
                        type=EventType.SESSION_RUN_FENCED,
                        session_id=session.id,
                        agent_name=session.agent_name,
                        environment_name=environment_name,
                        payload={
                            "previous_run_epoch": session_before_fence.run_epoch,
                            "run_epoch": session.run_epoch,
                            "inactive_for_seconds": inactive_for_seconds,
                            "reason": reason,
                            "metadata": metadata,
                        },
                    )
                )
            )

        provider_operation_addressed = False
        if session.status is SessionStatus.INTERRUPTING:
            provider_operation_addressed = (
                await self._model_completion_recovery.cancel_provider_operation_for_interruption(
                    session,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    invocation_context=invocation_context,
                )
                is not None
            )
            session = await self._recovery_ownership.require_session(session.id)
        active_model_stage = await self._session_store.load_active_model_completion_stage(
            session.id
        )
        if provider_operation_addressed and active_model_stage is not None:
            # Cancellation already resolved the exact in-flight operation. A
            # cancelled, pending, unavailable, or unconfirmed provider outcome
            # must not be reinterpreted as recoverable model completion before
            # the local interruption is finalized. If completion won, the
            # cancellation path promoted it and cleared the active stage above.
            model_boundary = ModelCompletionBoundaryReconciliation(
                state="none",
                session=session,
            )
        else:
            model_boundary = (
                await self._model_completion_recovery.reconcile_model_completion_boundary(
                    session,
                    invocation_context=invocation_context,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                )
            )
        session = model_boundary.session
        events.extend(copy_event(event) for event in model_boundary.recovery_events)
        if model_boundary.state == "provider_operation_pending":
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                events=tuple(events),
                message=(
                    "Provider operation is still pending; its exact durable dispatch remains "
                    "eligible for later recovery."
                ),
            )
        if model_boundary.state == "provider_operation_unavailable":
            if active_model_stage is None:
                raise RuntimeError(
                    "Unavailable provider operation has no active model-completion stage."
                )
            unavailable_event = next(
                (
                    event
                    for event in reversed(model_boundary.recovery_events)
                    if event.type == EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED
                ),
                None,
            )
            if unavailable_event is None:
                raise RuntimeError(
                    "Unavailable provider operation has no durable recovery evidence."
                )
            try:
                recovery_reason = ProviderOperationUnavailableReason(
                    unavailable_event.payload.get("recovery_reason")
                )
            except (TypeError, ValueError):
                raise RuntimeError(
                    "Unavailable provider operation has malformed recovery evidence."
                ) from None
            if session.status in {SessionStatus.PENDING, SessionStatus.RUNNING}:
                session = await self._session_store.transition_status(
                    session.id,
                    from_statuses={session.status},
                    to_status=SessionStatus.INTERRUPTED,
                )
            elif session.status is not SessionStatus.INTERRUPTED:
                raise RuntimeError(
                    "Unavailable provider operation cannot pause the current session status."
                )
            interrupted_event = event_with_runtime_generated_id(
                Event(
                    id=str(
                        uuid5(
                            _PROVIDER_OPERATION_UNAVAILABLE_INTERRUPT_NAMESPACE,
                            f"{session.id}\0{active_model_stage.stage.stage_id}\0"
                            f"{unavailable_event.id}",
                        )
                    ),
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session.id,
                    agent_name=session.agent_name,
                    environment_name=environment_name,
                    timestamp=unavailable_event.timestamp,
                    payload={
                        "interruption_type": "provider_operation_unavailable",
                        "stage_id": active_model_stage.stage.stage_id,
                        "recovery_reason": recovery_reason.value,
                        "duplicate_request_risk": provider_operation_duplicate_request_risk(
                            recovery_reason
                        ),
                    },
                )
            )
            persisted_interrupted = await self._event_writer.persist_exact_replay(interrupted_event)
            [interrupted_event] = await self._event_writer.fan_out_persisted(
                [persisted_interrupted]
            )
            events.append(interrupted_event)
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,),
                events=tuple(events),
                message=(
                    "Exact provider continuation is unavailable; explicit fallback retry or "
                    "failure is required."
                ),
            )
        if (
            model_boundary.state
            in {
                "promoted",
                "provider_operation_reconciled",
            }
            and model_boundary.completion_event is not None
        ):
            events.append(copy_event(model_boundary.completion_event))
        checkpoint = await self._session_store.load_checkpoint(session.id)
        pending_approval = pending_approval_reader.pending_approval_from_checkpoint(checkpoint)
        pending_user_input, _resolution_intent = user_input_lifecycle_authority_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            current_run_epoch=session.run_epoch,
            runtime_session=session,
        )
        pending_tool_round = pending_round_reader.pending_tool_round_from_checkpoint(checkpoint)
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
                    "Pending user-input recovery authority is ambiguous."
                )
        if pending_tool_round is None and await self._deferred_input.materialize_if_present(
            session.id
        ):
            actions.append(IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND)

        if (
            session.status is SessionStatus.RUNNING
            and invocation_context is not None
            and invocation_context.work_attempt is None
            and pending_tool_round is None
            and pending_approval is None
            and pending_user_input is None
        ):
            from cayu.runtime._producer_completion_replay import prepare_producer_completion_replay

            producer_replay = await prepare_producer_completion_replay(
                self._session_store, session, checkpoint, invocation_context, model_boundary
            )
            if producer_replay is not None:
                stage = model_boundary.completed_stage
                assert stage is not None
                semantics = model_completion_recovery_context_from_stage(stage)
                if (
                    semantics is None
                    or semantics.execution_profile_fingerprint
                    != invocation_context.profile.fingerprint
                    or semantics.interaction_id != invocation_context.binding.interaction_id
                    or semantics.task_id is not None
                ):
                    raise ModelCompletionManualRecoveryRequired(
                        "Producer completion lacks its original native run semantics."
                    )
                stream = self._run_session(
                    RecoverySessionRunRequest(
                        session=session,
                        invocation_context=invocation_context,
                        participant_context=participant_context,
                        messages=await self._session_store.load_transcript(session.id),
                        messages_to_append=[],
                        max_steps=semantics.max_steps,
                        limits=semantics.limits,
                        budget_limits=semantics.budget_limits,
                        retry_policy=semantics.retry_policy,
                        structured_output=semantics.structured_output,
                        tool_completion=semantics.tool_completion,
                        thinking=semantics.thinking,
                        request_metadata=semantics.request_metadata,
                        task_id=None,
                        task_worker_id=None,
                        task_handoff_id=None,
                        start_event_type=None,
                        start_event_payload={},
                        start_task_on_enter=False,
                        release_run_fence_on_exit=False,
                        run_limit_accounting=semantics.run_limit_accounting,
                        producer_replay=producer_replay,
                    )
                )
                async with _close_delegated_event_stream(stream) as owned:
                    async for event in owned:
                        events.append(event)
                current = await self._recovery_ownership.require_session(session.id)
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=current.status,
                    actions=(
                        *actions,
                        IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE
                        if current.status is SessionStatus.COMPLETED
                        else IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED
                        if current.status is SessionStatus.INTERRUPTED
                        else IncompleteSessionRecoveryAction.FAILED,
                    ),
                    events=tuple(events),
                    message="Replayed the retained producer model result without provider redispatch.",
                )

        if (
            session.status is SessionStatus.RUNNING
            and invocation_context is not None
            and invocation_context.work_attempt is not None
            and preserve_interaction_id == invocation_context.binding.interaction_id
            and pending_tool_round is not None
            and pending_approval is None
            and pending_user_input is None
            and has_recoverable_structured_output_round(pending_tool_round)
        ):
            # Replacement preserves this governed interaction. Reconcile its
            # reserved validation before interrupting the predecessor epoch so
            # that interruption cannot consume the original repair allowance.
            # This publishes validation evidence; only the replacement model
            # loop may dispatch a later repair under its execution claim.
            snapshot = await self._session_store.load_transcript_snapshot(session.id)
            transcript = [detach_message(record.message) for record in snapshot.records]
            async for event in self._pending_tool_round_recovery.recover_pending_tool_round(
                session=session,
                invocation_context=invocation_context,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                messages=transcript,
                execution_profile=invocation_context.profile,
                incomplete_recovery_claimed=True,
                expected_transcript_cursor=snapshot.cursor,
            ):
                events.append(event)
            checkpoint, pending_tool_round = await pending_round_reader.load_pending_tool_round(
                self._session_store,
                session.id,
            )
            actions.append(IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND)

        if (
            pending_tool_round is not None
            and pending_approval is None
            and pending_user_input is None
        ):
            pending_durable_children = await self._pending_durable_subagent_children(
                session=session,
                checkpoint=checkpoint,
                pending_round=pending_tool_round,
                registered_agent=registered_agent,
            )
            if pending_durable_children:
                statuses = sorted({child.status.value for child in pending_durable_children})
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=session.status,
                    actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                    events=tuple(events),
                    message=(
                        "Durable subagent work is still active; the parent tool round "
                        "remains pending until child completion is durable "
                        f"(children={len(pending_durable_children)}, "
                        f"statuses={','.join(statuses)})."
                    ),
                )

        if (
            session.status is SessionStatus.RUNNING
            and invocation_context is not None
            and invocation_context.work_attempt is None
            and pending_approval is None
            and pending_user_input is None
            and model_boundary.completed_stage is not None
        ):
            semantics = model_completion_recovery_context_from_stage(model_boundary.completed_stage)
            if semantics is not None and semantics.tool_completion is not None:
                policy = await load_recorded_tool_completion_policy(
                    self._session_store,
                    session,
                    checkpoint,
                    execution_profile=invocation_context.profile,
                    max_steps=semantics.max_steps,
                    limits=semantics.limits,
                    retry_policy=semantics.retry_policy,
                    context=semantics,
                )
                if (
                    policy is not None
                    and pending_tool_round is not None
                    and await pending_round_has_completion_success(
                        self._session_store,
                        session,
                        pending_tool_round,
                        policy=policy,
                        interaction_id=invocation_context.binding.interaction_id,
                    )
                ):
                    snapshot = await self._session_store.load_transcript_snapshot(session.id)
                    async for event in self._pending_tool_round_recovery.recover_pending_tool_round(
                        session=session,
                        invocation_context=invocation_context,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        messages=[detach_message(record.message) for record in snapshot.records],
                        execution_profile=invocation_context.profile,
                        incomplete_recovery_claimed=True,
                        expected_transcript_cursor=snapshot.cursor,
                    ):
                        events.append(event)
                    actions.append(IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND)
                result = await recorded_tool_completion_result(
                    self._session_store,
                    session,
                    policy=policy,
                    execution_profile=invocation_context.profile,
                    registered_agent=registered_agent,
                )
                if result is not None:
                    stream = self._run_session(
                        RecoverySessionRunRequest(
                            session=session,
                            invocation_context=invocation_context,
                            participant_context=participant_context,
                            messages=await self._session_store.load_transcript(session.id),
                            messages_to_append=[],
                            max_steps=semantics.max_steps,
                            limits=semantics.limits,
                            budget_limits=semantics.budget_limits,
                            retry_policy=semantics.retry_policy,
                            structured_output=semantics.structured_output,
                            tool_completion=policy,
                            tool_completion_replay=result,
                            thinking=semantics.thinking,
                            request_metadata=semantics.request_metadata,
                            task_id=semantics.task_id,
                            task_worker_id=None,
                            task_handoff_id=None,
                            start_event_type=None,
                            start_event_payload={},
                            start_task_on_enter=False,
                            release_run_fence_on_exit=False,
                            run_limit_accounting=semantics.run_limit_accounting,
                        )
                    )
                    async with _close_delegated_event_stream(stream) as owned:
                        async for event in owned:
                            events.append(event)
                    current = await self._recovery_ownership.require_session(session.id)
                    return IncompleteSessionRecoveryResult(
                        session_id=session.id,
                        previous_status=previous_status,
                        status=current.status,
                        actions=(
                            *actions,
                            IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
                        ),
                        events=tuple(events),
                        message="Completed from the recorded final tool without another tool or provider dispatch.",
                    )

        failed_with_recoverable_tool_round = (
            session.status is SessionStatus.FAILED
            and pending_tool_round is not None
            and pending_approval is None
            and pending_user_input is None
        )
        if session.status in {SessionStatus.PENDING, SessionStatus.RUNNING} or (
            failed_with_recoverable_tool_round
        ):
            if pending_approval is not None:
                interrupt_payload = {
                    "model_step_id": pending_approval.model_step_id,
                    "model_attempt_id": pending_approval.model_attempt_id,
                    "tool_round_id": pending_approval.tool_round_id,
                    "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                    **approval_support.bounded_pending_approval_event_payload(
                        pending_approval,
                        redactor=self._secret_redactor,
                    ),
                    "recovered": True,
                    "reason": reason,
                    "metadata": metadata,
                }
            elif pending_user_input is not None:
                interrupt_payload = {
                    "model_step_id": pending_user_input.model_step_id,
                    "model_attempt_id": pending_user_input.model_attempt_id,
                    "tool_round_id": pending_user_input.tool_round_id,
                    "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                    **pending_user_input_interruption_payload(pending_user_input),
                    "recovered": True,
                    "reason": reason,
                    "metadata": metadata,
                }
            elif pending_tool_round is not None:
                interrupt_payload = {
                    **pending_rounds.pending_tool_round_identity(pending_tool_round).payload(),
                    "reason": reason,
                    "metadata": metadata,
                    "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                    "recovered": True,
                }
            else:
                interrupt_payload = {
                    "reason": reason,
                    "metadata": metadata,
                    "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                    "recovered": True,
                }
            interrupt_payload["interruption_request_id"] = str(uuid4())
            try:
                session = await self._session_store.transition_status_and_checkpoint(
                    session.id,
                    from_statuses={session.status},
                    to_status=SessionStatus.INTERRUPTING,
                    checkpoint_transform=_checkpoint_with_pending_session_interrupt(
                        interrupt_payload, cascade_created_at=self._clock()
                    ),
                )
            except ValueError:
                session = await self._recovery_ownership.require_session(session.id)
                if session.status in _RECOVERY_RESUMABLE_SESSION_STATUSES:
                    return IncompleteSessionRecoveryResult(
                        session_id=session.id,
                        previous_status=previous_status,
                        status=session.status,
                        actions=(IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,),
                        events=(),
                        message="Session changed during recovery; recovery skipped.",
                    )
                raise
            session = await self._recovery_ownership.require_session(session.id)
            checkpoint = await self._session_store.load_checkpoint(session.id)
            pending_approval = pending_approval_reader.pending_approval_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
            )
            pending_tool_round = pending_round_reader.pending_tool_round_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
                runtime_session=session,
            )

        workspace_recovery_events = (
            await self._workspace_observation_recovery.recover_workspace_observations(
                session=session,
                registered_environment=registered_environment,
                execution_profile_snapshot=execution_profile_snapshot,
                invocation_context=invocation_context,
            )
        )
        if workspace_recovery_events:
            events.extend(workspace_recovery_events)
            actions.append(IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_OBSERVATION)
            checkpoint = await await_workspace_observation_store_read(
                lambda: self._session_store.load_checkpoint(session.id),
                operation="Post-recovery workspace observation checkpoint read",
            )
            pending_tool_round = pending_round_reader.pending_tool_round_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
                runtime_session=session,
            )

        if pending_tool_round is not None and pending_approval is None:
            transcript_snapshot = await self._session_store.load_transcript_snapshot(session.id)
            try:
                transcript = [
                    detach_message(record.message) for record in transcript_snapshot.records
                ]
                expected_transcript_cursor = transcript_snapshot.cursor
            finally:
                del transcript_snapshot
            try:
                async for event in self._pending_tool_round_recovery.recover_pending_tool_round(
                    session=session,
                    invocation_context=invocation_context,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    messages=transcript,
                    execution_profile=(
                        None
                        if execution_profile_snapshot is None
                        else execution_profile_snapshot.profile
                    ),
                    incomplete_recovery_claimed=True,
                    # This caller finishes the takeover when replay is elected.
                    elect_replay=True,
                    expected_transcript_cursor=expected_transcript_cursor,
                ):
                    events.append(event)
            except tool_call_replay.ToolCallReplayRequired as replay:
                # The continuation that took over replays the interrupted calls,
                # so finish this owner's interruption with the round still open.
                session = await self._recovery_ownership.require_session(session.id)
                session = await self._finalize_interrupting_for_recovery(
                    recovery_claim_id=claim_id,
                    preserve_interaction_id=preserve_interaction_id,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    events=events,
                    execution_profile=(
                        None
                        if execution_profile_snapshot is None
                        else execution_profile_snapshot.profile
                    ),
                    invocation_context=invocation_context,
                )
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=session.status,
                    actions=(*actions, IncompleteSessionRecoveryAction.PENDING_TOOL_EFFECT),
                    events=tuple(events),
                    message=str(replay),
                )
            except ToolEffectReconciliationRequired as unresolved:
                # Unknown external effects are a retained recovery pause. Finish
                # this owner's interruption before exposing the next explicit
                # receipt action; do not make that action drain our old marker.
                session = await self._recovery_ownership.require_session(session.id)
                session = await self._finalize_interrupting_for_recovery(
                    recovery_claim_id=claim_id,
                    preserve_interaction_id=preserve_interaction_id,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    events=events,
                    execution_profile=(
                        None
                        if execution_profile_snapshot is None
                        else execution_profile_snapshot.profile
                    ),
                    invocation_context=invocation_context,
                )
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=session.status,
                    actions=(
                        *actions,
                        (
                            IncompleteSessionRecoveryAction.PENDING_SUBAGENT
                            if isinstance(unresolved, ForegroundSubagentRecoveryRequired)
                            else IncompleteSessionRecoveryAction.PENDING_TOOL_EFFECT
                        ),
                    ),
                    events=tuple(events),
                    pending_subagent_session_ids=(
                        (unresolved.child_session_id,)
                        if isinstance(unresolved, ForegroundSubagentRecoveryRequired)
                        else ()
                    ),
                    message=(
                        str(unresolved)
                        if isinstance(unresolved, ForegroundSubagentRecoveryRequired)
                        else "External effect requires explicit receipt reconciliation or continuation."
                    ),
                )
            except ToolApprovalRequired:
                # Fail-closed planning of an ambiguous crash boundary may
                # atomically restore the human gate. Incomplete-session
                # recovery owns the outer interrupt transition, so retain the
                # paired approval and finish that transition below instead of
                # treating the pause as failure.
                pass
            except tool_round_recovery.UnsafeToolRoundContinuationError:
                if session.status not in _UNREPLAYABLE_TOOL_ROUND_ARCHIVE_SESSION_STATUSES:
                    raise
                await self._archive_unreplayable_tool_round(
                    session=session,
                    pending_round=pending_tool_round,
                )
                session = await self._finalize_interrupting_for_recovery(
                    recovery_claim_id=claim_id,
                    preserve_interaction_id=preserve_interaction_id,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    events=events,
                    execution_profile=(
                        None
                        if execution_profile_snapshot is None
                        else execution_profile_snapshot.profile
                    ),
                    invocation_context=invocation_context,
                )
                actions.append(IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED)
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=session.status,
                    actions=tuple(actions),
                    events=tuple(events),
                    message=(
                        "Archived an abandoned tool round whose opaque provider state "
                        "could not be replayed safely."
                    ),
                )
            actions.append(IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND)
            session = await self._recovery_ownership.require_session(session.id)
            checkpoint = await self._session_store.load_checkpoint(session.id)

        pending_approval = pending_approval_reader.pending_approval_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        if pending_approval is not None:
            # Approval recovery deliberately skips ordinary/native round
            # recovery, but a consumed external call still needs the same
            # durable uncertainty transition after worker loss. Keep the
            # human gate and selected terminals intact; never dispatch here.
            events.extend(
                await settle_prepared_tool_effects(
                    store=self._session_store,
                    writer=self._event_writer,
                    session=session,
                    pending=pending_approval,
                    profile=None
                    if execution_profile_snapshot is None
                    else execution_profile_snapshot.profile,
                )
            )
            unresolved_effect = await ToolEffectStateOwner(self._session_store).preserve_unresolved(
                session,
                tool_round_id=pending_approval.tool_round_id,
                tool_call_ids=tuple(call.tool_call_id for call in pending_approval.tool_calls),
            )
            pending_child_ids: tuple[str, ...] = ()
            if unresolved_effect:
                events.extend(
                    await self._pending_tool_round_recovery.deliver_pending_tool_effect_uncertainty(
                        session
                    )
                )
                actions.append(IncompleteSessionRecoveryAction.PENDING_TOOL_EFFECT)
                pending_child_ids = await self._pending_foreground_children_at_human_gate(
                    session=session,
                    checkpoint=checkpoint,
                    pending=pending_approval,
                    registered_agent=registered_agent,
                )
                if pending_child_ids:
                    actions.append(IncompleteSessionRecoveryAction.PENDING_SUBAGENT)
            if session.status == SessionStatus.FAILED:
                interrupt_payload = {
                    "model_step_id": pending_approval.model_step_id,
                    "model_attempt_id": pending_approval.model_attempt_id,
                    "tool_round_id": pending_approval.tool_round_id,
                    "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                    **approval_support.bounded_pending_approval_event_payload(
                        pending_approval,
                        redactor=self._secret_redactor,
                    ),
                    "recovered": True,
                    "reason": reason,
                    "metadata": metadata,
                    "interruption_request_id": str(uuid4()),
                }
                session = await self._session_store.transition_status_and_checkpoint(
                    session.id,
                    from_statuses={SessionStatus.FAILED},
                    to_status=SessionStatus.INTERRUPTING,
                    checkpoint_transform=_checkpoint_with_pending_session_interrupt(
                        interrupt_payload, cascade_created_at=self._clock()
                    ),
                )
            session = await self._finalize_interrupting_for_recovery(
                recovery_claim_id=claim_id,
                preserve_interaction_id=preserve_interaction_id,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                events=events,
                execution_profile=(
                    None
                    if execution_profile_snapshot is None
                    else execution_profile_snapshot.profile
                ),
                invocation_context=invocation_context,
            )
            actions.append(IncompleteSessionRecoveryAction.PENDING_APPROVAL)
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=tuple(actions),
                events=tuple(events),
                pending_approval_id=pending_approval.approval_id,
                pending_subagent_session_ids=pending_child_ids,
                message=(
                    "Foreground children require recovery; the pending approval remains intact."
                    if pending_child_ids
                    else "External effect remains unresolved; explicit receipt reconciliation is required."
                    if unresolved_effect
                    else "Session has a pending tool approval; resolve it with ToolApprovalRequest."
                ),
            )

        pending_user_input, _resolution_intent = user_input_lifecycle_authority_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            current_run_epoch=session.run_epoch,
            runtime_session=session,
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
                    "Pending user-input recovery authority changed before finalization."
                )
            # Like approval-owned rounds, a user-input pause bypasses ordinary
            # round recovery. Classify dispatched sibling effects without
            # consuming the answer or authorizing another tool invocation.
            events.extend(
                await settle_prepared_tool_effects(
                    store=self._session_store,
                    writer=self._event_writer,
                    session=session,
                    pending=pending_user_input,
                    profile=None
                    if execution_profile_snapshot is None
                    else execution_profile_snapshot.profile,
                )
            )
            unresolved_effect = await ToolEffectStateOwner(self._session_store).preserve_unresolved(
                session,
                tool_round_id=pending_user_input.tool_round_id,
                tool_call_ids=tuple(call.tool_call_id for call in pending_user_input.tool_calls),
            )
            pending_child_ids = ()
            if unresolved_effect:
                events.extend(
                    await self._pending_tool_round_recovery.deliver_pending_tool_effect_uncertainty(
                        session
                    )
                )
                actions.append(IncompleteSessionRecoveryAction.PENDING_TOOL_EFFECT)
                pending_child_ids = await self._pending_foreground_children_at_human_gate(
                    session=session,
                    checkpoint=checkpoint,
                    pending=pending_user_input,
                    registered_agent=registered_agent,
                )
                if pending_child_ids:
                    actions.append(IncompleteSessionRecoveryAction.PENDING_SUBAGENT)
            session = await self._finalize_interrupting_for_recovery(
                recovery_claim_id=claim_id,
                preserve_interaction_id=preserve_interaction_id,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                events=events,
                execution_profile=(
                    None
                    if execution_profile_snapshot is None
                    else execution_profile_snapshot.profile
                ),
                invocation_context=invocation_context,
            )
            actions.append(IncompleteSessionRecoveryAction.PENDING_USER_INPUT)
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=tuple(actions),
                events=tuple(events),
                pending_user_input_id=pending_user_input.input_id,
                pending_subagent_session_ids=pending_child_ids,
                message=(
                    "Foreground children require recovery; the pending user input remains intact."
                    if pending_child_ids
                    else "External effect remains unresolved; explicit receipt reconciliation is required."
                    if unresolved_effect
                    else "Session is awaiting user input; answer it with UserInputResponse."
                ),
            )

        if session.status == SessionStatus.INTERRUPTING:
            session = await self._finalize_interrupting_for_recovery(
                recovery_claim_id=claim_id,
                preserve_interaction_id=preserve_interaction_id,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                events=events,
                execution_profile=(
                    None
                    if execution_profile_snapshot is None
                    else execution_profile_snapshot.profile
                ),
                invocation_context=invocation_context,
            )
            actions.append(
                IncompleteSessionRecoveryAction.FINALIZED_INTERRUPT
                if previous_status == SessionStatus.INTERRUPTING
                else IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED
            )
        elif not actions:
            actions.append(IncompleteSessionRecoveryAction.SKIPPED_TERMINAL)

        message = "Recovered incomplete session."
        if actions == [IncompleteSessionRecoveryAction.SKIPPED_TERMINAL]:
            message = "Session is terminal; recovery skipped."
        return IncompleteSessionRecoveryResult(
            session_id=session.id,
            previous_status=previous_status,
            status=session.status,
            actions=tuple(actions),
            events=tuple(events),
            message=message,
        )

    async def _archive_unreplayable_tool_round(
        self,
        *,
        session: Session,
        pending_round: pending_rounds.PendingToolRound,
    ) -> None:
        """Retain quarantined evidence while removing it from resumable work."""

        expected_identity = pending_rounds.pending_tool_round_identity(pending_round)
        if session.status not in _UNREPLAYABLE_TOOL_ROUND_ARCHIVE_SESSION_STATUSES:
            raise RuntimeError(
                "An unreplayable tool round can only be abandoned from a recoverable "
                "terminal or interrupting session."
            )

        def archive(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            if (
                current_session.status is not session.status
                or current_session.run_epoch != session.run_epoch
            ):
                raise RuntimeError(
                    "Session changed before its unreplayable tool round was abandoned."
                )
            current = pending_round_reader.pending_tool_round_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
                runtime_session=current_session,
            )
            if (
                current is None
                or pending_rounds.pending_tool_round_identity(current) != expected_identity
            ):
                raise RuntimeError(
                    "Pending tool round changed before its unreplayable state was abandoned."
                )
            copied = copy_durable_record(checkpoint, "checkpoint")
            durable_round = copied.pop(pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY)
            pointer = model_completion_publication.model_step_publication_from_checkpoint(copied)
            if pointer is not None:
                if (
                    not pointer.assistant_message_deferred
                    or pointer.logical_step_id != expected_identity.model_step_id
                    or pointer.tool_round_id != expected_identity.tool_round_id
                ):
                    raise RuntimeError(
                        "Unreplayable tool round conflicts with its durable model-step pointer."
                    )
                copied.pop(model_completion_publication.LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY)
            return _retain_abandoned_unreplayable_tool_round(copied, durable_round)

        await self._session_store.transform_checkpoint(session.id, archive)

    async def _finalize_interrupting_for_recovery(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        events: list[Event],
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
        preserve_interaction_id: str | None = None,
        recovery_claim_id: str | None = None,
    ) -> Session:
        if session.status == SessionStatus.INTERRUPTING:
            async for event in self._session_finalization.interrupt_recovery(
                RecoveryInterruptionRequest(
                    recovery_claim_id=recovery_claim_id,
                    preserve_interaction_id=preserve_interaction_id,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                )
            ):
                events.append(event)
            session = await self._recovery_ownership.require_session(session.id)
        return session

    async def _pending_durable_subagent_children(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        pending_round: pending_rounds.PendingToolRound,
        registered_agent: runtime_records.RegisteredAgentState,
    ) -> tuple[Session, ...]:
        """Reconcile durable submissions without closing live child work as unknown."""

        lifecycle_events = await self._pending_tool_round_recovery.load_tool_round_lifecycle_events(
            session_id=session.id,
            pending_round=pending_round,
        )
        recorded_outcomes, _started_ids = tool_round_recovery.recorded_tool_outcomes(
            events=lifecycle_events,
            pending_round=pending_round,
        )
        children = await self._pending_tool_round_recovery.subagent_children_by_idempotency_key(
            session.id
        )
        pending: list[Session] = []
        for call in pending_round.tool_calls:
            if recorded_outcomes.get(call.tool_call_id) is not None:
                continue
            idempotency_key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_round_id=pending_round.tool_round_id,
                tool_call_id=call.tool_call_id,
            )
            recovery_arguments = (
                await self._pending_tool_round_recovery.subagent_recovery_arguments(
                    checkpoint=checkpoint,
                    parent_session=session,
                    tool_name=call.tool_name,
                    tool_round_id=pending_round.tool_round_id,
                    tool_call_id=call.tool_call_id,
                    idempotency_key=idempotency_key,
                    fallback=call.arguments,
                )
            )
            reconciled_result = await self._pending_tool_round_recovery.reconcile_subagent_child(
                children,
                idempotency_key=idempotency_key,
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                tool_round_id=pending_round.tool_round_id,
                arguments=recovery_arguments,
                parent_session=session,
                registered_agent=registered_agent,
            )
            if reconciled_result is not None:
                # The child recovery owner has confirmed the exact handoff.
                # Its queued child need not finish before the parent can settle;
                # normal round recovery below revalidates and publishes the result.
                continue
            child = children.get(idempotency_key)
            if child is None:
                continue
            subagent = child.metadata.get("subagent")
            if (
                isinstance(subagent, dict)
                and subagent.get("mode") == "durable"
                and child.status not in _RECOVERY_RESUMABLE_SESSION_STATUSES
            ):
                pending.append(child.model_copy(deep=True))
        return tuple(pending)

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
                redactor=self._tool_round_executor.redactor_for_tool_calls(
                    registered_agent=registered_agent, tool_calls=[tool_call]
                ),
            ):
                emitted.append(published)
        return emitted

    async def _pending_foreground_children_at_human_gate(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        pending: PendingToolApproval | PendingUserInput,
        registered_agent: runtime_records.RegisteredAgentState,
    ) -> tuple[str, ...]:
        """Read exact child obligations without repairing submissions or consuming a gate."""
        children = await self._pending_tool_round_recovery.subagent_children_by_idempotency_key(
            session.id
        )
        selected: list[str] = []
        owner = ToolEffectStateOwner(self._session_store)
        for call in pending.tool_calls:
            effect = await owner.resolve_call(
                session, tool_round_id=pending.tool_round_id, tool_call_id=call.tool_call_id
            )
            if effect is None or effect.state != "outcome_unknown":
                continue
            key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_round_id=pending.tool_round_id,
                tool_call_id=call.tool_call_id,
                approval_id=effect.intent.approval_id,
                pause_id=effect.intent.pause_id,
            )
            if effect.intent.idempotency_key != key or effect.intent.tool_name != call.tool_name:
                raise RuntimeError(
                    "Foreground child effect conflicts with its gated call identity."
                )
            child = children.get(key)
            if child is None:
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
            if not _matches_recoverable_subagent_child(
                child,
                idempotency_key=key,
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                arguments=arguments,
                parent_session=session,
                registered_agent=registered_agent,
            ):
                continue
            metadata = child.metadata.get("subagent")
            if (
                type(metadata) is dict
                and metadata.get("mode") == "foreground"
                and child.status
                in {SessionStatus.PENDING, SessionStatus.RUNNING, SessionStatus.INTERRUPTING}
            ):
                selected.append(child.id)
        return tuple(selected)


def _checkpoint_without_active_incomplete_recovery_claim(
    checkpoint: dict[str, Any] | None,
    *,
    now: datetime,
) -> dict[str, Any] | None:
    """Reject live recovery ownership and remove an expired internal marker."""
    _require_aware_datetime(now, "now")
    if checkpoint is None:
        return None
    updated = copy_durable_record(checkpoint, "checkpoint")
    existing = _incomplete_recovery_claim_from_checkpoint(updated)
    if existing is None:
        return updated
    if existing[1] > now:
        raise RuntimeError("Session has an active incomplete-session recovery operation.")
    updated.pop(_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY, None)
    return updated


def _incomplete_recovery_request_fingerprint(
    request: IncompleteSessionsRecoveryRequest,
) -> str:
    material = {
        "statuses": sorted(status.value for status in request.statuses),
        "inactive_for_seconds": request.inactive_for_seconds,
        "reason": request.reason,
        "metadata": request.metadata,
    }
    return sha256(
        canonical_durable_json_bytes(material, "incomplete sessions recovery cursor")
    ).hexdigest()


def _encode_incomplete_recovery_cursor(
    *,
    status: SessionStatus,
    session_cursor: str | None,
    request: IncompleteSessionsRecoveryRequest,
) -> str:
    if status not in request.statuses:
        raise ValueError("Recovery cursor status is not part of the request.")
    encoded_session_cursor: str | None = None
    if session_cursor is not None:
        session_cursor = require_clean_nonblank(session_cursor, "session cursor")
        session_cursor_bytes = session_cursor.encode("utf-8")
        if len(session_cursor_bytes) > MAX_SESSION_LIST_CURSOR_BYTES:
            raise ValueError(
                "Session-store recovery cursor exceeds its "
                f"{MAX_SESSION_LIST_CURSOR_BYTES}-byte contract."
            )
        encoded_session_cursor = base64.urlsafe_b64encode(session_cursor_bytes).decode("ascii")
    material = {
        "version": _INCOMPLETE_RECOVERY_CURSOR_VERSION,
        "status": status.value,
        "session_cursor_b64": encoded_session_cursor,
        "request_fingerprint": _incomplete_recovery_request_fingerprint(request),
    }
    encoded = base64.urlsafe_b64encode(
        canonical_durable_json_bytes(material, "incomplete sessions recovery cursor")
    ).decode("ascii")
    if len(encoded.encode("ascii")) > MAX_INCOMPLETE_SESSIONS_RECOVERY_CURSOR_BYTES:
        raise RuntimeError("Incomplete-session recovery cursor exceeds its byte limit.")
    return encoded


def _decode_incomplete_recovery_cursor(
    cursor: str,
    *,
    request: IncompleteSessionsRecoveryRequest,
) -> tuple[SessionStatus, str | None]:
    try:
        if len(cursor.encode("utf-8")) > MAX_INCOMPLETE_SESSIONS_RECOVERY_CURSOR_BYTES:
            raise ValueError("Incomplete-session recovery cursor exceeds its byte limit.")
        encoded_cursor = cursor.encode("ascii")
        raw = base64.b64decode(
            encoded_cursor,
            altchars=b"-_",
            validate=True,
        )
        if base64.urlsafe_b64encode(raw) != encoded_cursor:
            raise ValueError("Non-canonical incomplete-session recovery cursor.")
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeError, binascii.Error, ValueError) as exc:
        raise ValueError("Invalid incomplete-session recovery cursor.") from exc
    expected_keys = {
        "version",
        "status",
        "session_cursor_b64",
        "request_fingerprint",
    }
    if type(decoded) is not dict or set(decoded) != expected_keys:
        raise ValueError("Invalid incomplete-session recovery cursor.")
    if (
        type(decoded["version"]) is not int
        or decoded["version"] != _INCOMPLETE_RECOVERY_CURSOR_VERSION
    ):
        raise ValueError("Unsupported incomplete-session recovery cursor version.")
    try:
        status = SessionStatus(decoded["status"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid incomplete-session recovery cursor status.") from exc
    if status not in request.statuses:
        raise ValueError("Incomplete-session recovery cursor does not match the request.")
    if type(decoded["request_fingerprint"]) is not str or decoded[
        "request_fingerprint"
    ] != _incomplete_recovery_request_fingerprint(request):
        raise ValueError("Incomplete-session recovery cursor does not match the request.")
    encoded_session_cursor = decoded["session_cursor_b64"]
    session_cursor: str | None = None
    if encoded_session_cursor is not None:
        if type(encoded_session_cursor) is not str:
            raise ValueError("Invalid incomplete-session recovery cursor.")
        try:
            encoded_session_cursor_bytes = encoded_session_cursor.encode("ascii")
            session_cursor_bytes = base64.b64decode(
                encoded_session_cursor_bytes,
                altchars=b"-_",
                validate=True,
            )
            if base64.urlsafe_b64encode(session_cursor_bytes) != encoded_session_cursor_bytes:
                raise ValueError("Non-canonical session-store recovery cursor.")
            if len(session_cursor_bytes) > MAX_SESSION_LIST_CURSOR_BYTES:
                raise ValueError("Session-store recovery cursor exceeds its byte limit.")
            session_cursor = require_clean_nonblank(
                session_cursor_bytes.decode("utf-8"),
                "session cursor",
            )
        except (UnicodeError, binascii.Error, ValueError) as exc:
            raise ValueError("Invalid incomplete-session recovery cursor.") from exc
    return status, session_cursor


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


def _require_recovery_max_steps(value: int | None) -> int:
    if value is None:
        raise ValueError("Recovery requires recorded invocation max_steps.")
    return value
