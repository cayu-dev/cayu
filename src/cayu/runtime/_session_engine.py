"""Deep owner for session orchestration behind :class:`CayuApp`."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import sys
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, TypeVar
from uuid import NAMESPACE_URL, uuid4, uuid5

from cayu.resource_access import ResourceAccessPolicy, ResourceSessionStream, restore_session_stream
from cayu.runtime import _execution_profile_continuation as execution_profile_continuation
from cayu.runtime import (
    _prompt_transition,
    _session_checkpoint_admission,
    _session_execution_profile,
    _session_operation_state,
    _transcript,
)
from cayu.runtime._assistant_model_publication import AssistantModelPublication
from cayu.runtime._compaction.explicit import SessionCompaction
from cayu.runtime._durable_tool_round import DeferredInteractionInput, DurableToolRound
from cayu.runtime._durable_tool_round import _environment_name as _environment_name
from cayu.runtime._durable_tool_round import (
    _limit_reached_tool_call_event as _limit_reached_tool_call_event,
)
from cayu.runtime._durable_tool_round import (
    _limit_reached_tool_round_results as _limit_reached_tool_round_results,
)
from cayu.runtime._durable_tool_round import _limit_value_for_payload as _limit_value_for_payload
from cayu.runtime._execution_profile_classification import ExecutionProfileClassifier
from cayu.runtime._execution_profile_continuation import (
    ExecutionProfileContinuation,
    _execution_profile_decision_event,
    _execution_profile_decision_event_id,
)
from cayu.runtime._foreground_gate_continuation import ForegroundGatePolicyOwner
from cayu.runtime._interaction_events import (
    build_interaction_started_event,
    build_interaction_started_event_from_identity,
)
from cayu.runtime._model_completion_contracts import (
    ModelCompletionManualRecoveryRequired,
)
from cayu.runtime._model_completion_recovery import ModelCompletionRecovery
from cayu.runtime._model_policy import PolicySelection
from cayu.runtime._pending_tool_round_recovery import PendingToolRoundRecovery
from cayu.runtime._policy_wire import decode as decode_policy_evidence
from cayu.runtime._recovery_ownership import (
    RecoveryOwnership,
)
from cayu.runtime._session_finalization import (
    _INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE,
    _INTERACTION_TRANSITION_RUN_FENCE_ATTRIBUTE,
    _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
    SessionFinalization,
    _checkpoint_with_pending_session_interrupt,
    _close_async_iterator,
    _collect_through_event_type,
    _is_interaction_transition_run_fence,
    _pop_interaction_transition_cancellation_outcome,
    _raise_primary_with_secondary_failure,
    _run_interaction_transition_cancellation_cleanup_steps,
    _runtime_interruption_event,
    _task_event,
)
from cayu.runtime._session_outcome_publication import (
    SessionOutcomeContext,
    SessionOutcomePublication,
    SessionOutcomeState,
)
from cayu.runtime._session_preparation import (
    SessionPreparation,
    _require_native_structured_output_support,
)
from cayu.runtime._session_task_lifecycle import SessionTaskLifecycle
from cayu.runtime._session_tool_discovery import (
    _require_tool_discovery_view,
    _tool_discovery_operation_initializer,
)
from cayu.runtime._task_execution_admission import (
    require_ordinary_session_execution,
    verifier_aware_task_execution_outcome,
)
from cayu.runtime._terminal_event_publication import (
    TerminalEventPublication,
)
from cayu.runtime._terminal_evidence_finalization import TerminalEvidenceFinalization
from cayu.runtime._terminal_evidence_reader import TerminalEvidenceReader
from cayu.runtime._user_input_recovery_evidence import UserInputRecoveryEvidence
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader

if TYPE_CHECKING:
    from cayu.runtime._producer_completion_replay import _ProducerCompletionReplay
    from cayu.runtime._session_continuation_resume import _ResumeAdmissionHandoff


from cayu._exception_groups import (
    _failure_without_existing_exception_identities,
    exception_cause,
    exception_group_children,
    iter_exception_tree,
    set_exception_cause,
)
from cayu._exception_state import exception_state, pop_exception_state, set_exception_state
from cayu._task_wait import (
    CapturedAwaitableOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    consume_pending_task_cancellation,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_record,
    copy_json_value,
    require_clean_nonblank,
)
from cayu.approvals.tools import (
    PendingToolApproval,
    ResolutionActor,
    ResolutionActorSource,
    resolution_actor_payload,
)
from cayu.approvals.user_input import (
    AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY,
    USER_INPUT_SUPERSESSION_INTENT_KEY,
    AmbiguousPendingUserInput,
    PendingUserInput,
    ambiguous_pending_user_input_from_checkpoint,
    user_input_lifecycle_authority_from_checkpoint,
    user_input_supersession_intent_for,
)
from cayu.budgets._run_limit_accounting import (
    RunLimitAccountingContext,
    has_run_limit_accounting_authority,
    run_budget_authorities_from_context,
)
from cayu.budgets.base import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservationRecoveryContext,
    budget_limits_for_session,
    budget_reservation_authority_sha256,
    copy_budget_policy,
    request_budget_limits_for_session,
)
from cayu.budgets.billing import (
    BillingIdentity,
)
from cayu.budgets.run_limits import RunLimits, copy_run_limits, has_run_limits
from cayu.budgets.usage import (
    SessionUsageSummary,
)
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.peer_content import PeerContentAppendRequest, PeerContentReceipt
from cayu.configuration import RunDefaults
from cayu.context.base import (
    _runtime_authored_user_message_checkpoint_transform,
    validate_context_messages,
)
from cayu.context.footprints import (
    PromptContributionManifest,
    RequestFootprintConfig,
    RequestVariant,
    TargetedToolGrantFootprint,
    build_prompt_contribution_manifest,
    copy_request_footprint_config,
    targeted_tool_grant_footprint,
)
from cayu.context.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    NativeStructuredOutputUnsupported,
    StructuredOutputSpec,
    StructuredOutputStrategy,
    copy_structured_output_spec,
    require_secret_free_structured_output_spec,
    structured_output_repair_prompt,
    structured_output_tool_required_validation,
    validate_structured_output_text,
)
from cayu.context.thinking import ThinkingConfig
from cayu.deadlines import (
    ExecutionDeadlineExceeded,
    current_execution_deadline,
    deadline_stream,
    effective_deadline,
    expired_execution_deadline,
)
from cayu.egress.authority import EgressAuthorityChangeKind, EgressAuthorityTransitionState
from cayu.egress.runtime import (
    reactivate_parked_egress_environment,
    require_active_egress_transition_environment,
)
from cayu.egress.transitions import (
    EgressAuthorityAdoptionHandler,
    EgressAuthorityAdoptionResult,
    EgressAuthorityTransitionConflict,
    EgressAuthorityTransitionCoordinator,
    EgressAuthorityTransitionRecord,
    SessionCheckpointEgressAuthorityTransitionStore,
    _claim_parked_egress_authority_allocation,
    _find_parked_egress_authority_allocation,
    _has_runtime_egress_authority_adoption_result,
    _require_parked_egress_authority_allocation,
    require_egress_authority_transition_compatible_with_profile,
    require_exact_egress_authority_transition,
)
from cayu.environments.factory import EnvironmentFactoryOperation, EnvironmentFactoryResult
from cayu.events import (
    Event,
    EventType,
    copy_event,
    event_payload_authority_is_runtime_generated,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.exceptions import (
    TerminalEventPublicationUncertain,
    _is_runtime_interaction_lifecycle_publication_rejection,
)
from cayu.execution_profiles import (
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileComponentClass,
    ExecutionProfileDecision,
    ExecutionProfileDecisionKind,
    ExecutionProfileIdentity,
    changed_execution_profile_components,
    event_with_execution_profile_authority,
    execution_profile_with_component,
)
from cayu.execution_units import (
    ModelAttemptIdentity,
    ModelStepIdentity,
    new_model_step_identity,
)
from cayu.messages import (
    Message,
    MessageRole,
    detach_message,
)
from cayu.observability.hooks import (
    RuntimeHookPhase,
)
from cayu.providers._credential_boundary import (
    detach_credential_safe_provider_cancellation,
    provider_cancellation_admission_deadline,
    provider_cancellation_failures,
)
from cayu.providers.base import (
    NativeStructuredOutputSchemaInvalid,
)
from cayu.providers.hosted import HostedToolCapabilityError
from cayu.providers.operations import ProviderOperationMode
from cayu.providers.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _execution_profile_admission as execution_profile_admission
from cayu.runtime import _model_target as model_target
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _session_request_boundary as session_request_boundary
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._binding_cleanup import binding_finalize_explicit_cancellation
from cayu.runtime._delegated_event_stream import _close_delegated_event_stream
from cayu.runtime._diagnostics import (
    exception_diagnostic,
    task_update_error_payload,
)
from cayu.runtime._durable_subagents import (
    DURABLE_SUBAGENT_SUBMISSION_SEEDS_CHECKPOINT_KEY,
    DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY,
    durable_subagent_authority_rejected,
    durable_subagent_request_sha256,
    durable_subagent_submission_from_checkpoint,
    durable_subagent_worker_incompatible,
    is_durable_subagent_submission_unsettled,
)
from cayu.runtime._environment_lifecycle import (
    EnvironmentLifecycle,
    exception_failure_payload,
    render_initial_system_prompt_with_contributions,
)
from cayu.runtime._event_writer import (
    RuntimeEventWriter,
)
from cayu.runtime._execution_to_wait import _ExecutionToWait
from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait
from cayu.runtime._foreground_child_wait import (
    ForegroundChildActionRequired,
)
from cayu.runtime._foreground_subagent_recovery import ForegroundSubagentRecoveryRequired
from cayu.runtime._interruption_coordinator import (
    _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY,
    _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY,
    BackgroundInterruptionCoordinator,
    _copy_interruption_cascade_retry_request,
    _interruption_cascade_marker_datetime,
    _interruption_cascade_retry_event_payload,
    interruption_cascade_lease_seconds,
    interruption_cascade_suppressed,
)
from cayu.runtime._invocation_lifecycle import (
    InvocationContext,
    _authenticated_invocation_context,
    invocation_checkpoint_state_sha256,
)
from cayu.runtime._loop_policy_continuations import (
    before_stop_continuation_checkpoint_transform,
    before_stop_continuation_indices,
    before_stop_policy_key,
)
from cayu.runtime._message_redaction import redact_runtime_message_for_boundary
from cayu.runtime._model_completion_contracts import (
    ModelCompletionPublicationRequest,
    ModelCompletionPublicationResult,
    ModelCompletionRecoveryContext,
    model_completion_recovery_context_from_stage,
)
from cayu.runtime._model_errors import (
    _FallbackBillingCancellationStateCheckFailed,
    billing_identity_cancellation_failures,
    detach_billing_identity_cancellation,
    detach_billing_identity_cancellation_group,
)
from cayu.runtime._model_execution_selection import ModelExecutionSelection
from cayu.runtime._model_step_executor import (
    ModelStepExecutor,
    ModelStepFlowOutcome,
    _model_request_messages,
    _model_request_tools,
    _require_frozen_tool_exposure,
    _tool_capability_ceiling_exposure,
    reconstruct_assistant_step_result,
)
from cayu.runtime._producer_execution import _ProducerExecution
from cayu.runtime._recovery_claims import (
    _IncompleteRecoveryClaim,
)
from cayu.runtime._recovery_requests import (
    RecoveryAbandonedSessionRequest,
    RecoverySessionRunRequest,
)
from cayu.runtime._run_limit_accounting import (
    capture_run_limit_accounting_context,
    restore_run_limit_accounting_context,
)
from cayu.runtime._run_limits import (
    BudgetStepReservation,
    LimitEvaluation,
    RunLimitController,
    RunLimitGate,
    SessionUsageTracker,
)
from cayu.runtime._session_control import (
    INTERRUPT_REQUESTED_SESSION_STATUSES,
    ActiveSessionRun,
    SessionControl,
    SessionInterruptedByRequest,
    clear_current_task_cancellation,
)
from cayu.runtime._structured_output_tool_round import (
    _has_structured_output_tool_call,
    _redact_structured_output_validation,
    _structured_output_event,
    _structured_output_validating_event,
)
from cayu.runtime._task_store_operation_boundary import (
    raise_task_store_operation_failure,
)
from cayu.runtime._tool_completion import (
    _CheckpointState,
    load_recorded_tool_completion_policy,
    recorded_tool_completion_result,
)
from cayu.runtime._tool_effect_state import (
    ToolEffectReconciliationCleanupFailure,
    ToolEffectReconciliationRequired,
)
from cayu.runtime._tool_invocation.admission import ToolApprovalRequired
from cayu.runtime._tool_round_executor import (
    InterruptedToolRoundRequest,
    ToolRoundExecutor,
    UserInputRequired,
)
from cayu.runtime._work_attempt_session_mutation import (
    reconcile_work_attempt_initial_publication,
    record_work_attempt_execution_stop,
)
from cayu.runtime._workflow_structured_output_handoff import (
    WorkflowStructuredOutputHandoff,
)
from cayu.runtime.build_provenance import current_runtime_build_provenance
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import (
    ExecutionProfileAdoptionRejected,
    ExecutionProfileMigrationRequired,
    ExecutionProfileMismatchError,
    _ExecutionProfileAdmissionRequestRejected,
)
from cayu.runtime.loop_policies import (
    BeforeStopAction,
    BeforeStopContext,
    BeforeStopDecision,
    LoopPolicy,
    copy_before_stop_decision,
    validate_loop_policies,
)
from cayu.runtime.model_steps import (
    AssistantStepResult,
    StepClassification,
    classify_assistant_step,
    completion_requests_follow_up,
)
from cayu.runtime.provider_operations import (
    _PROVIDER_OPERATION_FALLBACK_ORDINALS_CHECKPOINT_KEY,
    _PROVIDER_OPERATION_PENDING_DISPOSITION_CHECKPOINT_KEY,
)
from cayu.runtime.stop_policy import StopDecision, StopLimit
from cayu.runtime.tool_completion import (
    ToolCompletionPolicy,
    ToolCompletionResult,
)
from cayu.sessions._checkpoint_mutations import (
    SessionRuntimePublicationConflict,
    runtime_publication_checkpoint_mutation,
)
from cayu.sessions._checkpoint_preservation import (
    _invocation_lifecycle_authority_mutation_scope,
    _invocation_lifecycle_authority_read_scope,
)
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
    execution_profile_from_session_metadata,
    model_failover_progress_for_session,
)
from cayu.sessions._execution_profile_rules import (
    SessionModelTransition,
    execution_profile_adoption_request_fingerprint,
    session_model_projection_cursor,
)
from cayu.sessions._foreground_child_checkpoint import (
    FOREGROUND_CHILD_TERMINAL_KEY,
    FOREGROUND_CHILD_WAIT_KEY,
    FOREGROUND_PARENT_CONTINUATION_KEY,
    ForegroundChildResumeRequest,
    ForegroundChildTerminal,
    ForegroundChildWait,
    ForegroundParentContinuation,
    foreground_child_state_from_checkpoint,
)
from cayu.sessions._interaction_transitions import (
    InteractionTransitionResult,
    InteractionTransitionSpec,
    QueuedInteractionProfileHandoff,
)
from cayu.sessions._invocation_lifecycle import (
    AdmitInvocationCommand,
    AdmittedInvocationBinding,
    CreateInvocationCommand,
    InvocationCheckpointPatch,
    InvocationMutationResult,
    PreparedInvocationBinding,
    RejectInvocationCommand,
    SettleInvocationCommand,
    invocation_lifecycle_receipt_history_present,
)
from cayu.sessions._invocation_scope import (
    _activate_session_interaction,
    _clear_session_interaction_settlement,
    _current_session_interaction_id,
    _current_session_invocation_terminal_event,
    _deactivate_session_interaction,
    _deactivate_session_run_fence,
    _latest_session_invocation_interaction_is_settled,
    _mark_session_invocation_terminal_event,
)
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalDecision,
    InvocationTerminalOutcome,
    build_invocation_terminal_decision,
    invocation_terminal_decision_from_checkpoint,
    invocation_terminal_event_id,
    settled_invocation_terminal_decision_from_checkpoint,
)
from cayu.sessions._model_completion_publication import (
    ModelCompletionStageDisposition,
    ModelCompletionStageSettlementRequest,
)
from cayu.sessions._terminal_evidence import (
    _SESSION_RUN_OPERATION_CHECKPOINT_KEY,
    interruption_request_id_from_payload,
    require_interruption_event_matches_pending_marker,
)
from cayu.sessions.base import (
    _QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY,
    INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY,
    ModelCompletionManualRecoveryRequest,
    ModelCompletionManualRecoveryResult,
    ModelCompletionStageSettlement,
    ModelFailoverPolicy,
    ModelTarget,
    SessionModelCompletionDispatchAlreadyAuthorized,
    SessionModelCompletionStageConflict,
    SessionRunFenced,
    SessionStore,
    _checkpoint_with_session_run_operation,
    _initial_transcript_pending_interaction_id,
    copy_model_completion_manual_recovery_request,
    model_completion_stage_settlement_request,
)
from cayu.sessions.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    CHECKPOINT_SCHEMA_VERSION_KEY,
    COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
)
from cayu.sessions.cleanup import RecoveryCleanupSupervisor
from cayu.sessions.creation_claims import bind_runtime_session_create_claim
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.forks import (
    session_fork_profile_relationship,
)
from cayu.sessions.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
    INTERACTION_TERMINAL_EVENT_TYPES,
)
from cayu.sessions.invocation import SessionExecutionSource
from cayu.sessions.messaging import (
    EnqueueSessionMessageRequest,
    EnqueueSessionMessageResult,
    SessionMessageDeliveryBatch,
    SessionQueuedMessagesPending,
    queued_session_message_input,
)
from cayu.sessions.records import (
    Session,
    SessionRuntimeIdentity,
    SessionStatus,
    SessionStatusConflict,
    _queued_dispatch_session_instance_fingerprint,
    copy_session_runtime_identity,
)
from cayu.sessions.recovery import (
    _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY,
    _incomplete_recovery_claim_from_checkpoint,
)
from cayu.sessions.requests import (
    CompactSessionRequest,
    InterruptSessionRequest,
    ResumeRequest,
    RunRequest,
    _authenticated_session_instance_id_for_run_request,
    _fork_initial_invocation_request_sha256,
    _runtime_resume_transport_metadata,
    run_request_with_runtime_session_instance_authority,
    runtime_prepared_session_authority,
    session_input_contract_evidence,
)
from cayu.sessions.transcript_input import (
    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
    session_input_messages_sha256,
)
from cayu.tasks.admission import (
    WorkAttemptExecutionClaimLost,
    WorkAttemptRecoveryRequired,
)
from cayu.tasks.contracts import TaskCompletionDecisionRequired
from cayu.tasks.dispatch import DispatchRequest
from cayu.tasks.records import TaskClaimLost
from cayu.tasks.store import TaskStore
from cayu.tools.catalogue import CALL_TOOL_NAME
from cayu.tools.discovery import (
    tool_discovery_view_initialization,
)
from cayu.tools.exposure import (
    ResolvedToolExposure,
    ToolCapabilityCeiling,
    ToolExposure,
    resolve_tool_capability_ceiling,
    resolved_tool_exposure_from_authority,
    tool_capability_ceiling_from_session_metadata,
)
from cayu.tools.grants import (
    PreparedTargetedToolGrant,
    TargetedToolGrantRecord,
    TargetedToolUseRejectionReason,
    build_targeted_tool_grant_record,
    copy_targeted_tool_grant_record,
    targeted_tool_grant_event,
    targeted_tool_view_generation_id,
)
from cayu.tools.targeted_projection import (
    TargetedToolProjectionKind,
    resolve_targeted_tool_projection,
    targeted_tool_projection_marker_message,
)
from cayu.vaults.redaction import SecretRedactor
from cayu.workspaces.observation_recovery import (
    retain_workspace_observation_pending_cancellation_requests,
    workspace_observation_pending_cancellation_requests,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _QueuedCompletionResult:
    outcome: Literal["continue", "complete", "stopped"]
    session: Session
    events: list[Event]
    invocation_context: InvocationContext | None
    completion_finalization_marker: dict[str, Any] | None = None
    continuation: AsyncGenerator[Event, None] | None = None


_INTERRUPTIBLE_SESSION_STATUSES = {
    SessionStatus.PENDING,
    SessionStatus.RUNNING,
}


_PROVIDER_CANCELLATION_FINALIZATION_FAILURE_STATE = (
    "_cayu_provider_cancellation_finalization_failure"
)

_PROVIDER_CANCELLATION_FINALIZATION_FAILURE_AUTHORITY = object()

_OperationResultT = TypeVar("_OperationResultT")


_INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED = "tool_approval_required"

_INTERRUPTION_TYPE_USER_INPUT_REQUIRED = "user_input_required"


@dataclass(frozen=True, slots=True)
class _ProviderCancellationFinalizationFailure:
    """Authenticated proof that runtime terminal publication did not finish."""

    authority: object


def _mark_provider_cancellation_finalization_failure(
    cancellation: asyncio.CancelledError,
) -> None:
    if not set_exception_state(
        cancellation,
        _PROVIDER_CANCELLATION_FINALIZATION_FAILURE_STATE,
        _ProviderCancellationFinalizationFailure(
            authority=_PROVIDER_CANCELLATION_FINALIZATION_FAILURE_AUTHORITY,
        ),
    ):
        raise RuntimeError("Could not retain provider cancellation finalization failure.")


def _has_provider_cancellation_finalization_failure(
    cancellation: asyncio.CancelledError,
) -> bool:
    handoff = exception_state(
        cancellation,
        _PROVIDER_CANCELLATION_FINALIZATION_FAILURE_STATE,
    )
    return (
        type(handoff) is _ProviderCancellationFinalizationFailure
        and handoff.authority is _PROVIDER_CANCELLATION_FINALIZATION_FAILURE_AUTHORITY
    )


def _copy_authenticated_cancellation_diagnostics_for_public(
    source: asyncio.CancelledError,
    detached: asyncio.CancelledError,
    *,
    finalization_failed: bool,
) -> None:
    """Retain runtime-owned diagnostics after credential-safe detachment."""

    notes = getattr(source, "__notes__", ())
    if type(notes) is list:
        for note in notes:
            if type(note) is str:
                detached.add_note(note)
    if finalization_failed and not set_exception_cause(
        detached,
        RuntimeError("Session interruption finalization failed after provider cancellation."),
    ):
        raise RuntimeError("Could not retain detached provider cancellation diagnostics.")


def _detach_provider_cancellation_for_public(
    cancellation: asyncio.CancelledError,
) -> asyncio.CancelledError | None:
    detached = detach_credential_safe_provider_cancellation(cancellation)
    finalization_failed = _has_provider_cancellation_finalization_failure(cancellation)
    if detached is None:
        if not finalization_failed:
            return None
        detached = asyncio.CancelledError("Provider operation cancelled")
    pending_cancellation_requests = workspace_observation_pending_cancellation_requests(
        cancellation
    )
    if pending_cancellation_requests:
        retain_workspace_observation_pending_cancellation_requests(
            detached,
            pending_cancellation_requests,
        )
    if finalization_failed:
        _mark_provider_cancellation_finalization_failure(detached)
    _copy_authenticated_cancellation_diagnostics_for_public(
        cancellation,
        detached,
        finalization_failed=finalization_failed,
    )
    return detached


def _detach_billing_cancellation_for_public(
    cancellation: asyncio.CancelledError,
) -> asyncio.CancelledError | None:
    detached = detach_billing_identity_cancellation(cancellation)
    if detached is None:
        return None
    pending_cancellation_requests = workspace_observation_pending_cancellation_requests(
        cancellation
    )
    if pending_cancellation_requests:
        retain_workspace_observation_pending_cancellation_requests(
            detached,
            pending_cancellation_requests,
        )
    finalization_failed = _has_provider_cancellation_finalization_failure(cancellation)
    if finalization_failed:
        _mark_provider_cancellation_finalization_failure(detached)
    _copy_authenticated_cancellation_diagnostics_for_public(
        cancellation,
        detached,
        finalization_failed=finalization_failed,
    )
    return detached


def _consume_authenticated_interaction_transition_run_fence(
    error: BaseException,
) -> bool:
    """Consume current transition fence markers without discarding evidence.

    A sibling transition boundary removes the marker from caller cancellation,
    but the exact store failure in its cause remains independently marked.  The
    enclosing run owner must recognize that authenticated child marker before
    deciding whether stale-epoch cleanup is legal.
    """

    authenticated = False
    for candidate in iter_exception_tree(error):
        if not _is_interaction_transition_run_fence(candidate):
            continue
        authenticated = True
        pop_exception_state(
            candidate,
            _INTERACTION_TRANSITION_RUN_FENCE_ATTRIBUTE,
        )
    return authenticated


_FORK_INITIAL_PROFILE_POLICY_ID = "cayu:fork-initial-invocation:v1"


def _fork_initial_profile_actor() -> ResolutionActor:
    return ResolutionActor(
        subject="cayu:fork-initial-invocation",
        source=ResolutionActorSource.SYSTEM,
    )


def _require_exact_fork_initial_invocation_profile(
    session: Session,
    request: ResumeRequest,
    *,
    queued_dispatch_id: str | None,
) -> ExecutionProfileIdentity | None:
    """Authenticate the dispatch-only first invocation frozen with an exact fork."""

    relationship = session_fork_profile_relationship(session)
    if (
        session.run_epoch != 0
        or relationship is None
        or relationship.initial_invocation_request_sha256 is None
    ):
        return None
    if queued_dispatch_id is None:
        raise SessionRunFenced(
            "Session fork first invocation must enter through its exact durable dispatch."
        )
    queued_dispatch_id = require_clean_nonblank(
        queued_dispatch_id,
        "queued_dispatch_id",
    )
    if relationship.initial_dispatch_id != queued_dispatch_id:
        raise SessionRunFenced(
            "Session fork first invocation dispatch identity does not match the exact fork."
        )
    if relationship.initial_invocation_request_sha256 != (
        _fork_initial_invocation_request_sha256(request)
    ):
        raise SessionRunFenced(
            "Session fork first invocation does not match the exact request frozen "
            "before child publication."
        )
    if relationship.initial_invocation_profile is None:  # pragma: no cover - model invariant
        raise RuntimeError("Session fork first invocation lost its execution profile.")
    return relationship.initial_invocation_profile


def _session_trace_event_fields(
    session: Session,
    request_metadata: dict[str, Any],
) -> dict[str, Any]:
    # Surface the parent linkage and any inbound W3C trace context on the session
    # start event so an OpenTelemetryEventSink can parent the root span. Additive:
    # other sinks ignore these keys.
    fields: dict[str, Any] = {}
    if session.parent_session_id:
        fields["parent_session_id"] = session.parent_session_id
    traceparent = request_metadata.get("traceparent")
    if isinstance(traceparent, str) and traceparent:
        fields["traceparent"] = traceparent
        tracestate = request_metadata.get("tracestate")
        if isinstance(tracestate, str) and tracestate:
            fields["tracestate"] = tracestate
    return fields


def _loop_policy_supports_before_stop(policy: LoopPolicy) -> bool:
    policy_method = type(policy).before_stop
    default_method = LoopPolicy.before_stop
    return policy_method is not default_method


@dataclass(frozen=True)
class _BeforeStopPolicySelection:
    decision: BeforeStopDecision
    policy_key: str


def _before_stop_policy_event(
    *,
    event_type: str,
    policy_name: str,
    scope: str,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    step: int,
    classification: StepClassification,
    payload: dict[str, Any],
) -> Event:
    event_payload = {
        "policy": policy_name,
        "scope": scope,
        "step": step,
        "classification": classification.payload(),
        **copy_json_value(payload, "payload"),
    }
    return Event(
        type=event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=_environment_name(registered_environment),
        payload=event_payload,
    )


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


def _replace_checkpoint_preserving_runtime_state(
    checkpoint: dict[str, Any],
):
    replacement = copy_durable_record(checkpoint, "checkpoint")

    def transform(_session: Session, current: dict[str, Any] | None) -> dict[str, Any]:
        updated = copy_durable_record(replacement, "checkpoint")
        updated[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION
        for key, field_name in (
            ("workspace_checkpoints", "workspace_checkpoints"),
            (
                ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
                "active_invocation_execution_profile",
            ),
            (
                INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
                "invocation_lifecycle_receipt",
            ),
            (
                INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
                "invocation_terminal_decision",
            ),
            (
                SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
                "settled_invocation_terminal_decision",
            ),
            (
                INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY,
                "initial_transcript_pending",
            ),
            (_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY, "pending_session_interrupt"),
            (_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY, "pending_interruption_cascade"),
            (_session_operation_state._SESSION_OPERATIONS_CHECKPOINT_KEY, "session_operations"),
            (_SESSION_RUN_OPERATION_CHECKPOINT_KEY, "session_run_operation"),
            (
                _QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY,
                "queued_dispatch_terminal_receipts",
            ),
            (
                _prompt_transition._PROMPT_ANATOMY_TRANSITION_INTENTS_CHECKPOINT_KEY,
                "prompt_anatomy_transition_intents",
            ),
            (
                _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY,
                "incomplete_session_recovery_claim",
            ),
            (
                DURABLE_SUBAGENT_SUBMISSION_SEEDS_CHECKPOINT_KEY,
                "durable_subagent_submission_seeds",
            ),
            (
                DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY,
                "durable_subagent_submissions",
            ),
            (
                _PROVIDER_OPERATION_FALLBACK_ORDINALS_CHECKPOINT_KEY,
                "provider_operation_fallback_dispatch_ordinals",
            ),
            (
                _PROVIDER_OPERATION_PENDING_DISPOSITION_CHECKPOINT_KEY,
                "provider_operation_pending_resolution_disposition",
            ),
            (
                COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
                "completion_result_event_publications",
            ),
            (
                "recovery_plan_execution",
                "recovery_plan_execution",
            ),
        ):
            updated.pop(key, None)
            if current is not None and key in current:
                updated[key] = copy_json_value(current[key], field_name)
        return updated

    return transform


def _knowledge_store(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.knowledge_store


def _knowledge_access_scope(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.knowledge_access_scope


def _user_tool_call_count(tool_calls: list[runtime_records.ToolCallRequest]) -> int:
    return sum(1 for tool_call in tool_calls if tool_call.name != STRUCTURED_OUTPUT_TOOL_NAME)


def _session_interruption_failure_with_additional_control(
    authoritative_failure: BaseException,
    additional_failure: BaseException,
    *,
    group_message: str,
    historical_cancellation_requests: int,
) -> BaseException:
    """Retain one later failure without losing interruption cancellation state."""

    if type(historical_cancellation_requests) is not int or historical_cancellation_requests < 0:
        raise ValueError("Historical cancellation requests must be a non-negative int.")
    current_task = asyncio.current_task()
    requests_before = 0 if current_task is None else current_task.cancelling()
    delivered_cancellation: asyncio.CancelledError | None = None
    if isinstance(additional_failure, asyncio.CancelledError):
        delivered_cancellation = consume_pending_task_cancellation(
            additional_failure,
            preserve_requests=historical_cancellation_requests,
        )
    requests_after = 0 if current_task is None else current_task.cancelling()
    requests_consumed = max(requests_before - requests_after, 0)

    failures: list[BaseException] = [additional_failure]
    if delivered_cancellation is not None and all(
        delivered_cancellation is not failure for failure in failures
    ):
        failures.append(delivered_cancellation)

    propagated_failure = authoritative_failure
    for failure in failures:
        existing_ids = {id(candidate) for candidate in iter_exception_tree(propagated_failure)}
        retained_failure = _failure_without_existing_exception_identities(
            failure,
            existing_ids,
        )
        if retained_failure is None:
            continue
        propagated_failure = BaseExceptionGroup(
            group_message,
            [propagated_failure, retained_failure],
        )

    pending_request_count = (
        workspace_observation_pending_cancellation_requests(authoritative_failure)
        + requests_consumed
    )
    if pending_request_count:
        retain_workspace_observation_pending_cancellation_requests(
            propagated_failure,
            pending_request_count,
        )
    return propagated_failure


def _session_interruption_cleanup_child_error(
    error: BaseException | None,
    *,
    operation: str,
    preserved_failure: BaseException | None = None,
) -> BaseException | None:
    """Classify cleanup-owned cancellation as an operational failure."""

    if error is None:
        return None
    if error is preserved_failure:
        return error
    if isinstance(error, asyncio.CancelledError):
        return unexpected_child_cancellation_error(
            error,
            operation=operation,
        )
    if not isinstance(error, BaseExceptionGroup):
        return error

    pending: list[tuple[BaseException, bool]] = [(error, False)]
    children_by_group: dict[int, tuple[BaseException, ...]] = {}
    classified_by_identity: dict[int, tuple[BaseException, bool]] = {}
    while pending:
        candidate, expanded = pending.pop()
        candidate_id = id(candidate)
        if candidate_id in classified_by_identity:
            continue
        if candidate is preserved_failure:
            classified_by_identity[candidate_id] = (candidate, True)
            continue
        if isinstance(candidate, asyncio.CancelledError):
            classified_by_identity[candidate_id] = (
                unexpected_child_cancellation_error(
                    candidate,
                    operation=operation,
                ),
                False,
            )
            continue
        if not isinstance(candidate, BaseExceptionGroup):
            classified_by_identity[candidate_id] = (candidate, False)
            continue
        if expanded:
            children = children_by_group.pop(candidate_id, ())
            classified_children: list[BaseException] = []
            contains_preserved_failure = False
            changed = False
            for child in children:
                classified_child, child_contains_preserved = classified_by_identity[id(child)]
                classified_children.append(classified_child)
                contains_preserved_failure = contains_preserved_failure or child_contains_preserved
                changed = changed or classified_child is not child
            if contains_preserved_failure or changed:
                classified_by_identity[candidate_id] = (
                    BaseExceptionGroup(
                        f"{operation} child failures.",
                        classified_children,
                    ),
                    contains_preserved_failure,
                )
            else:
                classified_by_identity[candidate_id] = (candidate, False)
            continue
        children = exception_group_children(candidate)
        if children is None:
            classified_by_identity[candidate_id] = (
                RuntimeError(f"{operation} returned an unreadable exception group."),
                False,
            )
            continue
        children_by_group[candidate_id] = children
        pending.append((candidate, True))
        pending.extend((child, False) for child in reversed(children))

    classified, _contains_preserved_failure = classified_by_identity[id(error)]
    return classified


def _failure_carries_interaction_publication_rejection(
    failure: Exception,
    *,
    session_id: str,
    interaction_id: str | None,
    runtime_authority: object,
) -> bool:
    cause = exception_cause(failure)
    return cause is not None and any(
        _is_runtime_interaction_lifecycle_publication_rejection(
            candidate,
            session_id=session_id,
            interaction_id=interaction_id,
            runtime_authority=runtime_authority,
        )
        for candidate in iter_exception_tree(cause)
    )


def _active_transition_matches_profile_decision(
    *,
    transition: EgressAuthorityTransitionRecord | None,
    session: Session,
    decision: ExecutionProfileDecision,
) -> bool:
    """Verify the complete durable activation tuple for profile admission."""

    return bool(
        transition is not None
        and transition.state is EgressAuthorityTransitionState.ACTIVE
        and transition.session_id == session.id
        and transition.environment_name == session.environment_name
        and transition.expected_authority == decision.expected_profile.egress_authority
        and transition.target_authority == decision.candidate_profile.egress_authority
        and transition.receipt is not None
    )


@dataclass(frozen=True)
class _ResumePreparation:
    """Validated continuation input retained across abandoned-owner recovery."""

    request: ResumeRequest
    task_id: str | None
    start_event_payload_extra: dict[str, Any]
    start_task_on_enter: bool
    adoption_request_fingerprint: str | None
    source_execution_profile: ExecutionProfileIdentity | None
    required_execution_profile: ExecutionProfileIdentity | None
    required_session_instance_fingerprint: str | None
    required_task_session_instance_id: str | None
    run_operation_id: str | None
    terminal_event_id: str | None
    queue_task_id: str | None
    queued_dispatch_id: str | None
    required_foreground_wait: ForegroundChildWait | None
    required_foreground_terminal: ForegroundChildTerminal | None
    required_foreground_continuation: ForegroundParentContinuation | None
    foreground_before_mutation: Callable[[], Awaitable[None]] | None
    continuation_handoff: _ResumeAdmissionHandoff | None
    participant_context: CollaborationAccessContext | None
    execution_to_wait: _ExternalExecutionToWait | None
    budget_policy: BudgetPolicy | None
    loaded_session: Session
    recovery_required: bool


class SessionEngine:
    """Own durable session operations and one run's end-to-end orchestration."""

    def __init__(
        self,
        *,
        session_preparation: SessionPreparation,
        session_tasks: SessionTaskLifecycle,
        session_outcomes: SessionOutcomePublication,
        compaction: SessionCompaction,
        session_store: SessionStore,
        resource_session_stream: ResourceSessionStream,
        require_participant_execution: Callable[
            [Session, CollaborationAccessContext | None], Awaitable[None]
        ],
        task_store: TaskStore | None,
        get_budget_policy: Callable[[], BudgetPolicy | None],
        event_writer: RuntimeEventWriter,
        environment_lifecycle: EnvironmentLifecycle,
        run_limit_controller: RunLimitController,
        session_control: SessionControl[SessionUsageTracker],
        model_step_executor: ModelStepExecutor,
        assistant_model_publication: AssistantModelPublication,
        execution_profile_continuation: ExecutionProfileContinuation,
        model_completion_recovery: ModelCompletionRecovery,
        user_input_evidence: UserInputRecoveryEvidence,
        request_footprint: RequestFootprintConfig,
        tool_round_executor: ToolRoundExecutor,
        recovery_ownership: RecoveryOwnership,
        terminal_evidence: TerminalEvidenceReader,
        terminal_finalization: TerminalEvidenceFinalization,
        session_finalization: SessionFinalization,
        foreground_gate_policy_owner: ForegroundGatePolicyOwner,
        terminal_event_publication: TerminalEventPublication,
        pending_tool_round_recovery: PendingToolRoundRecovery,
        deferred_input: DeferredInteractionInput,
        recovery_cleanup_supervisor: RecoveryCleanupSupervisor,
        background_interruption_coordinator: BackgroundInterruptionCoordinator,
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
        runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
        loop_policies: tuple[LoopPolicy, ...],
        loop_policy_execution_profile_identities: tuple[
            ExecutionProfileBehaviorIdentity | None, ...
        ],
        get_registered_agent: Callable[[str], runtime_records.RegisteredAgentState],
        get_registered_provider: Callable[[str | None], runtime_records.RegisteredProvider],
        get_registered_environment: Callable[
            [str | None], runtime_records.RegisteredEnvironment | None
        ],
        get_registered_environment_for_session: Callable[
            [str | None], runtime_records.RegisteredEnvironment | None
        ],
        effective_retry_policy: Callable[[RetryPolicy | None], RetryPolicy],
        application_run_defaults: RunDefaults,
        profile_classifier: ExecutionProfileClassifier,
        execution_profile_process_identity: str,
        resource_access_policy: ResourceAccessPolicy | None = None,
        egress_authority_adoption_handler: EgressAuthorityAdoptionHandler | None = None,
    ) -> None:
        self._session_preparation = session_preparation
        self._session_tasks = session_tasks
        self._session_outcomes = session_outcomes
        self._compaction = compaction
        self._resource_access_policy = resource_access_policy
        self.session_store = session_store
        self._resource_session_stream = resource_session_stream
        self._require_participant_execution = require_participant_execution
        self.task_store = task_store
        self._get_budget_policy = get_budget_policy
        self._event_writer = event_writer
        self._environment_lifecycle = environment_lifecycle
        self._run_limit_controller = run_limit_controller
        self._session_control = session_control
        self._model_step_executor = model_step_executor
        self._assistant_model_publication = assistant_model_publication
        self._request_footprint = copy_request_footprint_config(request_footprint)
        self._tool_round_executor = tool_round_executor
        self._pending_tool_round_recovery = pending_tool_round_recovery
        self._deferred_input = deferred_input
        self._recovery_ownership = recovery_ownership
        self._terminal_evidence = terminal_evidence
        self._terminal_finalization = terminal_finalization
        self._session_finalization = session_finalization
        self._foreground_gate_policy_owner = foreground_gate_policy_owner
        self._terminal_event_publication = terminal_event_publication
        if type(recovery_cleanup_supervisor) is not RecoveryCleanupSupervisor:
            raise TypeError("recovery_cleanup_supervisor must be a RecoveryCleanupSupervisor.")
        self._recovery_cleanup_supervisor = recovery_cleanup_supervisor
        self._background_interruption_coordinator = background_interruption_coordinator
        self._secret_redactor = secret_redactor
        self._workflow_structured_output_handoff = WorkflowStructuredOutputHandoff()
        self._clock = clock
        self._runtime_hooks = runtime_hooks
        self._loop_policies = loop_policies
        if len(loop_policies) != len(loop_policy_execution_profile_identities):
            raise ValueError("Loop-policy execution-profile identities are inconsistent.")
        self._loop_policy_execution_profile_identities = loop_policy_execution_profile_identities
        self._get_registered_agent = get_registered_agent

        def resolve_provider(
            name: str | None = None,
        ) -> runtime_records.RegisteredProvider:
            return get_registered_provider(name)

        self._get_registered_provider = resolve_provider

        self._get_registered_environment = get_registered_environment
        self._get_registered_environment_for_session = get_registered_environment_for_session
        self._effective_retry_policy = effective_retry_policy
        if type(application_run_defaults) is not RunDefaults:
            raise TypeError("application_run_defaults must be a RunDefaults.")
        self._application_run_defaults = application_run_defaults
        self._profile_classifier = profile_classifier
        self._egress_authority_adoption_handler = egress_authority_adoption_handler
        self._execution_profile_process_identity = require_clean_nonblank(
            execution_profile_process_identity,
            "execution_profile_process_identity",
        )
        self._execution_profile_continuation = execution_profile_continuation
        self._model_completion_recovery = model_completion_recovery
        self._user_input_evidence = user_input_evidence

    async def _require_active_egress_profile_cutover(
        self,
        *,
        session: Session,
        decision: ExecutionProfileDecision | None,
    ) -> EgressAuthorityAdoptionResult | None:
        """Gate profile admission on exact durable backend activation proof."""

        if (
            decision is None
            or decision.kind is not ExecutionProfileDecisionKind.ADOPTED
            or ExecutionProfileComponentClass.EGRESS_AUTHORITY
            not in decision.changed_component_classes
        ):
            return None
        expected = decision.expected_profile.egress_authority
        target = decision.candidate_profile.egress_authority
        if expected is None or target is None:
            raise EgressAuthorityTransitionConflict(
                "Egress profile adoption omitted its exact authority identities."
            )
        transition_store = SessionCheckpointEgressAuthorityTransitionStore(
            self.session_store,
            event_writer=self._event_writer,
        )
        handler = self._egress_authority_adoption_handler
        if handler is None:
            raise EgressAuthorityTransitionConflict(
                "Egress profile adoption requires an application-owned cutover handler."
            )
        environment_name = decision.event.environment_name
        if environment_name is None:
            raise EgressAuthorityTransitionConflict(
                "Egress profile adoption omitted its invocation environment."
            )
        try:
            expected_environment_fingerprint = (
                await self._environment_lifecycle.require_live_allocation_fingerprint(
                    session_id=session.id,
                    environment_name=environment_name,
                )
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise EgressAuthorityTransitionConflict(
                "Egress profile adoption has no verified pre-cutover allocation identity."
            ) from exc
        coordinator = EgressAuthorityTransitionCoordinator(
            transition_store,
            expected_source_environment_fingerprint=expected_environment_fingerprint,
        )
        parked_factory_result = _require_parked_egress_authority_allocation(
            handler,
            session_id=session.id,
            environment_name=environment_name,
            expected_fingerprint=expected_environment_fingerprint,
        )
        adoption = await handler.adopt(
            decision,
            coordinator=coordinator,
            expected_environment_fingerprint=expected_environment_fingerprint,
            factory_result=parked_factory_result,
        )
        if not _has_runtime_egress_authority_adoption_result(
            adoption,
            coordinator=coordinator,
        ):
            raise EgressAuthorityTransitionConflict(
                "Egress profile adoption did not return an environment handoff from the "
                "runtime-owned transition coordinator."
            )
        activation = adoption.transition
        if (
            activation.source_environment_fingerprint != expected_environment_fingerprint
            or activation.environment_fingerprint != expected_environment_fingerprint
        ):
            raise EgressAuthorityTransitionConflict(
                "Egress profile adoption activated a substituted backend allocation."
            )
        transition = await transition_store.load(session.id)
        if (
            not _active_transition_matches_profile_decision(
                transition=transition,
                session=session,
                decision=decision,
            )
            or transition != activation
        ):
            raise EgressAuthorityTransitionConflict(
                "The exact egress authority cutover is not durably active; "
                "profile admission remains parked."
            )
        await require_active_egress_transition_environment(
            environment=adoption.environment,
            transition=activation,
        )
        if adoption._factory_result is not parked_factory_result:
            raise EgressAuthorityTransitionConflict(
                "Egress adoption returned a substituted environment allocation."
            )
        return adoption

    def prepare_workflow_structured_output(self, session_id: str) -> None:
        """Opt one live workflow child into a process-local raw output handoff."""

        self._workflow_structured_output_handoff.prepare(session_id)

    def take_workflow_structured_output(self, session_id: str) -> tuple[bool, Any]:
        """Consume a live raw output without making it part of durable session state."""

        return self._workflow_structured_output_handoff.take(session_id)

    def _queued_dispatch_required_profile(
        self,
        *,
        session: Session,
        source_profile: ExecutionProfileIdentity,
        request: DispatchRequest,
    ) -> ExecutionProfileIdentity:
        """Resolve the governed profile for one new queued invocation."""

        if type(session) is not Session:
            raise TypeError("Queued dispatch profile resolution requires a Session.")
        if type(source_profile) is not ExecutionProfileIdentity:
            raise TypeError("Queued dispatch source profile has an invalid type.")
        if type(request) is not DispatchRequest:
            raise TypeError("Queued dispatch profile resolution requires a DispatchRequest.")
        if request.failover is not None or source_profile.model_failover is not None:
            self.session_store._require_model_failover_stage_protocol()
        initial_profile: ExecutionProfileIdentity | None = None
        if session.run_epoch == 0:
            initial_profile = _require_exact_fork_initial_invocation_profile(
                session,
                ResumeRequest(
                    session_id=request.session_id,
                    messages=request.messages,
                    target=request.target,
                    failover=request.failover,
                    tool_capability_ceiling=request.tool_capability_ceiling,
                    tool_grants=request.tool_grants,
                    profile_adoption=request.profile_adoption,
                    metadata=request.metadata,
                    max_steps=request.max_steps,
                    limits=request.limits,
                    budget_limits=request.budget_limits,
                    retry_policy=request.retry_policy,
                    structured_output=request.structured_output,
                    thinking=request.thinking,
                    loop_policies=request.loop_policies,
                ),
                queued_dispatch_id=request.dispatch_id,
            )
        target = request.target or ModelTarget(
            provider_name=session.provider_name,
            model=session.model,
        )
        registered_agent = self._get_registered_agent(session.agent_name)
        registered_provider = self._get_registered_provider(target.provider_name)

        def ordinary_queued_profile(
            candidate_provider: runtime_records.RegisteredProvider, candidate_model: str
        ) -> ExecutionProfileIdentity:
            candidate = _session_execution_profile._execution_profile_identity(
                registered_agent=registered_agent,
                provider_name=candidate_provider.name,
                registered_provider=candidate_provider,
                model=candidate_model,
                durable_system_prompt=None,
                redactor=self._secret_redactor,
                registered_environment=self._get_registered_environment_for_session(
                    session.environment_name
                ),
                process_identity=self._execution_profile_process_identity,
                runtime_hooks=self._runtime_hooks,
                loop_policies=self._loop_policies,
                loop_policy_execution_profile_identities=(
                    self._loop_policy_execution_profile_identities
                ),
                request_loop_policies=request.loop_policies,
                request_loop_policy_instance_identities=(
                    self._execution_profile_continuation.request_loop_policy_instance_identities(
                        request.loop_policies
                    )
                ),
                budget_policy=self._get_budget_policy(),
                request_budget_limits=request.budget_limits,
                causal_budget_id=session.causal_budget_id,
                structured_output=request.structured_output,
                thinking=request.thinking,
                max_steps=request.max_steps,
                limits=request.limits,
                retry_policy=self._effective_retry_policy(request.retry_policy),
                tool_capability_ceiling=resolve_tool_capability_ceiling(
                    request.tool_capability_ceiling,
                    registered_agent.tool_capabilities,
                    maximum=_session_execution_profile._session_tool_capability_ceiling(session),
                ),
            )
            candidate = execution_profile_with_component(
                candidate,
                source_profile.component(ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION),
            )
            if source_profile.schema_version >= 2:
                candidate = execution_profile_with_component(
                    candidate,
                    source_profile.component(ExecutionProfileComponentClass.INVOCATION_POLICIES),
                )
            return candidate

        policy = request.failover
        inherited = source_profile.model_failover
        target_changed = (
            target.provider_name != session.provider_name or target.model != session.model
        )
        if policy is None and inherited is not None and not target_changed:
            policy = ModelFailoverPolicy(
                fallbacks=tuple(
                    ModelTarget(provider_name=item.provider_name, model=item.model)
                    for item in inherited.plan.candidates[1:]
                ),
                max_total_attempts=inherited.plan.max_total_attempts,
            )
        if policy is None:
            candidate = ordinary_queued_profile(registered_provider, target.model)
        else:
            candidate = execution_profile_admission.resolve_model_failover_execution_profile(
                policy=policy,
                primary=target,
                registered_primary=registered_provider,
                resolve_provider=self._get_registered_provider,
                resolve_candidate_profile=ordinary_queued_profile,
                redactor=self._secret_redactor,
            ).profile
        if initial_profile is not None:
            if candidate != initial_profile:
                raise ExecutionProfileMismatchError(
                    session_id=session.id,
                    expected_profile_fingerprint=initial_profile.fingerprint,
                    candidate_profile_fingerprint=candidate.fingerprint,
                    changed_component_classes=changed_execution_profile_components(
                        initial_profile,
                        candidate,
                    ),
                    expected_profile=initial_profile,
                    candidate_profile=candidate,
                )
            return initial_profile
        return candidate

    async def _replay_execution_profile_decision(
        self,
        *,
        session: Session,
        candidate_profile: ExecutionProfileIdentity,
        intent: ExecutionProfileAdoptionIntent,
        adoption_request_fingerprint: str | None,
    ) -> Event | None:
        if adoption_request_fingerprint is None:
            raise AssertionError("Adoption replay requires a request fingerprint.")
        event_id, _ = _execution_profile_decision_event_id(
            session_id=session.id,
            run_epoch=session.run_epoch,
            expected_profile=candidate_profile,
            candidate_profile=candidate_profile,
            intent=intent,
        )
        records = await self.session_store.query_events(
            EventQuery(session_id=session.id, event_id=event_id, limit=1)
        )
        if not records:
            return None
        event = records[0].event
        payload = event.payload
        try:
            kind = ExecutionProfileDecisionKind(payload["decision"])
            expected_profile = ExecutionProfileIdentity.model_validate(payload["expected_profile"])
            persisted_candidate = ExecutionProfileIdentity.model_validate(
                payload["candidate_profile"]
            )
            changed = tuple(
                ExecutionProfileComponentClass(component)
                for component in payload["changed_component_classes"]
            )
            authority_decision = ExecutionProfileAuthorityDecision(payload["authority_decision"])
            persisted_egress_authority_change = (
                None
                if "egress_authority_change" not in payload
                else EgressAuthorityChangeKind(payload["egress_authority_change"])
            )
            persisted_actor = payload["actor"]
            policy_identity = payload["policy_identity"]
            policy_reason = payload["policy_reason"]
            reason = payload["reason"]
            idempotency_identity = payload["idempotency_identity"]
            persisted_request_fingerprint = payload["adoption_request_fingerprint"]
        except (KeyError, TypeError, ValueError):
            raise RuntimeError(
                "Durable execution-profile adoption evidence is malformed."
            ) from None
        if (
            event.session_id != session.id
            or event.agent_name != session.agent_name
            or event.environment_name != session.environment_name
            or persisted_candidate != candidate_profile
            or changed
            != changed_execution_profile_components(expected_profile, persisted_candidate)
            or persisted_actor != resolution_actor_payload(intent.requested_by)
            or reason != intent.reason
            or idempotency_identity != intent.idempotency_key
            or persisted_request_fingerprint != adoption_request_fingerprint
            or type(policy_identity) is not str
            or not policy_identity.strip()
            or type(policy_reason) is not str
        ):
            raise ValueError(
                "Execution-profile adoption idempotency key was already used for a "
                "different request."
            )
        replayed = ExecutionProfileDecision(
            kind=kind,
            expected_profile=expected_profile,
            candidate_profile=persisted_candidate,
            changed_component_classes=changed,
            policy_identity=policy_identity,
            policy_reason=policy_reason,
            authority_decision=authority_decision,
            egress_authority_change=persisted_egress_authority_change,
            idempotency_identity=idempotency_identity,
            adoption_request_fingerprint=persisted_request_fingerprint,
            actor=ResolutionActor.model_validate(persisted_actor),
            reason=reason,
            event=event,
        )
        if replayed.kind in {
            ExecutionProfileDecisionKind.MIGRATION_REQUIRED,
            ExecutionProfileDecisionKind.REJECTED,
        }:
            error_type = (
                ExecutionProfileMigrationRequired
                if replayed.kind is ExecutionProfileDecisionKind.MIGRATION_REQUIRED
                else ExecutionProfileAdoptionRejected
            )
            raise error_type(
                session_id=session.id,
                expected_profile_fingerprint=expected_profile.fingerprint,
                candidate_profile_fingerprint=persisted_candidate.fingerprint,
                changed_component_classes=changed,
                expected_profile=expected_profile,
                candidate_profile=persisted_candidate,
            )
        return copy_event(event)

    def discard_workflow_structured_output(self, session_id: str) -> None:
        self._workflow_structured_output_handoff.discard(session_id)

    def _record_workflow_structured_output(self, session_id: str, output: Any) -> None:
        self._workflow_structured_output_handoff.record(session_id, output)

    async def interruption_cascade_status(self, session_id: str) -> str:
        """Return the public control-plane state of a session's durable cascade."""
        durable_status = "none"

        def inspect(
            _session: Session,
            checkpoint: dict[str, Any] | None,
            store_now: datetime,
        ) -> None:
            nonlocal durable_status
            if checkpoint is None:
                return None
            marker = checkpoint.get(_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY)
            if marker is None:
                return None
            durable_status = "failed"
            if type(marker) is not dict:
                return None
            attempt_id = marker.get("attempt_id")
            interrupt_payload = marker.get("interrupt_payload")
            generation = marker.get("generation", 0)
            if (
                type(attempt_id) is not str
                or not attempt_id.strip()
                or type(interrupt_payload) is not dict
                or type(generation) is not int
                or generation < 0
            ):
                return None
            failure_recorded = marker.get("failure_recorded", False)
            if type(failure_recorded) is not bool:
                return None
            try:
                claim_id = marker.get("claim_id")
                claim_expires_at = _interruption_cascade_marker_datetime(
                    marker,
                    "claim_expires_at",
                )
                if claim_id is not None:
                    if (
                        type(claim_id) is not str
                        or not claim_id.strip()
                        or claim_expires_at is None
                        or generation < 1
                    ):
                        return None
                elif claim_expires_at is not None:
                    return None
                created_at = _interruption_cascade_marker_datetime(marker, "created_at")
            except ValueError:
                return None
            if claim_id is not None:
                assert claim_expires_at is not None
                durable_status = "pending" if claim_expires_at > store_now else "failed"
                return None
            if failure_recorded or created_at is None:
                return None
            unclaimed_grace = timedelta(seconds=interruption_cascade_lease_seconds())
            durable_status = "pending" if created_at + unclaimed_grace > store_now else "failed"
            return None

        await self.session_store.transform_checkpoint_with_store_time(session_id, inspect)
        if self._background_interruption_coordinator.is_pending(session_id):
            return "pending"
        return durable_status

    def _defer_background_interruption_cascade(
        self,
        *,
        parent_session_id: str,
        interrupt_payload: dict[str, Any],
        retry_after_seconds: float,
        drain_required: bool,
        retry_request: dict[str, Any] | None,
    ) -> None:
        self._background_interruption_coordinator.defer(
            parent_session_id=parent_session_id,
            interrupt_payload=interrupt_payload,
            retry_after_seconds=retry_after_seconds,
            drain_required=drain_required,
            retry_request=retry_request,
        )

    async def recover_model_completion_stage(
        self,
        request: ModelCompletionManualRecoveryRequest,
    ) -> ModelCompletionManualRecoveryResult:
        """Conservatively settle one ambiguous model dispatch before clearing it."""

        request = copy_model_completion_manual_recovery_request(request)
        if request.terminalization_only:
            from cayu.runtime._durable_model_terminalization import terminalize_dispatched_model

            return await terminalize_dispatched_model(self, request)
        return await self._recover_model_completion_stage(request)

    async def _recover_model_completion_stage(
        self,
        request: ModelCompletionManualRecoveryRequest,
        *,
        terminalization: tuple[_IncompleteRecoveryClaim, InvocationTerminalDecision, Event]
        | None = None,
    ) -> ModelCompletionManualRecoveryResult:
        from cayu.runtime._durable_model_terminalization import terminalization_plan_owner

        request = copy_model_completion_manual_recovery_request(request)
        session = await self.session_store.load(request.session_id)
        if session is None:
            raise KeyError(f"Session not found: {request.session_id}")
        stage = await self.session_store.load_model_completion_stage(
            request.session_id,
            request.stage_id,
        )
        if stage is None or stage.state != "in_flight":
            raise SessionModelCompletionStageConflict(
                "Manual model recovery requires one exact in-flight stage."
            )
        recovery_context = model_completion_recovery_context_from_stage(stage)
        if recovery_context is None:
            raise SessionModelCompletionStageConflict(
                "Manual model recovery requires durable accounting and profile authority."
            )
        interaction_id = stage.intent.get("interaction_id")
        provider_name = stage.intent.get("provider_name")
        pricing_provider_name = stage.intent.get("pricing_provider_name", provider_name)
        model_attempt_id = stage.intent.get("model_attempt_id")
        if not all(
            type(value) is str
            for value in (
                interaction_id,
                provider_name,
                pricing_provider_name,
                model_attempt_id,
            )
        ):
            raise SessionModelCompletionStageConflict(
                "Manual model recovery lost its interaction, provider, or attempt identity."
            )
        assert isinstance(interaction_id, str)
        assert isinstance(provider_name, str)
        assert isinstance(pricing_provider_name, str)
        assert isinstance(model_attempt_id, str)
        from cayu.runtime._model_failover_stage import model_failover_target_for_stored_stage

        selected_target = model_failover_target_for_stored_stage(session=session, stage=stage)
        if (
            stage.purpose == "assistant-turn"
            and selected_target is None
            and provider_name != session.provider_name
        ):
            raise SessionModelCompletionStageConflict(
                "Assistant model recovery provider conflicts with the session provider."
            )
        model_attempt_identity = ModelAttemptIdentity(
            model_step_id=stage.logical_step_id,
            model_attempt_id=model_attempt_id,
        )
        budget_dispatch_id = stage.stage_id
        budget_model = session.model if selected_target is None else selected_target.model
        if stage.purpose == "context-compaction":
            requested_model = stage.intent.get("requested_model")
            if type(requested_model) is not str:
                raise SessionModelCompletionStageConflict(
                    "Context-compaction recovery lost its requested model."
                )
            try:
                budget_model = require_clean_nonblank(requested_model, "requested_model")
            except ValueError as invalid_model:
                raise SessionModelCompletionStageConflict(
                    "Context-compaction recovery has an invalid requested model."
                ) from invalid_model
            budget_dispatch_id = model_attempt_id

        expected_settlement = model_completion_stage_settlement_request(
            stage,
            interaction_id=interaction_id,
            disposition=ModelCompletionStageDisposition.PROVIDER_EFFECT_OUTCOME_UNKNOWN,
            reason_code="operator_outcome_unknown",
            execution_profile_fingerprint=(recovery_context.execution_profile_fingerprint),
            settlement_run_epoch=request.expected_run_epoch,
            settled_reservation_ids=stage.reservation_ids,
        )

        def validate_replay(
            settlement: ModelCompletionStageSettlement,
        ) -> ModelCompletionStageSettlement:
            replay_request = ModelCompletionStageSettlementRequest.model_validate(
                settlement.model_dump(
                    mode="python",
                    include=set(ModelCompletionStageSettlementRequest.model_fields),
                )
            )
            if replay_request != expected_settlement:
                raise SessionModelCompletionStageConflict(
                    "Manual model recovery conflicts with the durable stage settlement."
                )
            return settlement

        prior = await self.session_store.load_model_completion_stage_settlement(
            request.session_id,
            request.stage_id,
        )
        checkpoint = await self.session_store.load_checkpoint(request.session_id)
        if terminalization is not None:
            from cayu.runtime._durable_model_terminalization import (
                require_terminalization_checkpoint,
            )

            active_profile = require_terminalization_checkpoint(session, checkpoint)
            if (
                active_profile.interaction_id != interaction_id
                or active_profile.profile.fingerprint
                != recovery_context.execution_profile_fingerprint
            ):
                raise SessionModelCompletionStageConflict(
                    "Model terminalization lost exact stage/profile authority."
                )
        else:
            try:
                registered_agent = self._get_registered_agent(session.agent_name)
                # A context-compaction stage may belong to a provider other than
                # the session's assistant provider. Invocation-profile authority
                # remains the session provider; the staged effect provider below
                # is used only for exact budget settlement.
                registered_provider = self._get_registered_provider(session.provider_name)
                registered_environment = self._get_registered_environment_for_session(
                    session.environment_name
                )
            except (KeyError, RuntimeError) as registration_error:
                raise ModelCompletionManualRecoveryRequired(
                    "Manual model recovery requires the original agent, provider, and "
                    "environment registrations."
                ) from registration_error
            budget_policy = self._get_budget_policy()
            active_profile = await self._execution_profile_continuation.validate(
                session=session,
                checkpoint=checkpoint,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                budget_policy=budget_policy,
                require_open_interaction=prior is None,
            )
            _authenticated_invocation_context(
                active_profile=active_profile,
                binding=AdmittedInvocationBinding(
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    interaction_id=active_profile.interaction_id,
                    run_epoch=session.run_epoch,
                    agent_name=session.agent_name,
                    provider_name=session.provider_name,
                    model=session.model,
                    runtime_name=session.runtime_name,
                    runtime_version=session.runtime_version,
                    runtime_build_provenance=session.runtime_build_provenance,
                    environment_name=session.environment_name,
                ),
                validated_profile=active_profile.profile,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                runtime_hooks=self._runtime_hooks,
                loop_policies=self._loop_policies,
                request_loop_policies=(),
                budget_policy=budget_policy,
                tool_capability_ceiling=_session_execution_profile._session_tool_capability_ceiling(
                    session
                ),
            )
        if prior is not None:
            budget_events = await self._run_limit_controller.recover_pending_budget_settlements(
                session_id=request.session_id
            )
            borrowed_budget_events = await (
                self._run_limit_controller.reconcile_borrowed_automatic_compaction_budget_authority(
                    session=session,
                    stage=stage,
                    allow_outcome_unknown=True,
                )
            )
            budget_events = list(
                {event.id: event for event in (*budget_events, *borrowed_budget_events)}.values()
            )
            await self._run_limit_controller.require_model_completion_reservation_settlements(
                reservation_ids=stage.reservation_ids,
                recovery_contexts=recovery_context.budget_reservations,
                dispatch_id=budget_dispatch_id,
            )
            current = await self.session_store.load(request.session_id)
            if current is None:
                raise KeyError(f"Session not found: {request.session_id}")
            if current.status is not request.terminal_status:
                raise SessionStatusConflict(
                    "Manual model recovery replay conflicts with the terminal session status."
                )
            return ModelCompletionManualRecoveryResult(
                session=current,
                settlement=validate_replay(prior),
                budget_events=tuple(budget_events),
                replayed=True,
            )

        if session.run_epoch != request.expected_run_epoch:
            raise SessionRunFenced(
                "Manual model recovery run epoch is stale: expected "
                f"{request.expected_run_epoch}, current {session.run_epoch}."
            )
        if session.status not in {
            SessionStatus.RUNNING,
            SessionStatus.INTERRUPTING,
            request.terminal_status,
        }:
            raise SessionStatusConflict(
                "Manual model recovery cannot replace a different terminal session outcome."
            )
        active = await self.session_store.load_active_model_completion_stage(request.session_id)
        if active is None or active.stage != stage:
            raise SessionModelCompletionStageConflict(
                "Manual model recovery no longer owns the exact active stage."
            )
        if stage.intent.get("provider_operation_start") is not None:
            raise ModelCompletionManualRecoveryRequired(
                "Background provider work must use provider-operation resolution."
            )
        dispatch = await self.session_store.load_model_completion_stage_dispatch(
            request.session_id,
            request.stage_id,
        )
        if dispatch is None:
            raise SessionModelCompletionStageConflict(
                "Receipt-less model completion must use automatic pre-provider recovery."
            )

        borrowed_budget_events = await (
            self._run_limit_controller.reconcile_borrowed_automatic_compaction_budget_authority(
                session=session,
                stage=stage,
                allow_outcome_unknown=True,
            )
        )
        stage_budget_events = (
            await self._run_limit_controller.reconcile_manual_model_completion_reservations(
                reservation_ids=stage.reservation_ids,
                recovery_contexts=recovery_context.budget_reservations,
                session=session,
                provider_name=pricing_provider_name,
                model=budget_model,
                model_attempt_identity=model_attempt_identity,
                dispatch_id=budget_dispatch_id,
                is_automatic_context_compaction=(stage.purpose == "context-compaction"),
            )
        )
        budget_events = list(
            {event.id: event for event in (*borrowed_budget_events, *stage_budget_events)}.values()
        )
        await self._run_limit_controller.require_model_completion_reservation_settlements(
            reservation_ids=stage.reservation_ids,
            recovery_contexts=recovery_context.budget_reservations,
            dispatch_id=budget_dispatch_id,
        )

        current = await self.session_store.load(request.session_id)
        current_active = await self.session_store.load_active_model_completion_stage(
            request.session_id
        )
        if current is None:
            raise KeyError(f"Session not found: {request.session_id}")
        if current.run_epoch != request.expected_run_epoch:
            raise SessionRunFenced(
                "Manual model recovery lost its exact run epoch after budget settlement."
            )
        if current_active is None or current_active.stage != stage:
            raise SessionModelCompletionStageConflict(
                "Manual model recovery lost its active stage after budget settlement."
            )

        recovery_event = event_with_runtime_envelope_authority(
            Event(
                id="evt_model_completion_manual_"
                + hashlib.sha256(
                    canonical_durable_json_bytes(
                        request.model_dump(mode="json"),
                        "model_completion_manual_recovery",
                    )
                ).hexdigest(),
                type=(
                    EventType.INTERACTION_FAILED
                    if request.terminal_status is SessionStatus.FAILED
                    else EventType.INTERACTION_INTERRUPTED
                ),
                session_id=request.session_id,
                interaction_id=interaction_id,
                agent_name=current.agent_name,
                environment_name=current.environment_name,
            ),
            "session_id",
            "interaction_id",
        )
        if terminalization is not None:
            decision = terminalization[1]
            recovery_event = recovery_event.model_copy(
                update={
                    "id": decision.interaction_event_id,
                    "timestamp": decision.observed_at,
                }
            )
        recovery_event = self._event_writer.prepare(recovery_event)
        try:
            transition_result = await self.session_store.apply_invocation_lifecycle_command(
                SettleInvocationCommand(
                    session_id=request.session_id,
                    expected_session_instance_id=current.instance_id,
                    expected_run_epoch=current.run_epoch,
                    expected_active_profile=active_profile,
                    recovery_claim_id=None
                    if terminalization is None
                    else terminalization[0].claim_id,
                    terminalization_only=terminalization is not None,
                    terminalization_plan_ownership=terminalization_plan_owner()
                    if terminalization is not None
                    else None,
                    transition=InteractionTransitionSpec(
                        event=recovery_event,
                        from_statuses=(current.status,),
                        to_status=request.terminal_status,
                        model_completion_stage_settlement=expected_settlement,
                        terminal_event=None if terminalization is None else terminalization[2],
                        terminal_decision=None if terminalization is None else terminalization[1],
                    ),
                )
            )
            if type(transition_result) is not InteractionTransitionResult:
                raise RuntimeError("Manual recovery settlement returned an invalid result.")
            transition = transition_result
        except Exception:
            reconciled = await self.session_store.load_model_completion_stage_settlement(
                request.session_id,
                request.stage_id,
            )
            if reconciled is None:
                raise
            validate_replay(reconciled)
            transitioned = await self.session_store.load(request.session_id)
            if transitioned is None:
                raise KeyError(f"Session not found: {request.session_id}") from None
            if transitioned.status is not request.terminal_status:
                raise SessionStatusConflict(
                    "Manual model recovery replay conflicts with the terminal session status."
                ) from None
            return ModelCompletionManualRecoveryResult(
                session=transitioned,
                settlement=reconciled,
                budget_events=tuple(budget_events),
                replayed=True,
            )

        await self._event_writer.fan_out_persisted(
            [transition.event]
            + ([] if transition.terminal_event is None else [transition.terminal_event])
        )
        settlement = await self.session_store.load_model_completion_stage_settlement(
            request.session_id,
            request.stage_id,
        )
        if settlement is None:
            raise RuntimeError("Manual model recovery committed without stage settlement.")
        if transition.session.status is not request.terminal_status:
            raise SessionStatusConflict(
                "Manual model recovery committed an unexpected terminal session status."
            )
        return ModelCompletionManualRecoveryResult(
            session=transition.session,
            settlement=validate_replay(settlement),
            budget_events=tuple(budget_events),
            replayed=transition.replayed,
        )

    async def _has_promoted_safe_stop(self, session: Session) -> bool:
        """Recognize this invocation's runtime-owned cooperative interruption.

        Unlike hard interruption, safe steering never sends Task.cancel(). A
        cancellation arriving during its settlement therefore still belongs to
        the caller, even though the durable interrupted outcome must finish.
        """
        from cayu.runtime._session_steering import steering_operation_key

        checkpoint = await self.session_store.load_checkpoint(session.id)
        profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        pending = (checkpoint or {}).get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
        return (
            profile is not None
            and profile.session_id == session.id
            and profile.run_epoch == session.run_epoch
            and type(pending) is dict
            and pending.get("interruption_request_id")
            == steering_operation_key(session.instance_id, profile.interaction_id)
        )

    async def _issue_targeted_tool_grants(
        self,
        *,
        session: Session,
        interaction_id: str,
        task_id: str | None,
        prepared: tuple[PreparedTargetedToolGrant, ...],
        issued_at: datetime,
    ) -> tuple[tuple[TargetedToolGrantRecord, ...], tuple[Event, ...]]:
        """Materialize one exact batch before any provider work can begin."""

        if not prepared:
            return (), ()
        codec = self.session_store.public_authority_alias_codec
        if codec is None:  # pragma: no cover - preflight capability invariant
            raise RuntimeError("Targeted tool grant alias authority is unavailable.")
        generation_id = targeted_tool_view_generation_id(
            session_id=session.id,
            root_invocation_id=session.invocation.root_invocation_id,
        )
        records = tuple(
            build_targeted_tool_grant_record(
                grant,
                session_id=session.id,
                interaction_id=interaction_id,
                generation_id=generation_id,
                agent_name=session.agent_name,
                task_id=task_id,
                environment_name=session.environment_name,
                principal=session.invocation.origin.subject,
                tenant=session.invocation.origin.tenant,
                issued_at=issued_at,
                codec=codec,
            )
            for grant in prepared
        )
        events = tuple(
            targeted_tool_grant_event(
                record,
                event_type=EventType.TARGETED_TOOL_GRANT_ISSUED,
                timestamp=issued_at,
                outcome="issued",
                event_id_suffix="issued",
            )
            for record in records
        )
        result = await self.session_store.issue_targeted_tool_grants(
            session.id,
            expected_run_epoch=session.run_epoch,
            records=records,
            events=events,
        )
        if len(result.records) != len(prepared) or len(result.events) != len(prepared):
            raise RuntimeError("Targeted tool grant issuance returned incomplete evidence.")
        return (
            tuple(copy_targeted_tool_grant_record(record) for record in result.records),
            result.events,
        )

    async def _reconstruct_targeted_tool_grants(
        self,
        *,
        session: Session,
        interaction_id: str,
        task_id: str | None,
        registered_agent: runtime_records.RegisteredAgentState,
        capability_ceiling: ToolCapabilityCeiling,
        observed_at: datetime,
        preserve_native_projection_snapshot: bool = False,
    ) -> tuple[tuple[TargetedToolGrantRecord, ...], tuple[Event, ...]]:
        """Revalidate durable references and optionally retain a native replay snapshot."""

        if not self.session_store.supports_targeted_tool_grants:
            return (), ()
        catalogue = registered_agent.tool_catalogue
        result = await self.session_store.reconstruct_targeted_tool_grants(
            session.id,
            expected_run_epoch=session.run_epoch,
            interaction_id=interaction_id,
            generation_id=targeted_tool_view_generation_id(
                session_id=session.id,
                root_invocation_id=session.invocation.root_invocation_id,
            ),
            agent_name=session.agent_name,
            task_id=task_id,
            environment_name=session.environment_name,
            principal=session.invocation.origin.subject,
            tenant=session.invocation.origin.tenant,
            catalogue_revision=catalogue.revision,
            descriptors_by_id={
                descriptor.tool_id: (
                    descriptor.name,
                    descriptor.version,
                    descriptor.schema_fingerprint,
                )
                for descriptor in catalogue.descriptors
            },
            capability_ceiling_names=frozenset(capability_ceiling.tool_names),
            observed_at=observed_at,
        )
        if result.events:
            await self._event_writer.fan_out_persisted(list(result.events))
        replay_only_rejections = (
            frozenset(
                {
                    TargetedToolUseRejectionReason.EXPIRED,
                    TargetedToolUseRejectionReason.REVOKED,
                }
            )
            if preserve_native_projection_snapshot
            else frozenset({TargetedToolUseRejectionReason.EXPIRED})
        )
        blocking_rejections = tuple(
            (grant_id, reason)
            for grant_id, reason in result.rejected
            if reason not in replay_only_rejections
        )
        if blocking_rejections:
            reasons = ", ".join(sorted({reason.value for _grant_id, reason in blocking_rejections}))
            raise RuntimeError("Targeted tool grant reconstruction failed closed: " + reasons)
        records = tuple(copy_targeted_tool_grant_record(record) for record in result.valid)
        if preserve_native_projection_snapshot and result.rejected:
            durable_records = await self.session_store.list_targeted_tool_grants(
                session.id,
                interaction_id=interaction_id,
            )
            durable_by_id = {record.grant_id: record for record in durable_records}
            if len(durable_by_id) != len(durable_records):
                raise RuntimeError(
                    "Targeted tool grant reconstruction found duplicate durable records."
                )
            expected_ids = {record.grant_id for record in result.valid} | {
                grant_id for grant_id, _reason in result.rejected
            }
            if durable_by_id.keys() != expected_ids:
                raise RuntimeError(
                    "Targeted tool grant reconstruction conflicts with durable state."
                )
            # OpenAI stateless replay must preserve the exact historical
            # additional_tools item. Expiry and revocation still reject a new
            # invocation in the atomic use-binding path; they do not rewrite a
            # provider item that was already dispatched.
            records = tuple(copy_targeted_tool_grant_record(record) for record in durable_records)
        return (
            records,
            result.events,
        )

    async def verifier_aware_task_execution_outcome(
        self,
        task_id: str | None,
        *,
        session_id: str | None = None,
        admit_session: bool = True,
        allow_missing_task: bool = False,
    ) -> tuple[bool, BaseException | None]:
        return await verifier_aware_task_execution_outcome(
            task_id,
            task_store=self.task_store,
            redactor=self._secret_redactor,
            session_id=session_id,
            admit_session=admit_session,
            allow_missing_task=allow_missing_task,
        )

    async def run(
        self,
        request: RunRequest,
        *,
        expected_execution_profile: ExecutionProfileIdentity | None = None,
        expected_registered_environment: runtime_records.RegisteredEnvironment | None = None,
        expected_context_policy: object | None = None,
        pause_after_initial_transcript: bool = False,
        participant_execution_key: str | None = None,
        participant_session_instance_id: str | None = None,
        participant_context: CollaborationAccessContext | None = None,
        participant_permit_operation: str | None = None,
        participant_permit_commitment: str | None = None,
        execution_to_wait: _ExecutionToWait | _ExternalExecutionToWait | None = None,
        producer_output: _ProducerExecution | None = None,
    ) -> AsyncGenerator[Event, None]:
        if type(pause_after_initial_transcript) is not bool:
            raise TypeError("pause_after_initial_transcript must be a bool.")
        if producer_output is not None and (
            type(producer_output) is not _ProducerExecution
            or producer_output.app._runtime_session_store is not self.session_store
            or participant_execution_key is None
            or execution_to_wait is not None
        ):
            raise PermissionError("Producer execution requires the exact native handoff.")
        if (participant_execution_key is None) != (participant_session_instance_id is None):
            raise ValueError(
                "Participant execution requires both an execution key and session incarnation."
            )
        if participant_execution_key is not None:
            if type(participant_execution_key) is not str or not participant_execution_key.strip():
                raise ValueError("Participant execution key must be non-blank.")
            if type(participant_session_instance_id) is not str:
                raise ValueError("Participant execution requires a session incarnation.")
            if participant_permit_operation is None or participant_permit_commitment is None:
                raise ValueError("Participant execution requires a durable permit.")
        retained_policy_selection = None
        if participant_execution_key is not None:
            if request.session_id is None:
                raise ValueError("Participant execution requires a session ID.")
            retained_session = await self.session_store.load(request.session_id)
            creation_receipt = await self.session_store.load_participant_session_creation_receipt(
                request.session_id
            )
            if (
                retained_session is None
                or retained_session.instance_id != participant_session_instance_id
                or creation_receipt is None
                or creation_receipt.binding.session_id != retained_session.id
                or creation_receipt.binding.session_instance_id != retained_session.instance_id
            ):
                raise SessionStatusConflict("Participant creation authority is unavailable.")
            evidence = creation_receipt.binding.policy_evidence_json
            if evidence is not None:
                selected = decode_policy_evidence(evidence.encode())
                if (
                    selected["provider_name"] != retained_session.provider_name
                    or selected["model"] != retained_session.model
                ):
                    raise SessionStatusConflict(
                        "Participant policy selection conflicts with its session."
                    )
                retained_policy_selection = PolicySelection(
                    target=ModelTarget(
                        provider_name=retained_session.provider_name, model=retained_session.model
                    ),
                    evidence=evidence.encode(),
                )
        preparation = self._session_preparation.prepare_initial_run(
            request,
            expected_execution_profile=expected_execution_profile,
            expected_registered_environment=expected_registered_environment,
            expected_context_policy=expected_context_policy,
            allow_existing_session_id=participant_execution_key is not None,
            retained_policy_selection=retained_policy_selection,
        )
        del request
        prepared = await preparation
        del preparation
        if prepared is None:
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        request = prepared.request
        registered_agent = prepared.registered_agent
        registered_provider = prepared.registered_provider
        registered_environment = prepared.registered_environment
        workspace_instructions = prepared.workspace_instructions
        rendered_system_prompt = prepared.rendered_system_prompt
        prompt_contributions = list(prepared.prompt_contributions)
        execution_profile = prepared.execution_profile
        tool_capability_ceiling = prepared.tool_capability_ceiling
        model_failover = prepared.model_failover
        tool_discovery_initializer = _tool_discovery_operation_initializer(
            registered_agent,
            tool_capability_ceiling,
        )
        targeted_tool_grants = prepared.targeted_tool_grants
        budget_policy = prepared.budget_policy
        session_identity = prepared.session_identity
        policy_evidence = prepared.policy_evidence
        # ``prepared`` also retains the registered provider, whose repr may contain
        # live credentials. Keep the deliberately scoped provider local below so
        # cancellation cleanup can drop it before a traceback escapes this frame.
        # The failover resolution retains frozen collaborators through its own
        # deliberately credential-free repr, not the prepared container's repr.
        del prepared
        session_id = request.session_id
        if session_id is None:
            raise AssertionError("Run request session identity was not assigned.")
        prepared_session_authority = runtime_prepared_session_authority(request)
        external_execution = None
        if type(execution_to_wait) is _ExternalExecutionToWait:
            if participant_execution_key is not None or prepared_session_authority is not None:
                raise PermissionError("External waits require ordinary root execution.")
            external_execution = await execution_to_wait.prepare_initial(
                request,
                execution_profile,
                request_sha256=session_request_boundary.work_attempt_source_request_sha256(
                    request,
                    kind="initial",
                    redactor=self._secret_redactor,
                    policy_instance_identity=self._execution_profile_continuation.policy_identity_registry.identity_for,
                ),
            )
        interaction_id = (
            external_execution.interaction_id
            if external_execution is not None
            else str(
                uuid5(
                    NAMESPACE_URL,
                    f"cayu-participant-session:{session_id}:{participant_execution_key}",
                )
            )
            if participant_execution_key is not None
            else (
                str(uuid4())
                if prepared_session_authority is None
                else prepared_session_authority.interaction_id
            )
        )
        interaction_started_event = build_interaction_started_event_from_identity(
            clock=self._clock,
            session_id=session_id,
            agent_name=registered_agent.spec.name,
            environment_name=_environment_name(registered_environment),
            interaction_id=interaction_id,
            policy_evidence=policy_evidence,
            event_id=(
                external_execution.interaction_event_id
                if external_execution is not None
                else str(
                    uuid5(
                        NAMESPACE_URL,
                        f"cayu-participant-session-event:{session_id}:{participant_execution_key}",
                    )
                )
                if participant_execution_key is not None
                else (
                    None
                    if prepared_session_authority is None
                    else prepared_session_authority.interaction_started_event_id
                )
            ),
            targeted_tool_grants=targeted_tool_grants,
        )
        if external_execution is not None:
            interaction_started_event = interaction_started_event.model_copy(
                update={
                    "timestamp": datetime.fromtimestamp(
                        external_execution.prepared_at_ms / 1000, tz=UTC
                    ),
                }
            )
        invocation_context: InvocationContext | None = None
        if participant_execution_key is not None:
            participant_session = await self.session_store.load(session_id)
            if (
                participant_session is None
                or participant_session.instance_id != participant_session_instance_id
            ):
                raise SessionStatusConflict("Participant session incarnation is unavailable.")
            participant_checkpoint = (
                await self.session_store.load_checkpoint(session_id)
                if producer_output is None
                else await producer_output.checkpoint(self.session_store, session_id)
            )
            if participant_session.status is SessionStatus.PENDING:
                if participant_checkpoint is not None and producer_output is None:
                    raise SessionRunFenced(
                        "Participant session has unexpected checkpoint authority."
                    )
                participant_active_profile = ActiveInvocationExecutionProfile(
                    session_id=session_id,
                    interaction_id=interaction_id,
                    run_epoch=participant_session.run_epoch + 1,
                    profile=execution_profile,
                )
                execution_profile = participant_active_profile.profile
                participant_invocation_context = _authenticated_invocation_context(
                    active_profile=participant_active_profile,
                    binding=PreparedInvocationBinding(
                        session_id=participant_session.id,
                        session_instance_id=participant_session.instance_id,
                        interaction_id=interaction_id,
                        run_epoch=participant_session.run_epoch + 1,
                        agent_name=participant_session.agent_name,
                        provider_name=participant_session.provider_name,
                        model=participant_session.model,
                        runtime_name=participant_session.runtime_name,
                        runtime_version=participant_session.runtime_version,
                        runtime_build_provenance=participant_session.runtime_build_provenance,
                        environment_name=participant_session.environment_name,
                    ),
                    validated_profile=execution_profile,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    runtime_hooks=self._runtime_hooks,
                    loop_policies=self._loop_policies,
                    request_loop_policies=request.loop_policies,
                    budget_policy=budget_policy,
                    tool_capability_ceiling=tool_capability_ceiling,
                    targeted_tool_grants=targeted_tool_grants,
                )
                admission_command = AdmitInvocationCommand(
                    session_id=session_id,
                    expected_session_instance_id=participant_session.instance_id,
                    expected_statuses=(SessionStatus.PENDING,),
                    expected_run_epoch=participant_session.run_epoch,
                    expected_checkpoint_sha256=invocation_checkpoint_state_sha256(
                        participant_checkpoint
                    ),
                    target_active_profile=participant_active_profile,
                    checkpoint_patch=InvocationCheckpointPatch(
                        mutation=runtime_publication_checkpoint_mutation(None, None)
                    ),
                    tool_capability_ceiling=tool_capability_ceiling,
                    interaction_source_messages=tuple(request.messages),
                    interaction_started_event=interaction_started_event,
                    defer_interaction_source=True,
                    allow_pending_initial_interaction=True,
                    participant_permit_operation=participant_permit_operation,
                    participant_permit_commitment=participant_permit_commitment,
                )
                admission_result = (
                    await self.session_store.apply_invocation_lifecycle_command(admission_command)
                    if producer_output is None
                    else await producer_output.admit(admission_command)
                )
                if type(admission_result) is not InvocationMutationResult:
                    raise RuntimeError(
                        "Participant invocation admission returned an invalid result."
                    )
                if admission_result.replayed:
                    return
                session = admission_result.session
                invocation_context = participant_invocation_context.with_admitted_session(session)
            elif participant_session.status is SessionStatus.RUNNING:
                active = active_invocation_execution_profile_from_checkpoint(participant_checkpoint)
                if (
                    active is None
                    or active.session_id != session_id
                    or active.interaction_id != interaction_id
                    or active.run_epoch != participant_session.run_epoch
                    or active.profile != execution_profile
                ):
                    raise SessionStatusConflict(
                        "Participant session has another active invocation."
                    )
                execution_profile = active.profile
                invocation_context = _authenticated_invocation_context(
                    active_profile=active,
                    binding=AdmittedInvocationBinding(
                        session_id=participant_session.id,
                        session_instance_id=participant_session.instance_id,
                        interaction_id=active.interaction_id,
                        run_epoch=participant_session.run_epoch,
                        agent_name=participant_session.agent_name,
                        provider_name=participant_session.provider_name,
                        model=participant_session.model,
                        runtime_name=participant_session.runtime_name,
                        runtime_version=participant_session.runtime_version,
                        runtime_build_provenance=participant_session.runtime_build_provenance,
                        environment_name=participant_session.environment_name,
                    ),
                    validated_profile=execution_profile,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    runtime_hooks=self._runtime_hooks,
                    loop_policies=self._loop_policies,
                    request_loop_policies=request.loop_policies,
                    budget_policy=budget_policy,
                    tool_capability_ceiling=tool_capability_ceiling,
                    targeted_tool_grants=targeted_tool_grants,
                )
                session = participant_session
            else:
                raise SessionStatusConflict("Participant session is no longer executable.")
        elif prepared_session_authority is None:
            request = run_request_with_runtime_session_instance_authority(
                request,
                session_instance_id=(
                    str(uuid4())
                    if external_execution is None
                    else external_execution.session_instance_id
                ),
            )
            bind_runtime_session_create_claim(
                request,
                identity=session_identity,
                interaction_started_event=interaction_started_event,
            )
            session_instance_id = _authenticated_session_instance_id_for_run_request(
                request,
                session_id=session_id,
            )
            if session_instance_id is None:
                raise AssertionError("Run preparation lost session-incarnation authority.")
            active_profile = ActiveInvocationExecutionProfile(
                session_id=session_id,
                interaction_id=interaction_id,
                run_epoch=1,
                profile=execution_profile,
            )
            # Pydantic defensively reconstructs nested profile models. From
            # this point the active wrapper's profile is the one live object
            # carried by the invocation context and every runtime phase.
            execution_profile = active_profile.profile
            invocation_context = _authenticated_invocation_context(
                active_profile=active_profile,
                binding=PreparedInvocationBinding(
                    session_id=session_id,
                    session_instance_id=session_instance_id,
                    interaction_id=interaction_id,
                    run_epoch=1,
                    agent_name=registered_agent.spec.name,
                    provider_name=registered_provider.name,
                    model=session_identity.model,
                    runtime_name=session_identity.runtime_name,
                    runtime_version=session_identity.runtime_version,
                    runtime_build_provenance=(session_identity.runtime_build_provenance),
                    environment_name=_environment_name(registered_environment),
                ),
                validated_profile=execution_profile,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                runtime_hooks=self._runtime_hooks,
                loop_policies=self._loop_policies,
                request_loop_policies=request.loop_policies,
                budget_policy=budget_policy,
                tool_capability_ceiling=tool_capability_ceiling,
                targeted_tool_grants=targeted_tool_grants,
            )
            tool_discovery_initialization = None
            if tool_discovery_initializer is not None:
                tool_discovery_initialization = tool_discovery_view_initialization(
                    session_id=session_id,
                    agent_name=registered_agent.spec.name,
                    catalogue=registered_agent.tool_catalogue,
                    ceiling=tool_capability_ceiling,
                )
            creation_command = CreateInvocationCommand(
                session_id=session_id,
                expected_session_instance_id=session_instance_id,
                request=request,
                identity=session_identity,
                active_profile=active_profile,
                interaction_started_event=interaction_started_event,
                interaction_source_messages=tuple(request.messages),
                tool_discovery_initialization=tool_discovery_initialization,
            )
            if type(execution_to_wait) is _ExternalExecutionToWait:
                assert external_execution is not None
                created = await execution_to_wait.create_invocation(
                    creation_command, external_execution
                )
            else:
                created = await self.session_store.apply_invocation_lifecycle_command(
                    creation_command
                )
            if type(created) is not InvocationMutationResult:
                raise RuntimeError("Invocation create returned an invalid result.")
            session = created.session
            invocation_context = invocation_context.with_admitted_session(session)
        else:

            def validate_prepared_submission(
                current_session: Session,
                checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any]:
                try:
                    intent = durable_subagent_submission_from_checkpoint(
                        checkpoint,
                        idempotency_key=prepared_session_authority.idempotency_key,
                    )
                except (TypeError, ValueError) as exc:
                    raise durable_subagent_authority_rejected() from exc
                if intent is None:
                    raise durable_subagent_authority_rejected()
                if intent.child_execution_profile != execution_profile:
                    raise durable_subagent_worker_incompatible()
                if (
                    intent.child_session_id != current_session.id
                    or intent.child_runtime_name != current_session.runtime_name
                    or intent.child_runtime_version != current_session.runtime_version
                    or intent.child_runtime_build_provenance
                    != current_session.runtime_build_provenance
                    or intent.queue_task_id != prepared_session_authority.queue_task_id
                    or intent.submission_sha256 != prepared_session_authority.submission_sha256
                    or intent.interaction_id != interaction_id
                    or intent.interaction_started_event_id != interaction_started_event.id
                    or intent.request_sha256 != durable_subagent_request_sha256(request)
                ):
                    raise durable_subagent_authority_rejected()
                return _checkpoint_with_session_run_operation(
                    checkpoint=checkpoint,
                    current_session=current_session,
                    operation_id=prepared_session_authority.dispatch_operation_id,
                    terminal_event_id=prepared_session_authority.terminal_event_id,
                    queue_task_id=prepared_session_authority.queue_task_id,
                )

            pending_session = await self.session_store.load(session_id)
            if pending_session is None:
                raise KeyError(f"Session not found: {session_id}")
            pending_checkpoint = await self.session_store.load_checkpoint(session_id)
            claimed_checkpoint = validate_prepared_submission(
                pending_session,
                pending_checkpoint,
            )
            prepared_active_profile = ActiveInvocationExecutionProfile(
                session_id=session_id,
                interaction_id=interaction_id,
                run_epoch=pending_session.run_epoch + 1,
                profile=execution_profile,
            )
            execution_profile = prepared_active_profile.profile
            prepared_invocation_context = _authenticated_invocation_context(
                active_profile=prepared_active_profile,
                binding=PreparedInvocationBinding(
                    session_id=pending_session.id,
                    session_instance_id=pending_session.instance_id,
                    interaction_id=interaction_id,
                    run_epoch=pending_session.run_epoch + 1,
                    agent_name=pending_session.agent_name,
                    provider_name=pending_session.provider_name,
                    model=pending_session.model,
                    runtime_name=pending_session.runtime_name,
                    runtime_version=pending_session.runtime_version,
                    runtime_build_provenance=(pending_session.runtime_build_provenance),
                    environment_name=pending_session.environment_name,
                ),
                validated_profile=execution_profile,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                runtime_hooks=self._runtime_hooks,
                loop_policies=self._loop_policies,
                request_loop_policies=request.loop_policies,
                budget_policy=budget_policy,
                tool_capability_ceiling=tool_capability_ceiling,
                targeted_tool_grants=targeted_tool_grants,
            )
            admission_command = AdmitInvocationCommand(
                session_id=session_id,
                expected_session_instance_id=pending_session.instance_id,
                expected_statuses=(SessionStatus.PENDING,),
                expected_run_epoch=pending_session.run_epoch,
                expected_checkpoint_sha256=(invocation_checkpoint_state_sha256(pending_checkpoint)),
                target_active_profile=prepared_active_profile,
                checkpoint_patch=InvocationCheckpointPatch(
                    mutation=runtime_publication_checkpoint_mutation(
                        pending_checkpoint,
                        claimed_checkpoint,
                    )
                ),
                tool_capability_ceiling=tool_capability_ceiling,
                interaction_source_messages=tuple(request.messages),
                interaction_started_event=interaction_started_event,
                defer_interaction_source=True,
                allow_pending_initial_interaction=True,
            )
            try:
                admission_result = await self.session_store.apply_invocation_lifecycle_command(
                    admission_command
                )
            except SessionRunFenced as admission_fence:
                # A CAS race is normally retryable. If the current store state
                # positively proves that the prepared child's durable parent
                # authority was removed or replaced, preserve that permanent
                # classification so the dispatcher terminally rejects it.
                try:
                    current_session = await self.session_store.load(session_id)
                    current_checkpoint = await self.session_store.load_checkpoint(session_id)
                except Exception:
                    raise admission_fence from None
                try:
                    if current_session is None:
                        raise durable_subagent_authority_rejected()
                    validate_prepared_submission(current_session, current_checkpoint)
                finally:
                    if type(current_checkpoint) is dict:
                        current_checkpoint.clear()
                    current_checkpoint = None
                raise admission_fence
            if type(admission_result) is not InvocationMutationResult:
                raise RuntimeError("Invocation admission returned an invalid result.")
            session = admission_result.session
            invocation_context = prepared_invocation_context.with_admitted_session(session)
        if invocation_context is None:
            raise AssertionError("Invocation admission produced no frozen context.")
        _activate_session_interaction(session.id, interaction_id)
        targeted_tool_projection = resolve_targeted_tool_projection(
            registered_agent.targeted_tool_mode,
            provider=registered_provider.provider,
            model=session.model,
        )
        current_task = asyncio.current_task()
        active_factory_run: ActiveSessionRun[SessionUsageTracker] | None = None
        if current_task is not None:
            active_factory_run = self._session_control.register_active_task(
                session.id,
                current_task,
                task_id=request.task_id,
                task_started=False,
                task_finished=False,
            )
        # The run fence belongs to this pre-run setup until control is handed to
        # _run_session. Every earlier exit must revoke it, including cancellation
        # and failures while recording the original setup failure.
        release_before_run = True
        pre_run_task_started = False
        interaction_started = True
        messages: list[Message] | None = None
        prompt_contribution_manifest: PromptContributionManifest | None = None
        try:
            await self._session_control.execution_presence.ensure(session, task_id=request.task_id)
            await _require_tool_discovery_view(
                self.session_store,
                session,
                registered_agent,
                tool_capability_ceiling,
                require_empty=True,
            )
            (
                targeted_tool_grant_records,
                targeted_tool_grant_events,
            ) = await self._issue_targeted_tool_grants(
                session=session,
                interaction_id=interaction_id,
                task_id=request.task_id,
                prepared=targeted_tool_grants,
                issued_at=interaction_started_event.timestamp,
            )
            await self._event_writer.fan_out_persisted([interaction_started_event])
            if targeted_tool_grant_events:
                await self._event_writer.fan_out_persisted(list(targeted_tool_grant_events))
            yield interaction_started_event
            for targeted_tool_grant_event_record in targeted_tool_grant_events:
                yield targeted_tool_grant_event_record
            factory_started_event = await self._environment_lifecycle.emit_factory_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            )
            if factory_started_event is not None:
                yield factory_started_event
                async for queued_event in self._session_control.drain_out_of_band_events(
                    session.id
                ):
                    yield queued_event
                if request.task_id is not None:
                    task = await self._session_tasks.start_task(
                        task_id=request.task_id,
                        session=session,
                        worker_id=request.task_worker_id,
                        lease_expires_at=request.task_lease_expires_at,
                    )
                    pre_run_task_started = True
                    if active_factory_run is not None:
                        active_factory_run.task_started = True
                    yield await self._event_writer.emit(
                        _task_event(
                            event_type=EventType.TASK_STARTED,
                            task=task,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                        )
                    )
            resolution = await self._environment_lifecycle.resolve_factory(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=factory_started_event,
                operation=EnvironmentFactoryOperation.CREATE,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            )
            registered_environment = resolution.registered_environment
            if registered_environment is not None:
                invocation_context = invocation_context.with_registered_environment(
                    registered_environment,
                    validated_profile=execution_profile,
                )
            for event in resolution.events:
                yield event
                async for queued_event in self._session_control.drain_out_of_band_events(
                    session.id
                ):
                    yield queued_event
            async for queued_event in self._session_control.drain_out_of_band_events(session.id):
                yield queued_event
            if resolution.error is not None:
                raise resolution.error

            if workspace_instructions is None:
                workspace_instructions = (
                    await self._environment_lifecycle.load_workspace_instructions(
                        registered_environment,
                    )
                )
            (
                resolved_system_prompt,
                resolved_prompt_contributions,
            ) = render_initial_system_prompt_with_contributions(
                agent_system_prompt=registered_agent.spec.system_prompt,
                workspace_instructions=workspace_instructions,
            )
            if resolved_system_prompt != rendered_system_prompt:
                raise RuntimeError(
                    "The durable system projection changed after the invocation profile "
                    "was frozen; session setup fails closed."
                )
            prompt_contributions = resolved_prompt_contributions
            if self._request_footprint.enabled:
                prompt_contribution_manifest = build_prompt_contribution_manifest(
                    rendered_system_prompt=rendered_system_prompt,
                    contributions=prompt_contributions,
                    config=self._request_footprint,
                )
            targeted_projection_marker = (
                targeted_tool_projection_marker_message(
                    targeted_tool_grants,
                    interaction_id=interaction_id,
                    generation_id=targeted_tool_view_generation_id(
                        session_id=session.id,
                        root_invocation_id=session.invocation.root_invocation_id,
                    ),
                )
                if targeted_tool_grants
                and targeted_tool_projection is TargetedToolProjectionKind.OPENAI_ADDITIONAL_TOOLS
                else None
            )
            initial_source_messages = [
                *request.messages,
                *(() if targeted_projection_marker is None else (targeted_projection_marker,)),
            ]
            messages = transcript_helpers.initial_messages(
                system_prompt=rendered_system_prompt,
                request_messages=initial_source_messages,
            )
            if targeted_projection_marker is None:
                # Preserve the established call shape for stores that do not
                # participate in native targeted-tool projection.
                await self.session_store.replace_initial_transcript_messages(
                    session.id,
                    request.messages,
                    messages,
                )
            else:
                await self.session_store.replace_initial_transcript_messages(
                    session.id,
                    request.messages,
                    messages,
                    runtime_suffix_count=1,
                )
            if pause_after_initial_transcript:
                # Native fixture preparation admits the exact transcript and
                # closes its interaction before any model-loop/provider work.
                # Keep setup ownership until normal terminal/fence cleanup.
                # This entrance returns before _run_session emits the origin;
                # retain that runtime-owned evidence before closing setup so a
                # later resume remains capturable as a complete trajectory.
                start_event = Event(
                    type=EventType.SESSION_STARTED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=_environment_name(registered_environment),
                    payload={
                        "agent_name": registered_agent.spec.name,
                        SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY: session_input_contract_evidence(
                            request,
                            message_start_index=(len(messages) - len(initial_source_messages)),
                        ),
                        **(
                            {}
                            if prompt_contribution_manifest is None
                            else {
                                "prompt_contribution_manifest": (
                                    prompt_contribution_manifest.model_dump(mode="json")
                                )
                            }
                        ),
                        **(
                            {}
                            if prepared_session_authority is None
                            else {
                                "dispatch_operation_id": (
                                    prepared_session_authority.dispatch_operation_id
                                ),
                                "queue_task_id": prepared_session_authority.queue_task_id,
                                "durable_subagent_submission_sha256": (
                                    prepared_session_authority.submission_sha256
                                ),
                            }
                        ),
                        **_session_trace_event_fields(session, request.metadata),
                    },
                )
                lineage_fields = (
                    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
                    *(
                        field_name
                        for field_name, expected in (
                            ("parent_session_id", session.parent_session_id),
                            ("causal_budget_id", session.causal_budget_id),
                        )
                        if expected is not None and start_event.payload.get(field_name) == expected
                    ),
                )
                yield await self._event_writer.emit(
                    event_with_runtime_payload_authority(start_event, *lineage_fields)
                )
                async for event in self._session_finalization.handle_session_interrupted(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=_environment_name(registered_environment),
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                ):
                    yield event
                return
            release_before_run = False
        except asyncio.CancelledError as cancellation:
            if await self._session_control.interrupt_requested(session.id):
                if interaction_started:
                    await self.session_store.materialize_deferred_interaction_input(session.id)
                async for event in self._session_finalization.handle_session_interrupted(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=_environment_name(registered_environment),
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                ):
                    yield event
                return
            await self._recovery_ownership.run_cleanup_steps(
                authoritative_failure=cancellation,
                steps=(
                    (
                        "cancelled pre-run session finalization",
                        lambda: self._session_finalization.finalize_abandoned_session_by_id(
                            session.id,
                            propagate_interaction_publication_rejection=True,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            execution_profile=execution_profile,
                            invocation_context=invocation_context,
                        ),
                    ),
                ),
            )
            raise
        except TerminalEventPublicationUncertain:
            raise
        except Exception as exc:
            if is_durable_subagent_submission_unsettled(
                exc,
                parent_session_id=session.id,
            ):
                raise
            if _failure_carries_interaction_publication_rejection(
                exc,
                session_id=session.id,
                interaction_id=_current_session_interaction_id(session.id),
                runtime_authority=self._session_finalization.interaction_publication_authority,
            ):
                raise
            try:
                failure_interaction_observed_at = (
                    await self._session_finalization.failure_interaction_observed_at(session)
                )
            except Exception as preflight_failure:
                _raise_primary_with_secondary_failure(
                    exc,
                    preflight_failure,
                    group_message=(
                        "Run failure evidence and interaction terminalization preflight failure."
                    ),
                )
            failure_diagnostic = exception_diagnostic(
                exc,
                redactor=self._secret_redactor,
            )
            if interaction_started:
                await self.session_store.materialize_deferred_interaction_input(session.id)
            (
                task_failure_event,
                task_failure_error,
            ) = await self._session_tasks.fail_task_for_run_setup_error(
                task_id=request.task_id,
                task_worker_id=request.task_worker_id,
                task_handoff_id=None,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                diagnostic=failure_diagnostic,
            )
            if task_failure_event is not None:
                yield task_failure_event
            (
                session,
                interaction_failed_event,
                _,
            ) = await self._session_finalization.publish_sibling_interaction_transition(
                session=session,
                invocation_context=invocation_context,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=_environment_name(registered_environment),
                to_status=SessionStatus.FAILED,
                observed_at=failure_interaction_observed_at,
                execution_profile=execution_profile,
            )
            if interaction_failed_event is not None:
                yield interaction_failed_event
            failure_payload = exception_failure_payload(
                exc,
                diagnostic=failure_diagnostic,
                redactor=self._secret_redactor,
            )
            if task_failure_error is not None:
                failure_payload.update(
                    task_update_error_payload(
                        task_failure_error,
                        redactor=self._secret_redactor,
                    )
                )
            async for event in self._terminal_event_publication.emit(
                event=Event(
                    type=EventType.SESSION_FAILED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=_environment_name(registered_environment),
                    payload=failure_payload,
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            ):
                yield event
                async for queued_event in self._session_control.drain_out_of_band_events(
                    session.id
                ):
                    yield queued_event
            return
        except BaseExceptionGroup as exc:
            if binding_finalize_explicit_cancellation(exc) is not None:

                async def prepare_interruption() -> None:
                    if interaction_started:
                        await self.session_store.materialize_deferred_interaction_input(session.id)

                (
                    _interruption_events,
                    propagated_failure,
                ) = await self._handle_session_interrupted_preserving_failure(
                    authoritative_failure=exc,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=_environment_name(registered_environment),
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                    prepare_interruption=prepare_interruption,
                )
                if propagated_failure is exc:
                    raise
                raise propagated_failure from None
            raise
        except GeneratorExit as abandonment:
            # This window is outside _run_session's own finalizer. Publish the
            # interaction closure and session interruption through the same
            # atomic boundary used by ordinary interruption.
            await self.session_store.materialize_deferred_interaction_input(session.id)
            current_session = await self.session_store.load(session.id)
            if current_session is not None:
                try:
                    async for _ in self._session_finalization.handle_session_interrupted(
                        session=current_session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    ):
                        pass
                except Exception as finalization_failure:
                    _raise_primary_with_secondary_failure(
                        abandonment,
                        finalization_failure,
                        group_message=(
                            "Setup stream abandonment and interaction terminalization failure."
                        ),
                    )
            raise
        finally:
            finalizer_failure = sys.exception()
            if finalizer_failure is not None and (
                isinstance(finalizer_failure, asyncio.CancelledError)
                or workspace_observation_pending_cancellation_requests(finalizer_failure)
            ):
                finalizer_steps: tuple[tuple[str, Callable[[], Awaitable[None]]], ...] = ()
                if release_before_run:
                    finalizer_steps = (
                        (
                            "Pre-run environment setup abort",
                            lambda: self._environment_lifecycle.abort_environment_setup(
                                session_id=session.id,
                                original_error=finalizer_failure,
                                execution_profile=execution_profile,
                                invocation_context=invocation_context,
                            ),
                        ),
                        (
                            "Pre-run environment run-fence release",
                            lambda: (
                                self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                                    session_id=session.id,
                                    execution_profile=execution_profile,
                                    invocation_context=invocation_context,
                                )
                            ),
                        ),
                    )
                propagated_finalizer_failure = (
                    await self._settle_session_finalizer_preserving_interruption_failure(
                        authoritative_failure=finalizer_failure,
                        steps=finalizer_steps,
                    )
                )
                if release_before_run:
                    # The owned child receives a copied context. Retire the
                    # caller's corresponding token after that child settles.
                    _deactivate_session_run_fence(session.id)
                try:
                    if current_task is not None and active_factory_run is not None:
                        self._session_control.unregister_active_task(
                            session.id,
                            current_task,
                        )
                finally:
                    if release_before_run:
                        _deactivate_session_interaction(session.id)
                if propagated_finalizer_failure is not finalizer_failure:
                    raise propagated_finalizer_failure from None
            else:
                try:
                    if release_before_run:
                        await self._environment_lifecycle.abort_environment_setup(
                            session_id=session.id,
                            original_error=finalizer_failure,
                            execution_profile=execution_profile,
                            invocation_context=invocation_context,
                        )
                finally:
                    try:
                        if release_before_run:
                            await self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                                session_id=session.id,
                                execution_profile=execution_profile,
                                invocation_context=invocation_context,
                            )
                    finally:
                        try:
                            if current_task is not None and active_factory_run is not None:
                                self._session_control.unregister_active_task(
                                    session.id,
                                    current_task,
                                )
                        finally:
                            if release_before_run:
                                _deactivate_session_interaction(session.id)

        try:
            if messages is None:
                raise RuntimeError("Initial transcript was not finalized during setup.")
            session_stream = self._run_session(
                session=session,
                execution_to_wait=execution_to_wait,
                participant_context=participant_context,
                invocation_context=invocation_context,
                model_failover=model_failover,
                messages=messages,
                messages_to_append=[],
                max_steps=request.max_steps,
                limits=request.limits,
                budget_limits=request.budget_limits,
                retry_policy=self._effective_retry_policy(request.retry_policy),
                structured_output=request.structured_output,
                thinking=request.thinking,
                request_metadata=request.metadata,
                request_trace_metadata=request.metadata,
                targeted_tool_grants=targeted_tool_grant_footprint(
                    targeted_tool_grant_records,
                    projection=targeted_tool_projection,
                ),
                task_id=request.task_id,
                task_worker_id=request.task_worker_id,
                task_lease_expires_at=request.task_lease_expires_at,
                task_handoff_id=None,
                start_event_type=EventType.SESSION_STARTED,
                start_event_payload={
                    "agent_name": registered_agent.spec.name,
                    **(
                        {}
                        if policy_evidence is None
                        else {"model_policy": decode_policy_evidence(policy_evidence)}
                    ),
                    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY: session_input_contract_evidence(
                        request,
                        message_start_index=(len(messages) - len(initial_source_messages)),
                    ),
                    **(
                        {}
                        if prompt_contribution_manifest is None
                        else {
                            "prompt_contribution_manifest": (
                                prompt_contribution_manifest.model_dump(mode="json")
                            )
                        }
                    ),
                    **(
                        {}
                        if prepared_session_authority is None
                        else {
                            "dispatch_operation_id": (
                                prepared_session_authority.dispatch_operation_id
                            ),
                            "queue_task_id": prepared_session_authority.queue_task_id,
                            "durable_subagent_submission_sha256": (
                                prepared_session_authority.submission_sha256
                            ),
                        }
                    ),
                },
                start_task_on_enter=not pre_run_task_started,
                tool_completion=request.tool_completion,
            )
        except BaseException as exc:
            try:
                await self._environment_lifecycle.abort_environment_setup(
                    session_id=session.id,
                    original_error=exc,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                )
            finally:
                try:
                    await self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                        session_id=session.id,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    )
                finally:
                    _deactivate_session_interaction(session.id)
            raise
        forwarded_stream = self._session_control.stream_with_out_of_band_events(
            session.id,
            session_stream,
        )
        billing_identity_cancellation: asyncio.CancelledError | None = None
        billing_identity_cancellation_group: BaseExceptionGroup | None = None
        propagated_cancellation: asyncio.CancelledError | None = None
        propagated_cancellation_group: BaseExceptionGroup | None = None
        try:
            async for event in forwarded_stream:
                yield event
        except asyncio.CancelledError as exc:
            billing_identity_cancellation = _detach_billing_cancellation_for_public(exc)
            if billing_identity_cancellation is None:
                propagated_cancellation = _detach_provider_cancellation_for_public(exc) or exc
        except BaseExceptionGroup as exc:
            billing_identity_cancellation_group = detach_billing_identity_cancellation_group(exc)
            if billing_identity_cancellation_group is None:
                propagated_cancellation_group = exc
        except GeneratorExit:
            # The forwarding iterator owns closure of the inner run stream.
            # Retain and close that layer so its finalizer cannot race a direct
            # close of the same inner generator.
            await _close_async_iterator(forwarded_stream)
            raise

        del forwarded_stream, session_stream
        # A registered provider can retain live credentials in its repr. Drop it
        # before any fallible interruption-store work and before raising a fresh
        # provider-free cancellation.
        del registered_provider

        if billing_identity_cancellation is not None:
            interruption_check_failed = False
            interruption_requested = False
            if self._session_control.is_interruption_request_active(session.id):
                try:
                    interruption_requested = await self._session_control.interrupt_requested(
                        session.id
                    )
                except Exception:
                    interruption_check_failed = True
            if interruption_check_failed:
                raise RuntimeError(
                    "Session interruption state check failed after provider billing cancellation"
                ) from None
            if interruption_requested:
                interruption_transition_failed = False
                try:
                    async for event in self._session_finalization.handle_session_interrupted(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    ):
                        yield event
                except Exception:
                    interruption_transition_failed = True
                if interruption_transition_failed:
                    raise RuntimeError(
                        "Session interruption transition failed after provider billing cancellation"
                    ) from None
                return
            raise billing_identity_cancellation

        if billing_identity_cancellation_group is not None:
            interruption_load_failed = False
            persisted_session = None
            if self._session_control.is_interruption_request_active(session.id):
                try:
                    persisted_session = await self.session_store.load(session.id)
                except Exception:
                    interruption_load_failed = True
            if interruption_load_failed:
                raise RuntimeError(
                    "Session interruption state load failed after provider billing cancellation"
                ) from None
            if (
                persisted_session is not None
                and persisted_session.status in _INTERRUPTIBLE_SESSION_STATUSES
            ):
                interruption_transition_failed = False
                try:
                    async for event in self._session_finalization.handle_session_interrupted(
                        session=persisted_session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    ):
                        yield event
                except Exception:
                    interruption_transition_failed = True
                if interruption_transition_failed:
                    raise RuntimeError(
                        "Session interruption transition failed after provider billing cancellation"
                    ) from None
            raise billing_identity_cancellation_group

        if propagated_cancellation is not None:
            if self._session_control.is_interruption_request_active(
                session.id
            ) and await self._session_control.interrupt_requested(session.id):
                async for event in self._session_finalization.handle_session_interrupted(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=_environment_name(registered_environment),
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                ):
                    yield event
                return
            raise propagated_cancellation

        if propagated_cancellation_group is not None:
            raise propagated_cancellation_group

    async def refresh_foreground_child_action(
        self,
        wait: ForegroundChildWait,
        event: Event,
        *,
        before_mutation: Callable[[], Awaitable[None]],
    ) -> None:
        from cayu.runtime._foreground_child_action_refresh import refresh_foreground_child_action

        agent = self._get_registered_agent(wait.parent_effect.agent_name)
        tool = agent.tools.get(wait.parent_effect.tool_name)
        if tool is None or tool.child_session_recovery is None:
            raise SessionRunFenced("Foreground action refresh lacks its registered child matcher.")
        await refresh_foreground_child_action(
            wait,
            event,
            store=self.session_store,
            writer=self._event_writer,
            matcher=tool.child_session_recovery,
            before_mutation=before_mutation,
        )

    async def compact_session(
        self,
        request: CompactSessionRequest,
        *,
        store_resolved_session_id: str | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        request = session_request_boundary.prepare_compact_session_request(
            request,
            redactor=self._secret_redactor,
            store_resolved_session_id=store_resolved_session_id,
        )
        session = await self.session_store.load(request.session_id)
        if session is not None:
            await self._require_participant_execution(session, participant_context)
        (
            requires_completion_decision,
            admission_failure,
        ) = await self.verifier_aware_task_execution_outcome(
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
        operation_stream = self._compaction.run(request)
        forwarded_stream = self._session_control.stream_with_out_of_band_events(
            request.session_id,
            operation_stream,
        )
        billing_identity_cancellation: asyncio.CancelledError | None = None
        billing_identity_cancellation_group: BaseExceptionGroup | None = None
        propagated_cancellation: asyncio.CancelledError | None = None
        try:
            async for event in forwarded_stream:
                yield event
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
        del forwarded_stream, operation_stream
        if billing_identity_cancellation is not None:
            raise billing_identity_cancellation
        if billing_identity_cancellation_group is not None:
            raise billing_identity_cancellation_group
        if propagated_cancellation is not None:
            raise propagated_cancellation

    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
        *,
        store_resolved_session_id: str | None = None,
        store_resolved_source_session_id: str | None = None,
        expected_authorized_target_instance_id: str | None = None,
    ) -> EnqueueSessionMessageResult:
        """Durably queue user steering for delivery by the active controller."""

        redacted_request = session_request_boundary.prepare_enqueue_message_request(
            request,
            redactor=self._secret_redactor,
            store_resolved_session_id=store_resolved_session_id,
            store_resolved_source_session_id=store_resolved_source_session_id,
            expected_authorized_target_instance_id=expected_authorized_target_instance_id,
        )
        await require_ordinary_session_execution(
            redacted_request.session_id,
            admit_session=False,
            task_store=self.task_store,
            redactor=self._secret_redactor,
        )
        if expected_authorized_target_instance_id is None:
            result = await self.session_store.enqueue_session_message(redacted_request)
        else:
            result = await self.session_store.enqueue_session_message(
                redacted_request,
                expected_authorized_target_instance_id=expected_authorized_target_instance_id,
            )
        if not result.replayed:
            await self._event_writer.fan_out_persisted([result.event])
        return result

    async def append_peer_content(
        self, request: PeerContentAppendRequest, *, pending_transcript_cursor: int | None = None
    ) -> PeerContentReceipt:
        """Admit peer content through the runtime session owner.

        Appending historical peer data never starts a model or tool. The
        store's target incarnation/cursor CAS remains the final mutation gate.
        """
        if type(request) is not PeerContentAppendRequest:
            raise TypeError("Peer append requires a PeerContentAppendRequest.")
        if request.append_key.creation_target is None:
            assert request.append_key.target_session_id is not None
            await require_ordinary_session_execution(
                request.append_key.target_session_id,
                admit_session=False,
                task_store=self.task_store,
                redactor=self._secret_redactor,
            )

        def qualify_target(target: Session) -> None:
            # Invoked against the exact transaction-owned target, including a
            # creation which won the race after the facade's initial read.
            profile = execution_profile_from_session_metadata(target.metadata)
            targets = [(target.provider_name, target.model)]
            if profile is not None and profile.model_failover is not None:
                targets = [
                    (candidate.provider_name, candidate.model)
                    for candidate in profile.model_failover.plan.candidates
                ]
            for provider_name, model in targets:
                provider = self._get_registered_provider(provider_name).provider
                qualification = provider.preflight_portable_messages(
                    model=model,
                    messages=[
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=(
                                request.occurrence.to_message_part(
                                    append_key=request.append_key,
                                    projection_id=request.append_key.projection_id,
                                    operation_key=request.operation_key,
                                ),
                            ),
                        )
                    ],
                    tools=[],
                )
                if qualification is not None:
                    import inspect

                    from cayu.collaboration.peer_content import PeerContentUnavailable

                    if inspect.iscoroutine(qualification):
                        qualification.close()
                    raise PeerContentUnavailable(
                        "Peer provider qualification must complete synchronously."
                    )

        return await self.session_store.append_peer_content(
            request,
            qualify_target=qualify_target,
            pending_transcript_cursor=pending_transcript_cursor,
        )

    async def _deliver_queued_session_messages(
        self,
        *,
        session_id: str,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        messages: list[Message],
        include_on_idle: bool,
        continue_active_interaction: bool = False,
        reject_only: bool = False,
        invocation_context: InvocationContext | None = None,
        predecessor_settlement_event: Event | None = None,
        before_successor_admission: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[list[Event], InvocationContext | None]:
        if invocation_context is not None and invocation_context.work_attempt is not None:
            # Acceptance may have come from another app without this TaskStore.
            # Do not publish input or a successor outside the exact admission.
            raise TaskCompletionDecisionRequired(
                "Queued steering cannot extend an admitted work attempt."
            )
        delivered_events: list[Event] = []
        # Pending peer admission requires fresh export-owner authority. The host
        # services the durable index through CayuApp.service_pending_peer_content;
        # ordinary execution must not turn historical requests into new grants.
        eligible_through: int | None = None
        profile_handoff: QueuedInteractionProfileHandoff | None = None
        if continue_active_interaction:
            delivery_interaction_id = _current_session_interaction_id(session_id)
            if delivery_interaction_id is None:
                raise RuntimeError("Queued input has no active interaction to continue.")
            interaction_started = True
            interaction_started_event = None
        else:
            if (
                invocation_context is None
                or predecessor_settlement_event is None
                or predecessor_settlement_event.type is not EventType.INTERACTION_COMPLETED
                or predecessor_settlement_event.interaction_id
                != invocation_context.binding.interaction_id
            ):
                raise SessionRunFenced(
                    "Queued interaction delivery lost its predecessor invocation settlement."
                )
            delivery_method = self.session_store.deliver_queued_session_messages
            store_type = type(delivery_method.__self__)
            delivery_owner = next(
                (
                    owner
                    for owner in type.__getattribute__(store_type, "__mro__")
                    if "deliver_queued_session_messages" in type.__getattribute__(owner, "__dict__")
                ),
                None,
            )
            if (
                getattr(
                    self.session_store,
                    "queued_interaction_profile_handoff_version",
                    None,
                )
                != 1
                or delivery_owner is None
                or (
                    type.__getattribute__(delivery_owner, "__dict__").get(
                        "queued_interaction_profile_handoff_version"
                    )
                    != 1
                )
            ):
                raise NotImplementedError(
                    "This SessionStore does not attest atomic queued interaction handoffs."
                )

            def prepare_first_delivery() -> tuple[str, QueuedInteractionProfileHandoff, Event]:
                assert invocation_context is not None and predecessor_settlement_event is not None
                interaction_id = str(uuid4())
                handoff = QueuedInteractionProfileHandoff(
                    expected_session_instance_id=session.instance_id,
                    predecessor_settlement_event_id=predecessor_settlement_event.id,
                    expected_active_profile=invocation_context.active_profile,
                    target_active_profile=ActiveInvocationExecutionProfile(
                        session_id=session.id,
                        interaction_id=interaction_id,
                        run_epoch=session.run_epoch,
                        profile=invocation_context.profile,
                    ),
                )
                return (
                    interaction_id,
                    handoff,
                    build_interaction_started_event(
                        clock=self._clock,
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        interaction_id=interaction_id,
                        queued_profile_handoff=handoff,
                    ),
                )

            interaction_started = False
            delivery_interaction_id, profile_handoff, interaction_started_event = (
                prepare_first_delivery()
            )
        delivery_batch_index = 0
        while True:
            delivery_id = (
                delivery_interaction_id
                if delivery_batch_index == 0
                else f"{delivery_interaction_id}:batch:{delivery_batch_index}"
            )
            if not continue_active_interaction and not interaction_started and not reject_only:
                # A settled predecessor/profile is not permission for a new
                # participant interaction. Recheck immediately before admission,
                # including a first delivery after a rejection-only batch.
                if before_successor_admission is None:
                    raise RuntimeError("Queued successor lacks its execution admission owner.")
                await before_successor_admission()
            try:
                try:
                    batch: SessionMessageDeliveryBatch = (
                        await self.session_store.deliver_queued_session_messages(
                            session_id,
                            include_on_idle=include_on_idle,
                            delivery_id=delivery_id,
                            eligible_through=eligible_through,
                            interaction_id=delivery_interaction_id,
                            interaction_started_event=(
                                None if interaction_started else interaction_started_event
                            ),
                            profile_handoff=(None if interaction_started else profile_handoff),
                            **({"reject_only": True} if reject_only else {}),
                        )
                    )
                except Exception as error:
                    if isinstance(error, SessionStatusConflict):
                        raise
                    batch = await self.session_store.deliver_queued_session_messages(
                        session_id,
                        include_on_idle=include_on_idle,
                        delivery_id=delivery_id,
                        eligible_through=eligible_through,
                        interaction_id=delivery_interaction_id,
                        interaction_started_event=(
                            None if interaction_started else interaction_started_event
                        ),
                        profile_handoff=(None if interaction_started else profile_handoff),
                        **({"reject_only": True} if reject_only else {}),
                    )
            except SessionStatusConflict:
                # An interrupt can win after the loop's durable status check,
                # between bounded delivery batches, or after completion detects
                # queued work. Preserve that lifecycle result through the normal
                # interruption finalizer instead of treating the store's delivery
                # fence as a runtime failure. Unrelated status conflicts still
                # propagate unchanged.
                await self._session_control.raise_if_interrupted(session_id)
                raise
            if eligible_through is None:
                eligible_through = batch.eligible_through
            messages.extend(
                queued_session_message_input(queued_message) for queued_message in batch.messages
            )
            if batch.messages and not continue_active_interaction and not interaction_started:
                active_invocation_profile = batch.active_invocation_profile
                if (
                    profile_handoff is None
                    or active_invocation_profile is None
                    or active_invocation_profile != profile_handoff.target_active_profile
                    or invocation_context is None
                    or predecessor_settlement_event is None
                    or interaction_started_event is None
                ):
                    raise SessionRunFenced(
                        "Queued delivery did not return its exact active-profile handoff."
                    )
                invocation_context = self._environment_lifecycle.accept_queued_interaction(
                    session=session,
                    invocation_context=invocation_context,
                    batch=batch,
                    profile_handoff=profile_handoff,
                    predecessor_settlement_event=predecessor_settlement_event,
                    interaction_started_event=interaction_started_event,
                )
                _activate_session_interaction(session_id, delivery_interaction_id)
            if batch.events:
                await self._event_writer.fan_out_persisted(list(batch.events))
                delivered_events.extend(batch.events)
            if batch.messages and not interaction_started:
                if interaction_started_event is None:
                    raise RuntimeError("Queued interaction start event was not prepared.")
                if not any(event.id == interaction_started_event.id for event in batch.events):
                    raise RuntimeError(
                        "Queued message delivery did not atomically persist interaction start."
                    )
                interaction_started = True
            if not batch.has_more:
                return delivered_events, invocation_context
            if not interaction_started:
                # A rejection receipt is immutable, but it did not start the
                # proposed interaction. The next batch is still a first delivery
                # and needs its own matching delivery/start/handoff identity.
                delivery_interaction_id, profile_handoff, interaction_started_event = (
                    prepare_first_delivery()
                )
                delivery_batch_index = 0
            else:
                delivery_batch_index += 1

    async def _complete_session_if_no_queued_messages(
        self,
        *,
        session: Session,
        invocation_context: InvocationContext,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        execution_profile: ExecutionProfileIdentity | None,
        task_id: str | None,
        tool_completion_result: ToolCompletionResult | None = None,
    ) -> tuple[Session, Event | None, bool, dict[str, Any] | None]:
        completion_marker: dict[str, Any] | None = None
        checkpoint_mutation: dict[str, Any] | None = None
        binding = (
            None if registered_environment is None else registered_environment.environment.binding
        )
        if (
            binding is not None
            and registered_environment is not None
            and registered_environment.bound_workspace is not None
            and binding._completion_requires_successful_finalization(
                registered_environment.bound_workspace
            )
        ):
            if execution_profile is None:
                raise RuntimeError(
                    "Completion-critical workspace finalization lost its execution profile."
                )
            (
                completion_marker,
                checkpoint_mutation,
            ) = self._environment_lifecycle.prepare_completion_finalization_transition(
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                task_id=task_id,
            )
        try:
            (
                transitioned,
                event,
                completed,
            ) = await self._session_finalization.publish_interaction_transition(
                session=session,
                invocation_context=invocation_context,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                to_status=(
                    SessionStatus.RUNNING
                    if checkpoint_mutation is not None
                    else SessionStatus.COMPLETED
                ),
                only_if_no_queued_messages=True,
                from_statuses={SessionStatus.RUNNING},
                checkpoint_mutation=checkpoint_mutation,
                tool_completion_result=tool_completion_result,
            )
            return transitioned, event, completed, completion_marker if completed else None
        except (SessionRunFenced, SessionStatusConflict):
            # An interrupt can win between the final provider response and the
            # atomic completion publication. Route that race through the normal
            # interrupt finalizer instead of the generic failure handler.
            await self._session_control.raise_if_interrupted(session.id)
            raise

    async def _handle_queued_messages_before_completion(
        self,
        *,
        finish_completion: Callable[[Session, dict[str, Any] | None], AsyncGenerator[Event, None]],
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        messages: list[Message],
        step: int,
        max_steps: int,
        run_started_at: float,
        turn_usage_tracker: SessionUsageTracker,
        active_run: ActiveSessionRun[SessionUsageTracker] | None,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None = None,
        predecessor_settlement_event: Event | None = None,
        task_id: str | None = None,
        participant_context: CollaborationAccessContext | None = None,
    ) -> AsyncGenerator[Event | _QueuedCompletionResult, None]:
        async def admit_successor() -> None:
            from cayu.runtime._producer_output_store import producer_owns_invocation

            try:
                with _invocation_lifecycle_authority_read_scope():
                    producer_checkpoint = await self.session_store.load_checkpoint(session.id)
                if producer_owns_invocation(producer_checkpoint):
                    # Preserve the completed producer's identity and leave queued
                    # input for separately authorized work after owner settlement.
                    raise SessionInterruptedByRequest(session.id)
                await self._require_participant_execution(session, participant_context)
            except asyncio.CancelledError as cancellation:
                # This read cannot dispatch a successor. The predecessor has
                # already settled, so the outer cancellation path deliberately
                # will not rewrite it. Settle only the abandoned session through
                # the existing supervised owner before propagating cancellation.
                async def finalize_admission() -> None:
                    terminal = await self._session_finalization.finalize_abandoned_session_run(
                        RecoveryAbandonedSessionRequest(
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            environment_name=environment_name,
                            run_started_at=run_started_at,
                            turn_usage_tracker=turn_usage_tracker,
                            active_run=active_run,
                            execution_profile=execution_profile,
                            invocation_context=invocation_context,
                            retain_terminal_publication_repair=True,
                        )
                    )
                    if terminal is not None:
                        _mark_session_invocation_terminal_event(terminal)

                await self._recovery_ownership.run_cleanup_steps(
                    authoritative_failure=cancellation,
                    steps=(("queued successor admission cancellation", finalize_admission),),
                )
                raise

        events: list[Event] = []
        delivery_method = self.session_store.deliver_queued_session_messages
        store_type = type(delivery_method.__self__)
        delivery_owner = next(
            (
                owner
                for owner in type.__getattribute__(store_type, "__mro__")
                if "deliver_queued_session_messages" in type.__getattribute__(owner, "__dict__")
            ),
            None,
        )
        declaration = (
            {} if delivery_owner is None else type.__getattribute__(delivery_owner, "__dict__")
        )
        version = declaration.get("session_message_lifecycle_version")
        store_version = getattr(store_type, "session_message_lifecycle_version", None)
        can_reject_only = (
            type(store_version) is int
            and store_version == 1
            and type(version) is int
            and version == 1
            and getattr(delivery_method, "__func__", None)
            is declaration.get("deliver_queued_session_messages")
        )
        while True:
            if step >= max_steps and not can_reject_only:
                # Ordinary unconditioned SDK steering does not require the new
                # lifecycle capability. Preserve its queue and existing limit stop.
                break
            message_count = len(messages)
            drained_events, rebound_context = await self._deliver_queued_session_messages(
                session_id=session.id,
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                messages=messages,
                include_on_idle=True,
                invocation_context=invocation_context,
                predecessor_settlement_event=predecessor_settlement_event,
                before_successor_admission=admit_successor,
                reject_only=step >= max_steps,
            )
            events.extend(drained_events)
            if len(messages) != message_count:
                yield _QueuedCompletionResult("continue", session, events, rebound_context)
                return
            if rebound_context is not invocation_context:
                raise SessionRunFenced("Terminal-only queue drain changed invocation authority.")

            # Expose rejection evidence before committing session completion.
            # A consumer closing here can release the unchanged predecessor
            # settlement without inventing terminal-session proof.
            for event in events:
                yield event
            events.clear()

            # The predecessor interaction already has an immutable settlement.
            # Complete only the session; replaying that settlement cannot make
            # a new empty-queue decision or install a finalization marker.
            marker: dict[str, Any] | None = None
            mutation: dict[str, Any] | None = None
            binding = (
                None
                if registered_environment is None
                else registered_environment.environment.binding
            )
            if (
                binding is not None
                and registered_environment is not None
                and registered_environment.bound_workspace is not None
                and binding._completion_requires_successful_finalization(
                    registered_environment.bound_workspace
                )
            ):
                if execution_profile is None:
                    raise RuntimeError("Completion-critical finalization lost its profile.")
                marker, mutation = (
                    self._environment_lifecycle.prepare_completion_finalization_transition(
                        registered_environment=registered_environment,
                        execution_profile=execution_profile,
                        task_id=task_id,
                    )
                )

            async def commit_and_finish_completion(
                marker: dict[str, Any] | None = marker,
                mutation: dict[str, Any] | None = mutation,
            ) -> _QueuedCompletionResult:
                # Mutation admission must atomically check the empty queue,
                # preserve epoch/profile, and fence the current store owner.
                completed_session = (
                    await self.session_store.transition_status_if_no_queued_messages(
                        session.id,
                        from_statuses={SessionStatus.RUNNING},
                        to_status=SessionStatus.RUNNING
                        if mutation is not None
                        else SessionStatus.COMPLETED,
                        **({"checkpoint_mutation": mutation} if mutation is not None else {}),
                    )
                )
                # Protect only the commit-to-terminal-proof window. Post-terminal
                # hooks keep their existing cancellation-aware owner; draining
                # them under this shield could make a cooperative hook unkillable.
                continuation = finish_completion(completed_session, marker)
                try:
                    completed_events, _ = await _collect_through_event_type(
                        continuation,
                        (
                            EventType.SESSION_COMPLETED,
                            EventType.SESSION_FAILED,
                            EventType.SESSION_INTERRUPTED,
                        ),
                        missing_message="Queued completion produced no terminal event.",
                    )
                    completed_session = await self.require_session(session.id)
                except BaseException as publication_close_failure:
                    await self._recovery_ownership.run_cleanup_steps(
                        authoritative_failure=publication_close_failure,
                        steps=(("queued completion publication close", continuation.aclose),),
                    )
                    raise
                return _QueuedCompletionResult(
                    "complete",
                    completed_session,
                    completed_events,
                    invocation_context,
                    marker,
                    continuation,
                )

            async def owned_completion() -> tuple[
                CapturedAwaitableOutcome[_QueuedCompletionResult], Event | None
            ]:
                captured = await capture_awaitable_outcome(commit_and_finish_completion)
                return captured, _current_session_invocation_terminal_event(session.id)

            continuation: AsyncGenerator[Event, None] | None = None
            try:
                # Reads before dispatch are abortable. Once dispatched, the
                # store revalidates the owner under its atomic mutation fence.
                await self._session_control.raise_if_interrupted(session.id)
                current = await self.session_store.load(session.id)
                current_profile = active_invocation_execution_profile_from_checkpoint(
                    await self.session_store.load_checkpoint(session.id)
                )
                if (
                    invocation_context is None
                    or current is None
                    or current.instance_id != session.instance_id
                    or current.run_epoch != session.run_epoch
                    or current_profile != invocation_context.active_profile
                ):
                    raise SessionRunFenced("Terminal-only completion lost invocation authority.")
                completion_task = asyncio.create_task(
                    owned_completion(), name="cayu-queued-session-completion"
                )
                outcome = await await_shielded_task_outcome(completion_task)
                if outcome.error is not None:
                    raise outcome.error
                assert outcome.result is not None
                captured, terminal_event = outcome.result
                if captured.result is not None:
                    continuation = captured.result.continuation
                if terminal_event is not None:
                    # The child owns publication; ContextVars do not merge back.
                    # Carry only its exact durable terminal proof to our finalizer.
                    _mark_session_invocation_terminal_event(terminal_event)
                if outcome.cancellation is not None:
                    cancellation = outcome.cancellation
                    current_task = asyncio.current_task()
                    requests_to_preserve = outcome.cancellation_requests_consumed + (
                        0 if current_task is None else current_task.cancelling()
                    )
                    if requests_to_preserve:
                        retain_workspace_observation_pending_cancellation_requests(
                            cancellation, requests_to_preserve
                        )
                    if captured.error is not None:
                        if isinstance(captured.error, Exception):
                            raise cancellation from captured.error
                        failure = BaseExceptionGroup(
                            "Queued completion failed with caller cancellation.",
                            [captured.error, cancellation],
                        )
                        retain_workspace_observation_pending_cancellation_requests(
                            failure, requests_to_preserve
                        )
                        raise failure from None
                    raise cancellation
                if captured.error is not None:
                    raise captured.error
                assert captured.result is not None
                assert continuation is not None
                for event in captured.result.events:
                    yield event
                async for event in continuation:
                    yield event
                yield _QueuedCompletionResult(
                    "complete", captured.result.session, [], invocation_context, marker
                )
            except SessionQueuedMessagesPending:
                if step >= max_steps:
                    break
                continue
            except (SessionRunFenced, SessionStatusConflict):
                await self._session_control.raise_if_interrupted(session.id)
                raise
            finally:
                if continuation is not None:
                    await self._recovery_ownership.run_cleanup_steps(
                        authoritative_failure=sys.exception(),
                        steps=(("queued completion continuation close", continuation.aclose),),
                    )
            return
        events.extend(
            [
                event
                async for event in self._stop_session_for_model_step_limit(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    messages=messages,
                    step=step,
                    max_steps=max_steps,
                    run_started_at=run_started_at,
                    turn_usage_tracker=turn_usage_tracker,
                    active_run=active_run,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                )
            ]
        )
        yield _QueuedCompletionResult("stopped", session, events, invocation_context)

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
                        and await self.interruption_cascade_status(loaded_session.id) == "failed"
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

    async def _latest_tool_exposure_profile_id(self, session_id: str) -> str | None:
        records = await self.session_store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.TOOL_EXPOSURE_RECORDED,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not records:
            return None

        event = records[0].event
        try:
            exposure = ToolExposure.model_validate(event.payload)
        except ValueError as exc:
            raise RuntimeError("Latest durable tool-exposure evidence is malformed.") from exc
        if not event_payload_authority_is_runtime_generated(
            event,
            field_name="profile_id",
            value=exposure.profile_id,
        ):
            raise RuntimeError("Latest durable tool-exposure profile is not runtime-attested.")
        return exposure.profile_id

    async def prepare_resume(
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
    ) -> _ResumePreparation:
        if request.failover is not None:
            self.session_store._require_model_failover_stage_protocol()
        # Resume profile admission and the resumed dispatch must observe one
        # application-budget snapshot even if the app configuration is changed
        # while the interaction-start event is being consumed.
        if (type(request) is ForegroundChildResumeRequest) != (
            required_foreground_wait is not None
        ):
            raise TypeError("Foreground continuation requires its private request and exact wait.")
        if (required_foreground_terminal is None) != (required_foreground_wait is None) or (
            required_foreground_terminal is not None
            and required_foreground_terminal.wait != required_foreground_wait
        ):
            raise TypeError("Foreground continuation requires the exact selected child terminal.")
        if (foreground_before_mutation is None) != (required_foreground_wait is None):
            raise TypeError("Foreground continuation requires its delivery ownership guard.")
        if required_foreground_continuation is not None and (
            required_foreground_continuation.terminal != required_foreground_terminal
            or required_foreground_continuation.request
            != request.model_copy(update={"loop_policies": ()})
        ):
            raise TypeError("Attached foreground continuation conflicts with its original request.")

        budget_policy = copy_budget_policy(self._get_budget_policy())
        if (source_execution_profile is None) != (required_execution_profile is None):
            raise ValueError(
                "Queued execution-profile authority requires both source and required profiles."
            )
        if (source_execution_profile is None) != (required_session_instance_fingerprint is None):
            raise ValueError(
                "Queued execution-profile authority requires a session-instance fingerprint."
            )
        if source_execution_profile is not None:
            if type(source_execution_profile) is not ExecutionProfileIdentity:
                raise TypeError("source_execution_profile must be an ExecutionProfileIdentity.")
            source_execution_profile = ExecutionProfileIdentity.model_validate(
                source_execution_profile.model_dump(mode="json")
            )
        if required_execution_profile is not None:
            if type(required_execution_profile) is not ExecutionProfileIdentity:
                raise TypeError("required_execution_profile must be an ExecutionProfileIdentity.")
            required_execution_profile = ExecutionProfileIdentity.model_validate(
                required_execution_profile.model_dump(mode="json")
            )
        if required_session_instance_fingerprint is not None and (
            len(required_session_instance_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in required_session_instance_fingerprint
            )
        ):
            raise ValueError(
                "required_session_instance_fingerprint must be a lowercase SHA-256 digest."
            )
        if terminal_event_id is not None and run_operation_id is None:
            raise ValueError("A terminal event identity requires a session run operation.")
        if queue_task_id is not None and terminal_event_id is None:
            raise ValueError("A queue task identity requires a terminal event identity.")
        if (queued_dispatch_id is None) != (run_operation_id is None):
            raise ValueError(
                "A queued dispatch identity requires a session run operation and vice versa."
            )
        (
            requires_completion_decision,
            admission_failure,
        ) = await self.verifier_aware_task_execution_outcome(
            task_id,
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
        require_secret_free_structured_output_spec(
            request.structured_output,
            redactor=self._secret_redactor,
            field_name="ResumeRequest.structured_output",
        )
        if request.profile_adoption is not None and adoption_request_fingerprint is None:
            adoption_request_fingerprint = execution_profile_adoption_request_fingerprint(
                request,
                redactor=self._secret_redactor,
            )
        loaded_session = await self.session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")
        await self._require_participant_execution(loaded_session, participant_context)
        if (
            required_task_session_instance_id is not None
            and loaded_session.instance_id != required_task_session_instance_id
        ):
            raise TaskClaimLost(
                "Attached task authority belongs to a different session incarnation."
            )
        if (
            required_session_instance_fingerprint is not None
            and _queued_dispatch_session_instance_fingerprint(loaded_session)
            != required_session_instance_fingerprint
        ):
            raise RuntimeError("Queued dispatch target session instance changed.")
        return _ResumePreparation(
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
            budget_policy=budget_policy,
            loaded_session=loaded_session,
            recovery_required=loaded_session.status
            in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING},
        )

    async def resume(self, preparation: _ResumePreparation) -> AsyncGenerator[Event, None]:
        """Execute a prepared continuation after abandoned-owner recovery settles."""
        request = preparation.request
        task_id = preparation.task_id
        start_event_payload_extra = preparation.start_event_payload_extra
        start_task_on_enter = preparation.start_task_on_enter
        adoption_request_fingerprint = preparation.adoption_request_fingerprint
        source_execution_profile = preparation.source_execution_profile
        required_execution_profile = preparation.required_execution_profile
        required_session_instance_fingerprint = preparation.required_session_instance_fingerprint
        required_task_session_instance_id = preparation.required_task_session_instance_id
        run_operation_id = preparation.run_operation_id
        terminal_event_id = preparation.terminal_event_id
        queue_task_id = preparation.queue_task_id
        queued_dispatch_id = preparation.queued_dispatch_id
        required_foreground_wait = preparation.required_foreground_wait
        required_foreground_terminal = preparation.required_foreground_terminal
        required_foreground_continuation = preparation.required_foreground_continuation
        foreground_before_mutation = preparation.foreground_before_mutation
        continuation_handoff = preparation.continuation_handoff
        participant_context = preparation.participant_context
        execution_to_wait = preparation.execution_to_wait
        budget_policy = preparation.budget_policy
        loaded_session = preparation.loaded_session

        def require_foreground_checkpoint(
            session: Session, checkpoint: dict[str, Any] | None
        ) -> None:
            if required_foreground_wait is None:
                return
            key = (
                FOREGROUND_CHILD_WAIT_KEY
                if required_foreground_continuation is None
                else FOREGROUND_PARENT_CONTINUATION_KEY
            )
            expected = (
                required_foreground_wait.model_dump(mode="json")
                if required_foreground_continuation is None
                else required_foreground_continuation.model_dump(mode="json")
            )
            if (
                session.id != required_foreground_wait.parent_effect.session_id
                or session.instance_id != required_foreground_wait.parent_effect.session_instance_id
                or checkpoint is None
                or canonical_durable_json_bytes(checkpoint.get(key), key)
                != canonical_durable_json_bytes(expected, key)
            ):
                raise SessionRunFenced(
                    "Foreground continuation lost its exact retained checkpoint."
                )

        if preparation.recovery_required:
            loaded_session = await self.session_store.load(request.session_id)
            if loaded_session is None:
                raise KeyError(f"Session not found: {request.session_id}")
        fork_initial_profile = _require_exact_fork_initial_invocation_profile(
            loaded_session,
            request,
            queued_dispatch_id=queued_dispatch_id,
        )
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
        loaded_projection_cursor = session_model_projection_cursor(loaded_session)

        stored_execution_profile = execution_profile_from_session_metadata(loaded_session.metadata)
        if stored_execution_profile.model_failover is not None:
            self.session_store._require_model_failover_stage_protocol()
        expected_failover_progress = model_failover_progress_for_session(
            session=loaded_session,
            execution_profile=stored_execution_profile,
            checkpoint=(
                await self.session_store.load_checkpoint(loaded_session.id)
                if stored_execution_profile.model_failover is not None
                else None
            ),
        )
        if expected_failover_progress is not None:
            loaded_projection_cursor = max(
                loaded_projection_cursor, expected_failover_progress.projection_cursor
            )
        registered_agent = self._get_registered_agent(loaded_session.agent_name)
        stored_tool_capability_ceiling = (
            _session_execution_profile._session_tool_capability_ceiling(
                loaded_session,
            )
        )
        effective_tool_capability_ceiling = resolve_tool_capability_ceiling(
            request.tool_capability_ceiling,
            registered_agent.tool_capabilities,
            maximum=stored_tool_capability_ceiling,
        )
        tool_capability_ceiling_narrowed = (
            effective_tool_capability_ceiling != stored_tool_capability_ceiling
        )
        if registered_agent.tool_discovery_mode is not None:
            if not self.session_store.supports_atomic_session_operation_initialization:
                raise RuntimeError(
                    "Tool discovery requires atomic session operation initialization."
                )
            if tool_capability_ceiling_narrowed:
                raise ValueError(
                    "A tool-discovery session cannot change its durable capability ceiling; "
                    "start a new session or fork with the narrower ceiling."
                )
            await _require_tool_discovery_view(
                self.session_store,
                loaded_session,
                registered_agent,
                effective_tool_capability_ceiling,
                require_empty=False,
            )
        requested_target = request.target
        target_changed = requested_target is not None and (
            requested_target.provider_name != loaded_session.provider_name
            or requested_target.model != loaded_session.model
        )
        request_projection = None
        if target_changed or loaded_projection_cursor:
            try:
                request_projection = model_target.project_portable_transcript(request.messages)
                if (
                    request_projection.provider_state_parts_dropped
                    or request_projection.thinking_parts_dropped
                ):
                    raise ValueError(
                        "New resume messages after model-target adoption cannot contain "
                        "provider state or thinking parts."
                    )
            except ValueError as exc:
                if queue_task_id is None:
                    raise
                raise _ExecutionProfileAdmissionRequestRejected(
                    "Queued dispatch target request contains non-portable provider material."
                ) from exc
        registered_provider = self._get_registered_provider(
            requested_target.provider_name
            if target_changed and requested_target is not None
            else loaded_session.provider_name
        )
        registered_environment = self._get_registered_environment_for_session(
            loaded_session.environment_name
        )

        from cayu.environments.snapshot_lifecycle import (
            _registry as execution_snapshot_registry,
        )
        from cayu.environments.snapshot_lifecycle import (
            _require_execution_snapshot_binding,
            ensure_execution_snapshot_binding,
        )

        if registered_environment is not None:
            # Reject before resume mutates controller position or appends input.
            await ensure_execution_snapshot_binding(
                self.session_store, loaded_session, registered_environment
            )

        def current_resume_profile(
            max_steps: int,
            candidate_provider: runtime_records.RegisteredProvider,
            candidate_model: str,
            runtime_identity: SessionRuntimeIdentity | None = None,
        ) -> ExecutionProfileIdentity:
            candidate = _session_execution_profile._execution_profile_identity(
                registered_agent=registered_agent,
                provider_name=candidate_provider.name,
                registered_provider=candidate_provider,
                model=candidate_model,
                durable_system_prompt=None,
                redactor=self._secret_redactor,
                registered_environment=registered_environment,
                process_identity=self._execution_profile_process_identity,
                runtime_hooks=self._runtime_hooks,
                loop_policies=self._loop_policies,
                loop_policy_execution_profile_identities=(
                    self._loop_policy_execution_profile_identities
                ),
                request_loop_policies=request.loop_policies,
                request_loop_policy_instance_identities=(
                    self._execution_profile_continuation.request_loop_policy_instance_identities(
                        request.loop_policies
                    )
                ),
                budget_policy=budget_policy,
                request_budget_limits=request.budget_limits,
                causal_budget_id=loaded_session.causal_budget_id,
                structured_output=request.structured_output,
                thinking=request.thinking,
                max_steps=max_steps,
                limits=request.limits,
                retry_policy=self._effective_retry_policy(request.retry_policy),
                tool_capability_ceiling=effective_tool_capability_ceiling,
                runtime_identity=runtime_identity,
                tool_completion=request.tool_completion,
            )
            return execution_profile_with_component(
                candidate,
                stored_execution_profile.component(
                    ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION
                ),
            )

        if request.profile_adoption is not None:
            replay_model = (
                requested_target.model
                if target_changed and requested_target is not None
                else loaded_session.model
            )
            replay_policy = request.failover
            inherited = stored_execution_profile.model_failover
            if replay_policy is None and inherited is not None and not target_changed:
                replay_policy = ModelFailoverPolicy(
                    fallbacks=tuple(
                        ModelTarget(provider_name=item.provider_name, model=item.model)
                        for item in inherited.plan.candidates[1:]
                    ),
                    max_total_attempts=inherited.plan.max_total_attempts,
                )
            if replay_policy is None:
                replay_candidate_profile = current_resume_profile(
                    request.max_steps, registered_provider, replay_model
                )
            else:
                replay_candidate_profile = (
                    execution_profile_admission.resolve_model_failover_execution_profile(
                        policy=replay_policy,
                        primary=ModelTarget(
                            provider_name=registered_provider.name, model=replay_model
                        ),
                        registered_primary=registered_provider,
                        resolve_provider=self._get_registered_provider,
                        resolve_candidate_profile=lambda provider, model: current_resume_profile(
                            request.max_steps, provider, model
                        ),
                        redactor=self._secret_redactor,
                    ).profile
                )
            replayed_profile_decision = await self._replay_execution_profile_decision(
                session=loaded_session,
                candidate_profile=replay_candidate_profile,
                intent=request.profile_adoption,
                adoption_request_fingerprint=adoption_request_fingerprint,
            )
            if replayed_profile_decision is not None:
                yield replayed_profile_decision
                return
        if loaded_session.status not in _session_checkpoint_admission._RESUMABLE_SESSION_STATUSES:
            if loaded_session.status in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}:
                from cayu.sessions.base import SessionExecutionInProgress

                raise SessionExecutionInProgress(loaded_session.id, loaded_session.status)
            raise SessionStatusConflict(
                "Session status transition not allowed: "
                f"{loaded_session.status} -> {SessionStatus.RUNNING}"
            )
        previous_tool_exposure_profile_id = await self._latest_tool_exposure_profile_id(
            loaded_session.id
        )
        if loaded_projection_cursor or self._secret_redactor.has_values:
            preflight_snapshot = await self.session_store.load_transcript_snapshot(
                loaded_session.id
            )
            preflight_transcript: list[Message] = []
            try:
                preflight_transcript = _transcript._transcript_snapshot_messages(preflight_snapshot)
                if loaded_projection_cursor:
                    _transcript._project_model_target_snapshot(
                        preflight_snapshot,
                        loaded_projection_cursor,
                    )
                if self._secret_redactor.has_values:
                    redacted_preflight, transcript_is_valid = (
                        session_request_boundary.redact_transcript(
                            preflight_transcript,
                            redactor=self._secret_redactor,
                            field_name="session.transcript",
                        )
                    )
                    redacted_preflight.clear()
                    if not transcript_is_valid:
                        raise ValueError(
                            "Session transcript contains a workload secret in execution "
                            "authority and cannot be resumed."
                        ) from None
            finally:
                preflight_transcript.clear()
                del preflight_snapshot

        run_operation_id = (
            (
                str(uuid4())
                if continuation_handoff is None
                else continuation_handoff.run_operation_id
            )
            if run_operation_id is None
            else require_clean_nonblank(run_operation_id, "run_operation_id")
        )
        continuing_execution_profile_snapshot: ActiveInvocationExecutionProfile | None = None
        egress_profile_activation: EgressAuthorityTransitionRecord | None = None
        egress_environment_handoff: EgressAuthorityAdoptionResult | None = None
        invocation_profile: ExecutionProfileIdentity | None = None

        async def require_no_incomplete_recovery_claim() -> None:
            observed_claim: tuple[str, datetime, datetime] | None = None

            def inspect(
                _session: Session,
                current_checkpoint: dict[str, Any] | None,
                store_now: datetime,
            ) -> None:
                nonlocal observed_claim
                existing = _incomplete_recovery_claim_from_checkpoint(current_checkpoint)
                if existing is not None:
                    observed_claim = (existing[0], existing[1], store_now)
                return None

            await self.session_store.transform_checkpoint_with_store_time(
                loaded_session.id,
                inspect,
            )
            if observed_claim is None:
                return
            claim_id, claim_expires_at, store_now = observed_claim
            if claim_expires_at > store_now:
                raise RuntimeError("Session has an active incomplete-session recovery operation.")
            current = await self.require_session(loaded_session.id)
            fenced = await self._recovery_ownership.fence_expired_incomplete_recovery_claim(
                session=current,
                claim_id=claim_id,
            )
            if not fenced:
                raise RuntimeError(
                    "Expired incomplete-session recovery ownership changed while resume "
                    "was fencing it; retry with current session state."
                ) from None
            current = await self.require_session(loaded_session.id)
            raise ValueError(
                "Session resume fenced an expired incomplete-session recovery owner; "
                f"retry with run epoch {current.run_epoch}."
            ) from None

        async def settle_expired_session_operation() -> None:
            def settle(
                _session: Session,
                current_checkpoint: dict[str, Any] | None,
                store_now: datetime,
            ) -> dict[str, Any] | None:
                if (
                    current_checkpoint is None
                    or _session_operation_state._SESSION_OPERATIONS_CHECKPOINT_KEY
                    not in current_checkpoint
                ):
                    return None
                updated = copy_durable_record(current_checkpoint, "checkpoint")
                operations = _session_operation_state._session_operation_state(updated)
                active_before = operations.get("active_operation_id")
                _session_operation_state._abandon_expired_session_operation(
                    operations, now=store_now
                )
                active_after = operations.get("active_operation_id")
                if active_after is not None:
                    raise RuntimeError(f"Session has an active durable operation: {active_after}")
                if active_before is None:
                    return None
                updated[_session_operation_state._SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
                return updated

            await self.session_store.transform_checkpoint_with_store_time(
                loaded_session.id,
                settle,
            )

        def validate_resumable_checkpoint(
            current_session: Session,
            current_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            if (
                request.failover is not None
                and pending_round_reader.pending_tool_round_from_checkpoint(
                    current_checkpoint,
                    redactor=self._secret_redactor,
                    runtime_session=current_session,
                )
                is not None
            ):
                execution_profile_admission.require_matching_recovery_failover_policy(
                    profile=stored_execution_profile, requested_policy=request.failover
                )
            if (
                model_failover_progress_for_session(
                    session=current_session,
                    execution_profile=stored_execution_profile,
                    checkpoint=current_checkpoint,
                )
                != expected_failover_progress
            ):
                raise SessionRunFenced("Model failover selection changed during resume admission.")
            if (
                required_session_instance_fingerprint is not None
                and _queued_dispatch_session_instance_fingerprint(current_session)
                != required_session_instance_fingerprint
            ):
                raise RuntimeError("Queued dispatch target session instance changed.")
            updated_checkpoint = (
                None
                if current_checkpoint is None
                else copy_durable_record(current_checkpoint, "checkpoint")
            )
            if _incomplete_recovery_claim_from_checkpoint(updated_checkpoint) is not None:
                raise RuntimeError("Session has an active incomplete-session recovery operation.")
            _session_checkpoint_admission._reject_pending_provider_operation_disposition(
                updated_checkpoint
            )
            if _initial_transcript_pending_interaction_id(updated_checkpoint) is not None:
                raise RuntimeError(
                    "Session setup did not publish its authoritative initial transcript; "
                    "resume fails closed. Start a new session."
                )
            if (
                updated_checkpoint is not None
                and _session_operation_state._SESSION_OPERATIONS_CHECKPOINT_KEY
                in updated_checkpoint
            ):
                operations = _session_operation_state._session_operation_state(updated_checkpoint)
                updated_checkpoint[_session_operation_state._SESSION_OPERATIONS_CHECKPOINT_KEY] = (
                    operations
                )
            active_operation_id = _session_operation_state._active_session_operation_id(
                updated_checkpoint
            )
            if active_operation_id is not None:
                raise RuntimeError(
                    f"Session has an active durable operation: {active_operation_id}"
                )
            _prompt_transition._reject_prepared_prompt_transition_intent(updated_checkpoint)
            if (
                pending_approval_reader.pending_approval_from_checkpoint(
                    updated_checkpoint,
                    redactor=self._secret_redactor,
                    consume_on_rejection=True,
                )
                is not None
            ):
                raise RuntimeError(
                    "Session has a pending tool approval. Resolve it with "
                    "resolve_tool_approval(...) before resuming with new messages."
                )
            pending_user_input, _ = user_input_lifecycle_authority_from_checkpoint(
                updated_checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
                current_run_epoch=current_session.run_epoch,
                runtime_session=current_session,
            )
            if pending_user_input is not None:
                raise RuntimeError(
                    "Session is awaiting user input. Answer it with "
                    "resolve_user_input(...) before resuming with new messages."
                )
            if target_changed or current_session.status == SessionStatus.COMPLETED:
                checkpoint_pending_round = pending_round_reader.pending_tool_round_from_checkpoint(
                    updated_checkpoint,
                    redactor=self._secret_redactor,
                    consume_on_rejection=True,
                    runtime_session=current_session,
                )
            else:
                checkpoint_pending_round = None
            if checkpoint_pending_round is not None:
                if target_changed:
                    raise RuntimeError(
                        "A session model target cannot change while tool-round recovery is pending."
                    )
                raise RuntimeError(
                    "Completed session has an inconsistent pending tool round. "
                    "Inspect or recover the session state before resuming it."
                )
            if (
                updated_checkpoint is not None
                and _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY in updated_checkpoint
            ):
                raise RuntimeError(
                    "Session has an incomplete background interruption cascade. "
                    "Retry the interruption before resuming with new messages."
                )
            return updated_checkpoint

        def claim_resumable_checkpoint(
            current_session: Session,
            current_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            updated_checkpoint = validate_resumable_checkpoint(
                current_session,
                current_checkpoint,
            )
            if invocation_profile is None:
                raise AssertionError("Session invocation admission lost its execution profile.")
            require_egress_authority_transition_compatible_with_profile(
                updated_checkpoint,
                invocation_profile.egress_authority,
            )
            if egress_profile_activation is not None:
                require_exact_egress_authority_transition(
                    updated_checkpoint,
                    egress_profile_activation,
                )
            return _checkpoint_with_session_run_operation(
                checkpoint=updated_checkpoint,
                current_session=current_session,
                operation_id=run_operation_id,
                terminal_event_id=terminal_event_id,
                queue_task_id=queue_task_id,
            )

        # Report deterministic checkpoint conflicts before claiming the session,
        # then repeat the same validation inside the atomic transition below so a
        # concurrent checkpoint update cannot bypass the guard.
        if request.failover is not None:
            recovery_checkpoint = await self.session_store.load_checkpoint(loaded_session.id)
            recovery_stage = await self._model_completion_recovery.load_model_completion_boundary(
                loaded_session
            )
            if (
                recovery_stage is not None
                or pending_round_reader.pending_tool_round_from_checkpoint(
                    recovery_checkpoint,
                    redactor=self._secret_redactor,
                    runtime_session=loaded_session,
                )
                is not None
            ):
                recovery_profile = active_invocation_execution_profile_from_checkpoint(
                    recovery_checkpoint
                )
                execution_profile_admission.require_matching_recovery_failover_policy(
                    profile=(
                        stored_execution_profile
                        if recovery_profile is None
                        else recovery_profile.profile
                    ),
                    requested_policy=request.failover,
                )
        await require_no_incomplete_recovery_claim()
        await settle_expired_session_operation()
        checkpoint = await self.session_store.load_checkpoint(loaded_session.id)
        checkpoint = await _prompt_transition._reconcile_committed_prompt_transition_intents(
            session_store=self.session_store,
            source_session_id=loaded_session.id,
            checkpoint=checkpoint,
            clock=self._clock,
            persist=False,
        )
        validate_resumable_checkpoint(loaded_session, checkpoint)
        active_invocation_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if active_invocation_profile is None and invocation_lifecycle_receipt_history_present(
            checkpoint
        ):
            raise SessionRunFenced(
                "Session resume lost its durable predecessor invocation authority."
            )
        if active_invocation_profile is not None and not (
            active_invocation_execution_profile_is_released(
                active_invocation_profile,
                session_id=loaded_session.id,
                run_epoch=loaded_session.run_epoch,
            )
        ):
            raise SessionRunFenced(
                "The previous invocation still owns terminal hooks or trailing cleanup."
            )
        # An in-flight provider dispatch has an ambiguous external outcome and
        # must reject resume without mutating the session. Check that boundary
        # before terminal-evidence reconciliation, which may perform a fenced
        # repair and advance the durable run epoch.
        model_failover: execution_profile_admission.ModelFailoverProfileResolution | None = None
        active_model_completion_boundary = (
            await self._model_completion_recovery.load_model_completion_boundary(loaded_session)
        )
        if active_model_completion_boundary is None:
            pending_model_completion = False
        else:
            continuing_profile_resolution = await self._execution_profile_continuation.resolve(
                session=loaded_session,
                checkpoint=checkpoint,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                request_failover=request.failover,
                tool_completion=request.tool_completion,
                request_loop_policies=request.loop_policies,
                budget_policy=budget_policy,
                request_budget_limits=request.budget_limits,
                structured_output=request.structured_output,
                thinking=request.thinking,
                max_steps=request.max_steps,
                limits=request.limits,
                retry_policy=self._effective_retry_policy(request.retry_policy),
                invocation_semantics_available=True,
            )
            continuing_execution_profile_snapshot = continuing_profile_resolution.snapshot
            model_failover = continuing_profile_resolution.model_failover
            pending_model_completion = (
                await self._model_completion_recovery.preflight_model_completion_boundary(
                    loaded_session,
                    registered_provider=registered_provider,
                    active_stage=active_model_completion_boundary,
                )
            )
        if target_changed and pending_model_completion:
            raise RuntimeError(
                "A session model target cannot change while model-completion recovery is pending."
            )
        (
            requires_completion_decision,
            admission_failure,
        ) = await self.verifier_aware_task_execution_outcome(
            task_id,
            session_id=request.session_id,
        )
        if admission_failure is not None:
            del request
            raise_task_store_operation_failure(admission_failure)
        if requires_completion_decision:
            del request
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        checkpoint = await self.session_store.load_checkpoint(loaded_session.id)
        checkpoint = await _prompt_transition._reconcile_committed_prompt_transition_intents(
            session_store=self.session_store,
            source_session_id=loaded_session.id,
            checkpoint=checkpoint,
            clock=self._clock,
        )
        validate_resumable_checkpoint(loaded_session, checkpoint)
        (
            loaded_session,
            checkpoint,
        ) = await self._terminal_finalization.reconcile_before_continuation(
            session=loaded_session,
            checkpoint=checkpoint,
        )
        validate_resumable_checkpoint(loaded_session, checkpoint)

        require_foreground_checkpoint(loaded_session, checkpoint)
        pending_round = pending_round_reader.pending_tool_round_from_checkpoint(checkpoint)
        continuing_recovery_boundary = (
            pending_round is not None
            or pending_model_completion
            or required_foreground_continuation is not None
        )
        if continuing_recovery_boundary and tool_capability_ceiling_narrowed:
            raise RuntimeError(
                "A tool capability ceiling cannot be narrowed while model or tool recovery "
                "is pending."
            )
        if continuing_recovery_boundary and request.profile_adoption is not None:
            raise RuntimeError(
                "An execution profile cannot be adopted while model or tool recovery is pending."
            )
        if continuing_recovery_boundary and request.tool_grants:
            raise RuntimeError(
                "Targeted tool grants cannot be issued while model or tool recovery is pending."
            )
        prepared_targeted_tool_grants = self._session_preparation._prepare_targeted_tool_grants(
            request.tool_grants,
            registered_agent=registered_agent,
            capability_ceiling=effective_tool_capability_ceiling,
        )
        candidate_runtime_identity: SessionRuntimeIdentity | None = None
        candidate_execution_profile: ExecutionProfileIdentity | None = None
        execution_profile_decision: ExecutionProfileDecision | None = None
        if not continuing_recovery_boundary:
            candidate_runtime_identity = SessionRuntimeIdentity(
                runtime_name="cayu",
                runtime_version=execution_profile_continuation._runtime_version(),
                runtime_build_provenance=current_runtime_build_provenance(),
            )
            expected_execution_profile = stored_execution_profile
            candidate_model = (
                requested_target.model
                if target_changed and requested_target is not None
                else loaded_session.model
            )

            def ordinary_resume_profile(
                max_steps: int,
                candidate_provider: runtime_records.RegisteredProvider,
                model: str | None = None,
            ) -> ExecutionProfileIdentity:
                # A resumed session executes the already-durable system
                # projection. The current AgentSpec prompt is not re-injected.
                return current_resume_profile(
                    max_steps,
                    candidate_provider,
                    candidate_model if model is None else model,
                    candidate_runtime_identity,
                )

            stored_failover = stored_execution_profile.model_failover
            if request.failover is not None or (stored_failover is not None and not target_changed):
                policy = request.failover
                if policy is None:
                    assert stored_failover is not None
                    policy = ModelFailoverPolicy(
                        fallbacks=tuple(
                            ModelTarget(provider_name=entry.provider_name, model=entry.model)
                            for entry in stored_failover.plan.candidates[1:]
                        ),
                        max_total_attempts=stored_failover.plan.max_total_attempts,
                    )
                model_failover = (
                    execution_profile_admission.resolve_model_failover_execution_profile(
                        policy=policy,
                        primary=ModelTarget(
                            provider_name=registered_provider.name, model=candidate_model
                        ),
                        registered_primary=registered_provider,
                        resolve_provider=self._get_registered_provider,
                        resolve_candidate_profile=lambda provider, model: ordinary_resume_profile(
                            request.max_steps, provider, model
                        ),
                        redactor=self._secret_redactor,
                    )
                )
                candidate_execution_profile = model_failover.profile
            else:
                candidate_execution_profile = ordinary_resume_profile(
                    request.max_steps, registered_provider
                )
            built_in_model_target_transition = False
            if target_changed:
                try:
                    source_registered_provider = self._get_registered_provider(
                        loaded_session.provider_name
                    )
                except (KeyError, ValueError):
                    source_registered_provider = None
                if source_registered_provider is not None:
                    source_registration_profile = _session_execution_profile._execution_profile_identity(
                        registered_agent=registered_agent,
                        provider_name=source_registered_provider.name,
                        registered_provider=source_registered_provider,
                        model=loaded_session.model,
                        durable_system_prompt=None,
                        redactor=self._secret_redactor,
                        registered_environment=registered_environment,
                        process_identity=self._execution_profile_process_identity,
                        runtime_hooks=self._runtime_hooks,
                        loop_policies=self._loop_policies,
                        loop_policy_execution_profile_identities=(
                            self._loop_policy_execution_profile_identities
                        ),
                        request_loop_policies=request.loop_policies,
                        request_loop_policy_instance_identities=(
                            self._execution_profile_continuation.request_loop_policy_instance_identities(
                                request.loop_policies
                            )
                        ),
                        budget_policy=budget_policy,
                        request_budget_limits=request.budget_limits,
                        causal_budget_id=loaded_session.causal_budget_id,
                        structured_output=request.structured_output,
                        thinking=request.thinking,
                        max_steps=request.max_steps,
                        limits=request.limits,
                        retry_policy=self._effective_retry_policy(request.retry_policy),
                        tool_capability_ceiling=stored_tool_capability_ceiling,
                        tool_completion=request.tool_completion,
                    )
                    source_registration_profile = execution_profile_with_component(
                        source_registration_profile,
                        expected_execution_profile.component(
                            ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION
                        ),
                    )
                    built_in_model_target_transition = (
                        source_registration_profile == expected_execution_profile
                    )
            if required_execution_profile is not None:
                assert source_execution_profile is not None
                if target_changed and expected_execution_profile != source_execution_profile:
                    raise ExecutionProfileMismatchError(
                        session_id=loaded_session.id,
                        expected_profile_fingerprint=(source_execution_profile.fingerprint),
                        candidate_profile_fingerprint=(expected_execution_profile.fingerprint),
                        changed_component_classes=changed_execution_profile_components(
                            source_execution_profile,
                            expected_execution_profile,
                        ),
                        expected_profile=source_execution_profile,
                        candidate_profile=expected_execution_profile,
                    )
                if candidate_execution_profile != required_execution_profile:
                    raise ExecutionProfileMismatchError(
                        session_id=loaded_session.id,
                        expected_profile_fingerprint=(required_execution_profile.fingerprint),
                        candidate_profile_fingerprint=(candidate_execution_profile.fingerprint),
                        changed_component_classes=changed_execution_profile_components(
                            required_execution_profile,
                            candidate_execution_profile,
                        ),
                        expected_profile=required_execution_profile,
                        candidate_profile=candidate_execution_profile,
                    )
            changed_profile_components = changed_execution_profile_components(
                expected_execution_profile,
                candidate_execution_profile,
            )
            if (
                fork_initial_profile is not None
                and fork_initial_profile != candidate_execution_profile
            ):
                raise ExecutionProfileMismatchError(
                    session_id=loaded_session.id,
                    expected_profile_fingerprint=fork_initial_profile.fingerprint,
                    candidate_profile_fingerprint=candidate_execution_profile.fingerprint,
                    changed_component_classes=changed_execution_profile_components(
                        fork_initial_profile,
                        candidate_execution_profile,
                    ),
                    expected_profile=fork_initial_profile,
                    candidate_profile=candidate_execution_profile,
                )
            if fork_initial_profile is not None and changed_profile_components:
                execution_profile_decision = _execution_profile_decision_event(
                    session=loaded_session,
                    expected_profile=expected_execution_profile,
                    candidate_profile=candidate_execution_profile,
                    changed_component_classes=changed_profile_components,
                    kind=ExecutionProfileDecisionKind.ADOPTED,
                    policy_identity=_FORK_INITIAL_PROFILE_POLICY_ID,
                    policy_reason=(
                        "The exact invocation was frozen with the durable session fork "
                        "before child creation."
                    ),
                    authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
                    intent=None,
                    adoption_request_fingerprint=None,
                    fallback_actor=_fork_initial_profile_actor(),
                    fallback_reason=("Apply the exact invocation admitted with the session fork."),
                    clock=(
                        self._clock
                        if continuation_handoff is None
                        else lambda: continuation_handoff.interaction_started_at
                    ),
                )
            else:
                execution_profile_decision = await self._profile_classifier.classify(
                    session=loaded_session,
                    expected_profile=expected_execution_profile,
                    candidate_profile=candidate_execution_profile,
                    changed_component_classes=changed_profile_components,
                    intent=request.profile_adoption,
                    adoption_request_fingerprint=adoption_request_fingerprint,
                    target_changed=target_changed,
                    target_provider_name=registered_provider.name,
                    target_model=(
                        requested_target.model
                        if target_changed and requested_target is not None
                        else loaded_session.model
                    ),
                    built_in_model_target_transition=(built_in_model_target_transition),
                    tool_capability_ceiling_narrowed=(tool_capability_ceiling_narrowed),
                    decision_clock=(
                        None
                        if continuation_handoff is None
                        else lambda: continuation_handoff.interaction_started_at
                    ),
                )
            # The decision is both admission authority and durable audit evidence.
            # Require the normal publication boundary to preserve it exactly before
            # an application cutover handler or the atomic admission store can see it.
            # This rejects workload secrets in policy/caller audit fields instead of
            # letting a later serializer or transition record persist them raw.
            self._event_writer.prepare_exact_replay(execution_profile_decision.event)
            if execution_profile_decision.kind in {
                ExecutionProfileDecisionKind.MIGRATION_REQUIRED,
                ExecutionProfileDecisionKind.REJECTED,
            }:
                rejection = await self.session_store.apply_invocation_lifecycle_command(
                    RejectInvocationCommand(
                        session_id=loaded_session.id,
                        expected_session_instance_id=loaded_session.instance_id,
                        expected_statuses=tuple(
                            _session_checkpoint_admission._RESUMABLE_SESSION_STATUSES
                        ),
                        expected_run_epoch=loaded_session.run_epoch,
                        expected_profile=expected_execution_profile,
                        candidate_profile=candidate_execution_profile,
                        event=execution_profile_decision.event,
                        decision=execution_profile_decision,
                    )
                )
                if not rejection.replayed:
                    await self._event_writer.fan_out_persisted([rejection.event])
                error_type = (
                    ExecutionProfileMigrationRequired
                    if execution_profile_decision.kind
                    is ExecutionProfileDecisionKind.MIGRATION_REQUIRED
                    else (
                        ExecutionProfileAdoptionRejected
                        if request.profile_adoption is not None
                        else ExecutionProfileMismatchError
                    )
                )
                raise error_type(
                    session_id=loaded_session.id,
                    expected_profile_fingerprint=expected_execution_profile.fingerprint,
                    candidate_profile_fingerprint=candidate_execution_profile.fingerprint,
                    changed_component_classes=changed_profile_components,
                    expected_profile=expected_execution_profile,
                    candidate_profile=candidate_execution_profile,
                )
        if continuing_recovery_boundary:
            continuing_profile_resolution = await self._execution_profile_continuation.resolve(
                session=loaded_session,
                checkpoint=checkpoint,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                request_failover=request.failover,
                tool_completion=request.tool_completion,
                request_loop_policies=request.loop_policies,
                budget_policy=budget_policy,
                request_budget_limits=request.budget_limits,
                structured_output=request.structured_output,
                thinking=request.thinking,
                max_steps=request.max_steps,
                limits=request.limits,
                retry_policy=self._effective_retry_policy(request.retry_policy),
                invocation_semantics_available=True,
                frozen_candidate_profile=(
                    None
                    if continuing_execution_profile_snapshot is None
                    else continuing_execution_profile_snapshot.profile
                ),
            )
            continuing_execution_profile_snapshot = continuing_profile_resolution.snapshot
            if continuing_profile_resolution.model_failover is not None:
                model_failover = continuing_profile_resolution.model_failover
            if (
                required_execution_profile is not None
                and continuing_execution_profile_snapshot.profile != required_execution_profile
            ):
                raise ExecutionProfileMismatchError(
                    session_id=loaded_session.id,
                    expected_profile_fingerprint=required_execution_profile.fingerprint,
                    candidate_profile_fingerprint=(
                        continuing_execution_profile_snapshot.profile.fingerprint
                    ),
                    changed_component_classes=changed_execution_profile_components(
                        required_execution_profile,
                        continuing_execution_profile_snapshot.profile,
                    ),
                    expected_profile=required_execution_profile,
                    candidate_profile=continuing_execution_profile_snapshot.profile,
                )
        # Provider-owned schema validation runs only after the current process's
        # execution profile has been frozen and any ordinary-resume drift has
        # been rejected. It remains before the durable status transition.
        try:
            _require_native_structured_output_support(
                request.structured_output, registered_provider=registered_provider
            )
        except (
            NativeStructuredOutputSchemaInvalid,
            NativeStructuredOutputUnsupported,
        ) as exc:
            if queue_task_id is None:
                raise
            raise _ExecutionProfileAdmissionRequestRejected(
                "Queued dispatch request failed deterministic provider preflight."
            ) from exc
        existing_deferred_input = await self.session_store.load_deferred_interaction_input(
            loaded_session.id
        )
        if target_changed and (pending_round is not None or existing_deferred_input is not None):
            raise RuntimeError(
                "A session model target cannot change while tool-round recovery or deferred "
                "interaction input is pending."
            )
        source_transcript: list[Message] | None = None
        source_transcript_digest: str | None = None
        source_transcript_cursor: int | None = None
        provider_state_parts_dropped = 0
        thinking_parts_dropped = 0
        if target_changed:
            if requested_target is None:
                raise AssertionError("Changed model target lost its requested identity.")
            source_snapshot = await self.session_store.load_transcript_snapshot(loaded_session.id)
            try:
                source_transcript = _transcript._transcript_snapshot_messages(source_snapshot)
                source_transcript_cursor = source_snapshot.cursor
            finally:
                del source_snapshot
            portable_projection = None
            try:
                source_transcript_digest = session_input_messages_sha256(source_transcript)
                portable_projection = model_target.project_portable_transcript(source_transcript)
                provider_state_parts_dropped = portable_projection.provider_state_parts_dropped
                thinking_parts_dropped = portable_projection.thinking_parts_dropped
                if request_projection is None:
                    raise AssertionError("Changed model target lost its request projection.")
                portable_source_messages, transcript_is_valid = (
                    session_request_boundary.redact_transcript(
                        list(portable_projection.messages),
                        redactor=self._secret_redactor,
                        field_name="session.transcript",
                    )
                )
                if not transcript_is_valid:
                    raise ValueError(
                        "Session transcript contains a workload secret in execution authority "
                        "and cannot be resumed."
                    ) from None
                portable_messages = [
                    *portable_source_messages,
                    *request_projection.messages,
                ]
                portable_source_messages.clear()
                validate_context_messages(portable_messages)
                hook_messages = _model_request_messages(
                    messages=[detach_message(message) for message in portable_messages],
                    structured_output=request.structured_output,
                )
                hook_tools: list[dict[str, Any]] = []
                portable_messages.clear()
                try:
                    hook_tools = _model_request_tools(
                        tool_exposure=_tool_capability_ceiling_exposure(
                            registered_agent,
                            effective_tool_capability_ceiling.tool_names,
                        ),
                        structured_output=request.structured_output,
                        targeted_tool_projection=resolve_targeted_tool_projection(
                            registered_agent.targeted_tool_mode,
                            provider=registered_provider.provider,
                            model=requested_target.model,
                        ),
                        tool_discovery_mode=registered_agent.tool_discovery_mode,
                    )
                    registered_provider.provider.preflight_portable_messages(
                        model=requested_target.model,
                        messages=hook_messages,
                        tools=hook_tools,
                    )
                finally:
                    hook_messages.clear()
                    hook_tools.clear()
            except ValueError as exc:
                if queue_task_id is None:
                    raise
                raise _ExecutionProfileAdmissionRequestRejected(
                    "Queued dispatch target provider rejected its portable request material."
                ) from exc
            finally:
                source_transcript.clear()
                portable_projection = None
        if continuing_recovery_boundary:
            interaction_id = await self._session_finalization.activate_latest_open_interaction(
                loaded_session.id
            )
            if interaction_id is None:
                raise RuntimeError(
                    "Pending model or tool recovery state has no open interaction. "
                    "Pre-interaction prerelease recovery state is unsupported."
                )
            if required_foreground_wait is not None and (
                interaction_id != required_foreground_wait.parent_effect.interaction_id
            ):
                raise SessionRunFenced("Foreground continuation cannot replace its interaction.")
            interaction_started_event = None
        else:
            interaction_id = (
                continuation_handoff.interaction_id
                if continuation_handoff is not None
                else str(uuid4())
            )
            interaction_started_event = build_interaction_started_event(
                clock=self._clock,
                session=loaded_session,
                registered_agent=registered_agent,
                environment_name=_environment_name(registered_environment),
                interaction_id=interaction_id,
                targeted_tool_grants=prepared_targeted_tool_grants,
                event_id=(
                    None
                    if continuation_handoff is None
                    else continuation_handoff.interaction_started_event_id
                ),
                started_at=(
                    None
                    if continuation_handoff is None
                    else continuation_handoff.interaction_started_at
                ),
            )
        model_transition = None
        model_transition_event = None
        if target_changed:
            if (
                requested_target is None
                or source_transcript_digest is None
                or source_transcript_cursor is None
            ):
                raise AssertionError("Changed model target lost its portable projection.")
            model_transition_event = event_with_runtime_payload_authority(
                event_with_runtime_envelope_authority(
                    event_with_runtime_generated_id(
                        Event(
                            id=(
                                str(uuid4())
                                if continuation_handoff is None
                                else continuation_handoff.model_transition_event_id
                            ),
                            type=EventType.SESSION_MODEL_SWITCHED,
                            session_id=loaded_session.id,
                            interaction_id=interaction_id,
                            timestamp=(
                                interaction_started_event.timestamp
                                if interaction_started_event is not None
                                else self._clock()
                            ),
                            agent_name=registered_agent.spec.name,
                            environment_name=_environment_name(registered_environment),
                            payload={
                                "source_provider_name": loaded_session.provider_name,
                                "source_model": loaded_session.model,
                                "target_provider_name": requested_target.provider_name,
                                "target_model": requested_target.model,
                                "provider_changed": (
                                    loaded_session.provider_name != requested_target.provider_name
                                ),
                                "model_changed": loaded_session.model != requested_target.model,
                                "provider_state_parts_dropped": provider_state_parts_dropped,
                                "thinking_parts_dropped": thinking_parts_dropped,
                                "source_transcript_cursor": source_transcript_cursor,
                                "cache_state_dropped": True,
                                "full_transcript_projection": True,
                            },
                        )
                    ),
                    "session_id",
                    "interaction_id",
                ),
                "source_provider_name",
                "source_model",
                "target_provider_name",
                "target_model",
            )
            model_transition = SessionModelTransition(
                target=requested_target,
                event=model_transition_event,
                source_transcript_digest=source_transcript_digest,
                source_transcript_cursor=source_transcript_cursor,
            )
        if (
            existing_deferred_input is not None
            and existing_deferred_input.interaction_id != interaction_id
        ):
            if continuing_recovery_boundary:
                _deactivate_session_interaction(loaded_session.id)
            raise RuntimeError("Deferred interaction input belongs to another interaction.")
        targeted_tool_projection = resolve_targeted_tool_projection(
            registered_agent.targeted_tool_mode,
            provider=registered_provider.provider,
            model=(
                requested_target.model
                if target_changed and requested_target is not None
                else loaded_session.model
            ),
        )
        targeted_projection_marker = (
            targeted_tool_projection_marker_message(
                prepared_targeted_tool_grants,
                interaction_id=interaction_id,
                generation_id=targeted_tool_view_generation_id(
                    session_id=loaded_session.id,
                    root_invocation_id=loaded_session.invocation.root_invocation_id,
                ),
            )
            if not continuing_recovery_boundary
            and prepared_targeted_tool_grants
            and targeted_tool_projection is TargetedToolProjectionKind.OPENAI_ADDITIONAL_TOOLS
            else None
        )
        interaction_source_messages = [
            *(
                existing_deferred_input.source_messages
                if existing_deferred_input is not None
                else []
            ),
            *request.messages,
            *(() if targeted_projection_marker is None else (targeted_projection_marker,)),
        ]
        invocation_profile = (
            continuing_execution_profile_snapshot.profile
            if continuing_execution_profile_snapshot is not None
            else candidate_execution_profile
        )
        if invocation_profile is None:
            raise AssertionError("Session invocation admission produced no execution profile.")
        egress_environment_handoff = await self._require_active_egress_profile_cutover(
            session=loaded_session,
            decision=execution_profile_decision,
        )
        parked_egress_factory_result = None
        if (
            egress_environment_handoff is None
            and self._egress_authority_adoption_handler is not None
            and invocation_profile.egress_authority is not None
            and (environment_name := _environment_name(registered_environment)) is not None
        ):
            try:
                retained_fingerprint = (
                    await self._environment_lifecycle.require_live_allocation_fingerprint(
                        session_id=loaded_session.id,
                        environment_name=environment_name,
                    )
                )
            except (TypeError, ValueError, RuntimeError):
                # A session with no retained process-local allocation follows
                # the ordinary durable reconnect path below.
                retained_fingerprint = None
            if retained_fingerprint is not None:
                parked_egress_factory_result = _find_parked_egress_authority_allocation(
                    self._egress_authority_adoption_handler,
                    session_id=loaded_session.id,
                    environment_name=environment_name,
                    expected_fingerprint=retained_fingerprint,
                )
        egress_profile_activation = (
            None if egress_environment_handoff is None else egress_environment_handoff.transition
        )
        # Egress cutover is a durable mutation that may advance the checkpoint
        # before invocation admission.  Admission must bind the resulting
        # state, rather than the snapshot loaded before that mutation.  A
        # competing write after this reload is still rejected by the atomic
        # command's expected checkpoint digest.
        checkpoint = await self.session_store.load_checkpoint(loaded_session.id)
        checkpoint = await _prompt_transition._reconcile_committed_prompt_transition_intents(
            session_store=self.session_store,
            source_session_id=loaded_session.id,
            checkpoint=checkpoint,
            clock=self._clock,
            persist=False,
        )
        admission_source_active_profile = active_invocation_execution_profile_from_checkpoint(
            checkpoint
        )
        require_foreground_checkpoint(loaded_session, checkpoint)
        if registered_environment is not None:
            # Admission below binds this checkpoint digest, closing the race
            # with a snapshot reservation after the initial read-only check.
            _require_execution_snapshot_binding(
                execution_snapshot_registry(checkpoint), registered_environment
            )
        claimed_checkpoint = claim_resumable_checkpoint(loaded_session, checkpoint)
        if required_foreground_terminal is not None and required_foreground_continuation is None:
            existing_terminal = claimed_checkpoint.get(FOREGROUND_CHILD_TERMINAL_KEY)
            selected_terminal = required_foreground_terminal.model_dump(mode="json")
            if existing_terminal is not None and existing_terminal != selected_terminal:
                raise SessionRunFenced(
                    "Foreground continuation selected a different child outcome."
                )
            claimed_checkpoint[FOREGROUND_CHILD_TERMINAL_KEY] = selected_terminal
        target_active_profile = ActiveInvocationExecutionProfile(
            session_id=loaded_session.id,
            interaction_id=interaction_id,
            run_epoch=loaded_session.run_epoch + 1,
            profile=invocation_profile,
        )
        invocation_profile = target_active_profile.profile
        target_model = (
            requested_target.model
            if target_changed and requested_target is not None
            else loaded_session.model
        )
        adopted_runtime_identity: SessionRuntimeIdentity | None = None
        if (
            execution_profile_decision is not None
            and execution_profile_decision.kind is ExecutionProfileDecisionKind.ADOPTED
            and ExecutionProfileComponentClass.RUNTIME
            in execution_profile_decision.changed_component_classes
        ):
            if candidate_runtime_identity is None:
                raise AssertionError(
                    "Runtime profile adoption lost its complete candidate identity."
                )
            adopted_runtime_identity = copy_session_runtime_identity(candidate_runtime_identity)
        prepared_invocation_context = _authenticated_invocation_context(
            active_profile=target_active_profile,
            binding=PreparedInvocationBinding(
                session_id=loaded_session.id,
                session_instance_id=loaded_session.instance_id,
                interaction_id=interaction_id,
                run_epoch=loaded_session.run_epoch + 1,
                agent_name=loaded_session.agent_name,
                provider_name=registered_provider.name,
                model=target_model,
                runtime_name=(
                    loaded_session.runtime_name
                    if adopted_runtime_identity is None
                    else adopted_runtime_identity.runtime_name
                ),
                runtime_version=(
                    loaded_session.runtime_version
                    if adopted_runtime_identity is None
                    else adopted_runtime_identity.runtime_version
                ),
                runtime_build_provenance=(
                    loaded_session.runtime_build_provenance
                    if adopted_runtime_identity is None
                    else adopted_runtime_identity.runtime_build_provenance
                ),
                environment_name=_environment_name(registered_environment),
            ),
            validated_profile=invocation_profile,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            runtime_hooks=self._runtime_hooks,
            loop_policies=self._loop_policies,
            request_loop_policies=request.loop_policies,
            budget_policy=budget_policy,
            tool_capability_ceiling=effective_tool_capability_ceiling,
            targeted_tool_grants=prepared_targeted_tool_grants,
        )
        try:
            if foreground_before_mutation is not None:
                await foreground_before_mutation()
            admission_command = AdmitInvocationCommand(
                session_id=loaded_session.id,
                expected_session_instance_id=loaded_session.instance_id,
                expected_statuses=tuple(_session_checkpoint_admission._RESUMABLE_SESSION_STATUSES),
                expected_run_epoch=loaded_session.run_epoch,
                expected_checkpoint_sha256=invocation_checkpoint_state_sha256(checkpoint),
                target_active_profile=target_active_profile,
                checkpoint_patch=InvocationCheckpointPatch(
                    mutation=runtime_publication_checkpoint_mutation(checkpoint, claimed_checkpoint)
                ),
                tool_capability_ceiling=effective_tool_capability_ceiling,
                interaction_started_event=interaction_started_event,
                continued_interaction_id=(interaction_id if continuing_recovery_boundary else None),
                interaction_source_messages=tuple(interaction_source_messages),
                defer_interaction_source=continuing_recovery_boundary,
                model_transition=model_transition,
                execution_profile_decision=execution_profile_decision,
                adopted_runtime_identity=adopted_runtime_identity,
                expected_active_profile=admission_source_active_profile,
            )
            if execution_to_wait is not None:
                if continuation_handoff is not None:
                    raise PermissionError(
                        "External parking cannot replace another admission handoff."
                    )
                admission_result = await execution_to_wait.admit_resume(
                    admission_command, prepared_invocation_context
                )
            elif continuation_handoff is None:
                admission_result = await self.session_store.apply_invocation_lifecycle_command(
                    admission_command
                )
            else:
                admission_result = await continuation_handoff.admit(
                    admission_command, prepared_invocation_context
                )
            if type(admission_result) is not InvocationMutationResult:
                raise RuntimeError("Invocation admission returned an invalid result.")
            if continuation_handoff is not None and admission_result.replayed:
                if continuing_recovery_boundary:
                    _deactivate_session_interaction(loaded_session.id)
                return
            session = admission_result.session
        except (SessionRunFenced, SessionStatusConflict, ValueError) as admission_error:
            if egress_profile_activation is not None:
                current_egress_transition = await SessionCheckpointEgressAuthorityTransitionStore(
                    self.session_store
                ).load(loaded_session.id)
                if current_egress_transition != egress_profile_activation:
                    if continuing_recovery_boundary:
                        _deactivate_session_interaction(loaded_session.id)
                    raise EgressAuthorityTransitionConflict(
                        "The exact egress authority transition changed before invocation admission."
                    ) from admission_error
            if request.profile_adoption is not None and candidate_execution_profile is not None:
                current_session = await self.session_store.load(loaded_session.id)
                if current_session is not None:
                    replayed_profile_decision = await self._replay_execution_profile_decision(
                        session=current_session,
                        candidate_profile=candidate_execution_profile,
                        intent=request.profile_adoption,
                        adoption_request_fingerprint=adoption_request_fingerprint,
                    )
                    if replayed_profile_decision is not None:
                        yield replayed_profile_decision
                        return
            if continuing_recovery_boundary:
                _deactivate_session_interaction(loaded_session.id)
            raise
        except BaseException:
            if continuing_recovery_boundary:
                _deactivate_session_interaction(loaded_session.id)
            raise
        invocation_context = prepared_invocation_context.with_admitted_session(session)
        if task_id is not None and required_task_session_instance_id is not None:
            try:
                await self._session_tasks.require_linked_continuation_task_authority(
                    task_id=task_id,
                    task_worker_id=request.task_worker_id,
                    task_handoff_id=request.task_handoff_id,
                    session_id=session.id,
                    session_instance_id=required_task_session_instance_id,
                )
            except BaseException as authority_failure:
                try:
                    await self._recovery_ownership.run_cleanup_steps(
                        authoritative_failure=authority_failure,
                        steps=(
                            (
                                "resume task-authority abandonment",
                                lambda: self._session_finalization.finalize_abandoned_session_by_id(
                                    session.id,
                                    registered_agent=registered_agent,
                                    registered_environment=registered_environment,
                                    execution_profile=invocation_profile,
                                    invocation_context=invocation_context,
                                    run_terminal_hooks=False,
                                ),
                            ),
                            (
                                "resume task-authority run-fence release",
                                lambda: (
                                    self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                                        session_id=session.id,
                                        execution_profile=invocation_profile,
                                        invocation_context=invocation_context,
                                    )
                                ),
                            ),
                        ),
                    )
                finally:
                    if continuing_recovery_boundary:
                        _deactivate_session_interaction(session.id)
                raise
        if not continuing_recovery_boundary:
            if interaction_id is None:
                raise RuntimeError("New interaction admission produced no interaction identity.")
            _activate_session_interaction(session.id, interaction_id)
        try:
            if continuation_handoff is not None:
                await continuation_handoff.after_admission(invocation_context)
            (
                issued_targeted_tool_grant_records,
                targeted_tool_grant_events,
            ) = await self._issue_targeted_tool_grants(
                session=session,
                interaction_id=interaction_id,
                task_id=task_id,
                prepared=prepared_targeted_tool_grants,
                issued_at=(
                    self._clock()
                    if interaction_started_event is None
                    else interaction_started_event.timestamp
                ),
            )
            (
                reconstructed_targeted_tool_grant_records,
                reconstructed_targeted_tool_grant_events,
            ) = (
                await self._reconstruct_targeted_tool_grants(
                    session=session,
                    interaction_id=interaction_id,
                    task_id=task_id,
                    registered_agent=registered_agent,
                    capability_ceiling=effective_tool_capability_ceiling,
                    observed_at=self._clock(),
                    preserve_native_projection_snapshot=(
                        targeted_tool_projection
                        is TargetedToolProjectionKind.OPENAI_ADDITIONAL_TOOLS
                    ),
                )
                if continuing_recovery_boundary
                else ((), ())
            )
            targeted_tool_grant_records = (
                reconstructed_targeted_tool_grant_records
                if continuing_recovery_boundary
                else issued_targeted_tool_grant_records
            )
            if session.run_epoch != loaded_session.run_epoch + 1:
                # Another complete invocation won admission after our preflight
                # snapshot. Refresh the adjacent profile while this invocation
                # owns the run fence so phase continuity follows durable order.
                previous_tool_exposure_profile_id = await self._latest_tool_exposure_profile_id(
                    session.id
                )
            admission_events = [
                event
                for event in (
                    (
                        None
                        if execution_profile_decision is None
                        else execution_profile_decision.event
                    ),
                    model_transition_event,
                    interaction_started_event,
                )
                if event is not None
            ]
            if admission_events:
                await self._event_writer.fan_out_persisted(admission_events)
                for admission_event in admission_events:
                    # Exact reuse is durable audit evidence but not a new public
                    # resume-stream boundary. Keep the long-standing stream
                    # ordering for ordinary exact resumes while still exposing
                    # changed-profile decisions and explicit idempotent replays.
                    if (
                        execution_profile_decision is not None
                        and execution_profile_decision.kind
                        is ExecutionProfileDecisionKind.EXACT_REUSE
                        and admission_event.id == execution_profile_decision.event.id
                    ):
                        continue
                    yield admission_event
            if targeted_tool_grant_events:
                await self._event_writer.fan_out_persisted(list(targeted_tool_grant_events))
                for targeted_tool_grant_event_record in targeted_tool_grant_events:
                    yield targeted_tool_grant_event_record
            for reconstructed_event in reconstructed_targeted_tool_grant_events:
                yield reconstructed_event
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
            for recovery_event in model_boundary.recovery_events:
                yield copy_event(recovery_event)
            if (
                model_boundary.state == "provider_operation_reconciled"
                and model_boundary.completion_event is not None
            ):
                yield copy_event(model_boundary.completion_event)
            if model_boundary.state in {
                "provider_operation_pending",
                "provider_operation_unavailable",
            }:
                # A queued dispatch must retain its exact queue/session handoff
                # until either provider reconciliation publishes the pinned
                # terminal event or terminal-evidence recovery transfers it to
                # a receipt. Clearing it before the following status write can
                # make acknowledgement loss look like a never-admitted failure
                # and terminally fail the queue task while provider work lives.
                if queue_task_id is None:
                    await self._terminal_event_publication.clear_run_operation(
                        session_id=session.id,
                        operation_id=run_operation_id,
                    )
                (
                    session,
                    pending_interaction_event,
                    _,
                ) = await self._session_finalization.publish_sibling_interaction_transition(
                    session=session,
                    invocation_context=invocation_context,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=_environment_name(registered_environment),
                    to_status=SessionStatus.INTERRUPTED,
                    execution_profile=invocation_profile,
                )
                if pending_interaction_event is not None:
                    yield pending_interaction_event
                await self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                    session_id=session.id,
                    execution_profile=invocation_profile,
                    invocation_context=invocation_context,
                )
                _deactivate_session_interaction(session.id)
                return
            projection_cursor = max(
                session_model_projection_cursor(session), loaded_projection_cursor
            )
            admitted_selection = model_failover_progress_for_session(
                session=session,
                execution_profile=invocation_profile,
                checkpoint=(
                    await self.session_store.load_checkpoint(session.id)
                    if model_failover is not None
                    else None
                ),
            )
            if admitted_selection is not None:
                projection_cursor = max(projection_cursor, admitted_selection.projection_cursor)
            if projection_cursor:
                transcript_snapshot = await self.session_store.load_transcript_snapshot(session.id)
                try:
                    transcript = list(
                        _transcript._project_model_target_snapshot(
                            transcript_snapshot,
                            projection_cursor,
                        ).messages
                    )
                finally:
                    del transcript_snapshot
            else:
                transcript = await self.session_store.load_transcript(session.id)
            transcript, transcript_is_valid = session_request_boundary.redact_transcript(
                transcript,
                redactor=self._secret_redactor,
                field_name="session.transcript",
            )
            if not transcript_is_valid:
                raise ValueError(
                    "Session transcript contains a workload secret in execution authority "
                    "and cannot be resumed."
                ) from None
        except asyncio.CancelledError as cancellation:
            try:
                await self._recovery_ownership.run_cleanup_steps(
                    authoritative_failure=cancellation,
                    steps=(
                        (
                            "cancelled resume setup finalization",
                            lambda: self._session_finalization.finalize_abandoned_session_by_id(
                                session.id,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                execution_profile=invocation_profile,
                                invocation_context=invocation_context,
                            ),
                        ),
                        (
                            "cancelled resume run-fence release",
                            lambda: (
                                self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                                    session_id=session.id,
                                    execution_profile=invocation_profile,
                                    invocation_context=invocation_context,
                                )
                            ),
                        ),
                    ),
                )
            finally:
                _deactivate_session_interaction(session.id)
            raise
        except Exception as exc:
            if is_durable_subagent_submission_unsettled(
                exc,
                parent_session_id=session.id,
            ):
                raise
            try:
                keep_recovery_interaction_open = False
                if continuing_recovery_boundary:
                    materialization_is_safe = False
                    try:
                        active_model_stage = (
                            await self.session_store.load_active_model_completion_stage(session.id)
                        )
                        if active_model_stage is not None:
                            keep_recovery_interaction_open = True
                        else:
                            (
                                _failure_checkpoint,
                                pending_failure_round,
                            ) = await pending_round_reader.load_pending_tool_round(
                                self.session_store,
                                session.id,
                                redactor=self._secret_redactor,
                                consume_on_rejection=True,
                                runtime_session=session,
                            )
                            materialization_is_safe = pending_failure_round is None
                            keep_recovery_interaction_open = pending_failure_round is not None
                    except Exception as inspection_error:
                        keep_recovery_interaction_open = True
                        exc.add_note(
                            "Deferred resume input remained private because recovery-state "
                            "inspection failed: "
                            f"{type(inspection_error).__name__}: {inspection_error}"
                        )
                    if materialization_is_safe:
                        try:
                            await self.session_store.materialize_deferred_interaction_input(
                                session.id
                            )
                        except Exception as materialization_error:
                            keep_recovery_interaction_open = True
                            exc.add_note(
                                "Deferred resume input remained private because materialization "
                                "failed: "
                                f"{type(materialization_error).__name__}: "
                                f"{materialization_error}"
                            )
                if keep_recovery_interaction_open:
                    session = await self.session_store.update_status(
                        session.id,
                        SessionStatus.FAILED,
                    )
                    interaction_failed_event = None
                else:
                    (
                        session,
                        interaction_failed_event,
                        _,
                    ) = await self._session_finalization.publish_sibling_interaction_transition(
                        session=session,
                        invocation_context=invocation_context,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                        to_status=SessionStatus.FAILED,
                        execution_profile=invocation_profile,
                    )
                if interaction_failed_event is not None:
                    yield interaction_failed_event
                failure_event = await self._terminal_event_publication.bind_to_run_operation(
                    Event(
                        type=EventType.SESSION_FAILED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=_environment_name(registered_environment),
                        payload=exception_failure_payload(
                            exc,
                            redactor=self._secret_redactor,
                        ),
                    ),
                    session=session,
                )
                persisted_failure = await self._event_writer.emit(failure_event)
                try:
                    await self._terminal_event_publication.clear_run_operation(
                        session_id=session.id,
                        operation_id=run_operation_id,
                        terminal_evidence_durable=True,
                    )
                except Exception as cleanup_failure:
                    logger.warning(
                        "Terminal evidence is durable but session run operation cleanup "
                        "remains pending: session_id=%s event_id=%s error_type=%s",
                        session.id,
                        persisted_failure.id,
                        type(cleanup_failure).__name__,
                    )
                yield persisted_failure
            finally:
                try:
                    await self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                        session_id=session.id,
                        execution_profile=invocation_profile,
                        invocation_context=invocation_context,
                    )
                finally:
                    _deactivate_session_interaction(session.id)
            return
        except BaseException:
            try:
                await self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                    session_id=session.id,
                    execution_profile=invocation_profile,
                    invocation_context=invocation_context,
                )
            finally:
                _deactivate_session_interaction(session.id)
            raise
        reconciled_pending_round = model_boundary.pending_tool_round or pending_round
        messages = (
            transcript + interaction_source_messages if continuing_recovery_boundary else transcript
        )
        session_stream = self._run_session(
            session=session,
            participant_context=participant_context,
            invocation_context=invocation_context,
            execution_to_wait=execution_to_wait,
            model_failover=model_failover,
            messages=messages,
            messages_to_append=(
                interaction_source_messages if continuing_recovery_boundary else request.messages
            ),
            max_steps=request.max_steps,
            limits=request.limits,
            budget_limits=request.budget_limits,
            retry_policy=self._effective_retry_policy(request.retry_policy),
            structured_output=request.structured_output,
            thinking=request.thinking,
            request_metadata=request.metadata,
            request_trace_metadata=(
                request.metadata
                if required_foreground_wait is not None
                else _runtime_resume_transport_metadata(request) or request.metadata
            ),
            foreground_wait=required_foreground_wait,
            targeted_tool_grants=targeted_tool_grant_footprint(
                targeted_tool_grant_records,
                projection=targeted_tool_projection,
            ),
            task_id=task_id,
            task_worker_id=request.task_worker_id,
            task_lease_expires_at=None,
            task_handoff_id=request.task_handoff_id,
            start_event_type=EventType.SESSION_RESUMED,
            start_event_payload={
                "agent_name": registered_agent.spec.name,
                "appended_messages": len(request.messages),
                **(
                    {
                        SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY: (
                            session_input_contract_evidence(
                                request,
                                message_start_index=(
                                    len(transcript) - len(interaction_source_messages)
                                ),
                            )
                        )
                    }
                    if not continuing_recovery_boundary and request.messages
                    else {}
                ),
                **copy_json_value(start_event_payload_extra, "start_event_payload_extra"),
            },
            start_task_on_enter=start_task_on_enter,
            messages_already_persisted=not continuing_recovery_boundary,
            messages_deferred=continuing_recovery_boundary,
            new_terminal_invocation=(
                loaded_session.status in {SessionStatus.COMPLETED, SessionStatus.FAILED}
                and not continuing_recovery_boundary
            ),
            deliver_queued_input_before_first_step=not continuing_recovery_boundary,
            pending_tool_round_source_transcript_cursor=(
                None
                if reconciled_pending_round is None
                else reconciled_pending_round.source_transcript_cursor
            ),
            run_limit_accounting=(
                required_foreground_continuation.run_limit_accounting
                if required_foreground_continuation is not None
                else pending_round.run_limit_accounting
                if required_foreground_wait is not None and pending_round is not None
                else None
            ),
            previous_tool_exposure_profile_id=previous_tool_exposure_profile_id,
            egress_environment_handoff=egress_environment_handoff,
            parked_egress_factory_result=parked_egress_factory_result,
            tool_completion=request.tool_completion,
        )
        authoritative_failure: BaseExceptionGroup | None = None
        try:
            try:
                if (
                    model_boundary.state == "promoted"
                    and model_boundary.completion_event is not None
                ):
                    yield copy_event(model_boundary.completion_event)
                async for event in session_stream:
                    yield event
            except BaseExceptionGroup as exc:
                detached_failure = detach_billing_identity_cancellation_group(exc)
                authoritative_failure = exc if detached_failure is None else detached_failure
            if authoritative_failure is None:
                return
            # A registered provider can retain live credentials in its repr.
            # Drop the registration before the detached failure crosses the
            # public resume boundary. Grouped interruption settlement already
            # ran inside _run_session while its run-fence and interaction
            # authority were still live.
            del registered_provider
            raise authoritative_failure from None
        except GeneratorExit:
            await session_stream.aclose()
            raise
        finally:
            _deactivate_session_interaction(session.id)

    async def continue_run(
        self,
        request: RecoverySessionRunRequest,
    ) -> AsyncGenerator[Event, None]:
        await self._require_participant_execution(request.session, request.participant_context)
        await self._session_tasks.require_continuation_task_authority(
            session_id=request.session.id,
            session_instance_id=(
                request.session.instance_id if request.task_id is not None else None
            ),
            task_id=request.task_id,
            task_worker_id=request.task_worker_id,
            task_handoff_id=request.task_handoff_id,
        )
        (
            requires_completion_decision,
            admission_failure,
        ) = await self.verifier_aware_task_execution_outcome(
            request.task_id,
            session_id=request.session.id,
        )
        if admission_failure is not None:
            del request
            raise_task_store_operation_failure(admission_failure)
        if requires_completion_decision:
            del request
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        stream = self._continue_admitted_run(request)
        stream = await self._resource_session_stream.restore(stream, request.session.id)
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def _continue_admitted_run(
        self, request: RecoverySessionRunRequest
    ) -> AsyncGenerator[Event, None]:
        """Continue an admitted invocation after its recovery gate has settled."""
        invocation_context = request.invocation_context
        invocation_context._validate()
        invocation_context.with_admitted_session(request.session)
        if (
            invocation_context.runtime_hooks is not self._runtime_hooks
            or invocation_context.loop_policies is not self._loop_policies
        ):
            raise RuntimeError("Recovery run substituted frozen invocation authority.")
        checkpoint = await self.session_store.load_checkpoint(request.session.id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        expected_profile = request.invocation_context.active_profile
        if (
            active_profile != expected_profile
            or expected_profile.session_id != request.session.id
            or expected_profile.run_epoch != request.session.run_epoch
        ):
            raise RuntimeError(
                "Recovery run does not reference the active invocation execution profile."
            )
        initial_tool_exposure = (
            None
            if request.initial_model_step_tool_exposure is None
            else resolved_tool_exposure_from_authority(
                request.initial_model_step_tool_exposure,
                invocation_context.registered_agent.tool_capabilities,
                catalogue_revision=invocation_context.registered_agent.tool_catalogue.revision,
            )
        )
        stream = self._run_recovered_session(
            session=request.session,
            admitted_checkpoint=checkpoint,
            participant_context=request.participant_context,
            invocation_context=invocation_context,
            messages=request.messages,
            messages_to_append=request.messages_to_append,
            max_steps=request.max_steps,
            limits=request.limits,
            budget_limits=request.budget_limits,
            retry_policy=request.retry_policy,
            structured_output=request.structured_output,
            tool_completion=request.tool_completion,
            thinking=request.thinking,
            request_metadata=request.request_metadata,
            task_id=request.task_id,
            task_worker_id=request.task_worker_id,
            task_handoff_id=request.task_handoff_id,
            start_event_type=request.start_event_type,
            start_event_payload=request.start_event_payload,
            start_task_on_enter=request.start_task_on_enter,
            release_run_fence_on_exit=request.release_run_fence_on_exit,
            deliver_queued_input_before_first_step=False,
            run_limit_accounting=request.run_limit_accounting,
            initial_model_step_identity=request.initial_model_step_identity,
            initial_model_step_number=request.initial_model_step_number,
            completed_tool_round_model_step=request.completed_tool_round_model_step,
            initial_model_step_tool_exposure=initial_tool_exposure,
            previous_tool_exposure_profile_id=(request.previous_tool_exposure_profile_id),
            preserve_failure_until_initial_provider_dispatch=(
                request.preserve_failure_until_initial_provider_dispatch
            ),
            producer_replay=request.producer_replay,
            tool_completion_replay=request.tool_completion_replay,
            execution_to_wait=request.execution_to_wait,
        )
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def _run_recovered_session(
        self,
        *,
        session: Session,
        invocation_context: InvocationContext,
        admitted_checkpoint: dict[str, Any] | None | _CheckpointState = _CheckpointState.NOT_LOADED,
        messages: list[Message],
        messages_to_append: list[Message],
        max_steps: int,
        limits: RunLimits,
        budget_limits: tuple[BudgetLimit, ...],
        retry_policy: RetryPolicy,
        structured_output: StructuredOutputSpec | None,
        thinking: ThinkingConfig | None,
        request_metadata: dict[str, Any],
        task_id: str | None,
        task_worker_id: str | None,
        task_handoff_id: str | None,
        start_event_type: EventType | None,
        start_event_payload: dict[str, Any],
        tool_completion: ToolCompletionPolicy | None = None,
        start_task_on_enter: bool = True,
        release_run_fence_on_exit: bool = True,
        deliver_queued_input_before_first_step: bool = True,
        run_limit_accounting: RunLimitAccountingContext | None = None,
        initial_model_step_identity: ModelStepIdentity | None = None,
        initial_model_step_number: int | None = None,
        completed_tool_round_model_step: int | None = None,
        initial_model_step_tool_exposure: ResolvedToolExposure | None = None,
        previous_tool_exposure_profile_id: str | None = None,
        preserve_failure_until_initial_provider_dispatch: bool = False,
        messages_deferred: bool = False,
        participant_context: CollaborationAccessContext | None = None,
        producer_replay: _ProducerCompletionReplay | None = None,
        tool_completion_replay: ToolCompletionResult | None = None,
        execution_to_wait: _ExternalExecutionToWait | None = None,
    ) -> AsyncGenerator[Event, None]:
        from cayu.sessions._session_continuation_store import has_unparked_external_interaction

        if type(invocation_context) is not InvocationContext:
            raise TypeError("invocation_context must be an authenticated InvocationContext.")
        # Recovery already authenticated this snapshot, with no intervening
        # async work. Only an omitted snapshot triggers a fresh load; an
        # explicitly absent checkpoint cannot acquire later evidence.
        checkpoint = admitted_checkpoint
        if checkpoint is _CheckpointState.NOT_LOADED:
            checkpoint = await self.session_store.load_checkpoint(session.id)
        # Approval/input resolution reconstructs native execution without the
        # application's external-wait adapter. Its durable ticket still needs
        # the completed turn's controls for later authenticated event servicing.
        # This only retains evidence; it grants no parking or service authority.
        retain_completion_controls = has_unparked_external_interaction(
            checkpoint, run_epoch=session.run_epoch
        )
        if tool_completion is None:
            tool_completion = await load_recorded_tool_completion_policy(
                self.session_store,
                session,
                checkpoint,
                execution_profile=invocation_context.profile,
                max_steps=max_steps,
                limits=limits,
                retry_policy=retry_policy,
            )
        registered_agent = invocation_context.registered_agent
        registered_provider = invocation_context.registered_provider
        model_failover = None
        if invocation_context.profile.model_failover is not None:
            continuation = await self._execution_profile_continuation.resolve(
                session=session,
                checkpoint=checkpoint,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                request_loop_policies=invocation_context.request_loop_policies,
                budget_policy=invocation_context.budget_policy,
                request_budget_limits=budget_limits,
                structured_output=structured_output,
                thinking=thinking,
                max_steps=max_steps,
                limits=limits,
                retry_policy=retry_policy,
                invocation_semantics_available=True,
            )
            if continuation.snapshot.profile != invocation_context.profile:
                raise RuntimeError(
                    "Recovered model configuration changed its invocation authority."
                )
            model_failover = continuation.model_failover
        interaction_id = _current_session_interaction_id(session.id)
        targeted_tool_projection = resolve_targeted_tool_projection(
            registered_agent.targeted_tool_mode,
            provider=registered_provider.provider,
            model=session.model,
        )
        targeted_tool_grant_records: tuple[TargetedToolGrantRecord, ...] = ()
        targeted_tool_grant_events: tuple[Event, ...] = ()
        if interaction_id is not None:
            (
                targeted_tool_grant_records,
                targeted_tool_grant_events,
            ) = await self._reconstruct_targeted_tool_grants(
                session=session,
                interaction_id=interaction_id,
                task_id=task_id,
                registered_agent=registered_agent,
                capability_ceiling=tool_capability_ceiling_from_session_metadata(session.metadata),
                observed_at=self._clock(),
                preserve_native_projection_snapshot=(
                    targeted_tool_projection is TargetedToolProjectionKind.OPENAI_ADDITIONAL_TOOLS
                ),
            )
        stream = self._run_session(
            session=session,
            participant_context=participant_context,
            invocation_context=invocation_context,
            model_failover=model_failover,
            messages=messages,
            messages_to_append=messages_to_append,
            messages_deferred=messages_deferred,
            max_steps=max_steps,
            limits=limits,
            budget_limits=budget_limits,
            retry_policy=retry_policy,
            structured_output=structured_output,
            thinking=thinking,
            request_metadata=request_metadata,
            request_trace_metadata=request_metadata,
            targeted_tool_grants=targeted_tool_grant_footprint(
                targeted_tool_grant_records,
                projection=targeted_tool_projection,
            ),
            task_id=task_id,
            task_worker_id=task_worker_id,
            task_lease_expires_at=None,
            task_handoff_id=task_handoff_id,
            start_event_type=start_event_type,
            start_event_payload=start_event_payload,
            start_task_on_enter=start_task_on_enter,
            release_run_fence_on_exit=release_run_fence_on_exit,
            deliver_queued_input_before_first_step=(deliver_queued_input_before_first_step),
            run_limit_accounting=run_limit_accounting,
            initial_model_step_identity=initial_model_step_identity,
            initial_model_step_number=initial_model_step_number,
            completed_tool_round_model_step=completed_tool_round_model_step,
            initial_model_step_tool_exposure=initial_model_step_tool_exposure,
            previous_tool_exposure_profile_id=previous_tool_exposure_profile_id,
            preserve_failure_until_initial_provider_dispatch=(
                preserve_failure_until_initial_provider_dispatch
            ),
            producer_replay=producer_replay,
            tool_completion_replay=tool_completion_replay,
            tool_completion=tool_completion,
            execution_to_wait=execution_to_wait,
            retain_completion_controls=retain_completion_controls,
        )
        for grant_event in targeted_tool_grant_events:
            yield grant_event
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    def _run_session(self, *, session: Session, **kwargs: Any) -> AsyncGenerator[Event, None]:
        # Internal admitted-work and continuation entrances also restore the
        # durable boundary. Preserve the unconfigured stream's existing owner.
        boundary = effective_deadline(current_execution_deadline(), session.execution_deadline)
        stream = self._run_session_with_deadline(session=session, **kwargs)
        if session.invocation.resource_access is not None:
            stream = restore_session_stream(
                stream,
                session=session,
                policy=self._resource_access_policy,
                store=self.session_store,
            )

        context = kwargs.get("invocation_context")
        work_attempt = context.work_attempt if isinstance(context, InvocationContext) else None
        task_id = kwargs.get("task_id")
        if (
            task_id is not None
            and isinstance(context, InvocationContext)
            and work_attempt is None
            and self.task_store is not None
            and self.task_store.supports_task_group_quiescence
        ):
            stream = self._run_session_with_group_observation(
                stream,
                task_id=task_id,
                invocation_context=context,
            )
        stopped = work_attempt.admission.execution_stop if work_attempt is not None else None
        if work_attempt is not None and (
            boundary.expired or (stopped is not None and stopped.request.reason == "elapsed_limit")
        ):
            # The first admission check below must settle this already-admitted
            # operation. A new zero-duration timer would cancel its stop write
            # and cleanup, rather than stop any execution (none can be admitted).
            return stream
        return stream if boundary.expires_at is None else deadline_stream(stream, boundary)

    async def _run_session_with_group_observation(
        self,
        stream: AsyncGenerator[Event, None],
        *,
        task_id: str,
        invocation_context: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        from cayu.runtime._task_group_invocation import (
            observe_owner_return,
            prepare_invocation_obligation,
        )

        observation = None
        failure: BaseException | None = None
        try:
            async with _close_delegated_event_stream(stream) as owned:
                observation = await prepare_invocation_obligation(
                    self.task_store,
                    task_id=task_id,
                    context=invocation_context,
                    redactor=self._secret_redactor,
                )
                async for event in owned:
                    yield event
        except BaseException as error:
            failure = error
            raise
        finally:
            # Only natural stream return acknowledges this execution owner.
            # A release receipt from a concurrent interruption/recovery owner
            # does not prove that this stream or provider cleanup returned.
            if failure is None and observation is not None:
                store = self.task_store
                assert store is not None
                # This acknowledgement owns its bounded observation and retry
                # handle. Do not put admission of that owner behind the general
                # cleanup supervisor, which may itself have exhausted capacity.
                await observe_owner_return(
                    store,
                    self.session_store,
                    observation,
                    redactor=self._secret_redactor,
                )

    async def _run_session_with_deadline(
        self,
        *,
        session: Session,
        invocation_context: InvocationContext,
        messages: list[Message],
        messages_to_append: list[Message],
        max_steps: int,
        limits: RunLimits,
        budget_limits: tuple[BudgetLimit, ...],
        retry_policy: RetryPolicy,
        structured_output: StructuredOutputSpec | None,
        thinking: ThinkingConfig | None,
        request_metadata: dict[str, Any],
        request_trace_metadata: dict[str, Any],
        targeted_tool_grants: TargetedToolGrantFootprint | None,
        task_id: str | None,
        task_worker_id: str | None,
        task_lease_expires_at: datetime | None,
        task_handoff_id: str | None,
        start_event_type: EventType | None,
        start_event_payload: dict[str, Any],
        tool_completion: ToolCompletionPolicy | None = None,
        start_task_on_enter: bool = True,
        release_run_fence_on_exit: bool = True,
        messages_already_persisted: bool = False,
        messages_deferred: bool = False,
        deliver_queued_input_before_first_step: bool = True,
        pending_tool_round_source_transcript_cursor: int | None = None,
        run_limit_accounting: RunLimitAccountingContext | None = None,
        initial_model_step_identity: ModelStepIdentity | None = None,
        initial_model_step_number: int | None = None,
        completed_tool_round_model_step: int | None = None,
        initial_model_step_tool_exposure: ResolvedToolExposure | None = None,
        previous_tool_exposure_profile_id: str | None = None,
        preserve_failure_until_initial_provider_dispatch: bool = False,
        egress_environment_handoff: EgressAuthorityAdoptionResult | None = None,
        parked_egress_factory_result: EnvironmentFactoryResult | None = None,
        new_terminal_invocation: bool = False,
        foreground_wait: ForegroundChildWait | None = None,
        model_failover: execution_profile_admission.ModelFailoverProfileResolution | None = None,
        participant_context: CollaborationAccessContext | None = None,
        execution_to_wait: _ExecutionToWait | _ExternalExecutionToWait | None = None,
        producer_replay: _ProducerCompletionReplay | None = None,
        tool_completion_replay: ToolCompletionResult | None = None,
        retain_completion_controls: bool = False,
    ) -> AsyncGenerator[Event, None]:
        outcome_state = SessionOutcomeState(session=session)

        def outcome_context() -> SessionOutcomeContext:
            return SessionOutcomeContext(
                state=outcome_state,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                environment_name=environment_name,
                run_started_at=run_started_at,
                turn_usage_tracker=turn_usage_tracker,
                active_run=active_run,
                task_id=task_id,
                task_worker_id=task_worker_id,
                task_lease_expires_at=task_lease_expires_at,
                task_handoff_id=task_handoff_id,
                request_loop_policies=request_loop_policies,
            )

        if type(invocation_context) is not InvocationContext:
            raise TypeError("invocation_context must be an authenticated InvocationContext.")
        if not isinstance(invocation_context.binding, AdmittedInvocationBinding):
            raise ValueError("Invocation execution requires an admitted context.")
        if invocation_context.work_attempt is not None:
            attempt_authority = invocation_context.work_attempt.admission
            execution_entry = attempt_authority.execution_entry
            if (
                task_id != attempt_authority.task_id
                or task_worker_id != attempt_authority.claim.worker_id
                or task_handoff_id is not None
                or execution_entry is None
                or execution_entry.request.claim_id != attempt_authority.claim.claim_id
                or execution_entry.request.generation != attempt_authority.claim.generation
                or execution_entry.request.run_epoch != outcome_state.session.run_epoch
            ):
                raise WorkAttemptExecutionClaimLost(
                    "Governed invocation conflicts with its exact task execution entry."
                )
            del attempt_authority
        registered_agent = invocation_context.registered_agent
        registered_provider = invocation_context.registered_provider
        registered_environment = invocation_context.registered_environment
        execution_profile = invocation_context.profile
        if model_failover is not None:
            if (
                type(model_failover)
                is not execution_profile_admission.ModelFailoverProfileResolution
            ):
                raise TypeError("Invocation failover configuration must be resolved.")
            model_failover = replace(model_failover)
            if (
                model_failover.profile != execution_profile
                or model_failover.registered_providers[0] is not registered_provider
            ):
                raise ValueError("Invocation failover configuration changed admitted authority.")
        elif execution_profile.model_failover is not None:
            raise RuntimeError("Routed invocation requires its resolved candidate configuration.")
        budget_policy = invocation_context.budget_policy
        request_loop_policies = invocation_context.request_loop_policies
        # Deep defense for internal recovery callers. Public entry points
        # validate before claiming mutable session ownership.
        require_secret_free_structured_output_spec(
            structured_output,
            redactor=self._secret_redactor,
            field_name="structured_output",
        )
        if targeted_tool_grants is not None:
            if type(targeted_tool_grants) is not TargetedToolGrantFootprint:
                raise TypeError(
                    "targeted_tool_grants must be a TargetedToolGrantFootprint or None."
                )
            targeted_tool_grants = TargetedToolGrantFootprint.model_validate(
                targeted_tool_grants.model_dump(mode="python")
            )
        if messages_already_persisted and messages_deferred:
            raise ValueError(
                "messages_already_persisted and messages_deferred are mutually exclusive."
            )
        if (
            initial_model_step_identity is not None
            and type(initial_model_step_identity) is not ModelStepIdentity
        ):
            raise TypeError("initial_model_step_identity must be a ModelStepIdentity.")
        if (initial_model_step_identity is None) != (initial_model_step_number is None):
            raise ValueError(
                "Initial model-step identity and step number must be supplied together."
            )
        if completed_tool_round_model_step is not None and (
            type(completed_tool_round_model_step) is not int
            or not 1 <= completed_tool_round_model_step <= max_steps
            or initial_model_step_identity is not None
        ):
            raise ValueError("Completed tool-round step conflicts with continuation bounds.")
        if initial_model_step_number is not None and (
            type(initial_model_step_number) is not int
            or not 1 <= initial_model_step_number <= max_steps
        ):
            raise ValueError("initial_model_step_number is outside this run's step bounds.")
        if (
            initial_model_step_tool_exposure is not None
            and previous_tool_exposure_profile_id is not None
        ):
            raise ValueError(
                "An initial frozen tool exposure and a previous profile cannot be supplied "
                "together."
            )
        if (initial_model_step_identity is None) != (initial_model_step_tool_exposure is None):
            raise ValueError(
                "An initial model-step retry identity and its frozen tool exposure must be "
                "supplied together."
            )
        if initial_model_step_tool_exposure is not None:
            initial_model_step_tool_exposure = _require_frozen_tool_exposure(
                initial_model_step_tool_exposure
            )
        if type(preserve_failure_until_initial_provider_dispatch) is not bool:
            raise TypeError("preserve_failure_until_initial_provider_dispatch must be a bool.")
        if preserve_failure_until_initial_provider_dispatch and (
            initial_model_step_identity is None
        ):
            raise ValueError(
                "Initial provider-dispatch failure preservation requires an initial "
                "model-step identity."
            )
        if egress_environment_handoff is not None and not (
            _has_runtime_egress_authority_adoption_result(egress_environment_handoff)
        ):
            raise EgressAuthorityTransitionConflict(
                "Invocation received an invalid egress environment handoff."
            )
        if (
            parked_egress_factory_result is not None
            and type(parked_egress_factory_result) is not EnvironmentFactoryResult
        ):
            raise TypeError("Parked egress handoff requires an EnvironmentFactoryResult.")
        if parked_egress_factory_result is not None and egress_environment_handoff is not None:
            raise EgressAuthorityTransitionConflict(
                "Invocation received two owners for one parked egress environment."
            )

        async def materialize_deferred_messages() -> None:
            nonlocal messages_deferred
            if not messages_deferred:
                return
            if (
                invocation_context.work_attempt is not None
                and invocation_context.work_attempt.admission.kind == "initial"
            ):
                # Initial admission retains the complete rendered projection,
                # not merely its caller-message suffix. The existing atomic
                # publisher verifies both against durable input and clears the
                # bootstrap marker together with transcript publication.
                if not await reconcile_work_attempt_initial_publication(
                    self.session_store,
                    admission=invocation_context.work_attempt.admission,
                    expected_source=messages_to_append,
                    expected_transcript=messages,
                    redactor=self._secret_redactor,
                ):
                    await self.session_store.replace_initial_transcript_messages(
                        outcome_state.session.id,
                        messages_to_append,
                        messages,
                    )
            else:
                await self.session_store.materialize_deferred_interaction_input(
                    outcome_state.session.id
                )
            messages_deferred = False

        async def materialize_deferred_messages_after_failure() -> None:
            if not messages_deferred:
                return
            checkpoint = await self.session_store.load_checkpoint(outcome_state.session.id)
            if (
                pending_round_reader.pending_tool_round_from_checkpoint(
                    checkpoint,
                    redactor=self._secret_redactor,
                    consume_on_rejection=True,
                    runtime_session=outcome_state.session,
                )
                is not None
            ):
                return
            await materialize_deferred_messages()

        provider = registered_provider.provider
        model_projection_cursor = session_model_projection_cursor(outcome_state.session)
        selected_progress = model_failover_progress_for_session(
            session=outcome_state.session,
            execution_profile=execution_profile,
            checkpoint=(
                await self.session_store.load_checkpoint(outcome_state.session.id)
                if model_failover is not None
                else None
            ),
        )
        if selected_progress is not None:
            model_projection_cursor = max(
                model_projection_cursor, selected_progress.projection_cursor
            )
        model_projection_source_prefix_count = 0
        model_projection_prefix_count = 0
        if model_projection_cursor:
            model_projection_snapshot = await self.session_store.load_transcript_snapshot(
                outcome_state.session.id
            )
            try:
                if model_projection_cursor > model_projection_snapshot.cursor:
                    raise ValueError(
                        "Model-target projection cursor exceeds the session transcript."
                    )
                deferred_tail_count = len(messages_to_append) if messages_deferred else 0
                retained_message_count = len(messages) - deferred_tail_count
                model_projection = _transcript._project_model_target_snapshot(
                    model_projection_snapshot,
                    model_projection_cursor,
                )
                model_projection_source_prefix_count = model_projection.source_prefix_count
                model_projection_prefix_count = model_projection.projected_prefix_count
                expected_retained_message_count = len(model_projection.messages)
                if (
                    retained_message_count < 0
                    or retained_message_count != expected_retained_message_count
                ):
                    raise RuntimeError(
                        "Model-facing transcript does not match the retained durable transcript."
                    )
            finally:
                del model_projection_snapshot
        # Resume admission already supplied the model-facing projection. A new
        # session cannot carry projection authority. Later durable transcript
        # reloads are projected from indexed snapshots below.
        messages = [
            redact_runtime_message_for_boundary(
                message,
                redactor=self._secret_redactor,
                field_name="message",
            )
            for message in messages
        ]
        messages_to_append = [
            redact_runtime_message_for_boundary(
                message,
                redactor=self._secret_redactor,
                field_name="message",
            )
            for message in messages_to_append
        ]
        # Per-run thinking override (RunRequest/ResumeRequest) wins over the agent's
        # default (AgentSpec.thinking); the agent default applies on every path,
        # including continuations that pass no override.
        effective_thinking = thinking if thinking is not None else registered_agent.spec.thinking
        environment_name = _environment_name(registered_environment)
        outcome_state.task_started = (
            task_id is not None
            and invocation_context.work_attempt is None
            and not start_task_on_enter
        )
        outcome_state.task_start_attempted = outcome_state.task_started
        outcome_state.task_finished = False
        tool_completion_result: ToolCompletionResult | None = None
        completion_finalization_marker: dict[str, Any] | None = None
        current_task = asyncio.current_task()
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None
        run_started_at = time.monotonic()

        def complete_session(
            completed_session: Session,
            completion_finalization_marker: dict[str, Any] | None,
        ) -> AsyncGenerator[Event, None]:
            return self._session_outcomes.complete_session(
                outcome_context(),
                completed_session,
                completion_finalization_marker,
                tool_completion_result,
            )

        # A fresh run means any earlier interrupt was fully handled before the
        # session transitioned back to RUNNING; drop a stale signal so it does
        # not force per-delta store polling for the whole resumed run.
        self._session_control.discard_interrupt_signal(outcome_state.session.id)
        limits = copy_run_limits(limits)
        budget_policy = copy_budget_policy(budget_policy)
        budget_limits = request_budget_limits_for_session(
            limits=budget_limits,
            agent_name=registered_agent.spec.name,
            causal_budget_id=outcome_state.session.causal_budget_id,
        )
        hosted_search_budget_limits = (
            *budget_limits,
            *budget_limits_for_session(
                policy=budget_policy,
                agent_name=registered_agent.spec.name,
                causal_budget_id=outcome_state.session.causal_budget_id,
            ),
        )
        if registered_agent.hosted_tools and any(
            not limit.allow_unpriced for limit in hosted_search_budget_limits
        ):
            raise HostedToolCapabilityError(
                "Strict cost budgets cannot admit OpenAI hosted web search because "
                "the Responses API exposes no hard per-response search-call ceiling. "
                "Use an explicitly allow_unpriced budget or disable hosted search."
            )
        retry_policy = copy_retry_policy(retry_policy)
        structured_output = copy_structured_output_spec(structured_output)
        request_loop_policies = validate_loop_policies(
            request_loop_policies,
            field_name="request_loop_policies",
        )
        structured_output_retries = 0
        run_baseline: SessionUsageSummary | None = None
        run_budget_authorities = None
        if run_limit_accounting is not None:
            if not has_run_limit_accounting_authority(limits, budget_limits):
                raise ValueError("Run-limit accounting requires active run-scoped authority.")
            run_started_at, run_baseline, run_budget_authorities = (
                restore_run_limit_accounting_context(
                    run_limit_accounting,
                    session_id=outcome_state.session.id,
                    budget_limits=budget_limits,
                    now=self._clock(),
                )
            )
        turn_usage_tracker = self._run_limit_controller.usage_tracker(outcome_state.session.id)
        baseline_events: list[Event] = []
        request_budget_notify_events: list[Event] = []
        close_new_pending_round_on_interrupt = False
        initial_provider_dispatch_started = False

        async def emit_structured_event(event: Event) -> Event:
            if isinstance(execution_to_wait, _ExternalExecutionToWait):
                return await execution_to_wait.publish_structured_event(
                    invocation_context, self._event_writer, event
                )
            return await self._event_writer.emit(event)

        async def park_external_completion() -> AsyncIterator[Event]:
            # Successful structured/final-tool outcomes are whole-turn boundaries,
            # just like before-stop COMPLETE. Park before terminalizing the writer.
            assert isinstance(execution_to_wait, _ExternalExecutionToWait)
            await execution_to_wait.park(invocation_context)
            async for event in self._session_outcomes.publish_stop_interruption(
                outcome_context(),
                BeforeStopDecision(
                    action=BeforeStopAction.INTERRUPT,
                    reason="Explicit continuation wait reached a whole-turn boundary.",
                ),
            ):
                yield event

        try:
            work_attempt = invocation_context.work_attempt
            if work_attempt is not None:
                admitted = work_attempt.admission
                stopped = admitted.execution_stop
                if stopped is not None and stopped.request.reason == "elapsed_limit":
                    if admitted.run_semantics is None:
                        raise RuntimeError("Elapsed execution stop has no admitted run semantics.")
                    raise ExecutionDeadlineExceeded(admitted.run_semantics.deadline, "session")
            if (
                isinstance(execution_to_wait, _ExternalExecutionToWait)
                and invocation_context.recovery_claim_id is not None
            ):
                recovered_wait = await execution_to_wait.prepare(invocation_context)
                if recovered_wait.ticket.state == "WAITING":
                    await turn_usage_tracker.mark_current_position()
                    # The native park already accepted the entire turn, including
                    # stop policies. Finish its interrupted publication/cleanup;
                    # do not reevaluate policies or dispatch another model step.
                    async for event in self._session_outcomes.publish_stop_interruption(
                        outcome_context(),
                        BeforeStopDecision(
                            action=BeforeStopAction.INTERRUPT,
                            reason="Explicit continuation wait reached a whole-turn boundary.",
                        ),
                    ):
                        yield event
                    return
            effective_deadline(
                current_execution_deadline(), outcome_state.session.execution_deadline
            ).require_admission("session")
            if structured_output is not None and (
                STRUCTURED_OUTPUT_TOOL_NAME in registered_agent.tools
            ):
                raise ValueError(
                    f"Tool name is reserved for structured output: {STRUCTURED_OUTPUT_TOOL_NAME}"
                )
            if current_task is not None:
                active_run = self._session_control.register_active_task(
                    outcome_state.session.id,
                    current_task,
                    task_id=task_id,
                    task_started=outcome_state.task_started,
                    task_finished=outcome_state.task_finished,
                    turn_registered_agent=registered_agent,
                    turn_environment_name=environment_name,
                    turn_started_at=run_started_at,
                    turn_usage_tracker=turn_usage_tracker,
                )
                await self._session_control.execution_presence.ensure(
                    outcome_state.session, task_id=task_id
                )
            await turn_usage_tracker.mark_current_position()
            if execution_to_wait is not None:
                await execution_to_wait.prepare(invocation_context)
            if outcome_state.task_started and task_id is not None:
                from cayu.runtime._task_group_invocation import bind_invocation

                await bind_invocation(
                    self.task_store,
                    task_id=task_id,
                    context=invocation_context,
                    redactor=self._secret_redactor,
                )
            if limits.scope == "run" and has_run_limits(limits) and run_limit_accounting is None:
                run_baseline = await self._run_limit_controller.session_usage_summary(
                    outcome_state.session.id
                )

            model_boundary = (
                await self._model_completion_recovery.reconcile_model_completion_boundary(
                    outcome_state.session,
                    invocation_context=invocation_context,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                )
            )
            outcome_state.session = model_boundary.session
            recovered_assistant_step: AssistantStepResult | None = None
            recovered_structured_events = model_boundary.structured_output_events
            governed_closed_tool_round = (
                invocation_context.work_attempt is not None
                and model_boundary.pointer is not None
                and model_boundary.pointer.tool_round_id is not None
                and model_boundary.pending_tool_round is None
                and model_boundary.transcript_cursor > model_boundary.pointer.transcript_end_cursor
            )
            if (
                execution_to_wait is not None
                and invocation_context.recovery_claim_id is not None
                and isinstance(execution_to_wait, _ExternalExecutionToWait)
            ):
                recovered_semantics = await execution_to_wait.require_recovered_result(
                    invocation_context, model_boundary
                )
                stage = model_boundary.completed_stage
                assert stage is not None and model_boundary.pointer is not None
                if model_boundary.pointer.tool_round_id is not None:
                    # A closed round reuses its publication. An unfinished one
                    # goes through recover_pending_tool_round below, which owns
                    # effect reconciliation and fail-closed policy recovery.
                    # Neither case reconstructs tool calls for fresh execution.
                    if model_boundary.closed_tool_receipt is not None:
                        recovered_structured_events = await self._model_completion_recovery.load_closed_structured_output_events(
                            outcome_state.session,
                            stage,
                            invocation_context,
                            model_boundary.closed_tool_receipt,
                            expected_spec=recovered_semantics.structured_output,
                        )
                    completion_event = model_boundary.completion_event
                    completed_step = (
                        None if completion_event is None else completion_event.payload.get("step")
                    )
                    if (
                        completion_event is None
                        or completion_event.interaction_id
                        != invocation_context.binding.interaction_id
                        or type(completed_step) is not int
                        or completed_step < 1
                    ):
                        raise RuntimeError("Closed external-wait tool round lost its model step.")
                    initial_model_step_number = completed_step + 1
                else:
                    recovered_assistant_step = reconstruct_assistant_step_result(
                        stage=stage,
                        pointer=model_boundary.pointer,
                        pending_round=None,
                        session_id=outcome_state.session.id,
                        interaction_id=invocation_context.binding.interaction_id,
                        source_run_epoch=stage.source_run_epoch,
                    )
                    if recovered_assistant_step is None:
                        raise RuntimeError(
                            "External wait recovery lost its committed assistant result."
                        )
            elif producer_replay is not None:
                from cayu.runtime._producer_completion_replay import _ProducerCompletionReplay

                if type(producer_replay) is not _ProducerCompletionReplay:
                    raise PermissionError("Producer replay requires its native recovery owner.")
                producer_replay.require(invocation_context, model_boundary)
                stage = model_boundary.completed_stage
                assert stage is not None and model_boundary.pointer is not None
                recovered_assistant_step = reconstruct_assistant_step_result(
                    stage=stage,
                    pointer=model_boundary.pointer,
                    pending_round=None,
                    session_id=outcome_state.session.id,
                    interaction_id=invocation_context.binding.interaction_id,
                    source_run_epoch=stage.source_run_epoch,
                )
                if recovered_assistant_step is None:
                    raise RuntimeError("Producer recovery has no committed assistant result.")
            elif (
                model_boundary.blocks_provider_dispatch or governed_closed_tool_round
            ) and not messages_to_append:
                work_attempt = invocation_context.work_attempt
                stage = model_boundary.completed_stage
                if work_attempt is None or stage is None or model_boundary.pointer is None:
                    raise ModelCompletionManualRecoveryRequired(
                        "The latest model completion is already durable at the transcript tail. "
                        "Provider redispatch requires an explicit new message."
                    )
                admission = work_attempt.admission
                recovery_context = model_completion_recovery_context_from_stage(stage)
                if (
                    admission.recovery_evidence_sha256 is None
                    or admission.execution_entry is None
                    or stage.source_run_epoch > admission.execution_entry.request.run_epoch
                    or recovery_context is None
                    or recovery_context.execution_profile_fingerprint
                    != admission.source_execution_profile_fingerprint
                ):
                    raise WorkAttemptRecoveryRequired(
                        "Completed model replay lacks exact recovered invocation authority."
                    )
                # The reconciliation owner validated the immutable stage, receipt
                # and current transcript pointer. The old stage epoch remains
                # valid after cleanup/rebinding advances the invocation epoch.
                if governed_closed_tool_round:
                    completion_event = model_boundary.completion_event
                    completed_step = (
                        None if completion_event is None else completion_event.payload.get("step")
                    )
                    if (
                        completion_event is None
                        or completion_event.interaction_id != admission.interaction_id
                        or type(completed_step) is not int
                        or completed_step < 1
                    ):
                        raise WorkAttemptRecoveryRequired(
                            "Closed tool publication lacks its original governed model step."
                        )
                    initial_model_step_number = completed_step + 1
                else:
                    recovered_assistant_step = reconstruct_assistant_step_result(
                        stage=stage,
                        pointer=model_boundary.pointer,
                        pending_round=model_boundary.pending_tool_round,
                        session_id=admission.session_id,
                        interaction_id=admission.interaction_id,
                        source_run_epoch=stage.source_run_epoch,
                    )
                    if recovered_assistant_step is None:
                        raise RuntimeError(
                            "Recovered governed model completion is a non-turn outcome."
                        )
            if model_boundary.state == "promoted" and model_boundary.completion_event is not None:
                yield copy_event(model_boundary.completion_event)
            factory_started_event = await self._environment_lifecycle.emit_factory_started(
                session=outcome_state.session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            )
            if factory_started_event is not None:
                yield factory_started_event
                task_start_event = await self._session_outcomes.start_linked_task_if_needed(
                    outcome_context(),
                )
                if task_start_event is not None:
                    yield task_start_event
            adopted_factory_result = None
            parked_egress_renewal = None
            if egress_environment_handoff is not None:
                handler = self._egress_authority_adoption_handler
                if handler is None:
                    raise EgressAuthorityTransitionConflict(
                        "Egress environment handoff lost its adoption handler."
                    )
                activation = egress_environment_handoff.transition
                parked_result = _claim_parked_egress_authority_allocation(
                    handler,
                    session_id=outcome_state.session.id,
                    environment_name=activation.environment_name,
                    expected_fingerprint=activation.source_environment_fingerprint,
                    expected_factory_result=(egress_environment_handoff._factory_result),
                )
                adopted_factory_result = egress_environment_handoff._claim_factory_result()
                if adopted_factory_result is not parked_result:
                    raise EgressAuthorityTransitionConflict(
                        "Egress environment handoff claimed a substituted allocation."
                    )
            elif parked_egress_factory_result is not None:
                handler = self._egress_authority_adoption_handler
                if handler is None or execution_profile is None or registered_environment is None:
                    raise EgressAuthorityTransitionConflict(
                        "Parked egress reuse lost its exact invocation authority."
                    )
                authority = execution_profile.egress_authority
                environment_name = _environment_name(registered_environment)
                if authority is None or environment_name is None:
                    raise EgressAuthorityTransitionConflict(
                        "Parked egress reuse lost its exact invocation authority."
                    )
                fingerprint = parked_egress_factory_result.reconnect_metadata.get(
                    "allocation_fingerprint"
                )
                if type(fingerprint) is not str:
                    raise EgressAuthorityTransitionConflict(
                        "Parked egress reuse lost its allocation fingerprint."
                    )
                adopted_factory_result = _claim_parked_egress_authority_allocation(
                    handler,
                    session_id=outcome_state.session.id,
                    environment_name=environment_name,
                    expected_fingerprint=fingerprint,
                    expected_factory_result=parked_egress_factory_result,
                )
                factory = registered_environment.factory
                if factory is None:
                    raise EgressAuthorityTransitionConflict(
                        "Parked egress reuse lost its registered environment factory."
                    )
                parked_egress_renewal = (
                    adopted_factory_result,
                    factory,
                    authority,
                    environment_name,
                    execution_profile.fingerprint,
                )
            factory_resolution = await self._environment_lifecycle.resolve_factory(
                session=outcome_state.session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=factory_started_event,
                operation=(
                    EnvironmentFactoryOperation.CREATE
                    if start_event_type is EventType.SESSION_STARTED
                    else EnvironmentFactoryOperation.RECONNECT
                ),
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                adopted_factory_result=adopted_factory_result,
                new_terminal_invocation=new_terminal_invocation,
            )
            registered_environment = factory_resolution.registered_environment
            if registered_environment is not None:
                invocation_context = invocation_context.with_registered_environment(
                    registered_environment,
                    validated_profile=execution_profile,
                )
            environment_name = _environment_name(registered_environment)
            if active_run is not None:
                active_run.turn_environment_name = environment_name
            if factory_resolution.error is None:
                if parked_egress_renewal is not None:
                    (
                        parked_result,
                        parked_factory,
                        parked_authority,
                        parked_environment_name,
                        parked_execution_profile_fingerprint,
                    ) = parked_egress_renewal
                    await reactivate_parked_egress_environment(
                        factory_result=parked_result,
                        factory=parked_factory,
                        expected_authority=parked_authority,
                        session_id=outcome_state.session.id,
                        environment_name=parked_environment_name,
                        agent_name=registered_agent.spec.name,
                        execution_profile_fingerprint=(parked_execution_profile_fingerprint),
                    )
                active_egress_transition = await SessionCheckpointEgressAuthorityTransitionStore(
                    self.session_store
                ).load(outcome_state.session.id)
                if (
                    active_egress_transition is not None
                    and active_egress_transition.state is EgressAuthorityTransitionState.ACTIVE
                ):
                    if registered_environment is None:
                        raise EgressAuthorityTransitionConflict(
                            "The active egress profile resolved no invocation environment."
                        )
                    await require_active_egress_transition_environment(
                        environment=registered_environment.environment,
                        transition=active_egress_transition,
                    )
            for event in factory_resolution.events:
                yield event
            if factory_resolution.error is not None:
                raise factory_resolution.error
            binding_started_event = await self._environment_lifecycle.emit_binding_started(
                session=outcome_state.session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            )
            if binding_started_event is not None:
                yield binding_started_event
                task_start_event = await self._session_outcomes.start_linked_task_if_needed(
                    outcome_context(),
                )
                if task_start_event is not None:
                    yield task_start_event
            binding_result = await self._environment_lifecycle.bind(
                session=outcome_state.session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=binding_started_event,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            )
            registered_environment = binding_result.registered_environment
            if registered_environment is not None:
                invocation_context = invocation_context.with_registered_environment(
                    registered_environment,
                    validated_profile=execution_profile,
                )
            environment_name = _environment_name(registered_environment)
            if active_run is not None:
                active_run.turn_environment_name = environment_name
            for event in binding_result.events:
                yield event
            if binding_result.error is not None:
                raise binding_result.error
            async for event in self._tool_round_executor.emit_mcp_manifest_checks(
                session=outcome_state.session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                invocation_context=invocation_context,
            ):
                yield event
            if start_event_type is not None:
                interaction_started_event = (
                    await self._session_outcomes.emit_interaction_started_if_needed(
                        session=outcome_state.session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                    )
                )
                if interaction_started_event is not None:
                    yield interaction_started_event
                start_event = Event(
                    type=start_event_type,
                    session_id=outcome_state.session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        **start_event_payload,
                        "run_epoch": outcome_state.session.run_epoch,
                        **(
                            {"execution_deadline": current_execution_deadline().inspection()}
                            if current_execution_deadline().expires_at is not None
                            else {}
                        ),
                        **_session_trace_event_fields(
                            outcome_state.session, request_trace_metadata
                        ),
                    },
                )
                lineage_fields = tuple(
                    field_name
                    for field_name, expected in (
                        ("parent_session_id", outcome_state.session.parent_session_id),
                        ("causal_budget_id", outcome_state.session.causal_budget_id),
                    )
                    if expected is not None and start_event.payload.get(field_name) == expected
                )
                if (
                    start_event_type in {EventType.SESSION_STARTED, EventType.SESSION_RESUMED}
                    and type(start_event.payload.get(SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY))
                    is str
                ):
                    lineage_fields = (
                        *lineage_fields,
                        SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
                    )
                if lineage_fields:
                    start_event = event_with_runtime_payload_authority(
                        start_event,
                        *lineage_fields,
                    )
                yield await self._event_writer.emit(start_event)
            recovered_structured_outcome = next(
                (
                    event
                    for event in recovered_structured_events
                    if event.type
                    in {EventType.STRUCTURED_OUTPUT_VALIDATED, EventType.STRUCTURED_OUTPUT_FAILED}
                ),
                None,
            )
            recovered_structured_retry = any(
                event.type is EventType.STRUCTURED_OUTPUT_RETRY
                for event in recovered_structured_events
            )
            recovery_tail_message_count = len(messages_to_append)
            recovery_expected_transcript_cursor: int | None = None
            if pending_tool_round_source_transcript_cursor is not None:
                (
                    _checkpoint,
                    entrance_pending_round,
                ) = await pending_round_reader.load_pending_tool_round(
                    self.session_store,
                    outcome_state.session.id,
                )
                if entrance_pending_round is not None:
                    structured_output_retries = max(
                        structured_output_retries,
                        entrance_pending_round.structured_output_retries,
                    )
                pending_result_cursor = pending_tool_round_source_transcript_cursor + int(
                    entrance_pending_round is None
                    or entrance_pending_round.assistant_message_state == "published"
                )
                pending_snapshot = await self.session_store.load_transcript_snapshot(
                    outcome_state.session.id
                )
                try:
                    pending_result_position = pending_snapshot.retained_position(
                        pending_result_cursor
                    )
                    if model_projection_cursor:
                        pending_projection = _transcript._project_model_target_snapshot(
                            pending_snapshot,
                            model_projection_cursor,
                        )
                        if pending_result_position < model_projection_source_prefix_count:
                            raise RuntimeError(
                                "Pending tool round predates the model-target projection boundary."
                            )
                        pending_result_position = (
                            pending_projection.projected_prefix_count
                            + pending_result_position
                            - model_projection_source_prefix_count
                        )
                        retained_pending_message_count = len(pending_projection.messages)
                    else:
                        retained_pending_message_count = len(pending_snapshot.records)
                    expected_resume_message_count = retained_pending_message_count + (
                        len(messages_to_append) if messages_deferred else 0
                    )
                    if len(messages) != expected_resume_message_count:
                        raise RuntimeError(
                            "Pending tool round resume context does not exactly match the "
                            "retained transcript and deferred interaction input."
                        )
                    recovery_expected_transcript_cursor = pending_snapshot.cursor
                except ValueError as exc:
                    raise RuntimeError(
                        "Pending tool round transcript cursor is unavailable in the "
                        "retained resume context."
                    ) from exc
                finally:
                    del pending_snapshot
                recovery_tail_message_count = len(messages) - pending_result_position
            if (
                has_run_limit_accounting_authority(limits, budget_limits)
                and run_limit_accounting is None
            ):
                run_limit_accounting = capture_run_limit_accounting_context(
                    session_id=outcome_state.session.id,
                    run_started_at=run_started_at,
                    run_baseline=run_baseline,
                    budget_limits=budget_limits,
                    now=self._clock(),
                )
                run_budget_authorities = run_budget_authorities_from_context(
                    run_limit_accounting,
                    budget_limits=budget_limits,
                )
            # Recovery may rebind `session` below; the tool-round runner builds its
            # own gate afterwards, so this one serves only replay admission and is
            # built only when a replay asks for admission.
            replay_gate_session = outcome_state.session
            replay_limit_gate: RunLimitGate | None = None
            replay_limit_evaluations: list[LimitEvaluation] = []

            async def admit_replay(identity: ModelAttemptIdentity) -> bool:
                # Replay retains the logical call identity already counted by its
                # start event, but still obeys current run/session and budget limits,
                # through the same gate as ordinary tool admission.
                nonlocal replay_limit_gate
                if replay_limit_gate is None:
                    replay_limit_gate = RunLimitGate(
                        self._run_limit_controller,
                        session=replay_gate_session,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        limits=limits,
                        budget_limits=budget_limits,
                        run_started_at=run_started_at,
                        run_baseline=run_baseline,
                        budget_baseline_events=baseline_events,
                        budget_notify_events=request_budget_notify_events,
                        run_budget_authorities=run_budget_authorities,
                        pricing_provider_name=(
                            registered_provider.provider.billing_provider_name
                            or registered_provider.name
                        ),
                        execution_profile_fingerprint=(
                            None if execution_profile is None else execution_profile.fingerprint
                        ),
                    )
                evaluation = await replay_limit_gate.evaluate_limits(execution_identity=identity)
                replay_limit_evaluations.append(evaluation)
                return evaluation.decision is None

            async for event in self._pending_tool_round_recovery.recover_pending_tool_round(
                session=outcome_state.session,
                invocation_context=invocation_context,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                messages=messages,
                execution_profile=execution_profile,
                tail_message_count=recovery_tail_message_count,
                admit_replay=admit_replay,
                expected_transcript_cursor=recovery_expected_transcript_cursor,
            ):
                if event.type in {
                    EventType.STRUCTURED_OUTPUT_VALIDATED,
                    EventType.STRUCTURED_OUTPUT_FAILED,
                }:
                    recovered_structured_outcome = event
                elif event.type == EventType.STRUCTURED_OUTPUT_RETRY:
                    recovered_structured_retry = True
                yield event
            if recovered_structured_retry:
                if (
                    recovered_structured_outcome is None
                    or recovered_structured_outcome.type != EventType.STRUCTURED_OUTPUT_FAILED
                ):
                    raise RuntimeError("Recovered structured-output retry has no failed outcome.")
                recovered_attempt = recovered_structured_outcome.payload.get("attempt")
                if type(recovered_attempt) is not int:
                    raise RuntimeError("Recovered structured-output retry has no valid attempt.")
                structured_output_retries = max(
                    structured_output_retries,
                    recovered_attempt,
                )
            elif (
                recovered_structured_outcome is not None
                and recovered_structured_outcome.type == EventType.STRUCTURED_OUTPUT_FAILED
            ):
                recovered_attempt = recovered_structured_outcome.payload.get("attempt")
                raise RuntimeError(
                    f"Structured output validation failed after {recovered_attempt} attempt(s)."
                )
            # Typed backstop for a missed entrance: every entry point already
            # preflights this before touching persisted state. Here the session
            # is running, so a raise fails it cleanly instead of preventing it.
            _require_native_structured_output_support(
                structured_output, registered_provider=registered_provider
            )
            task_start_event = await self._session_outcomes.start_linked_task_if_needed(
                outcome_context(),
            )
            if task_start_event is not None:
                yield task_start_event
            if messages_deferred:
                await materialize_deferred_messages()
            elif not messages_already_persisted:
                await self.session_store.append_transcript_messages(
                    outcome_state.session.id,
                    messages_to_append,
                )
            for evaluation in replay_limit_evaluations:
                async for event in self._session_finalization.apply_limit_evaluation(
                    evaluation=evaluation,
                    session=outcome_state.session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    messages=messages,
                    run_started_at=run_started_at,
                    turn_usage_tracker=turn_usage_tracker,
                    active_run=active_run,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                ):
                    yield event
                if evaluation.decision is not None:
                    return
            tool_completion_result = (
                await recorded_tool_completion_result(
                    self.session_store,
                    outcome_state.session,
                    policy=tool_completion,
                    execution_profile=execution_profile,
                    registered_agent=registered_agent,
                )
                if not messages_to_append
                else None
            )
            if (
                tool_completion_replay is not None
                and tool_completion_result != tool_completion_replay
            ):
                raise RuntimeError("Final-tool recovery lost its exact durable completion basis.")
            skip_model_steps = False
            if (
                (
                    recovered_structured_outcome is not None
                    and recovered_structured_outcome.type == EventType.STRUCTURED_OUTPUT_VALIDATED
                )
                or tool_completion_result is not None
            ) and not messages_to_append:
                if tool_completion_result is not None:
                    await self._session_finalization.stop_at_requested_tool_round_boundary(
                        outcome_state.session
                    )
                if isinstance(execution_to_wait, _ExternalExecutionToWait):
                    async for event in park_external_completion():
                        yield event
                    return
                (
                    outcome_state.session,
                    interaction_completed_event,
                    session_completed,
                    completion_finalization_marker,
                ) = await self._complete_session_if_no_queued_messages(
                    session=outcome_state.session,
                    invocation_context=invocation_context,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    execution_profile=execution_profile,
                    task_id=task_id,
                    tool_completion_result=tool_completion_result,
                )
                if interaction_completed_event is not None:
                    yield interaction_completed_event
                if not session_completed:
                    if producer_replay is not None or tool_completion_replay is not None:
                        raise SessionInterruptedByRequest(outcome_state.session.id)
                    recovered_step = (
                        recovered_structured_outcome.payload.get("step")
                        if recovered_structured_outcome is not None
                        else completed_tool_round_model_step or 1
                    )
                    if type(recovered_step) is not int:
                        raise RuntimeError(
                            "Recovered structured-output result has no valid model step."
                        ) from None
                    queued_completion = None
                    async with contextlib.aclosing(
                        self._handle_queued_messages_before_completion(
                            participant_context=participant_context,
                            finish_completion=complete_session,
                            session=outcome_state.session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            environment_name=environment_name,
                            messages=messages,
                            step=recovered_step,
                            max_steps=max_steps,
                            run_started_at=run_started_at,
                            turn_usage_tracker=turn_usage_tracker,
                            active_run=active_run,
                            execution_profile=execution_profile,
                            invocation_context=invocation_context,
                            predecessor_settlement_event=interaction_completed_event,
                            task_id=task_id,
                        )
                    ) as queued_stream:
                        async for queued_item in queued_stream:
                            if isinstance(queued_item, Event):
                                yield queued_item
                            else:
                                queued_completion = queued_item
                    assert queued_completion is not None
                    outcome_state.session = queued_completion.session
                    completion_finalization_marker = (
                        queued_completion.completion_finalization_marker
                    )
                    rebound_invocation_context = queued_completion.invocation_context
                    if rebound_invocation_context is None:
                        raise RuntimeError("Queued handoff lost live invocation authority.")
                    if rebound_invocation_context is not invocation_context:
                        if rebound_invocation_context.profile.component(
                            ExecutionProfileComponentClass.FINALIZATION
                        ) != invocation_context.profile.component(
                            ExecutionProfileComponentClass.FINALIZATION
                        ):
                            tool_completion = None
                        tool_completion_result = None
                    invocation_context = rebound_invocation_context
                    for event in queued_completion.events:
                        yield event
                    if queued_completion.outcome in {"stopped", "complete"}:
                        return
                else:
                    skip_model_steps = True
            limit_gate = RunLimitGate(
                self._run_limit_controller,
                session=outcome_state.session,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                limits=limits,
                budget_limits=budget_limits,
                run_started_at=run_started_at,
                run_baseline=run_baseline,
                budget_baseline_events=baseline_events,
                budget_notify_events=request_budget_notify_events,
                run_budget_authorities=run_budget_authorities,
                pricing_provider_name=(
                    registered_provider.provider.billing_provider_name or registered_provider.name
                ),
                execution_profile_fingerprint=(
                    None if execution_profile is None else execution_profile.fingerprint
                ),
            )
            tool_round_runner = self._tool_round_executor.create_run(
                session=outcome_state.session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                limit_gate=limit_gate,
                request_metadata=request_metadata,
                task_id=task_id,
                structured_output=structured_output,
                thinking=effective_thinking,
                max_steps=max_steps,
                limits=limits,
                budget_limits=budget_limits,
                retry_policy=retry_policy,
                run_started_at=run_started_at,
                turn_usage_tracker=turn_usage_tracker,
                active_run=active_run,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                run_limit_accounting=run_limit_accounting,
            )

            def model_completion_recovery_context(
                billing_identity: BillingIdentity | None,
                reservations: tuple[BudgetStepReservation, ...],
            ) -> ModelCompletionRecoveryContext | None:
                execution_profile_fingerprint = (
                    None if execution_profile is None else execution_profile.fingerprint
                )
                interaction_id = _current_session_interaction_id(outcome_state.session.id)
                if interaction_id is None:
                    raise RuntimeError(
                        "Model-completion recovery context requires an active interaction."
                    )
                background_provider_operation = (
                    registered_provider.provider.provider_operation_mode
                    is ProviderOperationMode.BACKGROUND
                )
                budget_reservation_contexts = tuple(
                    BudgetReservationRecoveryContext(
                        reservation_id=reservation.record.reservation_id,
                        budget_limit_id=reservation.record.budget_limit_id,
                        limit=(reservation.limit if background_provider_operation else None),
                        reservation_authority_sha256=(
                            budget_reservation_authority_sha256(reservation.record)
                        ),
                    )
                    for reservation in reservations
                )
                if (
                    not background_provider_operation
                    and tool_completion is None
                    and not isinstance(execution_to_wait, _ExternalExecutionToWait)
                    and not retain_completion_controls
                ):
                    return ModelCompletionRecoveryContext(
                        interaction_id=interaction_id,
                        execution_profile_fingerprint=execution_profile_fingerprint,
                        budget_reservations=budget_reservation_contexts,
                        max_steps=max_steps,
                        limits=limits,
                        retry_policy=retry_policy,
                        tool_completion=tool_completion,
                    )
                context = ModelCompletionRecoveryContext(
                    interaction_id=interaction_id,
                    execution_profile_fingerprint=execution_profile_fingerprint,
                    task_id=task_id,
                    request_metadata=request_metadata,
                    structured_output=structured_output,
                    thinking=effective_thinking,
                    max_steps=max_steps,
                    limits=limits,
                    run_limit_accounting=run_limit_accounting,
                    budget_limits=budget_limits,
                    budget_reservations=budget_reservation_contexts,
                    retry_policy=retry_policy,
                    structured_output_attempt=(
                        structured_output_retries + 1
                        if (
                            structured_output is not None
                            and structured_output.strategy == StructuredOutputStrategy.TOOL
                        )
                        else None
                    ),
                    billing_identity=billing_identity,
                    tool_completion=tool_completion,
                )
                payload = context.model_dump(mode="json")
                if self._secret_redactor.redact_json(payload) != payload:
                    raise ValueError(
                        "Model-completion recovery semantics contain a workload secret and "
                        "cannot be persisted exactly; the provider operation was not started."
                    )
                return context

            async def publish_model_completion(
                publication: ModelCompletionPublicationRequest,
            ) -> ModelCompletionPublicationResult:
                return await self._assistant_model_publication.publish(
                    publication,
                    session=outcome_state.session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    task_id=task_id,
                    request_metadata=request_metadata,
                    structured_output=structured_output,
                    thinking=effective_thinking,
                    max_steps=max_steps,
                    limits=limits,
                    budget_limits=budget_limits,
                    retry_policy=retry_policy,
                    structured_output_attempt=(
                        structured_output_retries + 1
                        if (
                            structured_output is not None
                            and structured_output.strategy == StructuredOutputStrategy.TOOL
                            and publication.assistant_step_result is not None
                            and _has_structured_output_tool_call(
                                publication.assistant_step_result.tool_calls
                            )
                        )
                        else None
                    ),
                    structured_output_retries=structured_output_retries,
                    run_limit_accounting=run_limit_accounting,
                )

            live_model_semantic_components = (
                ExecutionProfileComponentClass.CONTEXT_SELECTION,
                ExecutionProfileComponentClass.AUTOMATIC_RECALL,
                ExecutionProfileComponentClass.CONTEXT_COMPACTION,
                ExecutionProfileComponentClass.PROVIDER_ADAPTER,
                ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY,
            )
            request_loop_policy_instance_identities = (
                self._execution_profile_continuation.request_loop_policy_instance_identities(
                    request_loop_policies
                )
            )

            def live_candidate_profile(
                candidate_provider: runtime_records.RegisteredProvider, candidate_model: str
            ) -> ExecutionProfileIdentity:
                candidate = _session_execution_profile._execution_profile_identity(
                    registered_agent=registered_agent,
                    provider_name=candidate_provider.name,
                    registered_provider=candidate_provider,
                    model=candidate_model,
                    durable_system_prompt=None,
                    redactor=self._secret_redactor,
                    registered_environment=registered_environment,
                    process_identity=self._execution_profile_process_identity,
                    runtime_hooks=self._runtime_hooks,
                    loop_policies=self._loop_policies,
                    loop_policy_execution_profile_identities=(
                        self._loop_policy_execution_profile_identities
                    ),
                    request_loop_policies=request_loop_policies,
                    request_loop_policy_instance_identities=(
                        request_loop_policy_instance_identities
                    ),
                    budget_policy=budget_policy,
                    request_budget_limits=budget_limits,
                    causal_budget_id=outcome_state.session.causal_budget_id,
                    structured_output=structured_output,
                    thinking=effective_thinking,
                    max_steps=max_steps,
                    limits=limits,
                    retry_policy=retry_policy,
                    tool_capability_ceiling=_session_execution_profile._session_tool_capability_ceiling(
                        outcome_state.session,
                    ),
                    tool_completion=tool_completion,
                )
                return execution_profile_with_component(
                    candidate,
                    execution_profile.component(
                        ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION
                    ),
                )

            def validate_live_model_semantics() -> None:
                """Fail before mutable registered model semantics can drift."""

                if model_failover is None:
                    candidate = live_candidate_profile(
                        registered_provider, outcome_state.session.model
                    )
                else:
                    # Resolve from the invocation's frozen collaborators, not
                    # registrations installed after the invocation was admitted.
                    providers = {item.name: item for item in model_failover.registered_providers}
                    binding = execution_profile.model_failover
                    if binding is None:
                        raise RuntimeError("Routed invocation lost its profile binding.")
                    candidate = (
                        execution_profile_admission.resolve_model_failover_execution_profile(
                            policy=ModelFailoverPolicy(
                                fallbacks=tuple(
                                    ModelTarget(provider_name=item.provider_name, model=item.model)
                                    for item in model_failover.plan.candidates[1:]
                                ),
                                max_total_attempts=model_failover.plan.max_total_attempts,
                            ),
                            primary=ModelTarget(
                                provider_name=registered_provider.name,
                                model=outcome_state.session.model,
                            ),
                            registered_primary=registered_provider,
                            resolve_provider=providers.__getitem__,
                            resolve_candidate_profile=live_candidate_profile,
                            redactor=self._secret_redactor,
                        ).profile
                    )
                changed = tuple(
                    component_class
                    for component_class in live_model_semantic_components
                    if candidate.component(component_class)
                    != execution_profile.component(component_class)
                )
                if changed:
                    raise ExecutionProfileMismatchError(
                        session_id=outcome_state.session.id,
                        expected_profile_fingerprint=execution_profile.fingerprint,
                        candidate_profile_fingerprint=candidate.fingerprint,
                        changed_component_classes=changed,
                        expected_profile=execution_profile,
                        candidate_profile=candidate,
                    )

            model_step_run = self._model_step_executor.create_run(
                provider=provider,
                session=outcome_state.session,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                environment_name=environment_name,
                structured_output=structured_output,
                thinking=effective_thinking,
                knowledge_store=_knowledge_store(registered_environment),
                knowledge_access_scope=_knowledge_access_scope(registered_environment),
                request_metadata=request_metadata,
                retry_policy=retry_policy,
                request_budget_limits=budget_limits,
                limit_gate=limit_gate,
                budget_policy=budget_policy,
                run_started_at=run_started_at,
                turn_usage_tracker=turn_usage_tracker,
                active_run=active_run,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                validate_live_model_semantics=validate_live_model_semantics,
                initial_tool_exposure=initial_model_step_tool_exposure,
                previous_tool_exposure_profile_id=previous_tool_exposure_profile_id,
                targeted_tool_grants=targeted_tool_grants,
                interaction_id=_current_session_interaction_id(outcome_state.session.id),
                model_completion_recovery_context_factory=(model_completion_recovery_context),
                model_completion_publisher=publish_model_completion,
                model_execution_selection=(
                    None
                    if model_failover is None
                    else ModelExecutionSelection(
                        invocation_context=invocation_context,
                        resolution=model_failover,
                        candidate_index=0
                        if selected_progress is None
                        else selected_progress.candidate_index,
                    )
                ),
            )

            def install_queued_invocation_context(
                rebound: InvocationContext | None,
            ) -> None:
                nonlocal invocation_context, tool_completion_result, tool_completion
                tool_completion_result = None
                if (
                    rebound is not None
                    and rebound is not invocation_context
                    and (
                        rebound.profile.component(ExecutionProfileComponentClass.FINALIZATION)
                        != invocation_context.profile.component(
                            ExecutionProfileComponentClass.FINALIZATION
                        )
                    )
                ):
                    tool_completion = None
                if rebound is None:
                    raise RuntimeError("Queued handoff lost live invocation authority.")
                if rebound is invocation_context:
                    return
                model_step_run.rebind_queued_interaction(rebound)
                tool_round_runner.rebind_queued_interaction(rebound)
                invocation_context = rebound

            if foreground_wait is not None:
                from cayu.runtime._foreground_child_continuation import (
                    load_attached_foreground_continuation,
                )

                attached = await load_attached_foreground_continuation(
                    outcome_state.session, store=self.session_store
                )
                if (
                    attached is None
                    or attached.terminal.wait != foreground_wait
                    or (
                        foreground_wait.parent_effect.interaction_id
                        != invocation_context.binding.interaction_id
                    )
                ):
                    raise SessionRunFenced(
                        "Foreground model continuation lost its attached result."
                    )
                initial_model_step_number = max(
                    initial_model_step_number or 1, attached.completed_model_step + 1
                )
            if completed_tool_round_model_step is not None:
                initial_model_step_number = max(
                    initial_model_step_number or 1, completed_tool_round_model_step + 1
                )
            first_model_step = (
                recovered_assistant_step.step
                if recovered_assistant_step is not None
                else initial_model_step_number or 1
            )
            # Retained final-tool completion checked steering before its settlement.
            if tool_completion_result is None:
                await self._session_finalization.stop_at_requested_tool_round_boundary(
                    outcome_state.session
                )
            model_steps = () if skip_model_steps else range(first_model_step, max_steps + 1)
            # A resumed round can have consumed the final allowed step already.
            # The empty-loop limit path must report that consumed step, not an
            # uninitialized loop variable or a newly granted model request.
            step = first_model_step - 1
            for step in model_steps:
                replayed_step = recovered_assistant_step if step == first_model_step else None
                if producer_replay is not None and replayed_step is None:
                    # Completion recovery is not a new execution grant. A policy
                    # asking for further work must leave this recovery boundary.
                    raise SessionInterruptedByRequest(outcome_state.session.id)
                model_step_identity = (
                    ModelStepIdentity(model_step_id=replayed_step.model_step_id)
                    if replayed_step is not None
                    else initial_model_step_identity
                    if step == first_model_step and initial_model_step_identity is not None
                    else new_model_step_identity()
                )
                await self._session_control.raise_if_interrupted(outcome_state.session.id)
                if (
                    step == first_model_step
                    and deliver_queued_input_before_first_step
                    and replayed_step is None
                    and (invocation_context is None or invocation_context.work_attempt is None)
                ):
                    queued_events, retained_context = await self._deliver_queued_session_messages(
                        session_id=outcome_state.session.id,
                        session=outcome_state.session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        messages=messages,
                        include_on_idle=False,
                        continue_active_interaction=True,
                        invocation_context=invocation_context,
                    )
                    if retained_context is not invocation_context:
                        raise RuntimeError(
                            "Active-interaction queued delivery changed invocation authority."
                        )
                    for event in queued_events:
                        yield event
                # Recovery paths can rematerialize the durable transcript. Reapply the
                # stable retained-prefix projection to the already boundary-sanitized
                # in-memory messages without sanitizing runtime tool controls twice.
                if model_projection_cursor:
                    if len(messages) < model_projection_prefix_count:
                        raise RuntimeError(
                            "Model-facing transcript lost its retained projection prefix."
                        )
                    messages[:] = model_target.project_portable_transcript_prefix(
                        messages,
                        model_projection_prefix_count,
                    ).messages
                source_transcript_cursor = await self.session_store.load_transcript_cursor(
                    outcome_state.session.id
                )
                model_step_flow_outcome: ModelStepFlowOutcome | None = None
                close_new_pending_round_on_interrupt = True
                if replayed_step is not None:
                    # This model attempt already ran and its accounting was
                    # reconciled above. Retain the common post-model gates below,
                    # but do not admit/reserve/dispatch the provider a second time.
                    model_step_flow_outcome = ModelStepFlowOutcome(
                        assistant_step_result=replayed_step
                    )
                else:
                    budget_evaluation = await limit_gate.evaluate_budget(
                        budget_policy,
                        execution_identity=model_step_identity,
                    )
                    async for event in self._session_finalization.apply_budget_evaluation(
                        evaluation=budget_evaluation,
                        session=outcome_state.session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=environment_name,
                        messages=messages,
                        run_started_at=run_started_at,
                        turn_usage_tracker=turn_usage_tracker,
                        active_run=active_run,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    ):
                        yield event
                    if budget_evaluation.check is not None:
                        return
                    limit_evaluation = await limit_gate.evaluate_limits(
                        execution_identity=model_step_identity,
                    )
                    async for event in self._session_finalization.apply_limit_evaluation(
                        evaluation=limit_evaluation,
                        session=outcome_state.session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=environment_name,
                        messages=messages,
                        run_started_at=run_started_at,
                        turn_usage_tracker=turn_usage_tracker,
                        active_run=active_run,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    ):
                        yield event
                    if limit_evaluation.decision is not None:
                        return
                    request_variant = (
                        RequestVariant.STRUCTURED_OUTPUT_REPAIR
                        if structured_output_retries
                        else RequestVariant.INITIAL
                    )
                    model_step_events = model_step_run.execute(
                        step=step,
                        messages=messages,
                        source_transcript_cursor=source_transcript_cursor,
                        model_step_identity=model_step_identity,
                        request_variant=request_variant,
                    )
                    try:
                        async for event, flow_outcome in model_step_events:
                            if event is not None:
                                if (
                                    preserve_failure_until_initial_provider_dispatch
                                    and not initial_provider_dispatch_started
                                    and event.type is EventType.PROVIDER_OPERATION_STARTING
                                    and event.payload.get("model_step_id")
                                    == model_step_identity.model_step_id
                                ):
                                    # Durable replacement stage transfers failure ownership.
                                    initial_provider_dispatch_started = True
                                yield event
                            if flow_outcome is not None:
                                if model_step_flow_outcome is not None:
                                    raise RuntimeError(
                                        "Model step produced more than one terminal flow outcome."
                                    )
                                model_step_flow_outcome = flow_outcome
                    finally:
                        await _close_async_iterator(model_step_events)
                if model_step_flow_outcome is None:
                    await self._session_control.raise_if_interrupted(outcome_state.session.id)
                    raise RuntimeError("Model step finished without a terminal flow outcome.")
                if model_step_flow_outcome.stop_session:
                    return
                if model_step_flow_outcome.portable_history_cursor is not None:
                    if (
                        model_failover is None
                        or model_step_flow_outcome.portable_history_cursor
                        != source_transcript_cursor
                    ):
                        raise RuntimeError("Model failover returned a different history boundary.")
                    # The executor returns an explicit projection boundary;
                    # it never mutates this loop's borrowed history. Preserve
                    # neutral turns and runtime controls, and keep the newly
                    # accepted provider's parts outside the invalidated prefix.
                    model_projection_cursor = source_transcript_cursor
                    snapshot = await self.session_store.load_transcript_snapshot(
                        outcome_state.session.id
                    )
                    durable_projection = _transcript._project_model_target_snapshot(
                        snapshot, model_projection_cursor
                    )
                    model_projection_source_prefix_count = durable_projection.source_prefix_count
                    portable = model_target.project_portable_transcript(messages)
                    messages[:] = portable.messages
                    model_projection_prefix_count = portable.projected_prefix_count
                await self._session_control.raise_if_interrupted(outcome_state.session.id)
                assistant_step_result = model_step_flow_outcome.assistant_step_result
                if assistant_step_result is None:
                    raise RuntimeError("Successful model step finished without a result.")
                completed_model_attempt_identity = ModelAttemptIdentity(
                    model_step_id=assistant_step_result.model_step_id,
                    model_attempt_id=assistant_step_result.model_attempt_id,
                )
                raw_assistant_message = assistant_step_result.assistant_message
                tool_round_identity = assistant_step_result.tool_round_identity
                assistant_message = (
                    transcript_helpers.redact_untrusted_assistant_message_for_boundary(
                        raw_assistant_message,
                        tool_round_identity=tool_round_identity,
                        redactor=self._secret_redactor,
                        field_name="assistant_message",
                    )
                    if raw_assistant_message is not None
                    else None
                )
                tool_calls = assistant_step_result.tool_calls
                tool_round_id = (
                    None if tool_round_identity is None else tool_round_identity.tool_round_id
                )
                pending_tool_round = None

                if tool_calls:
                    if tool_round_identity is None:
                        raise RuntimeError("Tool calls require a tool-round identity.")
                    (
                        _checkpoint,
                        pending_tool_round,
                    ) = await pending_round_reader.load_pending_tool_round(
                        self.session_store,
                        outcome_state.session.id,
                    )
                    if pending_tool_round is None:
                        raise RuntimeError(
                            "Model completion with tool calls lost its durable pending marker."
                        )
                    if (
                        pending_rounds.pending_tool_round_identity(pending_tool_round)
                        != tool_round_identity
                    ):
                        raise RuntimeError(
                            "Model completion tool-round identity conflicts with its "
                            "durable pending marker."
                        )
                    durable_tool_calls = tool_round_recovery.pending_round_tool_calls(
                        pending_tool_round
                    )
                    expected_durable_tool_calls = tool_calls
                    if (
                        structured_output is not None
                        and structured_output.strategy == StructuredOutputStrategy.TOOL
                        and _has_structured_output_tool_call(tool_calls)
                    ):
                        expected_durable_tool_calls = []
                        for call in tool_calls:
                            arguments = self._secret_redactor.redact_json_values(call.arguments)
                            if type(arguments) is not dict:
                                raise AssertionError(
                                    "Structured-output arguments redacted as a non-object."
                                )
                            if (
                                call.name == CALL_TOOL_NAME
                                and call.targeted_tool_grant_id is not None
                                and type(call.arguments.get("tool_ref")) is str
                            ):
                                arguments["tool_ref"] = call.arguments["tool_ref"]
                            expected_durable_tool_calls.append(
                                runtime_records.copy_tool_call_request(
                                    call,
                                    arguments=arguments,
                                )
                            )
                    if durable_tool_calls != expected_durable_tool_calls:
                        raise RuntimeError(
                            "Model completion tool calls conflict with its durable pending marker."
                        )
                if (
                    assistant_message is not None
                    and (
                        pending_tool_round is None
                        or pending_tool_round.assistant_message_state == "published"
                    )
                    and (replayed_step is None or model_boundary.state == "promoted")
                ):
                    messages.append(assistant_message)
                limit_evaluation = await limit_gate.evaluate_limits(
                    pending_tool_calls=_user_tool_call_count(tool_calls),
                    execution_identity=completed_model_attempt_identity,
                )
                async for event in self._session_finalization.apply_limit_evaluation(
                    evaluation=limit_evaluation,
                    session=outcome_state.session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    messages=messages,
                    tool_calls=tool_calls,
                    tool_round_identity=tool_round_identity,
                    run_started_at=run_started_at,
                    turn_usage_tracker=turn_usage_tracker,
                    active_run=active_run,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                ):
                    yield event
                if limit_evaluation.decision is not None:
                    return

                budget_evaluation = await limit_gate.evaluate_budget(
                    budget_policy,
                    execution_identity=completed_model_attempt_identity,
                )
                async for event in self._session_finalization.apply_budget_evaluation(
                    evaluation=budget_evaluation,
                    session=outcome_state.session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    messages=messages,
                    tool_calls=tool_calls,
                    tool_round_identity=tool_round_identity,
                    run_started_at=run_started_at,
                    turn_usage_tracker=turn_usage_tracker,
                    active_run=active_run,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                ):
                    yield event
                if budget_evaluation.check is not None:
                    return

                if not tool_calls and completion_requests_follow_up(
                    assistant_step_result.completion
                ):
                    continue

                if (
                    structured_output is not None
                    and structured_output.strategy == StructuredOutputStrategy.TOOL
                    and _has_structured_output_tool_call(tool_calls)
                ):
                    if (
                        pending_tool_round is None
                        or tool_round_identity is None
                        or tool_round_id is None
                    ):
                        raise RuntimeError(
                            "Structured-output tool round lost its durable pending marker."
                        )
                    round_owner = DurableToolRound(
                        session=outcome_state.session,
                        tool_round_identity=tool_round_identity,
                        session_store=self.session_store,
                        event_writer=self._event_writer,
                    )
                    structured_publication = None
                    async with contextlib.aclosing(
                        round_owner.publish_structured(
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            messages=messages,
                            tool_calls=tool_calls,
                            pending_round=pending_tool_round,
                            spec=structured_output,
                            step=step,
                            attempt=structured_output_retries + 1,
                            retry_allowed=step < max_steps,
                            execution_profile=execution_profile,
                            redactor=self._secret_redactor,
                            tool_redactor=self._tool_round_executor._secret_redactor,
                        )
                    ) as structured_events:
                        async for item in structured_events:
                            if isinstance(item, Event):
                                yield item
                            else:
                                structured_publication = item
                    assert structured_publication is not None
                    validation = structured_publication.validation
                    cancellation = structured_publication.cancellation
                    close_new_pending_round_on_interrupt = False
                    if cancellation is not None:
                        raise cancellation
                    await self._session_control.raise_if_interrupted(outcome_state.session.id)
                    for event in structured_publication.events:
                        yield copy_event(event)
                    await self._session_finalization.stop_at_requested_tool_round_boundary(
                        outcome_state.session
                    )

                    if validation.valid:
                        self._record_workflow_structured_output(
                            outcome_state.session.id,
                            validation.output,
                        )
                        if isinstance(execution_to_wait, _ExternalExecutionToWait):
                            async for event in park_external_completion():
                                yield event
                            return
                        (
                            outcome_state.session,
                            interaction_completed_event,
                            session_completed,
                            completion_finalization_marker,
                        ) = await self._complete_session_if_no_queued_messages(
                            session=outcome_state.session,
                            invocation_context=invocation_context,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            environment_name=environment_name,
                            execution_profile=execution_profile,
                            task_id=task_id,
                            tool_completion_result=None,
                        )
                        if interaction_completed_event is not None:
                            yield interaction_completed_event
                        if not session_completed:
                            if producer_replay is not None:
                                raise SessionInterruptedByRequest(outcome_state.session.id)
                            queued_completion = None
                            async with contextlib.aclosing(
                                self._handle_queued_messages_before_completion(
                                    participant_context=participant_context,
                                    finish_completion=complete_session,
                                    session=outcome_state.session,
                                    registered_agent=registered_agent,
                                    registered_environment=registered_environment,
                                    environment_name=environment_name,
                                    messages=messages,
                                    step=step,
                                    max_steps=max_steps,
                                    run_started_at=run_started_at,
                                    turn_usage_tracker=turn_usage_tracker,
                                    active_run=active_run,
                                    execution_profile=execution_profile,
                                    invocation_context=invocation_context,
                                    predecessor_settlement_event=interaction_completed_event,
                                    task_id=task_id,
                                )
                            ) as queued_stream:
                                async for queued_item in queued_stream:
                                    if isinstance(queued_item, Event):
                                        yield queued_item
                                    else:
                                        queued_completion = queued_item
                            assert queued_completion is not None
                            outcome_state.session = queued_completion.session
                            completion_finalization_marker = (
                                queued_completion.completion_finalization_marker
                            )
                            install_queued_invocation_context(queued_completion.invocation_context)
                            for event in queued_completion.events:
                                yield event
                            if queued_completion.outcome == "stopped":
                                return
                            if queued_completion.outcome == "complete":
                                return
                            continue
                        break
                    if structured_output_retries >= structured_output.max_retries:
                        raise RuntimeError(
                            "Structured output validation failed after "
                            f"{structured_output_retries + 1} attempt(s)."
                        )
                    if step >= max_steps:
                        raise RuntimeError(
                            "Structured output validation failed after "
                            f"{structured_output_retries + 1} attempt(s): "
                            "maximum model steps reached before repair."
                        )
                    structured_output_retries += 1
                    continue

                if not tool_calls:
                    close_new_pending_round_on_interrupt = False
                    await self._session_finalization.stop_at_requested_tool_round_boundary(
                        outcome_state.session
                    )
                    if structured_output is not None:
                        yield await emit_structured_event(
                            event_with_execution_profile_authority(
                                _structured_output_validating_event(
                                    session=outcome_state.session,
                                    registered_agent=registered_agent,
                                    environment_name=environment_name,
                                    spec=structured_output,
                                    step=step,
                                    attempt=structured_output_retries + 1,
                                    execution_identity=completed_model_attempt_identity,
                                ),
                                execution_profile,
                            )
                        )
                        if structured_output.strategy == StructuredOutputStrategy.NATIVE:
                            if assistant_step_result is None:
                                raise RuntimeError(
                                    "Native structured output validation requires an "
                                    "assistant step result."
                                )
                            validation = validate_structured_output_text(
                                assistant_step_result.text_content,
                                structured_output,
                            )
                            if validation.valid:
                                self._record_workflow_structured_output(
                                    outcome_state.session.id,
                                    validation.output,
                                )
                                yield await emit_structured_event(
                                    event_with_execution_profile_authority(
                                        _structured_output_event(
                                            event_type=EventType.STRUCTURED_OUTPUT_VALIDATED,
                                            session=outcome_state.session,
                                            registered_agent=registered_agent,
                                            environment_name=environment_name,
                                            spec=structured_output,
                                            validation=validation,
                                            step=step,
                                            attempt=structured_output_retries + 1,
                                            execution_identity=completed_model_attempt_identity,
                                            redactor=self._secret_redactor,
                                        ),
                                        execution_profile,
                                    )
                                )
                                if isinstance(execution_to_wait, _ExternalExecutionToWait):
                                    async for event in park_external_completion():
                                        yield event
                                    return
                                (
                                    outcome_state.session,
                                    interaction_completed_event,
                                    session_completed,
                                    completion_finalization_marker,
                                ) = await self._complete_session_if_no_queued_messages(
                                    session=outcome_state.session,
                                    invocation_context=invocation_context,
                                    registered_agent=registered_agent,
                                    registered_environment=registered_environment,
                                    environment_name=environment_name,
                                    execution_profile=execution_profile,
                                    task_id=task_id,
                                    tool_completion_result=None,
                                )
                                if interaction_completed_event is not None:
                                    yield interaction_completed_event
                                if not session_completed:
                                    if producer_replay is not None:
                                        raise SessionInterruptedByRequest(outcome_state.session.id)
                                    queued_completion = None
                                    async with contextlib.aclosing(
                                        self._handle_queued_messages_before_completion(
                                            participant_context=participant_context,
                                            finish_completion=complete_session,
                                            session=outcome_state.session,
                                            registered_agent=registered_agent,
                                            registered_environment=registered_environment,
                                            environment_name=environment_name,
                                            messages=messages,
                                            step=step,
                                            max_steps=max_steps,
                                            run_started_at=run_started_at,
                                            turn_usage_tracker=turn_usage_tracker,
                                            active_run=active_run,
                                            execution_profile=execution_profile,
                                            invocation_context=invocation_context,
                                            predecessor_settlement_event=(
                                                interaction_completed_event
                                            ),
                                            task_id=task_id,
                                        )
                                    ) as queued_stream:
                                        async for queued_item in queued_stream:
                                            if isinstance(queued_item, Event):
                                                yield queued_item
                                            else:
                                                queued_completion = queued_item
                                    assert queued_completion is not None
                                    outcome_state.session = queued_completion.session
                                    completion_finalization_marker = (
                                        queued_completion.completion_finalization_marker
                                    )
                                    install_queued_invocation_context(
                                        queued_completion.invocation_context
                                    )
                                    for event in queued_completion.events:
                                        yield event
                                    if queued_completion.outcome == "stopped":
                                        return
                                    if queued_completion.outcome == "complete":
                                        return
                                    continue
                                break
                        else:
                            validation = structured_output_tool_required_validation()
                        yield await emit_structured_event(
                            event_with_execution_profile_authority(
                                _structured_output_event(
                                    event_type=EventType.STRUCTURED_OUTPUT_FAILED,
                                    session=outcome_state.session,
                                    registered_agent=registered_agent,
                                    environment_name=environment_name,
                                    spec=structured_output,
                                    validation=validation,
                                    step=step,
                                    attempt=structured_output_retries + 1,
                                    execution_identity=completed_model_attempt_identity,
                                    redactor=self._secret_redactor,
                                ),
                                execution_profile,
                            )
                        )
                        if structured_output_retries >= structured_output.max_retries:
                            raise RuntimeError(
                                "Structured output validation failed after "
                                f"{structured_output_retries + 1} attempt(s)."
                            )
                        if step >= max_steps:
                            raise RuntimeError(
                                "Structured output validation failed after "
                                f"{structured_output_retries + 1} attempt(s): "
                                "maximum model steps reached before repair."
                            )
                        structured_output_retries += 1
                        redacted_validation = _redact_structured_output_validation(
                            validation,
                            self._secret_redactor,
                        )
                        repair_message = Message.text(
                            "user",
                            structured_output_repair_prompt(
                                spec=structured_output,
                                validation=redacted_validation,
                            ),
                        )
                        messages.append(repair_message)
                        runtime_message_transform = (
                            _runtime_authored_user_message_checkpoint_transform(
                                anchor_index=len(messages) - 1,
                                message=repair_message,
                            )
                        )
                        await (
                            self.session_store.append_transcript_messages_and_transform_checkpoint(
                                outcome_state.session.id,
                                [repair_message],
                                runtime_message_transform,
                            )
                        )
                        yield await emit_structured_event(
                            event_with_execution_profile_authority(
                                _structured_output_event(
                                    event_type=EventType.STRUCTURED_OUTPUT_RETRY,
                                    session=outcome_state.session,
                                    registered_agent=registered_agent,
                                    environment_name=environment_name,
                                    spec=structured_output,
                                    validation=validation,
                                    step=step,
                                    attempt=structured_output_retries,
                                    execution_identity=completed_model_attempt_identity,
                                    redactor=self._secret_redactor,
                                ),
                                execution_profile,
                            )
                        )
                        continue
                    if assistant_step_result is None:
                        raise RuntimeError("Before-stop policies require an assistant step result.")
                    before_stop_decision: BeforeStopDecision | None = None
                    selected_policy_key: str | None = None
                    async for event, policy_decision in self._run_before_stop_policies(
                        session=outcome_state.session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        step_result=assistant_step_result,
                        step=step,
                        max_steps=max_steps,
                        request_metadata=request_metadata,
                        request_loop_policies=request_loop_policies,
                        invocation_context=invocation_context,
                    ):
                        yield event
                        if policy_decision is not None:
                            before_stop_decision = policy_decision.decision
                            selected_policy_key = policy_decision.policy_key
                    if execution_to_wait is not None and (
                        before_stop_decision is None
                        or before_stop_decision.action == BeforeStopAction.COMPLETE
                    ):
                        await execution_to_wait.park(invocation_context)
                        before_stop_decision = BeforeStopDecision(
                            action=BeforeStopAction.INTERRUPT,
                            reason="Explicit continuation wait reached a whole-turn boundary.",
                        )
                    if before_stop_decision is not None:
                        if before_stop_decision.action == BeforeStopAction.CONTINUE:
                            if step >= max_steps:
                                async for event in self._stop_session_for_model_step_limit(
                                    session=outcome_state.session,
                                    registered_agent=registered_agent,
                                    registered_environment=registered_environment,
                                    environment_name=environment_name,
                                    messages=messages,
                                    step=step,
                                    max_steps=max_steps,
                                    run_started_at=run_started_at,
                                    turn_usage_tracker=turn_usage_tracker,
                                    active_run=active_run,
                                    execution_profile=execution_profile,
                                    invocation_context=invocation_context,
                                ):
                                    yield event
                                return
                            if before_stop_decision.message is None:
                                raise RuntimeError(
                                    "Before-stop continue decision requires a message."
                                )
                            repair_message = before_stop_decision.message
                            if selected_policy_key is None:
                                raise RuntimeError("Before-stop continuation lost its policy.")
                            messages.append(repair_message)
                            runtime_message_transform = (
                                _runtime_authored_user_message_checkpoint_transform(
                                    anchor_index=len(messages) - 1,
                                    message=repair_message,
                                )
                            )
                            continuation_transform = before_stop_continuation_checkpoint_transform(
                                snapshot=await self.session_store.load_transcript_snapshot(
                                    outcome_state.session.id
                                ),
                                message=repair_message,
                                policy_key=selected_policy_key,
                                checkpoint_transform=runtime_message_transform,
                            )
                            await self.session_store.append_transcript_messages_and_transform_checkpoint(
                                outcome_state.session.id,
                                [repair_message],
                                continuation_transform,
                            )
                            del continuation_transform
                            continue
                        if before_stop_decision.action == BeforeStopAction.INTERRUPT:
                            async for event in self._session_outcomes.publish_stop_interruption(
                                outcome_context(), before_stop_decision
                            ):
                                yield event
                            return
                        if before_stop_decision.action == BeforeStopAction.FAIL:
                            raise RuntimeError(
                                f"Before-stop policy failed session: {before_stop_decision.reason}"
                            )
                    (
                        outcome_state.session,
                        interaction_completed_event,
                        session_completed,
                        completion_finalization_marker,
                    ) = await self._complete_session_if_no_queued_messages(
                        session=outcome_state.session,
                        invocation_context=invocation_context,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=environment_name,
                        execution_profile=execution_profile,
                        task_id=task_id,
                        tool_completion_result=None,
                    )
                    if interaction_completed_event is not None:
                        yield interaction_completed_event
                    if not session_completed:
                        if producer_replay is not None:
                            # A retained result may complete its own interaction,
                            # but cannot consume a queue or admit its successor.
                            raise SessionInterruptedByRequest(outcome_state.session.id)
                        queued_completion = None
                        async with contextlib.aclosing(
                            self._handle_queued_messages_before_completion(
                                participant_context=participant_context,
                                finish_completion=complete_session,
                                session=outcome_state.session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                environment_name=environment_name,
                                messages=messages,
                                step=step,
                                max_steps=max_steps,
                                run_started_at=run_started_at,
                                turn_usage_tracker=turn_usage_tracker,
                                active_run=active_run,
                                execution_profile=execution_profile,
                                invocation_context=invocation_context,
                                predecessor_settlement_event=interaction_completed_event,
                                task_id=task_id,
                            )
                        ) as queued_stream:
                            async for queued_item in queued_stream:
                                if isinstance(queued_item, Event):
                                    yield queued_item
                                else:
                                    queued_completion = queued_item
                        assert queued_completion is not None
                        outcome_state.session = queued_completion.session
                        completion_finalization_marker = (
                            queued_completion.completion_finalization_marker
                        )
                        install_queued_invocation_context(queued_completion.invocation_context)
                        for event in queued_completion.events:
                            yield event
                        if queued_completion.outcome == "stopped":
                            return
                        if queued_completion.outcome == "complete":
                            return
                        continue
                    break

                if tool_round_identity is None:
                    raise RuntimeError("Ordinary tool calls require a tool-round identity.")
                # The round may yield interruption evidence after cancellation.
                # Observe its close before session cleanup releases the run fence;
                # otherwise async-generator finalization can orphan its failure.
                round_stream = tool_round_runner.run(
                    messages=messages,
                    tool_calls=tool_calls,
                    tool_round_identity=tool_round_identity,
                    model_step=step,
                )
                async with contextlib.aclosing(round_stream) as owned_round_stream:
                    async for event in owned_round_stream:
                        # The deadline stream cannot publish after its timer expires.
                        # Drain the round's interruption evidence until it re-raises
                        # cancellation instead of suspending it at a yield that the
                        # deadline owner must immediately close with GeneratorExit.
                        if expired_execution_deadline() is None:
                            yield event
                close_new_pending_round_on_interrupt = False
                if tool_round_runner.user_input_pause_superseded:
                    # A remote operator-interruption transaction already removed
                    # the exact pause and owns its round disposition. Re-observe
                    # durable interruption only after disabling ordinary-round
                    # cleanup, which would otherwise require a marker that the
                    # supersession transaction intentionally cleared.
                    await self._session_control.raise_if_interrupted(outcome_state.session.id)
                    raise RuntimeError(
                        "Superseded user-input pause lost durable interruption authority."
                    )
                if tool_round_runner.stopped_for_limit:
                    return
                tool_completion_result = await recorded_tool_completion_result(
                    self.session_store,
                    outcome_state.session,
                    policy=tool_completion,
                    execution_profile=execution_profile,
                    registered_agent=registered_agent,
                )
                await self._session_finalization.stop_at_requested_tool_round_boundary(
                    outcome_state.session
                )
                if tool_completion_result is not None:
                    if isinstance(execution_to_wait, _ExternalExecutionToWait):
                        async for event in park_external_completion():
                            yield event
                        return
                    (
                        outcome_state.session,
                        interaction_completed_event,
                        session_completed,
                        completion_finalization_marker,
                    ) = await self._complete_session_if_no_queued_messages(
                        session=outcome_state.session,
                        invocation_context=invocation_context,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=environment_name,
                        execution_profile=execution_profile,
                        task_id=task_id,
                        tool_completion_result=tool_completion_result,
                    )
                    if interaction_completed_event is not None:
                        yield interaction_completed_event
                    if not session_completed:
                        if producer_replay is not None:
                            # A retained result may complete its own interaction,
                            # but cannot consume a queue or admit its successor.
                            raise SessionInterruptedByRequest(outcome_state.session.id)
                        queued_completion = None
                        async with contextlib.aclosing(
                            self._handle_queued_messages_before_completion(
                                participant_context=participant_context,
                                finish_completion=complete_session,
                                session=outcome_state.session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                environment_name=environment_name,
                                messages=messages,
                                step=step,
                                max_steps=max_steps,
                                run_started_at=run_started_at,
                                turn_usage_tracker=turn_usage_tracker,
                                active_run=active_run,
                                execution_profile=execution_profile,
                                invocation_context=invocation_context,
                                predecessor_settlement_event=interaction_completed_event,
                                task_id=task_id,
                            )
                        ) as queued_stream:
                            async for queued_item in queued_stream:
                                if isinstance(queued_item, Event):
                                    yield queued_item
                                else:
                                    queued_completion = queued_item
                        assert queued_completion is not None
                        outcome_state.session = queued_completion.session
                        completion_finalization_marker = (
                            queued_completion.completion_finalization_marker
                        )
                        install_queued_invocation_context(queued_completion.invocation_context)
                        for event in queued_completion.events:
                            yield event
                        if queued_completion.outcome == "stopped":
                            return
                        if queued_completion.outcome == "complete":
                            return
                        continue
                    break
            else:
                if not skip_model_steps:
                    async for event in self._stop_session_for_model_step_limit(
                        session=outcome_state.session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=environment_name,
                        messages=messages,
                        step=step,
                        max_steps=max_steps,
                        run_started_at=run_started_at,
                        turn_usage_tracker=turn_usage_tracker,
                        active_run=active_run,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    ):
                        yield event
                    return

            async with contextlib.aclosing(
                self._session_outcomes.complete_session(
                    outcome_context(),
                    outcome_state.session,
                    completion_finalization_marker,
                    tool_completion_result,
                )
            ) as completion_stream:
                async for event in completion_stream:
                    yield event
        except ToolApprovalRequired as exc:
            async with contextlib.aclosing(
                self._session_outcomes.pause_for_tool_approval(
                    outcome_context(),
                    exc=exc,
                    materialize_deferred_messages_after_failure=materialize_deferred_messages_after_failure,
                )
            ) as outcome_stream:
                async for event in outcome_stream:
                    yield event
        except UserInputRequired as exc:
            async with contextlib.aclosing(
                self._session_outcomes.pause_for_user_input(
                    outcome_context(),
                    exc=exc,
                    materialize_deferred_messages_after_failure=materialize_deferred_messages_after_failure,
                )
            ) as outcome_stream:
                async for event in outcome_stream:
                    yield event
        except ForegroundChildActionRequired as exc:
            async with contextlib.aclosing(
                self._session_outcomes.pause_for_foreground_child(
                    outcome_context(),
                    exc=exc,
                    materialize_deferred_messages_after_failure=materialize_deferred_messages_after_failure,
                )
            ) as outcome_stream:
                async for event in outcome_stream:
                    yield event
        except ToolEffectReconciliationRequired as unresolved:
            authoritative_failure = (
                unresolved.failures
                if isinstance(unresolved, ToolEffectReconciliationCleanupFailure)
                else unresolved
            )
            events, propagated = await self._handle_session_interrupted_preserving_failure(
                authoritative_failure=authoritative_failure,
                foreground_subagent_pending=(
                    unresolved
                    if isinstance(unresolved, ForegroundSubagentRecoveryRequired)
                    else None
                ),
                prepare_interruption=lambda: (
                    self._pending_tool_round_recovery.deliver_pending_tool_effect_uncertainty(
                        outcome_state.session
                    )
                ),
                session=outcome_state.session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                run_started_at=run_started_at,
                turn_usage_tracker=turn_usage_tracker,
                active_run=active_run,
            )
            for event in events:
                yield event
            if propagated is not authoritative_failure:
                raise propagated from None
            if isinstance(unresolved, ToolEffectReconciliationCleanupFailure):
                raise authoritative_failure from None
            return
        except SessionInterruptedByRequest as interruption:
            from cayu.runtime._session_steering import SessionSteeringBoundaryReached

            promote_rejected_stop = isinstance(interruption, SessionSteeringBoundaryReached)

            async def prepare_requested_interruption() -> list[Event]:
                if promote_rejected_stop:
                    await self._session_finalization.promote_rejected_completion_stop(
                        outcome_state.session
                    )
                await materialize_deferred_messages_after_failure()
                prepared_events: list[Event] = []
                if close_new_pending_round_on_interrupt:
                    async for event in self._close_durable_pending_tool_round_after_interrupt(
                        session=outcome_state.session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    ):
                        prepared_events.append(event)
                return prepared_events

            # A detached observer may redeliver its interrupt while terminal
            # publication is awaiting storage. Finish that owned publication
            # and retain its events before propagating concurrent cancellation.
            # Acknowledge this task's handled request before handing settlement
            # to a child task. A deferred observer signal belongs to the parent;
            # the child's publication marker cannot acknowledge it on its own.
            self._session_control.begin_emitting_interrupted(outcome_state.session.id)
            try:
                (
                    interruption_events,
                    propagated,
                ) = await self._handle_session_interrupted_preserving_failure(
                    authoritative_failure=interruption,
                    prepare_interruption=prepare_requested_interruption,
                    include_interrupted_session=True,
                    session=outcome_state.session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                    run_started_at=run_started_at,
                    turn_usage_tracker=turn_usage_tracker,
                    active_run=active_run,
                )
            finally:
                self._session_control.end_emitting_interrupted(outcome_state.session.id)
            for event in interruption_events:
                yield event
            if propagated is not interruption:
                additional_failure = _failure_without_existing_exception_identities(
                    propagated, {id(interruption)}
                )
                if additional_failure is not None:
                    if (
                        isinstance(additional_failure, BaseExceptionGroup)
                        and len(additional_failure.exceptions) == 1
                    ):
                        # Removing the handled request leaves the original
                        # sole failure, including process-control exceptions.
                        additional_failure = additional_failure.exceptions[0]
                    pending_requests = workspace_observation_pending_cancellation_requests(
                        propagated
                    )
                    if pending_requests:
                        retain_workspace_observation_pending_cancellation_requests(
                            additional_failure, pending_requests
                        )
                    raise additional_failure from None
            return
        except asyncio.CancelledError as cancellation:
            provider_cancellation_diagnostics = (
                provider_cancellation_failures(cancellation)
                + billing_identity_cancellation_failures(cancellation)
            )[:2]
            if _consume_authenticated_interaction_transition_run_fence(cancellation):
                # The owned transition attempt proved that this worker lost
                # authority. Preserve caller cancellation without attempting
                # abandoned-run writes under the stale epoch. The transition
                # handoff is runtime-only evidence for cleanup that this branch
                # must not perform, so do not let it escape on the public error.
                pop_exception_state(
                    cancellation,
                    _INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE,
                )
                raise
            transition_cancellation_outcome = _pop_interaction_transition_cancellation_outcome(
                cancellation
            )
            interaction_transition_failures: list[dict[str, Any]] = []
            interaction_transition: InteractionTransitionSpec | None = None
            if transition_cancellation_outcome is not None:
                (
                    transition_settled,
                    interaction_transition_failures,
                    interaction_transition,
                ) = transition_cancellation_outcome
            else:
                transition_settled = False
            if interaction_transition_failures and interaction_transition is not None:
                committed_transition_recorded = False

                async def record_committed_transition_cancellation() -> None:
                    nonlocal committed_transition_recorded
                    committed_transition_recorded = await self._session_finalization.record_committed_interaction_transition_cancellation(
                        RecoveryAbandonedSessionRequest(
                            session=outcome_state.session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            environment_name=environment_name,
                            run_started_at=run_started_at,
                            turn_usage_tracker=turn_usage_tracker,
                            active_run=active_run,
                            interaction_transition_failures=tuple(interaction_transition_failures),
                            interaction_transition=interaction_transition,
                            execution_profile=execution_profile,
                            invocation_context=invocation_context,
                        )
                    )

                diagnostic_failures = await _run_interaction_transition_cancellation_cleanup_steps(
                    cancellation,
                    supervisor=self._recovery_cleanup_supervisor,
                    steps=(
                        (
                            "committed interaction-transition cancellation diagnostics",
                            record_committed_transition_cancellation,
                        ),
                    ),
                )
                if committed_transition_recorded or diagnostic_failures:
                    # Exact committed evidence, or inability to classify it,
                    # forbids a later cleanup branch from rewriting status.
                    raise
            if transition_settled or _latest_session_invocation_interaction_is_settled(
                outcome_state.session.id
            ):
                # The authoritative interaction/session decision already
                # committed. Its durable side-effect handoff remains
                # recoverable, so abandoned-run cleanup must not rewrite it.
                raise
            billing_identity_cancellation = detach_billing_identity_cancellation(cancellation)
            if (
                billing_identity_cancellation is not None
                and preserve_failure_until_initial_provider_dispatch
                and self._session_control.is_interruption_request_active(outcome_state.session.id)
            ):
                # Provider-operation fallback recovery has no equivalent outer
                # run boundary. Drop provider-bearing locals before fallible
                # interruption work and surface only a detached cancellation.
                del provider, registered_provider
                try:
                    interruption_requested = await self._session_control.interrupt_requested(
                        outcome_state.session.id
                    )
                except Exception:
                    raise RuntimeError(
                        "Session interruption state check failed after provider billing "
                        "cancellation"
                    ) from None
                if interruption_requested:
                    clear_current_task_cancellation()
                    try:
                        await materialize_deferred_messages_after_failure()
                        interruption_events = []
                        if close_new_pending_round_on_interrupt:
                            async for (
                                event
                            ) in self._close_durable_pending_tool_round_after_interrupt(
                                session=outcome_state.session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                execution_profile=execution_profile,
                                invocation_context=invocation_context,
                            ):
                                interruption_events.append(event)
                        async for event in self._session_finalization.handle_session_interrupted(
                            session=outcome_state.session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            environment_name=environment_name,
                            execution_profile=execution_profile,
                            invocation_context=invocation_context,
                            run_started_at=run_started_at,
                            turn_usage_tracker=turn_usage_tracker,
                            active_run=active_run,
                            interaction_transition_failures=tuple(interaction_transition_failures),
                            provider_cancellation_failures=(provider_cancellation_diagnostics),
                        ):
                            interruption_events.append(event)
                    except Exception:
                        raise RuntimeError(
                            "Session interruption transition failed after provider billing "
                            "cancellation"
                        ) from None
                    for event in interruption_events:
                        yield event
                    return
                raise billing_identity_cancellation from None
            provider_cancellation_authenticated = (
                billing_identity_cancellation is not None
                or detach_credential_safe_provider_cancellation(cancellation) is not None
            )
            interruption_classification: tuple[bool, bool] | None = None

            async def classify_cancelled_interruption() -> None:
                nonlocal interruption_classification
                requested = await self._session_control.interrupt_requested(
                    outcome_state.session.id
                )
                promoted = (
                    await self._has_promoted_safe_stop(outcome_state.session)
                    if requested
                    else False
                )
                interruption_classification = (requested, promoted)

            # A completed owned mutation may restore cancellation requests before
            # propagating its original signal. These reads are part of cancellation
            # cleanup too: redelivery must not skip terminal settlement or turn the
            # ensuing fence release into the primary failure.
            cancellation_had_cause = exception_cause(cancellation) is not None
            classification_failures = await self._recovery_ownership.run_cleanup_steps(
                authoritative_failure=cancellation,
                steps=(("cancelled interruption classification", classify_cancelled_interruption),),
            )
            if classification_failures or interruption_classification is None:
                if len(classification_failures) == 1 and not cancellation_had_cause:
                    # Keep the established single-read failure shape; retain the
                    # supervisor's aggregate when earlier failures also exist.
                    raise cancellation from classification_failures[0][1]
                raise cancellation
            interruption_requested, preserve_caller_cancellation = interruption_classification
            if interruption_requested:
                interruption_events = []

                async def finish_requested_interruption() -> None:
                    await materialize_deferred_messages_after_failure()
                    if close_new_pending_round_on_interrupt:
                        async for event in self._close_durable_pending_tool_round_after_interrupt(
                            session=outcome_state.session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            execution_profile=execution_profile,
                            invocation_context=invocation_context,
                        ):
                            interruption_events.append(event)
                    async for event in self._session_finalization.handle_session_interrupted(
                        session=outcome_state.session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=environment_name,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                        run_started_at=run_started_at,
                        turn_usage_tracker=turn_usage_tracker,
                        active_run=active_run,
                        interaction_transition_failures=tuple(interaction_transition_failures),
                        provider_cancellation_failures=provider_cancellation_diagnostics,
                    ):
                        interruption_events.append(event)

                if preserve_caller_cancellation:
                    await _run_interaction_transition_cancellation_cleanup_steps(
                        cancellation,
                        supervisor=self._recovery_cleanup_supervisor,
                        steps=(
                            ("cooperative interruption settlement", finish_requested_interruption),
                        ),
                    )
                    raise cancellation
                clear_current_task_cancellation()
                await finish_requested_interruption()
                for event in interruption_events:
                    yield event
                return
            if (
                preserve_failure_until_initial_provider_dispatch
                and not initial_provider_dispatch_started
            ):
                # Ordinary caller cancellation does not revoke an accepted
                # fallback. A positively authenticated interrupt request above
                # remains authoritative and completes its typed outcome.
                raise

            async def close_cancelled_tool_round() -> None:
                if not close_new_pending_round_on_interrupt:
                    return
                async for _ in self._close_durable_pending_tool_round_after_interrupt(
                    session=outcome_state.session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                ):
                    pass

            # Exception targets are cleared when their handler exits. Retained
            # cleanup needs only the safe deadline evidence, not the exception.
            native_admission_deadline = provider_cancellation_admission_deadline(cancellation)
            cancellation_terminal_event: Event | None = None
            deadline_expired = expired_execution_deadline() is not None

            async def finalize_cancelled_session() -> None:
                nonlocal cancellation_terminal_event
                if deadline_expired and invocation_context.work_attempt is not None:
                    if self.task_store is None:
                        raise RuntimeError("Governed deadline stop requires its task store.")
                    await record_work_attempt_execution_stop(
                        self.task_store,
                        admission=invocation_context.work_attempt.admission,
                        reason="elapsed_limit",
                        redactor=self._secret_redactor,
                    )
                cancellation_terminal_event = (
                    await self._session_finalization.finalize_abandoned_session_run(
                        RecoveryAbandonedSessionRequest(
                            session=outcome_state.session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            environment_name=environment_name,
                            run_started_at=run_started_at,
                            turn_usage_tracker=turn_usage_tracker,
                            active_run=active_run,
                            interaction_transition_failures=tuple(interaction_transition_failures),
                            interaction_transition=interaction_transition,
                            provider_cancellation_failures=(provider_cancellation_diagnostics),
                            native_admission_deadline=native_admission_deadline,
                            execution_profile=execution_profile,
                            invocation_context=invocation_context,
                        )
                    )
                )

            cancellation_cleanup_steps = (
                (
                    "cancelled tool-round finalization",
                    close_cancelled_tool_round,
                ),
                (
                    "cancelled session finalization",
                    finalize_cancelled_session,
                ),
            )
            if interaction_transition_failures:
                cancellation_cleanup_failures = (
                    await _run_interaction_transition_cancellation_cleanup_steps(
                        cancellation,
                        steps=cancellation_cleanup_steps,
                        supervisor=self._recovery_cleanup_supervisor,
                    )
                )
            else:
                cancellation_cleanup_failures = await self._recovery_ownership.run_cleanup_steps(
                    authoritative_failure=cancellation,
                    steps=cancellation_cleanup_steps,
                )
            if cancellation_terminal_event is not None:
                _mark_session_invocation_terminal_event(cancellation_terminal_event)
            if provider_cancellation_authenticated and cancellation_cleanup_failures:
                _mark_provider_cancellation_finalization_failure(cancellation)
            raise
        except GeneratorExit as abandonment:
            if (
                preserve_failure_until_initial_provider_dispatch
                and not initial_provider_dispatch_started
            ):
                # Closing an observer stream is not an operator decision to discard
                # an already accepted fallback. Its outer owner releases the fence.
                raise
            # The consumer closed the event stream (client disconnect / abandoned
            # async generator) while the session was still live. Finalize instead of
            # stranding it in RUNNING; an async generator must not yield while
            # handling GeneratorExit, so the terminal emission is drained, not
            # streamed.
            if _latest_session_invocation_interaction_is_settled(outcome_state.session.id):
                # Closing after an authoritative transition was exposed must
                # not rewrite the completed/paused interaction decision.
                raise
            await materialize_deferred_messages_after_failure()
            if await self._session_control.interrupt_requested(outcome_state.session.id):
                # A consumer may close while receiving interrupted tool outcomes.
                # Preserve the accepted interruption identity, so its requester
                # observes the same terminal receipt instead of timing out on an
                # unrelated abandonment receipt. Cleanup cannot yield here.
                _, propagated_failure = await self._handle_session_interrupted_preserving_failure(
                    authoritative_failure=abandonment,
                    session=outcome_state.session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                    run_started_at=run_started_at,
                    turn_usage_tracker=turn_usage_tracker,
                    active_run=active_run,
                )
                if propagated_failure is not abandonment:
                    raise propagated_failure from None
                raise
            try:
                await self._session_finalization.finalize_abandoned_session_run(
                    RecoveryAbandonedSessionRequest(
                        session=outcome_state.session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=environment_name,
                        run_started_at=run_started_at,
                        turn_usage_tracker=turn_usage_tracker,
                        active_run=active_run,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    )
                )
            except Exception as finalization_failure:
                _raise_primary_with_secondary_failure(
                    abandonment,
                    finalization_failure,
                    group_message=("Stream abandonment and interaction terminalization failure."),
                )
            raise
        except TerminalEventPublicationUncertain:
            raise
        except Exception as exc:
            if (
                preserve_failure_until_initial_provider_dispatch
                and not initial_provider_dispatch_started
            ):
                # An accepted provider-operation fallback remains the durable retry
                # owner until its replacement dispatch is staged. Do not convert a
                # setup, request-construction, billing, or store failure into an
                # unrelated terminal session that the pending disposition cannot
                # finish after restart.
                raise
            async with contextlib.aclosing(
                self._session_outcomes.fail(
                    outcome_context(),
                    exc=exc,
                    materialize_deferred_messages_after_failure=materialize_deferred_messages_after_failure,
                )
            ) as outcome_stream:
                async for event in outcome_stream:
                    yield event
        except BaseExceptionGroup as failure:
            if preserve_failure_until_initial_provider_dispatch and (
                not initial_provider_dispatch_started
            ):
                if detach_billing_identity_cancellation_group(failure) is not None:
                    # The recovery fallback bypasses the ordinary run/resume
                    # boundary that normally strips provider-bearing traceback
                    # state from grouped billing cancellation. Drop provider
                    # registrations before any fallible store access; the recovery
                    # entrypoint publishes the detached group after this frame exits.
                    del provider, registered_provider
                    try:
                        interruption_requested = await self._session_control.interrupt_requested(
                            outcome_state.session.id
                        )
                    except Exception:
                        raise _FallbackBillingCancellationStateCheckFailed from None
                    if not interruption_requested:
                        raise
                elif not await self._session_control.interrupt_requested(outcome_state.session.id):
                    raise
            # ``ExceptionGroup`` is also an ``Exception`` and is handled by
            # the ordinary failure branch above. Only groups carrying a true
            # ``BaseException`` control leaf reach this interruption path.
            if binding_finalize_explicit_cancellation(failure) is None:
                raise
            if _latest_session_invocation_interaction_is_settled(outcome_state.session.id):
                # A terminal interaction decision already committed. Preserve
                # the grouped control outcome without rewriting that decision.
                raise
            if any(
                _is_interaction_transition_run_fence(candidate)
                for candidate in iter_exception_tree(failure)
            ):
                # This worker has positive evidence that its durable mutation
                # authority was lost. Do not attempt a terminal write under a
                # stale run epoch.
                raise

            async def prepare_group_interruption() -> list[Event]:
                await materialize_deferred_messages_after_failure()
                prepared_events: list[Event] = []
                if close_new_pending_round_on_interrupt:
                    async for event in self._close_durable_pending_tool_round_after_interrupt(
                        session=outcome_state.session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    ):
                        prepared_events.append(event)
                return prepared_events

            (
                _interruption_events,
                propagated_failure,
            ) = await self._handle_session_interrupted_preserving_failure(
                authoritative_failure=failure,
                session=outcome_state.session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                prepare_interruption=prepare_group_interruption,
                run_started_at=run_started_at,
                turn_usage_tracker=turn_usage_tracker,
                active_run=active_run,
            )
            if propagated_failure is failure:
                raise
            raise propagated_failure from None
        finally:
            finalizer_failure = sys.exception()
            if finalizer_failure is not None and (
                isinstance(finalizer_failure, asyncio.CancelledError)
                or workspace_observation_pending_cancellation_requests(finalizer_failure)
            ):

                async def discard_interrupt_signal() -> None:
                    self._session_control.discard_interrupt_signal(outcome_state.session.id)

                finalizer_steps: list[tuple[str, Callable[[], Awaitable[None]]]] = [
                    (
                        "Session environment setup abort",
                        lambda: self._environment_lifecycle.abort_environment_setup(
                            session_id=outcome_state.session.id,
                            original_error=finalizer_failure,
                            execution_profile=execution_profile,
                            invocation_context=invocation_context,
                        ),
                    ),
                    (
                        "Session interrupt-signal cleanup",
                        discard_interrupt_signal,
                    ),
                ]
                if release_run_fence_on_exit:
                    finalizer_steps.append(
                        (
                            "Session environment run-fence release",
                            lambda: (
                                self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                                    session_id=outcome_state.session.id,
                                    execution_profile=execution_profile,
                                    invocation_context=invocation_context,
                                )
                            ),
                        )
                    )
                propagated_finalizer_failure = (
                    await self._settle_session_finalizer_preserving_interruption_failure(
                        authoritative_failure=finalizer_failure,
                        steps=tuple(finalizer_steps),
                    )
                )
                if release_run_fence_on_exit:
                    # The owned child receives a copied context. Retire the
                    # caller's corresponding token after that child settles.
                    _deactivate_session_run_fence(outcome_state.session.id)
                try:
                    if current_task is not None:
                        self._session_control.unregister_active_task(
                            outcome_state.session.id,
                            current_task,
                        )
                finally:
                    _clear_session_interaction_settlement(outcome_state.session.id)
                    _deactivate_session_interaction(outcome_state.session.id)
                if propagated_finalizer_failure is not finalizer_failure:
                    raise propagated_finalizer_failure from None
            else:
                try:
                    await self._environment_lifecycle.abort_environment_setup(
                        session_id=outcome_state.session.id,
                        original_error=finalizer_failure,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    )
                finally:
                    self._session_control.discard_interrupt_signal(outcome_state.session.id)
                    try:
                        if release_run_fence_on_exit:
                            await self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                                session_id=outcome_state.session.id,
                                execution_profile=execution_profile,
                                invocation_context=invocation_context,
                            )
                    finally:
                        try:
                            if current_task is not None:
                                self._session_control.unregister_active_task(
                                    outcome_state.session.id,
                                    current_task,
                                )
                        finally:
                            _clear_session_interaction_settlement(outcome_state.session.id)
                            _deactivate_session_interaction(outcome_state.session.id)

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

    async def _stop_session_for_model_step_limit(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        messages: list[Message],
        step: int,
        max_steps: int,
        run_started_at: float,
        turn_usage_tracker: SessionUsageTracker,
        active_run: ActiveSessionRun[SessionUsageTracker] | None,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        usage_summary = await self._run_limit_controller.session_usage_summary(session.id)
        decision = StopDecision(
            limit=StopLimit.MODEL_STEPS,
            maximum=max_steps,
            actual=step,
            message=(
                "Run limit reached: the configured model-step allowance was "
                f"exhausted ({step} >= {max_steps})."
            ),
        )
        async for event in self._session_finalization.stop_session_for_limit_reached(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            decision=decision,
            usage_summary=usage_summary,
            cost_summary=None,
            messages=messages,
            tool_calls=[],
            completed_tool_outcomes=[],
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
            reconcile_transition_cancellation=False,
        ):
            yield event

    async def _clear_pending_tool_round_if_matches(
        self,
        session_id: str,
        pending_round: pending_rounds.PendingToolRound,
    ) -> None:
        checkpoint = await self.session_store.load_checkpoint(session_id)
        if checkpoint is None:
            return
        copied_checkpoint = copy_durable_record(checkpoint, "checkpoint")
        current = pending_round_reader.pending_tool_round_from_checkpoint(
            copied_checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        if current is None or current.tool_round_id != pending_round.tool_round_id:
            return
        copied_checkpoint.pop(pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY, None)
        await self.session_store.transform_checkpoint(
            session_id,
            _replace_checkpoint_preserving_runtime_state(copied_checkpoint),
        )

    async def _clear_pending_interruption_cascade(self, session_id: str) -> str | None:
        cleared_attempt_id: str | None = None

        def transform(_session: Session, checkpoint: dict[str, Any] | None) -> dict[str, Any]:
            nonlocal cleared_attempt_id
            if checkpoint is None:
                return {}
            copied_checkpoint = copy_durable_record(checkpoint, "checkpoint")
            marker = copied_checkpoint.pop(_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY, None)
            if marker is None:
                return copied_checkpoint
            if type(marker) is not dict or type(marker.get("attempt_id")) is not str:
                raise ValueError("Pending interruption cascade checkpoint is invalid.")
            cleared_attempt_id = marker["attempt_id"]
            return copied_checkpoint

        await self.session_store.transform_checkpoint(session_id, transform)
        return cleared_attempt_id

    async def require_session(self, session_id: str) -> Session:
        loaded = await self.session_store.load(session_id)
        if loaded is None:
            raise KeyError(f"Session not found: {session_id}") from None
        return loaded

    async def _handle_session_interrupted_preserving_failure(
        self,
        *,
        authoritative_failure: BaseException,
        foreground_subagent_pending: ForegroundSubagentRecoveryRequired | None = None,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
        prepare_interruption: Callable[[], Awaitable[Iterable[Event] | None]] | None = None,
        include_interrupted_session: bool = False,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
    ) -> tuple[list[Event], BaseException]:
        """Settle interruption durably without losing later caller control."""

        current_task = asyncio.current_task()
        retained_request_count = workspace_observation_pending_cancellation_requests(
            authoritative_failure
        )
        current_request_count = 0 if current_task is None else current_task.cancelling()
        historical_request_count = max(current_request_count - retained_request_count, 0)
        entry_request_count = max(retained_request_count, current_request_count)
        concurrent_controls: list[BaseException] = []
        if current_task is not None and current_request_count > historical_request_count:
            # Requests already represented by the authoritative failure were
            # restored before it crossed this boundary. Consume only those
            # owned requests; an older, previously handled cancellation count
            # is task history, not a newly delivered control signal.
            consume_pending_task_cancellation(
                preserve_requests=historical_request_count,
            )

        events: list[Event] = []
        owned_cleanup_failures: list[BaseException] = []

        async def collect_events() -> None:
            persisted_session = await self.session_store.load(session.id)
            allowed_statuses = _INTERRUPTIBLE_SESSION_STATUSES | {SessionStatus.INTERRUPTING}
            if include_interrupted_session:
                allowed_statuses = allowed_statuses | {SessionStatus.INTERRUPTED}
            if persisted_session is None or persisted_session.status not in allowed_statuses:
                return
            if prepare_interruption is not None:
                try:
                    prepared_events = await prepare_interruption()
                except BaseException as preparation_failure:
                    # Preparatory tool-round or deferred-input cleanup must not
                    # prevent the authoritative session transition. Retain the
                    # exact failure and continue to terminalize the session.
                    owned_cleanup_failures.append(preparation_failure)
                else:
                    if prepared_events is not None:
                        events.extend(prepared_events)
            async for event in self._session_finalization.handle_session_interrupted(
                session=persisted_session,
                foreground_subagent_pending=foreground_subagent_pending,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                run_started_at=run_started_at,
                turn_usage_tracker=turn_usage_tracker,
                active_run=active_run,
                terminal_finalization_handoff_source_task=current_task,
            ):
                events.append(event)

        cleanup_task = asyncio.create_task(
            capture_awaitable_outcome(collect_events),
            name="cayu-session-interruption-cleanup",
        )
        cancellation_requests_consumed = 0
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError as cancellation:
                requests_before = 0 if current_task is None else current_task.cancelling()
                retained_cancellation = consume_pending_task_cancellation(
                    cancellation,
                    preserve_requests=historical_request_count,
                )
                requests_after = 0 if current_task is None else current_task.cancelling()
                cancellation_requests_consumed += max(
                    requests_before - requests_after,
                    0,
                )
                concurrent_controls.append(retained_cancellation or cancellation)
            except BaseException as control:
                # A supervisory exit cannot orphan the durable interruption
                # mutation. Preserve it as data until the exact cleanup task
                # reaches a terminal outcome.
                concurrent_controls.append(control)

        captured = cleanup_task.result()
        for event in events:
            if event.type in {
                EventType.SESSION_COMPLETED,
                EventType.SESSION_FAILED,
                EventType.SESSION_INTERRUPTED,
            }:
                _mark_session_invocation_terminal_event(event)
        classified_cleanup_failures: list[BaseException] = []
        for cleanup_failure in (*owned_cleanup_failures, captured.error):
            cleanup_error = _session_interruption_cleanup_child_error(
                cleanup_failure,
                operation="Session interruption cleanup",
            )
            if cleanup_error is not None and all(
                cleanup_error is not retained for retained in classified_cleanup_failures
            ):
                classified_cleanup_failures.append(cleanup_error)

        failures: list[BaseException] = [authoritative_failure]
        for additional_failure in (*concurrent_controls, *classified_cleanup_failures):
            existing_ids = {
                id(candidate)
                for retained_failure in failures
                for candidate in iter_exception_tree(retained_failure)
            }
            retained_failure = _failure_without_existing_exception_identities(
                additional_failure,
                existing_ids,
            )
            if retained_failure is not None:
                failures.append(retained_failure)
        propagated_failure = (
            authoritative_failure
            if len(failures) == 1
            else BaseExceptionGroup(
                "Session interruption cleanup received additional failures.",
                failures,
            )
        )
        pending_request_count = entry_request_count + cancellation_requests_consumed
        if pending_request_count:
            retain_workspace_observation_pending_cancellation_requests(
                propagated_failure,
                pending_request_count,
            )
        return events, propagated_failure

    async def _settle_session_finalizer_preserving_interruption_failure(
        self,
        *,
        authoritative_failure: BaseException,
        steps: tuple[tuple[str, Callable[[], Awaitable[None]]], ...],
    ) -> BaseException:
        """Settle finalizer awaits without replacing the authoritative interruption."""

        if isinstance(authoritative_failure, asyncio.CancelledError) and not (
            workspace_observation_pending_cancellation_requests(authoritative_failure)
        ):
            # Ordinary caller cancellation retains its public exception shape.
            # A failed classifier can leave no terminal settlement; fence-release
            # failure is then secondary, not a replacement for cancellation.
            await self._recovery_ownership.run_cleanup_steps(
                authoritative_failure=authoritative_failure, steps=steps
            )
            return authoritative_failure

        propagated_failure = authoritative_failure
        current_task = asyncio.current_task()
        retained_request_count = workspace_observation_pending_cancellation_requests(
            authoritative_failure
        )
        current_request_count = 0 if current_task is None else current_task.cancelling()
        historical_request_count = max(current_request_count - retained_request_count, 0)
        if current_task is not None and current_request_count > historical_request_count:
            # Python 3.11 and 3.12 do not rescind an undelivered cancellation
            # when ``uncancel()`` reaches zero. Observe the pending injection
            # first, then normalize only requests already represented by the
            # authoritative failure. Preserve and aggregate a genuinely new
            # request that arrives during this checkpoint.
            pending_control: asyncio.CancelledError | None = None
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError as control:
                pending_control = control
            if pending_control is not None:
                request_count_after_checkpoint = current_task.cancelling()
                new_request_count = max(
                    request_count_after_checkpoint - current_request_count,
                    0,
                )
                requests_to_preserve = historical_request_count + new_request_count
                consume_pending_task_cancellation(
                    pending_control,
                    preserve_requests=requests_to_preserve,
                )
                if new_request_count:
                    propagated_failure = _session_interruption_failure_with_additional_control(
                        propagated_failure,
                        pending_control,
                        group_message=(
                            "Session interruption finalization received additional control signals."
                        ),
                        historical_cancellation_requests=requests_to_preserve,
                    )
                    retain_workspace_observation_pending_cancellation_requests(
                        propagated_failure,
                        retained_request_count + new_request_count,
                    )

        for operation, step in steps:
            current_task = asyncio.current_task()
            historical_cancellation_requests = (
                0 if current_task is None else current_task.cancelling()
            )
            cleanup_task = asyncio.create_task(
                capture_awaitable_outcome(step),
                name="cayu-session-interruption-finalizer",
            )
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except BaseException as control:
                    propagated_failure = _session_interruption_failure_with_additional_control(
                        propagated_failure,
                        control,
                        group_message=(
                            "Session interruption finalization received additional control signals."
                        ),
                        historical_cancellation_requests=(historical_cancellation_requests),
                    )

            captured = cleanup_task.result()
            cleanup_error = _session_interruption_cleanup_child_error(
                captured.error,
                operation=operation,
                preserved_failure=authoritative_failure,
            )
            if cleanup_error is not None:
                propagated_failure = _session_interruption_failure_with_additional_control(
                    propagated_failure,
                    cleanup_error,
                    group_message="Session interruption finalization failed.",
                    historical_cancellation_requests=(historical_cancellation_requests),
                )
        return propagated_failure

    async def _close_durable_pending_tool_round_after_interrupt(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Close a round when interruption precedes creation of its live runner."""
        _checkpoint, pending_round = await pending_round_reader.load_pending_tool_round(
            self.session_store,
            session.id,
        )
        if pending_round is None:
            return
        messages = await self.session_store.load_transcript(session.id)
        request = InterruptedToolRoundRequest(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            messages=messages,
            tool_calls=tool_round_recovery.pending_round_tool_calls(pending_round),
            tool_outcomes=[],
            tool_round_identity=pending_rounds.pending_tool_round_identity(pending_round),
            cancellation_artifacts=None,
            cancellation_artifacts_by_id=None,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
        )
        async for event in self._pending_tool_round_recovery.close_interrupted_tool_round(request):
            yield event

    async def _interrupt_background_subagent_children(
        self,
        *,
        parent_session_id: str,
        interrupt_payload: dict[str, Any],
        create_if_missing: bool = True,
        retry_request: dict[str, Any] | None = None,
    ) -> None:
        await self._background_interruption_coordinator.run_cascade(
            parent_session_id=parent_session_id,
            interrupt_payload=interrupt_payload,
            create_if_missing=create_if_missing,
            retry_request=retry_request,
        )

    async def _run_before_stop_policies(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        step_result: AssistantStepResult,
        step: int,
        max_steps: int,
        request_metadata: dict[str, Any],
        request_loop_policies: tuple[LoopPolicy, ...],
        invocation_context: InvocationContext | None = None,
    ) -> AsyncGenerator[tuple[Event, _BeforeStopPolicySelection | None], None]:
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or registered_agent is not invocation_context.registered_agent
            or registered_environment is not invocation_context.registered_environment
            or len(invocation_context.request_loop_policies) != len(request_loop_policies)
            or any(
                frozen is not supplied
                for frozen, supplied in zip(
                    invocation_context.request_loop_policies,
                    request_loop_policies,
                    strict=True,
                )
            )
        ):
            raise RuntimeError("Loop policy substituted frozen invocation authority.")
        classification = classify_assistant_step(step_result)
        policy_groups = (
            (
                "app",
                self._loop_policies
                if invocation_context is None
                else invocation_context.loop_policies,
            ),
            ("agent", registered_agent.loop_policies),
            (
                "request",
                validate_loop_policies(
                    request_loop_policies
                    if invocation_context is None
                    else invocation_context.request_loop_policies,
                    field_name="request_loop_policies",
                ),
            ),
        )
        snapshot = None
        checkpoint = None
        for scope, policies in policy_groups:
            for position, policy in enumerate(policies):
                if not _loop_policy_supports_before_stop(policy):
                    continue
                if snapshot is None:
                    snapshot = await self.session_store.load_transcript_snapshot(session.id)
                    checkpoint = await self.session_store.load_checkpoint(session.id)
                policy_key = before_stop_policy_key(policy, scope=scope, position=position)
                policy_name = require_clean_nonblank(policy.name, "loop_policy.name")
                yield (
                    await self._event_writer.emit(
                        _before_stop_policy_event(
                            event_type="custom.loop.before_stop.started",
                            policy_name=policy_name,
                            scope=scope,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            step=step,
                            classification=classification,
                            payload={},
                        )
                    ),
                    None,
                )
                context = BeforeStopContext(
                    session=session,
                    step_result=step_result,
                    classification=classification,
                    step=step,
                    max_steps=max_steps,
                    metadata=request_metadata,
                    messages=[record.message for record in snapshot.records],
                    continuation_message_indices=before_stop_continuation_indices(
                        checkpoint,
                        snapshot=snapshot,
                        session_id=session.id,
                        policy_key=policy_key,
                    ),
                )
                try:
                    decision = copy_before_stop_decision(await policy.before_stop(context))
                except Exception as exc:
                    diagnostic = exception_diagnostic(
                        exc,
                        empty_message="loop policy failed",
                        nonportable_message=("Loop policy failed with a non-portable diagnostic."),
                        redactor=self._secret_redactor,
                    )
                    yield (
                        await self._event_writer.emit(
                            _before_stop_policy_event(
                                event_type="custom.loop.before_stop.failed",
                                policy_name=policy_name,
                                scope=scope,
                                session=session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                step=step,
                                classification=classification,
                                payload=diagnostic.payload_fields(),
                            )
                        ),
                        None,
                    )
                    raise
                yield (
                    await self._event_writer.emit(
                        _before_stop_policy_event(
                            event_type="custom.loop.before_stop.completed",
                            policy_name=policy_name,
                            scope=scope,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            step=step,
                            classification=classification,
                            payload={
                                "action": decision.action.value,
                                "reason": decision.reason,
                                "metadata": copy_json_value(decision.metadata, "metadata"),
                            },
                        )
                    ),
                    None,
                )
                if decision.action != BeforeStopAction.COMPLETE:
                    yield (
                        await self._event_writer.emit(
                            _before_stop_policy_event(
                                event_type="custom.loop.before_stop.selected",
                                policy_name=policy_name,
                                scope=scope,
                                session=session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                step=step,
                                classification=classification,
                                payload={
                                    "action": decision.action.value,
                                    "reason": decision.reason,
                                    "metadata": copy_json_value(
                                        decision.metadata,
                                        "metadata",
                                    ),
                                },
                            )
                        ),
                        _BeforeStopPolicySelection(decision=decision, policy_key=policy_key),
                    )
                    return
