"""Deep owner for session orchestration behind :class:`CayuApp`."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import sys
import threading
import time
import traceback as traceback_module
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal, NoReturn, TypeVar
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from cayu._exception_groups import (
    exception_cause,
    exception_group_children,
    iter_exception_tree,
    rebuild_exception_group,
    set_exception_cause,
)
from cayu._exception_state import pop_exception_state, set_exception_state
from cayu._task_wait import (
    ShieldedTaskOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    consume_pending_task_cancellation,
    restore_task_cancellation_requests,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    MIN_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    copy_durable_json_value,
    copy_json_value,
    require_clean_nonblank,
)
from cayu.core.billing import (
    UNRESOLVED_BILLING_IDENTITY,
    BillingIdentity,
    BillingIdentityState,
    ResolvedBillingIdentity,
    resolved_billing_identity,
)
from cayu.core.events import (
    Event,
    EventType,
    copy_event,
    event_id_is_runtime_generated,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_nested_payload_authority,
    event_with_runtime_payload_authority,
)
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.messages import (
    Message,
    MessageRole,
    detach_message,
)
from cayu.core.thinking import ThinkingConfig, thinking_config_payload
from cayu.core.tools import (
    ToolResult,
)
from cayu.environments import (
    EnvironmentFactoryOperation,
    WorkspaceInstructions,
)
from cayu.providers import (
    CacheBreakpoint,
    CachePolicy,
    ModelProvider,
    ModelRequest,
    NativeStructuredOutputSchemaInvalid,
    ProviderOperationMode,
    UsageDialect,
    copy_usage_dialect,
)
from cayu.providers.base import privacy_safe_provider_option_projection
from cayu.runtime import _approval_publication as approval_publication
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _execution_profile_admission as execution_profile_admission
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _model_completion_publication as model_completion_publication
from cayu.runtime import _model_target as model_target
from cayu.runtime import _resume_ledger as resume_ledger
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _session_request_boundary as session_request_boundary
from cayu.runtime import _tool_argument_publication as tool_argument_publication
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_results as tool_results
from cayu.runtime import _tool_round_publication as tool_round_publication
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._binding_cleanup import binding_finalize_explicit_cancellation
from cayu.runtime._child_session_identity import (
    ChildSessionKind,
    generate_child_session_id,
)
from cayu.runtime._diagnostics import (
    ExceptionDiagnostic,
    exception_diagnostic,
    task_failure_payload_from_diagnostic,
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
    _reconcile_exact_persisted_event,
)
from cayu.runtime._execution_profile_identity_validation import (
    copy_secret_free_execution_profile_behavior_identity,
)
from cayu.runtime._fork_source_snapshot import fork_source_checkpoint_sha256
from cayu.runtime._interruption_coordinator import (
    _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY,
    _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY,
    BackgroundInterruptionCoordinator,
    _copy_interruption_cascade_retry_request,
    _interruption_cascade_marker_datetime,
    _interruption_cascade_retry_event_payload,
    interruption_cascade_lease_seconds,
    interruption_cascade_suppressed,
    suppress_interruption_cascade,
)
from cayu.runtime._message_redaction import redact_runtime_message_for_boundary
from cayu.runtime._model_errors import (
    _FallbackBillingCancellationStateCheckFailed,
    detach_billing_identity_cancellation,
    detach_billing_identity_cancellation_group,
)
from cayu.runtime._model_step_executor import (
    ModelCompletionPublicationRequest,
    ModelCompletionPublicationResult,
    ModelCompletionRecoveryContext,
    ModelStepBudgetEvaluationRequest,
    ModelStepBudgetReservationFailureRequest,
    ModelStepExecutor,
    ModelStepFlowOutcome,
    ModelStepLimitEvaluationRequest,
    _all_registered_tool_exposure,
    _detach_model_request,
    _event_with_model_identity_authority,
    _model_request_messages,
    _model_request_tools,
    _require_frozen_tool_exposure,
    _session_agent_spec,
    is_ambiguous_provider_operation_start_error,
    model_completion_recovery_context_from_stage,
)
from cayu.runtime._recovery_coordinator import (
    ModelCompletionManualRecoveryRequired,
    RecoveryAbandonedSessionRequest,
    RecoveryCoordinator,
    _checkpoint_without_active_incomplete_recovery_claim,
    _run_recovery_cleanup_steps,
)
from cayu.runtime._run_limit_accounting import (
    RunLimitAccountingContext,
    capture_run_limit_accounting_context,
    has_run_limit_accounting_authority,
    restore_run_limit_accounting_context,
    run_budget_authorities_from_context,
)
from cayu.runtime._run_limits import (
    BudgetEvaluation,
    BudgetReservationIdentityGuard,
    BudgetReservationLeaseLost,
    BudgetReservationLeaseLostBeforeModelDispatch,
    BudgetStepReservation,
    LimitEvaluation,
    RunLimitController,
    RunLimitGate,
    SessionUsageTracker,
    add_budget_failure_note,
    budget_limit_reached_payload,
    budget_provider_cleanup_failure,
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
    _structured_output_tool_round_outcomes,
    _structured_output_tool_terminal_event,
    _structured_output_validating_event,
    _StructuredOutputToolRoundPublicationExtension,
    _validate_structured_output_tool_round,
)
from cayu.runtime._terminal_evidence import interruption_request_id_from_payload
from cayu.runtime._tool_round_executor import (
    InterruptedToolRoundRequest,
    ToolApprovalRequired,
    ToolRoundExecutor,
    ToolRoundLimitRequest,
    UserInputRequired,
    ordered_tool_result_messages,
)
from cayu.runtime._workflow_structured_output_handoff import (
    WorkflowStructuredOutputHandoff,
)
from cayu.runtime.approvals import (
    PendingToolApproval,
    ResolutionActor,
    ResolutionActorSource,
    ToolApprovalDecision,
    resolution_actor_payload,
)
from cayu.runtime.budgets import (
    BudgetCheck,
    BudgetLimit,
    BudgetPolicy,
    BudgetReservationRecoveryContext,
    BudgetReservationResult,
    _effective_budget_limit_id,
    budget_check_payload,
    budget_limits_for_session,
    budget_reservation_payload,
    copy_budget_policy,
    has_deferred_contextual_price,
    request_budget_limits_for_session,
)
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.runtime.context import (
    _COMPACTION_ATTEMPT_ID_KEY,
    CheckpointCompactionContextPolicy,
    ContextBuildError,
    ContextBuildResult,
    ContextCompactionTelemetry,
    ContextRequest,
    _attach_context_build_termination_diagnostics,
    _automatic_compaction_dispatch_runner_scope,
    _compaction_completion_publisher_scope,
    _compaction_model_attempt_identity_scope,
    _compaction_model_completed_payload,
    _CompactionAccountingUsageError,
    _context_secret_redactor_scope,
    _defer_billing_identity_cancellation_scope,
    _durable_compaction_completion_evidence,
    _runtime_authored_user_message_checkpoint_transform,
    context_build_termination_compaction_telemetry,
    project_compaction_invocation_checkpoint,
    sanitize_context_build_error_checkpoint,
    sanitize_context_build_result_checkpoint,
    sanitize_context_compaction_telemetry,
    validate_context_messages,
)
from cayu.runtime.costs import (
    SessionCostSummary,
)
from cayu.runtime.dispatch import DispatchRequest
from cayu.runtime.errors import (
    TerminalEventPublicationUncertain,
    _is_runtime_interaction_lifecycle_publication_rejection,
    _runtime_interaction_lifecycle_publication_rejected,
)
from cayu.runtime.execution_profiles import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    EXECUTION_PROFILE_METADATA_KEY,
    ActiveInvocationExecutionProfile,
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAdoptionRejected,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileComponentClass,
    ExecutionProfileDecision,
    ExecutionProfileDecisionKind,
    ExecutionProfileIdentity,
    ExecutionProfileMigrationRequired,
    ExecutionProfileMismatchError,
    ExecutionProfilePolicy,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyError,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
    _ExecutionProfileAdmissionRequestRejected,
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
    active_invocation_execution_profile_matches_session_epoch,
    changed_execution_profile_components,
    checkpoint_with_active_invocation_execution_profile,
    copy_execution_profile_policy_result,
    event_with_execution_profile_authority,
    execution_profile_changes_authority,
    execution_profile_decision_payload,
    execution_profile_from_session_metadata,
    execution_profile_session_metadata,
    execution_profile_with_component,
    execution_profile_with_durable_system_projection_digest,
    unavailable_execution_profile_components,
)
from cayu.runtime.execution_units import (
    ModelAttemptIdentity,
    ModelStepIdentity,
    ToolRoundIdentity,
    copy_model_attempt_identity,
    copy_model_step_identity,
    copy_tool_round_identity,
    new_model_step_identity,
    strip_runtime_owned_execution_identity,
)
from cayu.runtime.hooks import (
    RuntimeHook,
    RuntimeHookContext,
    RuntimeHookPhase,
    RuntimeHookRuntime,
    _runtime_hook_supports_phase,
)
from cayu.runtime.hooks import (
    _runtime_hook_event as _build_runtime_hook_event,
)
from cayu.runtime.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
    INTERACTION_SUMMARY_EVENT_TYPES,
    INTERACTION_TERMINAL_EVENT_TYPES,
    InteractionStatus,
    InteractionSummaryEvidence,
    interaction_usage_summary,
)
from cayu.runtime.invocation import SessionInvocationBinding
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
    ProviderOperationEvidenceError,
    ProviderOperationInspectionStatus,
    inspect_provider_operation,
    pending_provider_operation_disposition_from_checkpoint,
    provider_operation_resolution_outcome_event_id,
    validate_provider_operation_resolution_outcome_event,
)
from cayu.runtime.request_footprints import (
    PromptContributionManifest,
    RequestFootprintConfig,
    RequestVariant,
    analyze_request_footprint,
    build_prompt_contribution_manifest,
    copy_request_footprint_config,
)
from cayu.runtime.retry_policy import (
    RetryPolicy,
    copy_retry_policy,
)
from cayu.runtime.sessions import (
    _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY,
    _QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY,
    _SESSION_RUN_OPERATION_CHECKPOINT_KEY,
    _SESSION_RUN_OPERATION_ID_PAYLOAD_KEY,
    FORK_EXECUTION_PROFILE_METADATA_KEY,
    FORK_GROUP_SOURCE_SNAPSHOT_METADATA_KEY,
    FORK_TRANSCRIPT_VALIDATION_ERROR,
    INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY,
    PROMPT_ANATOMY_TRANSITION_METADATA_KEY,
    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
    CompactSessionRequest,
    EnqueueSessionMessageRequest,
    EnqueueSessionMessageResult,
    EventOrder,
    EventQuery,
    ForkExecutionProfileDecisionRecord,
    ForkExecutionProfileSelection,
    ForkSessionRequest,
    ForkSystemPromptPolicy,
    ForkSystemPromptReplacement,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionRecoveryResult,
    IncompleteSessionsRecoveryPage,
    IncompleteSessionsRecoveryRequest,
    InteractionTransitionResult,
    InteractionTransitionSpec,
    InterruptSessionRequest,
    ModelTarget,
    ProfiledSessionForkResult,
    PromptAnatomyTransitionReceipt,
    ResumeRequest,
    RunRequest,
    RuntimePublicationRequest,
    Session,
    SessionForkActiveModelStageConflict,
    SessionForkProfileRelationship,
    SessionForkSourceNotFound,
    SessionIdentity,
    SessionInvocationAdmission,
    SessionMessageDeliveryBatch,
    SessionModelTransition,
    SessionOperationPublication,
    SessionOrder,
    SessionQuery,
    SessionRunFenced,
    SessionStatus,
    SessionStatusConflict,
    SessionStore,
    TranscriptQuery,
    TranscriptSnapshot,
    _activate_session_interaction,
    _checkpoint_after_session_run_operation_cleanup,
    _checkpoint_with_session_run_operation,
    _clear_session_interaction_recovered_active_through,
    _clear_session_interaction_settlement,
    _close_session_interaction,
    _current_session_interaction_id,
    _current_session_interaction_recovered_active_through,
    _current_session_interaction_started_at,
    _current_session_invocation_interaction_ids,
    _deactivate_session_interaction,
    _deactivate_session_run_fence,
    _event_with_session_run_operation,
    _fork_group_initial_invocation_request_sha256,
    _incomplete_recovery_claim_from_checkpoint,
    _initial_transcript_pending_interaction_id,
    _latest_session_invocation_interaction_is_settled,
    _mark_session_interaction_settled,
    _queued_dispatch_session_instance_fingerprint,
    _runtime_resume_transport_metadata,
    _session_metadata_with_model_projection,
    _session_run_operation_from_checkpoint,
    _set_session_interaction_recovered_active_through,
    apply_fork_system_prompt_replacement,
    attribute_event_to_current_interaction,
    attribute_events_to_current_interaction,
    bind_runtime_session_create_claim,
    copy_compact_session_request,
    copy_fork_session_request,
    copy_incomplete_session_recovery_request,
    copy_incomplete_sessions_recovery_request,
    copy_interaction_transition_spec,
    copy_profiled_session_fork_result,
    copy_resume_request,
    copy_run_request,
    execution_profile_adoption_request_fingerprint,
    fork_session_invocation,
    run_request_with_task_invocation,
    runtime_prepared_session_authority,
    runtime_publication_checkpoint_mutation,
    runtime_publication_checkpoint_value_digest,
    session_fork_profile_relationship,
    session_input_contract_evidence,
    session_input_messages_sha256,
    session_model_projection_cursor,
    session_prompt_anatomy_transition,
    system_prompt_messages_sha256,
    validate_profiled_fork_evidence,
)
from cayu.runtime.stop_policy import (
    RunLimits,
    StopDecision,
    StopLimit,
    copy_run_limits,
    has_run_limits,
)
from cayu.runtime.structured_output import (
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
from cayu.runtime.structured_output import (
    _require_native_structured_output_support as _require_provider_native_output_support,
)
from cayu.runtime.tasks import (
    Task,
    TaskQuery,
    TaskStatus,
    TaskStore,
    TaskTerminalizationRequest,
    TaskTerminalKind,
    _task_invocation_for_attachment,
    _terminalize_claimed_task,
)
from cayu.runtime.tool_exposure import (
    ResolvedToolExposure,
    validate_resolved_tool_exposure_authority,
)
from cayu.runtime.tool_policy import (
    metadata_with_taint_labels,
    taint_labels_from_metadata,
)
from cayu.runtime.usage import (
    ModelCompletionPurpose,
    SessionUsageSummary,
    aggregate_usage_metrics_payload,
    combine_session_usage_summaries,
    session_usage_summary,
    session_usage_summary_payload,
)
from cayu.runtime.user_input import (
    event_with_pending_user_input_authority,
    pending_user_input_from_checkpoint,
    pending_user_input_interruption_payload,
)
from cayu.runtime.workspace_observation_recovery import (
    retain_workspace_observation_pending_cancellation_requests,
    workspace_observation_pending_cancellation_requests,
)
from cayu.vaults import (
    SecretRedactor,
)
from cayu.workspaces import WorkspaceForkLineage, WorkspaceForkLineageStatus

logger = logging.getLogger(__name__)


class _SessionCompactionReplay(Exception):
    def __init__(self, event_ids: Iterable[str]) -> None:
        self.event_ids = tuple(event_ids)
        super().__init__("Replay an existing durable session compaction outcome.")


class _ExpiredIncompleteRecoveryClaim(Exception):
    def __init__(self, claim_id: str) -> None:
        self.claim_id = claim_id
        super().__init__("Fence an expired incomplete-session recovery owner.")


class SessionCompactionAttemptSuperseded(RuntimeError):
    """A recovered compaction attempt owns the durable operation claim."""


_RESUMABLE_SESSION_STATUSES = {
    SessionStatus.COMPLETED,
    SessionStatus.FAILED,
    SessionStatus.INTERRUPTED,
}

_FORKABLE_SESSION_STATUSES = _RESUMABLE_SESSION_STATUSES
_PROMPT_ANATOMY_TRANSITION_INTENTS_CHECKPOINT_KEY = "prompt_anatomy_transition_intents"
_PROMPT_ANATOMY_TRANSITION_INTENT_LIMIT = 256


class _PromptTransitionIntentStatus(StrEnum):
    PREPARED = "prepared"
    COMPLETED = "completed"


class _PromptTransitionIntent(BaseModel):
    """Typed durable authority for one prompt-succession effect."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transition_id: str
    request_sha256: str
    source_session_id: str
    descendant_session_id: str
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_snapshot_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    status: _PromptTransitionIntentStatus
    requested_at: datetime
    descendant_created_at: datetime | None = None
    completed_at: datetime | None = None

    @classmethod
    def prepared(
        cls,
        *,
        receipt: PromptAnatomyTransitionReceipt,
        source_run_epoch: int,
        source_snapshot_cursor: int,
        requested_at: datetime,
    ) -> _PromptTransitionIntent:
        return cls(
            transition_id=receipt.transition_id,
            request_sha256=receipt.request_sha256,
            source_session_id=receipt.source_session_id,
            descendant_session_id=receipt.descendant_session_id,
            source_run_epoch=source_run_epoch,
            source_transcript_cursor=receipt.source_transcript_cursor,
            source_snapshot_cursor=source_snapshot_cursor,
            status=_PromptTransitionIntentStatus.PREPARED,
            requested_at=requested_at,
        )

    @field_validator(
        "transition_id",
        "request_sha256",
        "source_session_id",
        "descendant_session_id",
    )
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        if info.field_name in {"transition_id", "request_sha256"} and (
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @field_validator("requested_at", "descendant_created_at", "completed_at")
    @classmethod
    def validate_timestamps(cls, value: datetime | None, info) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value

    @model_validator(mode="after")
    def validate_lifecycle(self) -> _PromptTransitionIntent:
        completed = self.status is _PromptTransitionIntentStatus.COMPLETED
        if completed != (self.descendant_created_at is not None and self.completed_at is not None):
            raise ValueError("Prompt-transition intent completion evidence is inconsistent.")
        return self

    def same_identity(self, other: _PromptTransitionIntent) -> bool:
        return (
            self.transition_id,
            self.request_sha256,
            self.source_session_id,
            self.descendant_session_id,
            self.source_run_epoch,
            self.source_transcript_cursor,
            self.source_snapshot_cursor,
        ) == (
            other.transition_id,
            other.request_sha256,
            other.source_session_id,
            other.descendant_session_id,
            other.source_run_epoch,
            other.source_transcript_cursor,
            other.source_snapshot_cursor,
        )


class _PromptTransitionIntentLedger(BaseModel):
    """Bounded parser and state machine for prompt-transition checkpoint authority."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    records: dict[str, _PromptTransitionIntent] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_records(self) -> _PromptTransitionIntentLedger:
        if len(self.records) > _PROMPT_ANATOMY_TRANSITION_INTENT_LIMIT:
            raise ValueError("Prompt-transition intent ledger exceeds its bounded record limit.")
        if any(key != record.transition_id for key, record in self.records.items()):
            raise ValueError("Prompt-transition intent ledger key conflicts with its record.")
        return self

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: dict[str, Any] | None,
        *,
        required: bool,
    ) -> _PromptTransitionIntentLedger:
        raw = (
            None
            if checkpoint is None
            else checkpoint.get(_PROMPT_ANATOMY_TRANSITION_INTENTS_CHECKPOINT_KEY)
        )
        if raw is None:
            if required:
                raise RuntimeError("Prompt-anatomy transition intent disappeared.")
            return cls()
        try:
            return cls.model_validate(
                copy_json_value(raw, _PROMPT_ANATOMY_TRANSITION_INTENTS_CHECKPOINT_KEY)
            )
        except ValidationError as exc:
            raise ValueError("Prompt-anatomy transition intent ledger is malformed.") from exc

    def store_in(self, checkpoint: dict[str, Any]) -> None:
        checkpoint[_PROMPT_ANATOMY_TRANSITION_INTENTS_CHECKPOINT_KEY] = self.model_dump(mode="json")

    def require_exact(self, candidate: _PromptTransitionIntent) -> _PromptTransitionIntent:
        existing = self.records.get(candidate.transition_id)
        if existing is None or not existing.same_identity(candidate):
            raise RuntimeError("Prompt-anatomy transition intent conflicts with the exact request.")
        return existing

    def prepare(self, candidate: _PromptTransitionIntent) -> None:
        for record in self.records.values():
            if (
                record.descendant_session_id == candidate.descendant_session_id
                or record.request_sha256 == candidate.request_sha256
            ) and not record.same_identity(candidate):
                raise RuntimeError(
                    "Existing prompt-anatomy transition intent conflicts with the exact request."
                )
        existing = self.records.get(candidate.transition_id)
        if existing is not None:
            self.require_exact(candidate)
            return
        if len(self.records) >= _PROMPT_ANATOMY_TRANSITION_INTENT_LIMIT:
            completed = sorted(
                (
                    record
                    for record in self.records.values()
                    if record.status is _PromptTransitionIntentStatus.COMPLETED
                ),
                key=lambda record: record.completed_at or record.requested_at,
            )
            if completed:
                del self.records[completed[0].transition_id]
        if len(self.records) >= _PROMPT_ANATOMY_TRANSITION_INTENT_LIMIT:
            raise RuntimeError("Prompt-transition intent ledger has no bounded capacity.")
        self.records[candidate.transition_id] = candidate

    def complete(
        self,
        candidate: _PromptTransitionIntent,
        *,
        descendant_created_at: datetime,
        completed_at: datetime,
    ) -> None:
        existing = self.require_exact(candidate)
        if (
            existing.status is _PromptTransitionIntentStatus.COMPLETED
            and existing.descendant_created_at == descendant_created_at
        ):
            return
        self.records[candidate.transition_id] = existing.model_copy(
            update={
                "status": _PromptTransitionIntentStatus.COMPLETED,
                "descendant_created_at": descendant_created_at,
                "completed_at": completed_at,
            }
        )

    def complete_from_receipt(
        self,
        receipt: PromptAnatomyTransitionReceipt,
        *,
        descendant_created_at: datetime,
        completed_at: datetime,
    ) -> bool:
        """Complete surviving prepared authority from the durable descendant receipt."""

        existing = self.records.get(receipt.transition_id)
        if existing is None:
            return False
        if (
            existing.request_sha256,
            existing.source_session_id,
            existing.descendant_session_id,
            existing.source_transcript_cursor,
        ) != (
            receipt.request_sha256,
            receipt.source_session_id,
            receipt.descendant_session_id,
            receipt.source_transcript_cursor,
        ):
            raise RuntimeError(
                "Prompt-anatomy transition receipt conflicts with its durable intent."
            )
        if existing.status is _PromptTransitionIntentStatus.COMPLETED:
            if existing.descendant_created_at != descendant_created_at:
                raise RuntimeError(
                    "Prompt-anatomy transition completion time conflicts with its descendant."
                )
            return False
        self.records[receipt.transition_id] = existing.model_copy(
            update={
                "status": _PromptTransitionIntentStatus.COMPLETED,
                "descendant_created_at": descendant_created_at,
                "completed_at": completed_at,
            }
        )
        return True

    def has_prepared_intent(self) -> bool:
        return any(
            record.status is _PromptTransitionIntentStatus.PREPARED
            for record in self.records.values()
        )


def _reject_prepared_prompt_transition_intent(
    checkpoint: dict[str, Any] | None,
) -> None:
    ledger = _PromptTransitionIntentLedger.from_checkpoint(
        checkpoint,
        required=False,
    )
    if ledger.has_prepared_intent():
        raise RuntimeError(
            "Session has a prepared prompt-anatomy succession intent. Retry the exact "
            "fork request before continuing or compacting the source session."
        )


async def _reconcile_committed_prompt_transition_intents(
    *,
    session_store: SessionStore,
    source_session_id: str,
    checkpoint: dict[str, Any] | None,
    clock: Callable[[], datetime],
) -> dict[str, Any] | None:
    """Complete prepared intents whose descendants already prove the durable effect."""

    ledger = _PromptTransitionIntentLedger.from_checkpoint(
        checkpoint,
        required=False,
    )
    committed_receipts: list[tuple[PromptAnatomyTransitionReceipt, datetime]] = []
    for record in ledger.records.values():
        if record.status is not _PromptTransitionIntentStatus.PREPARED:
            continue
        descendant = await session_store.load(record.descendant_session_id)
        if descendant is None:
            continue
        receipt = session_prompt_anatomy_transition(descendant)
        if receipt is None:
            raise RuntimeError(
                "Prepared prompt-anatomy descendant has no exact transition receipt."
            )
        committed_receipts.append((receipt, descendant.created_at))
    if not committed_receipts:
        return checkpoint

    reconciled_checkpoint: dict[str, Any] | None = None

    def reconcile(
        _current_source: Session,
        current_checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        nonlocal reconciled_checkpoint
        if current_checkpoint is None:
            raise RuntimeError("Prompt-anatomy transition intent disappeared.")
        updated = copy_json_value(current_checkpoint, "checkpoint")
        current_ledger = _PromptTransitionIntentLedger.from_checkpoint(
            updated,
            required=True,
        )
        changed = False
        for receipt, descendant_created_at in committed_receipts:
            changed = (
                current_ledger.complete_from_receipt(
                    receipt,
                    descendant_created_at=descendant_created_at,
                    completed_at=clock(),
                )
                or changed
            )
        if changed:
            current_ledger.store_in(updated)
        reconciled_checkpoint = updated
        return updated

    await session_store.publish_checkpoint_and_events(
        source_session_id,
        checkpoint_transform=reconcile,
        events=[],
    )
    return reconciled_checkpoint


def _prompt_fork_matches_prepared(existing: Session, prepared: Session) -> bool:
    return (
        existing.id,
        existing.agent_name,
        existing.provider_name,
        existing.model,
        existing.parent_session_id,
        existing.causal_budget_id,
        existing.runtime_name,
        existing.runtime_version,
        existing.environment_name,
        existing.status,
        existing.labels,
        existing.metadata,
        existing.run_epoch,
    ) == (
        prepared.id,
        prepared.agent_name,
        prepared.provider_name,
        prepared.model,
        prepared.parent_session_id,
        prepared.causal_budget_id,
        prepared.runtime_name,
        prepared.runtime_version,
        prepared.environment_name,
        prepared.status,
        prepared.labels,
        prepared.metadata,
        prepared.run_epoch,
    )


@dataclass(frozen=True, slots=True)
class _PromptSuccessionWorkflow:
    """Cohesive durable state and event projection for one prompt succession."""

    request: ForkSessionRequest
    request_sha256: str
    receipt: PromptAnatomyTransitionReceipt

    @classmethod
    def recover(
        cls,
        *,
        request: ForkSessionRequest,
        accepted_request_sha256s: tuple[str, ...],
        descendant: Session,
    ) -> _PromptSuccessionWorkflow:
        receipt = session_prompt_anatomy_transition(descendant)
        if receipt is None or receipt.request_sha256 not in accepted_request_sha256s:
            raise RuntimeError(
                "Existing prompt-anatomy descendant conflicts with the exact request."
            )
        return cls(
            request=request,
            request_sha256=receipt.request_sha256,
            receipt=receipt,
        )

    def prepared_intent(
        self,
        *,
        source_run_epoch: int,
        source_snapshot_cursor: int,
        requested_at: datetime,
    ) -> _PromptTransitionIntent:
        return _PromptTransitionIntent.prepared(
            receipt=self.receipt,
            source_run_epoch=source_run_epoch,
            source_snapshot_cursor=source_snapshot_cursor,
            requested_at=requested_at,
        )

    def raw_fork_event(self, descendant: Session) -> Event:
        receipt = self.receipt
        payload: dict[str, Any] = {
            "source_session_id": receipt.source_session_id,
            "source_status": receipt.source_status.value,
            "parent_session_id": descendant.parent_session_id,
            "causal_budget_id": descendant.causal_budget_id,
            "transcript_cursor": self.request.transcript_cursor,
            "copy_checkpoint": receipt.copy_checkpoint,
            "agent_name": descendant.agent_name,
            "provider_name": descendant.provider_name,
            "model": descendant.model,
            "environment_name": descendant.environment_name,
            "inherited_taint_labels": list(receipt.inherited_taint_labels),
            "prompt_anatomy_transition": {
                "transition_id": receipt.transition_id,
                "source_transcript_cursor": receipt.source_transcript_cursor,
                "source_prompt_sha256": receipt.source_prompt_sha256,
                "child_prompt_sha256": receipt.child_prompt_sha256,
                "source_system_prompt_retained": receipt.source_system_prompt_retained,
                "policy": receipt.policy.value,
            },
        }
        return event_with_runtime_generated_id(
            Event(
                id=f"prompt-transition-{receipt.transition_id}",
                type=EventType.SESSION_FORKED,
                session_id=descendant.id,
                timestamp=descendant.created_at,
                agent_name=descendant.agent_name,
                environment_name=descendant.environment_name,
                payload=payload,
            )
        )


@dataclass(slots=True)
class _ForkPromptWorkflow:
    """Own the prompt-policy branch of session forking behind one strategy seam."""

    request: ForkSessionRequest
    request_sha256: str
    prompt_replacement: ForkSystemPromptReplacement | None = None
    child_prompt_messages: tuple[Message, ...] = ()
    rendered_child_prompt: str | None = None
    prompt_environment_name: str | None = None
    succession: _PromptSuccessionWorkflow | None = None
    intent: _PromptTransitionIntent | None = None

    @classmethod
    async def prepare(
        cls,
        *,
        request: ForkSessionRequest,
        request_sha256: str,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_lifecycle: EnvironmentLifecycle,
    ) -> _ForkPromptWorkflow:
        workflow = cls(request=request, request_sha256=request_sha256)
        if request.system_prompt_policy is not ForkSystemPromptPolicy.CURRENT_AGENT:
            return workflow
        if request.copy_checkpoint:
            raise ValueError(
                "Explicit prompt-anatomy succession cannot copy checkpoint state; "
                "set copy_checkpoint=False."
            )
        if registered_environment is None or registered_environment.factory is not None:
            raise ValueError(
                "Explicit prompt-anatomy succession requires a concrete registered "
                "environment so current workspace instructions can be rendered."
            )
        workflow.prompt_environment_name = registered_environment.spec.name
        workspace_instructions = await environment_lifecycle.load_workspace_instructions(
            registered_environment,
        )
        rendered_child_prompt, _prompt_contributions = (
            render_initial_system_prompt_with_contributions(
                agent_system_prompt=registered_agent.spec.system_prompt,
                workspace_instructions=workspace_instructions,
            )
        )
        workflow.rendered_child_prompt = rendered_child_prompt
        workflow.child_prompt_messages = tuple(
            transcript_helpers.initial_messages(
                system_prompt=rendered_child_prompt,
                request_messages=[],
            )
        )
        workflow.prompt_replacement = ForkSystemPromptReplacement(
            message=(workflow.child_prompt_messages[0] if workflow.child_prompt_messages else None),
        )
        return workflow

    def requires_transcript_snapshot(
        self,
        *,
        target_changed: bool,
        source_projection_cursor: int,
    ) -> bool:
        return (
            self.prompt_replacement is not None
            or self.request.transcript_cursor is not None
            or target_changed
            or bool(source_projection_cursor)
        )

    def require_inherited_system_projection(
        self,
        *,
        source_messages: list[Message],
        selected_messages: list[Message],
    ) -> None:
        """Keep an inherited profile coherent with the copied transcript."""

        if self.prompt_replacement is not None:
            return
        source_system = [
            message for message in source_messages if message.role is MessageRole.SYSTEM
        ]
        selected_system = [
            message for message in selected_messages if message.role is MessageRole.SYSTEM
        ]
        if selected_system != source_system:
            raise ValueError(
                "A partial fork that inherits the parent execution profile must retain "
                "the complete durable system projection. Increase transcript_cursor or "
                "select the current child prompt explicitly."
            )

    def project_selected_messages(
        self,
        messages: list[Message],
        interaction_ids: list[str | None],
    ) -> tuple[list[Message], str | None]:
        if self.prompt_replacement is None:
            return messages, None
        projected, projected_interaction_ids = apply_fork_system_prompt_replacement(
            messages,
            interaction_ids,
            self.prompt_replacement,
        )
        projected_interaction_ids.clear()
        return projected, system_prompt_messages_sha256(self.child_prompt_messages)

    def requires_portability_validation(self, *, target_changed: bool) -> bool:
        return target_changed or self.prompt_replacement is not None

    def attach_receipt(
        self,
        metadata: dict[str, Any],
        *,
        source_session: Session,
        destination_session_id: str,
        selected_source_cursor: int,
        source_prompt_sha256: str | None,
        child_prompt_sha256: str | None,
        child_agent_name: str,
        provider_name: str,
        model: str,
        model_changed: bool,
        inherited_taint_labels: set[str] | frozenset[str],
    ) -> None:
        if self.prompt_replacement is None:
            return
        if source_prompt_sha256 is None or child_prompt_sha256 is None:
            raise AssertionError("Prompt succession lost its prompt digests.")
        if self.prompt_environment_name is None:
            raise AssertionError("Prompt succession lost its concrete environment identity.")
        receipt = PromptAnatomyTransitionReceipt.create(
            request_sha256=self.request_sha256,
            source_session_id=source_session.id,
            descendant_session_id=destination_session_id,
            source_status=source_session.status,
            source_transcript_cursor=selected_source_cursor,
            source_prompt_sha256=source_prompt_sha256,
            child_prompt_sha256=child_prompt_sha256,
            child_agent_name=child_agent_name,
            child_environment_name=self.prompt_environment_name,
            copy_checkpoint=self.request.copy_checkpoint,
            source_provider_name=source_session.provider_name,
            source_model=source_session.model,
            provider_name=provider_name,
            model=model,
            model_target_changed=model_changed,
            portability_preflight=(
                "provider_portable_transcript_preflight"
                if model_changed
                else "context_messages_validated"
            ),
            inherited_taint_labels=tuple(sorted(inherited_taint_labels)),
        )
        self.succession = _PromptSuccessionWorkflow(
            request=self.request,
            request_sha256=self.request_sha256,
            receipt=receipt,
        )
        metadata[PROMPT_ANATOMY_TRANSITION_METADATA_KEY] = receipt.model_dump(mode="json")

    def require_prepared_intent(self, source_checkpoint: dict[str, Any] | None) -> None:
        if self.succession is None:
            return
        if self.intent is None or source_checkpoint is None:
            raise RuntimeError("Prompt-anatomy transition intent disappeared.")
        intent_ledger = _PromptTransitionIntentLedger.from_checkpoint(
            source_checkpoint,
            required=True,
        )
        intent_ledger.require_exact(self.intent)

    async def prepare_effect(
        self,
        *,
        session_store: SessionStore,
        source_session: Session,
        source_snapshot_cursor: int,
        fork_session: Session,
        validate_source_checkpoint: Callable[
            [Session, dict[str, Any] | None], dict[str, Any] | None
        ],
        clock: Callable[[], datetime],
    ) -> Session:
        if self.succession is None:
            return fork_session
        self.intent = self.succession.prepared_intent(
            source_run_epoch=source_session.run_epoch,
            source_snapshot_cursor=source_snapshot_cursor,
            requested_at=clock(),
        )

        def persist_prompt_transition_intent(
            current_source: Session,
            source_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            validated = validate_source_checkpoint(current_source, source_checkpoint)
            updated = {} if validated is None else copy_json_value(validated, "checkpoint")
            intent_ledger = _PromptTransitionIntentLedger.from_checkpoint(
                updated,
                required=False,
            )
            if self.intent is None:
                raise AssertionError("Prompt succession lost its durable intent identity.")
            intent_ledger.prepare(self.intent)
            intent_ledger.store_in(updated)
            return updated

        await session_store.publish_checkpoint_and_events(
            source_session.id,
            checkpoint_transform=persist_prompt_transition_intent,
            events=[],
            expected_statuses=_FORKABLE_SESSION_STATUSES,
            expected_run_epoch=source_session.run_epoch,
            expected_transcript_cursor=source_snapshot_cursor,
        )
        descendant_created_at = clock()
        return fork_session.model_copy(
            update={
                "created_at": descendant_created_at,
                "updated_at": descendant_created_at,
            }
        )

    async def recover_created_after_error(
        self,
        *,
        session_store: SessionStore,
        prepared: Session,
        error: Exception,
    ) -> Session:
        if self.succession is None:
            raise error
        existing = await session_store.load(prepared.id)
        if existing is None or not _prompt_fork_matches_prepared(existing, prepared):
            raise error
        existing_transition = session_prompt_anatomy_transition(existing)
        if existing_transition != self.succession.receipt:
            raise RuntimeError(
                "Existing prompt-anatomy descendant conflicts with the exact request."
            ) from error
        return existing

    async def complete_effect(
        self,
        *,
        session_store: SessionStore,
        source_session_id: str,
        created: Session,
        clock: Callable[[], datetime],
    ) -> None:
        if self.succession is None:
            return
        if self.intent is None:
            raise AssertionError("Prompt succession lost its durable intent identity.")

        def complete_prompt_transition_intent(
            _current_source: Session,
            source_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            if source_checkpoint is None:
                raise RuntimeError("Prompt-anatomy transition intent disappeared.")
            updated = copy_json_value(source_checkpoint, "checkpoint")
            intent_ledger = _PromptTransitionIntentLedger.from_checkpoint(
                updated,
                required=True,
            )
            if self.intent is None:
                raise AssertionError("Prompt succession lost its durable intent identity.")
            intent_ledger.complete(
                self.intent,
                descendant_created_at=created.created_at,
                completed_at=clock(),
            )
            intent_ledger.store_in(updated)
            return updated

        await session_store.publish_checkpoint_and_events(
            source_session_id,
            checkpoint_transform=complete_prompt_transition_intent,
            events=[],
        )

    def raw_fork_event(
        self,
        *,
        created: Session,
        ordinary_event: Event,
    ) -> Event:
        if self.succession is None:
            return ordinary_event
        return self.succession.raw_fork_event(created)

    async def publish_event(
        self,
        *,
        event_writer: RuntimeEventWriter,
        event: Event,
    ) -> Event:
        if self.succession is None:
            return await event_writer.emit(event)
        persisted = await event_writer.persist_exact_replay(event)
        delivered = await event_writer.fan_out_persisted([persisted])
        return delivered[0]


def _fork_event_with_runtime_authority(event: Event) -> Event:
    payload_fields = [
        "source_session_id",
        "parent_session_id",
        "causal_budget_id",
        "execution_profile_selection",
        "selected_profile_fingerprint",
        "source_profile_fingerprint",
        "fork_request_sha256",
        "system_prompt_policy",
    ]
    if "initial_invocation_request_sha256" in event.payload:
        payload_fields.extend(
            (
                "initial_invocation_request_sha256",
                "initial_invocation_profile_fingerprint",
            )
        )
    return event_with_runtime_payload_authority(
        event_with_runtime_envelope_authority(event, "session_id"),
        *payload_fields,
    )


_INTERRUPTIBLE_SESSION_STATUSES = {
    SessionStatus.PENDING,
    SessionStatus.RUNNING,
}

_SESSION_OPERATIONS_CHECKPOINT_KEY = "session_operations"

_CONTEXT_COMPACTION_OPERATION_KIND = "context_compaction"

_SESSION_OPERATION_CLAIM_LEASE = timedelta(minutes=5)

_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS = 30.0

_SESSION_OPERATION_CLAIM_HEARTBEAT_RETRY_SECONDS = 5.0

_SESSION_OPERATION_CLAIM_HEARTBEAT_STOP_TIMEOUT_SECONDS = 0.1

_SESSION_OPERATION_STORE_WAIT_TIMEOUT_SECONDS = 5.0

_INTERACTION_TRANSITION_REPLAY_MAX_ATTEMPTS = 3

_INTERACTION_TRANSITION_REPLAY_WINDOW_SECONDS = 30.0

_INTERACTION_TRANSITION_RUN_FENCE_ATTRIBUTE = "_cayu_interaction_transition_run_fence"

_INTERACTION_TRANSITION_RUN_FENCE_AUTHORITY = object()

_INTERACTION_TRANSITION_REPLAY_FAILURE_AUTHORITY = object()

_INTERACTION_TRANSITION_REPLAY_ATTEMPTS_ATTRIBUTE = "_cayu_interaction_transition_replay_attempts"

_INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE = (
    "_cayu_interaction_transition_cancellation_outcome"
)

_INTERACTION_TRANSITION_CANCELLATION_OUTCOME_AUTHORITY = object()

_INTERACTION_TRANSITION_DIAGNOSTIC_MAX_DEPTH = 8

_INTERACTION_TRANSITION_DIAGNOSTIC_MAX_DESCENDANTS = 32

_ExceptionT = TypeVar("_ExceptionT", bound=BaseException)

_INTERRUPTION_TYPE_OPERATOR_REQUESTED = "operator_requested"

_INTERRUPTION_TYPE_RUNTIME_INTERRUPTED = "runtime_interrupted"

_INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED = "tool_approval_required"

_INTERRUPTION_TYPE_USER_INPUT_REQUIRED = "user_input_required"

_INTERRUPTION_TYPE_LIMIT_REACHED = "limit_reached"


@dataclass(frozen=True, slots=True)
class _InteractionTransitionCancellationOutcome:
    """Authenticated immutable evidence carried to the outer run boundary."""

    authority: object
    transition_settled: bool
    encoded_failure_diagnostics: str
    encoded_transition_spec: str | None


def _interaction_transition_failure_contains_run_fence(error: BaseException) -> bool:
    return any(isinstance(candidate, SessionRunFenced) for candidate in iter_exception_tree(error))


def _mark_interaction_transition_run_fence(error: BaseException) -> None:
    BaseException.__setattr__(
        error,
        _INTERACTION_TRANSITION_RUN_FENCE_ATTRIBUTE,
        _INTERACTION_TRANSITION_RUN_FENCE_AUTHORITY,
    )


def _is_interaction_transition_run_fence(error: BaseException) -> bool:
    try:
        authority = BaseException.__getattribute__(
            error,
            _INTERACTION_TRANSITION_RUN_FENCE_ATTRIBUTE,
        )
    except (AttributeError, TypeError):
        return False
    return authority is _INTERACTION_TRANSITION_RUN_FENCE_AUTHORITY


def _interaction_transition_replay_failure(
    failures: list[Exception],
) -> Exception:
    """Return one ordered terminal failure without flattening attempt groups."""

    if not failures:
        raise AssertionError("Interaction transition replay had no failures.")
    unique_failures = _unique_exception_identities(failures)
    if len(failures) == 1:
        failure = unique_failures[0]
    else:
        failure = ExceptionGroup(
            "Interaction transition publication failed across replay attempts.",
            unique_failures,
        )
        BaseException.__setattr__(
            failure,
            "_cayu_interaction_transition_replay_failure",
            _INTERACTION_TRANSITION_REPLAY_FAILURE_AUTHORITY,
        )
        BaseException.__setattr__(
            failure,
            _INTERACTION_TRANSITION_REPLAY_ATTEMPTS_ATTRIBUTE,
            tuple(failures),
        )
    return failure


def _interaction_transition_replay_failures(
    error: BaseException,
) -> tuple[Exception, ...] | None:
    """Return runtime-owned replay failures for durable diagnostics."""

    try:
        authority = BaseException.__getattribute__(
            error,
            "_cayu_interaction_transition_replay_failure",
        )
    except (AttributeError, TypeError):
        return None
    if authority is not _INTERACTION_TRANSITION_REPLAY_FAILURE_AUTHORITY or not isinstance(
        error,
        ExceptionGroup,
    ):
        return None
    try:
        attempts = BaseException.__getattribute__(
            error,
            _INTERACTION_TRANSITION_REPLAY_ATTEMPTS_ATTRIBUTE,
        )
    except (AttributeError, TypeError):
        return None
    if (
        type(attempts) is not tuple
        or len(attempts) < 2
        or any(not isinstance(attempt, Exception) for attempt in attempts)
    ):
        return None
    return attempts


def _interaction_transition_failure_diagnostics(
    failures: tuple[BaseException, ...],
    *,
    redactor: SecretRedactor,
) -> list[dict[str, Any]]:
    """Return bounded ordered diagnostics without flattening attempt groups."""

    remaining_descendants = _INTERACTION_TRANSITION_DIAGNOSTIC_MAX_DESCENDANTS
    root_identities = {id(failure) for failure in failures}
    seen_identities = set(root_identities)

    def project(error: BaseException, *, depth: int) -> dict[str, Any]:
        nonlocal remaining_descendants
        projected = exception_diagnostic(error, redactor=redactor).payload_fields()
        if not isinstance(error, BaseExceptionGroup):
            return projected
        children = exception_group_children(error)
        if children is None:
            projected["children_unavailable"] = True
            return projected
        if depth >= _INTERACTION_TRANSITION_DIAGNOSTIC_MAX_DEPTH:
            projected["children_truncated"] = len(children)
            return projected

        projected_children: list[dict[str, Any]] = []
        duplicate_children = 0
        truncated_children = 0
        for child in children:
            child_identity = id(child)
            if child_identity in seen_identities:
                duplicate_children += 1
                continue
            if remaining_descendants <= 0:
                truncated_children += 1
                continue
            seen_identities.add(child_identity)
            remaining_descendants -= 1
            projected_children.append(project(child, depth=depth + 1))
        if projected_children:
            projected["children"] = projected_children
        if duplicate_children:
            projected["duplicate_children_omitted"] = duplicate_children
        if truncated_children:
            projected["children_truncated"] = truncated_children
        return projected

    return [project(failure, depth=0) for failure in failures]


def _unique_exception_identities(
    failures: Iterable[_ExceptionT],
) -> list[_ExceptionT]:
    """Preserve first-seen order without recording one exception object twice."""

    unique: list[_ExceptionT] = []
    for failure in failures:
        if not any(existing is failure for existing in unique):
            unique.append(failure)
    return unique


async def _run_interaction_transition_cancellation_cleanup_steps(
    cancellation: asyncio.CancelledError,
    *,
    steps: tuple[tuple[str, Callable[[], Awaitable[None]]], ...],
) -> tuple[tuple[str, BaseException], ...]:
    """Keep this cancellation's attempt and cleanup failures jointly visible.

    The shared recovery helper intentionally keeps historical causes behind the
    current cleanup group. Interaction-transition acknowledgement failures are
    different: they and the cancellation cleanup belong to one bounded replay
    handoff, so this narrow boundary exposes both without changing sibling
    recovery classification contracts.
    """

    attempt_failures = exception_cause(cancellation)
    cleanup_failures = await _run_recovery_cleanup_steps(
        authoritative_failure=cancellation,
        steps=steps,
    )
    cleanup_cause = exception_cause(cancellation)
    if (
        cleanup_failures
        and attempt_failures is not None
        and cleanup_cause is not None
        and cleanup_cause is not attempt_failures
    ):
        set_exception_cause(cleanup_cause, None)
        set_exception_cause(
            cancellation,
            BaseExceptionGroup(
                "Interaction transition attempt and cancellation cleanup failures.",
                _unique_exception_identities([attempt_failures, cleanup_cause]),
            ),
        )
    return cleanup_failures


def _normalize_interaction_transition_child_cancellation(
    error: BaseException | None,
) -> BaseException | None:
    """Classify store-owned cancellation without fabricating caller cancellation."""

    if isinstance(error, asyncio.CancelledError):
        return unexpected_child_cancellation_error(
            error,
            operation="Interaction transition publication",
        )
    if not isinstance(error, BaseExceptionGroup) or not any(
        isinstance(candidate, asyncio.CancelledError) for candidate in iter_exception_tree(error)
    ):
        return error
    return rebuild_exception_group(
        error,
        group_message="Interaction transition publication child failures.",
        leaf_mapper=lambda child: (
            unexpected_child_cancellation_error(
                child,
                operation="Interaction transition publication",
            )
            if isinstance(child, asyncio.CancelledError)
            else child
        ),
        invalid_leaf_factory=lambda: RuntimeError(
            "Interaction transition publication returned an unreadable exception group."
        ),
    )


def _validated_interaction_transition_result(
    value: object,
    *,
    transition: InteractionTransitionSpec,
) -> InteractionTransitionResult:
    """Detach and verify one extension-controlled publication result."""

    if type(value) is not InteractionTransitionResult:
        raise TypeError("Session store returned an invalid interaction transition result.")
    try:
        result = InteractionTransitionResult(
            session=value.session,
            event=value.event,
            status_changed=value.status_changed,
            replayed=value.replayed,
        )
    except (AttributeError, TypeError, ValueError):
        raise TypeError(
            "Session store returned an invalid interaction transition result."
        ) from None
    if result.event != transition.event:
        raise RuntimeError("Interaction transition publication returned a conflicting event.")
    if result.session.id != transition.event.session_id:
        raise RuntimeError("Interaction transition publication returned a conflicting session.")
    if result.status_changed:
        if result.session.status is not transition.to_status:
            raise RuntimeError(
                "Interaction transition publication changed status does not match its session."
            )
    elif (
        not transition.only_if_no_queued_messages
        or result.session.status not in transition.from_statuses
    ):
        raise RuntimeError("Interaction transition publication unchanged status is inconsistent.")
    return result


def _attach_interaction_transition_cancellation_outcome(
    cancellation: asyncio.CancelledError,
    *,
    transition_settled: bool,
    failure_diagnostics: list[dict[str, Any]],
    transition: InteractionTransitionSpec | None,
) -> None:
    """Carry only authenticated, durable-safe transition settlement evidence."""

    copied_diagnostics = copy_durable_json_value(
        failure_diagnostics,
        "interaction transition cancellation diagnostics",
    )
    if type(copied_diagnostics) is not list:
        raise TypeError("Interaction transition cancellation diagnostics must be a list.")
    outcome = _InteractionTransitionCancellationOutcome(
        authority=_INTERACTION_TRANSITION_CANCELLATION_OUTCOME_AUTHORITY,
        transition_settled=transition_settled,
        encoded_failure_diagnostics=json.dumps(
            copied_diagnostics,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoded_transition_spec=(
            None
            if transition is None
            else json.dumps(
                copy_interaction_transition_spec(transition).model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
    )
    if not set_exception_state(
        cancellation,
        _INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE,
        outcome,
    ):
        raise RuntimeError("Could not attach interaction transition cancellation outcome.")


def _pop_interaction_transition_cancellation_outcome(
    cancellation: asyncio.CancelledError,
) -> tuple[bool, list[dict[str, Any]], InteractionTransitionSpec | None] | None:
    """Consume transition evidence only when it came from the owned store boundary."""

    outcome = pop_exception_state(
        cancellation,
        _INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE,
    )
    if (
        type(outcome) is not _InteractionTransitionCancellationOutcome
        or outcome.authority is not _INTERACTION_TRANSITION_CANCELLATION_OUTCOME_AUTHORITY
        or type(outcome.transition_settled) is not bool
        or type(outcome.encoded_failure_diagnostics) is not str
        or (
            outcome.encoded_transition_spec is not None
            and type(outcome.encoded_transition_spec) is not str
        )
    ):
        return None
    try:
        decoded = json.loads(outcome.encoded_failure_diagnostics)
        copied = copy_durable_json_value(
            decoded,
            "interaction transition cancellation diagnostics",
        )
        transition = (
            None
            if outcome.encoded_transition_spec is None
            else InteractionTransitionSpec.model_validate(
                json.loads(outcome.encoded_transition_spec)
            )
        )
    except Exception:
        return None
    if type(copied) is not list or any(type(item) is not dict for item in copied):
        return None
    return outcome.transition_settled, copied, transition


def _raise_interaction_transition_cancellation(
    cancellation: asyncio.CancelledError,
    secondary_failures: Iterable[BaseException] = (),
    *,
    transition_settled: bool = False,
    failure_diagnostics: list[dict[str, Any]] | None = None,
    transition: InteractionTransitionSpec | None = None,
) -> NoReturn:
    """Redeliver caller cancellation with every settled secondary outcome."""

    current_failures = list(secondary_failures)
    if any(
        _interaction_transition_failure_contains_run_fence(failure) for failure in current_failures
    ):
        _mark_interaction_transition_run_fence(cancellation)
    _attach_interaction_transition_cancellation_outcome(
        cancellation,
        transition_settled=transition_settled,
        failure_diagnostics=[] if failure_diagnostics is None else failure_diagnostics,
        transition=transition,
    )
    preserved: list[BaseException] = []
    existing_cause = exception_cause(cancellation)
    if existing_cause is not None:
        preserved.append(existing_cause)
    preserved = _unique_exception_identities([*preserved, *current_failures])
    if preserved:
        cause: BaseException = (
            preserved[0]
            if len(preserved) == 1
            else BaseExceptionGroup(
                "Interaction transition cancellation and store failures.",
                preserved,
            )
        )
        if not set_exception_cause(cancellation, cause):
            raise BaseExceptionGroup(
                "Interaction transition cancellation and store failures.",
                [cancellation, cause],
            ) from None
    raise cancellation from exception_cause(cancellation)


def _interaction_transition_replay_is_authoritatively_rejected(
    error: BaseException,
) -> bool:
    """Return whether retry cannot reconcile the rejected transition."""

    leaves = tuple(
        candidate
        for candidate in iter_exception_tree(error)
        if not isinstance(candidate, BaseExceptionGroup)
    )
    return bool(leaves) and all(isinstance(leaf, SessionStatusConflict) for leaf in leaves)


def _transcript_snapshot_messages(snapshot: TranscriptSnapshot) -> list[Message]:
    return [detach_message(record.message) for record in snapshot.records]


def _project_model_target_snapshot(
    snapshot: TranscriptSnapshot,
    projection_cursor: int,
) -> model_target.PortableTranscriptProjection:
    """Project retained rows preceding one permanent absolute cursor."""

    if projection_cursor > snapshot.cursor:
        raise ValueError("Model-target projection cursor exceeds the session transcript.")
    prefix_count = sum(record.index < projection_cursor for record in snapshot.records)
    return model_target.project_portable_transcript_prefix(
        _transcript_snapshot_messages(snapshot),
        prefix_count,
    )


def _session_operation_heartbeat_failure(task: asyncio.Task[None]) -> BaseException:
    if task.cancelled():
        try:
            task.result()
        except asyncio.CancelledError as cancellation:
            concurrent_failure = exception_cause(cancellation)
            if concurrent_failure is not None:
                return concurrent_failure
        return SessionCompactionAttemptSuperseded(
            "Session compaction operation heartbeat was cancelled unexpectedly."
        )
    failure = task.exception()
    if failure is None:
        return SessionCompactionAttemptSuperseded(
            "Session compaction operation heartbeat stopped unexpectedly."
        )
    return failure


def _completed_session_operation_child_failure(
    task: asyncio.Task[Any],
    *,
    operation: str,
) -> BaseException | None:
    try:
        task.result()
    except asyncio.CancelledError as child_cancellation:
        return unexpected_child_cancellation_error(
            child_cancellation,
            operation=operation,
        )
    except BaseException as child_failure:
        return child_failure
    return None


def _consume_detached_session_operation_task(task: asyncio.Task[Any]) -> None:
    """Observe a renewal task that was detached after its lease deadline."""

    # Cancelled tasks do not emit an un-retrieved-exception warning, and calling
    # result() here would consume their cancellation message before the owning
    # heartbeat can classify and preserve it.
    if task.cancelled():
        return
    with contextlib.suppress(BaseException):
        task.result()


class _SessionOperationClaimHeartbeatState:
    """Expose a heartbeat store call that may still own the session boundary."""

    def __init__(self) -> None:
        self.pending_store_task: asyncio.Task[Any] | None = None
        self.confirmed_claim_expires_at: datetime | None = None

    def confirm_claim(self, expires_at: datetime) -> None:
        confirmed = self.confirmed_claim_expires_at
        if confirmed is None or expires_at > confirmed:
            self.confirmed_claim_expires_at = expires_at

    def observe_store_task(self, task: asyncio.Task[Any]) -> None:
        if self.pending_store_task is task:
            self.pending_store_task = None
        _consume_detached_session_operation_task(task)


def _require_unexpired_session_operation_commit(
    *,
    clock: Callable[[], datetime],
    claim_expires_at: datetime,
    operation: str,
) -> None:
    commit_at = clock()
    if commit_at.tzinfo is None or commit_at.utcoffset() is None:
        raise ValueError(f"{operation} commit time must be timezone-aware.")
    if claim_expires_at <= commit_at:
        raise SessionCompactionAttemptSuperseded(f"{operation} claim expired before commit.")


def _require_unexpired_session_operation_claim(
    *,
    clock: Callable[[], datetime],
    claim_expires_at: datetime,
    operation: str,
) -> None:
    observed_at = clock()
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError(f"{operation} observation time must be timezone-aware.")
    if claim_expires_at.tzinfo is None or claim_expires_at.utcoffset() is None:
        raise ValueError(f"{operation} claim expiry must be timezone-aware.")
    if claim_expires_at <= observed_at:
        raise SessionCompactionAttemptSuperseded(
            f"{operation} returned a claim that had already expired."
        )


async def _run_while_session_operation_claimed(
    operation: Callable[[], Awaitable[tuple[ContextBuildResult, BaseException | None]]],
    *,
    heartbeat_task: asyncio.Task[None],
) -> tuple[ContextBuildResult, BaseException | None]:
    heartbeat_observed = False
    heartbeat_failure: BaseException | None = None

    def observe_heartbeat_failure() -> BaseException:
        nonlocal heartbeat_failure
        nonlocal heartbeat_observed
        if not heartbeat_observed:
            heartbeat_failure = _session_operation_heartbeat_failure(heartbeat_task)
            heartbeat_observed = True
        if heartbeat_failure is None:
            raise AssertionError("Completed session operation heartbeat had no outcome.")
        return heartbeat_failure

    async def run_operation() -> tuple[ContextBuildResult, BaseException | None]:
        if heartbeat_task.done():
            raise observe_heartbeat_failure()
        return await operation()

    if heartbeat_task.done():
        raise observe_heartbeat_failure()
    operation_task = asyncio.create_task(run_operation())
    try:
        done, _pending = await asyncio.wait(
            {operation_task, heartbeat_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if heartbeat_task not in done:
            try:
                return operation_task.result()
            except asyncio.CancelledError as child_cancellation:
                raise unexpected_child_cancellation_error(
                    child_cancellation,
                    operation="Session compaction operation",
                ) from child_cancellation

        heartbeat_failure = observe_heartbeat_failure()
        if not operation_task.done():
            operation_task.cancel()
        outcome = await await_shielded_task_outcome(operation_task)
        if outcome.result is not None:
            _attach_context_build_termination_diagnostics(
                heartbeat_failure,
                compaction_telemetry=outcome.result[0].compaction_telemetry,
            )
        if outcome.error is not None:
            _preserve_context_build_termination_diagnostics(
                outcome.error,
                heartbeat_failure,
            )
        if outcome.cancellation is not None:
            outcome.cancellation.add_note(
                "Session compaction operation ownership was also lost: "
                f"{type(heartbeat_failure).__name__}: {heartbeat_failure}"
            )
            raise outcome.cancellation from heartbeat_failure
        if outcome.error is not None and not isinstance(
            outcome.error,
            asyncio.CancelledError,
        ):
            heartbeat_failure.add_note(
                "Compaction also failed while operation ownership loss was handled: "
                f"{type(outcome.error).__name__}: {outcome.error}"
            )
            raise heartbeat_failure from outcome.error
        raise heartbeat_failure
    except asyncio.CancelledError as exc:
        if not operation_task.done():
            operation_task.cancel()
        outcome = await await_shielded_task_outcome(
            operation_task,
            cancellation=exc,
        )
        cancellation = outcome.cancellation or exc
        if outcome.result is not None:
            _attach_context_build_termination_diagnostics(
                cancellation,
                compaction_telemetry=outcome.result[0].compaction_telemetry,
            )
        if outcome.error is not None:
            _preserve_context_build_termination_diagnostics(
                outcome.error,
                cancellation,
            )
        if heartbeat_task.done():
            heartbeat_failure = observe_heartbeat_failure()
            cancellation.add_note(
                "Session compaction operation ownership was also lost during caller "
                f"cancellation: {type(heartbeat_failure).__name__}: {heartbeat_failure}"
            )
        if outcome.error is not None and not isinstance(
            outcome.error,
            asyncio.CancelledError,
        ):
            cancellation.add_note(
                "Compaction also failed while caller cancellation was handled: "
                f"{type(outcome.error).__name__}: {outcome.error}"
            )
        raise cancellation from exception_cause(cancellation)
    finally:
        if not operation_task.done():
            operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)


def _preserve_context_build_termination_diagnostics(
    source: BaseException,
    target: BaseException,
) -> None:
    telemetry = (
        source.compaction_telemetry
        if isinstance(source, ContextBuildError)
        else context_build_termination_compaction_telemetry(source)
    )
    if telemetry:
        _attach_context_build_termination_diagnostics(
            target,
            compaction_telemetry=list(telemetry),
        )


def _compact_session_request_digest(request: CompactSessionRequest) -> str:
    payload = request.model_dump(mode="json")
    # The authenticated caller is audit data for each attempt, not part of the
    # operation's semantic identity. A different operator must be able to
    # recover the same idempotent request after its lease expires.
    payload.pop("requested_by", None)
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _optional_text_digest(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode()).hexdigest()


def _session_operation_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    stored = checkpoint.get(_SESSION_OPERATIONS_CHECKPOINT_KEY)
    if stored is None:
        return {"version": 1, "active_operation_id": None, "records": {}}
    if type(stored) is not dict:
        raise ValueError("Session operation checkpoint must be an object.")
    operations = copy_json_value(stored, _SESSION_OPERATIONS_CHECKPOINT_KEY)
    if operations.get("version") != 1:
        raise ValueError("Unsupported session operation checkpoint version.")
    records = operations.get("records")
    if type(records) is not dict:
        raise ValueError("Session operation checkpoint records must be an object.")
    active_operation_id = operations.get("active_operation_id")
    if active_operation_id is not None:
        require_clean_nonblank(active_operation_id, "active_operation_id")
    return operations


def _active_session_operation_id(checkpoint: dict[str, Any] | None) -> str | None:
    if checkpoint is None or _SESSION_OPERATIONS_CHECKPOINT_KEY not in checkpoint:
        return None
    active_operation_id = _session_operation_state(checkpoint).get("active_operation_id")
    return active_operation_id if type(active_operation_id) is str else None


def _operation_claim_expiry(record: dict[str, Any]) -> datetime | None:
    value = record.get("claim_expires_at")
    if type(value) is not str:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _abandon_expired_session_operation(
    operations: dict[str, Any],
    *,
    now: datetime,
) -> str | None:
    active_operation_id = operations.get("active_operation_id")
    if type(active_operation_id) is not str:
        return None
    records = operations.get("records")
    if type(records) is not dict:
        raise ValueError("Session operation checkpoint records must be an object.")
    active_record = next(
        (
            record
            for record in records.values()
            if type(record) is dict and record.get("operation_id") == active_operation_id
        ),
        None,
    )
    if active_record is None:
        raise RuntimeError("Active durable session operation record is missing.")
    expiry = _operation_claim_expiry(active_record)
    if active_record.get("status") != "running" or expiry is None or expiry > now:
        return None
    active_record["status"] = "abandoned"
    active_record["abandoned_at"] = now.isoformat()
    active_record["updated_at"] = now.isoformat()
    active_record.pop("claim_expires_at", None)
    operations["active_operation_id"] = None
    return active_operation_id


def _session_compaction_replay_event_ids(
    record: dict[str, Any],
) -> tuple[str, ...] | None:
    status = record.get("status")
    if status in {"running", "abandoned"}:
        return None
    if status not in {"completed", "failed"}:
        raise RuntimeError(f"Unsupported session compaction operation status: {status}")
    stored_event_ids = record.get("event_ids")
    if type(stored_event_ids) is not list or not all(
        type(event_id) is str and event_id for event_id in stored_event_ids
    ):
        raise RuntimeError("Completed session compaction operation is missing replay event ids.")
    return tuple(stored_event_ids)


def _store_session_operation_state(
    checkpoint: dict[str, Any],
    operations: dict[str, Any],
) -> None:
    records = operations.get("records")
    if type(records) is not dict:
        raise ValueError("Session operation checkpoint records must be an object.")
    if records or operations.get("active_operation_id") is not None:
        checkpoint[_SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
    else:
        checkpoint.pop(_SESSION_OPERATIONS_CHECKPOINT_KEY, None)


def _archive_inactive_session_operation_records(
    checkpoint: dict[str, Any],
    *,
    except_idempotency_key: str,
) -> dict[str, dict[str, Any]]:
    """Move inactive operation records out of the live checkpoint."""

    operations = _session_operation_state(checkpoint)
    records = operations["records"]
    archived: dict[str, dict[str, Any]] = {}
    for key, record in list(records.items()):
        if key == except_idempotency_key:
            continue
        if type(record) is not dict:
            raise ValueError("Session operation checkpoint records must be objects.")
        if record.get("status") == "running":
            raise RuntimeError("Checkpoint contains an untracked running session operation.")
        archived[key] = records.pop(key)
    _store_session_operation_state(checkpoint, operations)
    return archived


def _reject_pending_provider_operation_disposition(
    checkpoint: dict[str, Any] | None,
) -> None:
    if pending_provider_operation_disposition_from_checkpoint(checkpoint) is not None:
        raise RuntimeError(
            "Session has an accepted provider-operation resolution pending. "
            "Recover that disposition before starting other session work."
        )


def _reject_unresumable_session_checkpoint(
    session: Session,
    checkpoint: dict[str, Any] | None,
    *,
    redactor: SecretRedactor,
    allow_active_operation: bool = False,
) -> None:
    _reject_prepared_prompt_transition_intent(checkpoint)
    _reject_pending_provider_operation_disposition(checkpoint)
    if _initial_transcript_pending_interaction_id(checkpoint) is not None:
        raise RuntimeError(
            "Session setup did not publish its authoritative initial transcript; "
            "resume, fork, and compaction fail closed. Start a new session."
        )
    if (
        approval_support.pending_approval_from_checkpoint(
            checkpoint,
            redactor=redactor,
            consume_on_rejection=True,
        )
        is not None
    ):
        raise RuntimeError("Session has a pending tool approval.")
    if (
        pending_user_input_from_checkpoint(
            checkpoint,
            redactor=redactor,
            consume_on_rejection=True,
        )
        is not None
    ):
        raise RuntimeError("Session is awaiting user input.")
    if (
        tool_round_recovery.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=redactor,
            consume_on_rejection=True,
        )
        is not None
    ):
        raise RuntimeError("Session has a pending tool round.")
    if checkpoint is not None and _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY in checkpoint:
        raise RuntimeError("Session has an incomplete background interruption cascade.")
    if checkpoint is not None and not allow_active_operation:
        operations = _session_operation_state(checkpoint)
        active_operation_id = operations.get("active_operation_id")
        if active_operation_id is not None:
            raise RuntimeError(f"Session has an active durable operation: {active_operation_id}")


def _application_compaction_causal_payload(
    *,
    request: CompactSessionRequest,
    operation_id: str,
    attempt_id: str,
    source_cursor: int,
    compactor: str,
    result_cursor: Any = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "operation_id": operation_id,
        "attempt_id": attempt_id,
        "request_id": request.idempotency_key,
        "reason": request.reason,
        "source_run_epoch": request.expected_run_epoch,
        "source_transcript_cursor": source_cursor,
        "compactor": _application_compaction_event_text(compactor) or "ContextCompactor",
        "mode": "bounded",
        "instruction_present": request.instructions is not None,
        "instruction_digest": _optional_text_digest(request.instructions),
        "actor": resolution_actor_payload(request.requested_by),
    }
    if type(result_cursor) is int and result_cursor >= 0:
        payload["result_transcript_cursor"] = result_cursor
    return payload


_APPLICATION_COMPACTION_EVENT_TEXT_MAX_BYTES = 512


def _application_compaction_event_text(value: Any) -> str | None:
    if type(value) is not str or not value or value != value.strip():
        return None
    if any(
        0xD800 <= ord(char) <= 0xDFFF or ord(char) < 0x20 or ord(char) == 0x7F for char in value
    ):
        return None
    if len(value.encode("utf-8")) > _APPLICATION_COMPACTION_EVENT_TEXT_MAX_BYTES:
        return None
    return value


def _model_execution_identity_payload(
    identity: ModelStepIdentity | ModelAttemptIdentity,
) -> dict[str, str]:
    if type(identity) is ModelAttemptIdentity:
        return copy_model_attempt_identity(identity).payload()
    if type(identity) is ModelStepIdentity:
        return copy_model_step_identity(identity).payload()
    raise TypeError("Model execution identity has an unsupported type.")


def _require_application_compaction_event_text(value: Any, field_name: str) -> str:
    value = require_clean_nonblank(value, field_name)
    if _application_compaction_event_text(value) is None:
        raise ValueError(
            f"`{field_name}` must contain valid Unicode without control characters "
            f"and be at most {_APPLICATION_COMPACTION_EVENT_TEXT_MAX_BYTES} UTF-8 bytes."
        )
    return value


def _application_compaction_event(
    *,
    telemetry: ContextCompactionTelemetry,
    request: CompactSessionRequest,
    operation_id: str,
    attempt_id: str,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    compactor: str,
    execution_identity: ModelStepIdentity | ModelAttemptIdentity | None = None,
) -> Event:
    sanitized = sanitize_context_compaction_telemetry(telemetry)
    payload = copy_json_value(sanitized.payload, "compaction_telemetry_payload")
    strip_runtime_owned_execution_identity(payload)
    if sanitized.event_type == EventType.CONTEXT_COMPACTION_FAILED:
        payload.pop("compacted_transcript_cursor", None)
    if execution_identity is not None:
        payload.update(_model_execution_identity_payload(execution_identity))
    payload.update(
        _application_compaction_causal_payload(
            request=request,
            operation_id=operation_id,
            attempt_id=attempt_id,
            source_cursor=request.expected_transcript_cursor,
            result_cursor=payload.get("compacted_transcript_cursor"),
            compactor=compactor,
        )
    )
    return Event(
        type=sanitized.event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload=payload,
    )


def _application_compaction_budget_event(
    *,
    check: BudgetCheck,
    request: CompactSessionRequest,
    operation_id: str,
    attempt_id: str,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    compactor: str,
    execution_identity: ModelStepIdentity | ModelAttemptIdentity,
) -> Event:
    return Event(
        type=EventType.BUDGET_LIMIT_REACHED,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload={
            **budget_check_payload(check),
            **_model_execution_identity_payload(execution_identity),
            **_application_compaction_causal_payload(
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                source_cursor=request.expected_transcript_cursor,
                compactor=compactor,
            ),
        },
    )


def _application_compaction_ledger_event(
    *,
    event_type: EventType,
    payload: dict[str, Any],
    request: CompactSessionRequest,
    operation_id: str,
    attempt_id: str,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    compactor: str,
) -> Event:
    return Event(
        type=event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload={
            **copy_json_value(payload, "compaction_budget_payload"),
            **_application_compaction_causal_payload(
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                source_cursor=request.expected_transcript_cursor,
                compactor=compactor,
            ),
        },
    )


def _budget_check_identity(
    check: BudgetCheck,
) -> str:
    return check.budget_limit_id


def _complete_session_operation_checkpoint(
    *,
    checkpoint: dict[str, Any] | None,
    persisted_record: dict[str, Any] | None,
    compacted_checkpoint: dict[str, Any],
    idempotency_key: str,
    operation_id: str,
    attempt_id: str,
    event_ids: list[str],
    result_cursor: Any,
    completed_at: datetime,
    on_terminalize: Callable[[datetime], None],
) -> SessionOperationPublication:
    if completed_at.tzinfo is None or completed_at.utcoffset() is None:
        raise ValueError("Session compaction completion time must be timezone-aware.")
    if checkpoint is None:
        raise SessionCompactionAttemptSuperseded(
            "Session compaction attempt was superseded before publication."
        )
    updated = copy_json_value(checkpoint, "checkpoint")
    operations = _session_operation_state(updated)
    record = operations["records"].get(idempotency_key)
    if type(record) is not dict or record.get("operation_id") != operation_id:
        if persisted_record is not None and persisted_record.get("operation_id") == operation_id:
            raise SessionCompactionAttemptSuperseded(
                "Session compaction attempt was superseded before publication."
            )
        raise RuntimeError("Session compaction operation claim was lost before publication.")
    if (
        record.get("status") != "running"
        or record.get("current_attempt_id") != attempt_id
        or operations.get("active_operation_id") != operation_id
    ):
        raise SessionCompactionAttemptSuperseded(
            "Session compaction attempt was superseded before publication."
        )
    claim_expires_at = _operation_claim_expiry(record)
    if claim_expires_at is None or claim_expires_at <= completed_at:
        raise SessionCompactionAttemptSuperseded(
            "Session compaction operation claim expired before terminal publication."
        )
    if persisted_record is not None:
        if persisted_record.get("operation_id") != operation_id:
            raise RuntimeError(
                "Session compaction operation replay record does not match its claim."
            )
        if persisted_record.get("status") != "abandoned":
            raise SessionCompactionAttemptSuperseded(
                "Session compaction attempt was superseded before publication."
            )
    on_terminalize(claim_expires_at)
    compacted_state = copy_json_value(compacted_checkpoint, "compacted_checkpoint")
    compacted_context = compacted_state.get(_CONTEXT_COMPACTION_OPERATION_KIND)
    if type(compacted_context) is not dict:
        raise ValueError("Compacted checkpoint is missing context compaction state.")
    updated[_CONTEXT_COMPACTION_OPERATION_KIND] = compacted_context
    record["status"] = "completed"
    existing_event_ids = record.get("event_ids", [])
    record["event_ids"] = [
        *existing_event_ids,
        *(event_id for event_id in event_ids if event_id not in existing_event_ids),
    ]
    record["result_transcript_cursor"] = result_cursor
    record["completed_at"] = completed_at.isoformat()
    record["updated_at"] = completed_at.isoformat()
    record.pop("claim_expires_at", None)
    operations["active_operation_id"] = None
    terminal_record = operations["records"].pop(idempotency_key)
    _store_session_operation_state(updated, operations)
    return SessionOperationPublication(
        checkpoint=updated,
        operation_records={idempotency_key: terminal_record},
    )


def _renew_session_operation_claim_publication(
    *,
    checkpoint: dict[str, Any] | None,
    idempotency_key: str,
    operation_id: str,
    attempt_id: str,
    event_ids: list[str],
    renewed_at: datetime,
    persisted_record: dict[str, Any] | None,
    on_renew: Callable[[datetime], None] | None = None,
    on_renewed: Callable[[datetime], None] | None = None,
) -> tuple[SessionOperationPublication, datetime]:
    if renewed_at.tzinfo is None or renewed_at.utcoffset() is None:
        raise ValueError("Session compaction claim renewal time must be timezone-aware.")
    if checkpoint is None:
        raise SessionCompactionAttemptSuperseded(
            "Session compaction attempt was superseded before claim renewal."
        )
    updated = copy_json_value(checkpoint, "checkpoint")
    operations = _session_operation_state(updated)
    record = operations["records"].get(idempotency_key)
    if type(record) is not dict or record.get("operation_id") != operation_id:
        if persisted_record is not None and persisted_record.get("operation_id") == operation_id:
            raise SessionCompactionAttemptSuperseded(
                "Session compaction attempt was superseded before claim renewal."
            )
        raise RuntimeError("Session compaction operation claim was lost before renewal.")
    if (
        record.get("status") != "running"
        or record.get("current_attempt_id") != attempt_id
        or operations.get("active_operation_id") != operation_id
    ):
        raise SessionCompactionAttemptSuperseded(
            "Session compaction attempt was superseded before claim renewal."
        )
    current_expiry = _operation_claim_expiry(record)
    if current_expiry is None or current_expiry <= renewed_at:
        raise SessionCompactionAttemptSuperseded(
            "Session compaction operation claim expired before renewal."
        )
    if on_renew is not None:
        on_renew(current_expiry)
    existing_event_ids = record.get("event_ids", [])
    record["event_ids"] = [
        *existing_event_ids,
        *(event_id for event_id in event_ids if event_id not in existing_event_ids),
    ]
    renewed_until = renewed_at + _SESSION_OPERATION_CLAIM_LEASE
    record["updated_at"] = renewed_at.isoformat()
    record["claim_expires_at"] = renewed_until.isoformat()
    if on_renewed is not None:
        on_renewed(renewed_until)
    updated[_SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
    return SessionOperationPublication(checkpoint=updated), renewed_until


def _append_session_operation_attempt_events(
    *,
    idempotency_key: str,
    operation_id: str,
    attempt_id: str,
    event_ids: list[str],
    clock: Callable[[], datetime],
    on_renew: Callable[[datetime], None],
    on_renewed: Callable[[datetime], None],
) -> Callable[
    [Session, dict[str, Any] | None, dict[str, Any] | None],
    SessionOperationPublication,
]:
    def append_events(
        _session: Session,
        checkpoint: dict[str, Any] | None,
        persisted_record: dict[str, Any] | None,
    ) -> SessionOperationPublication:
        publication, _renewed_until = _renew_session_operation_claim_publication(
            checkpoint=checkpoint,
            idempotency_key=idempotency_key,
            operation_id=operation_id,
            attempt_id=attempt_id,
            event_ids=event_ids,
            renewed_at=clock(),
            persisted_record=persisted_record,
            on_renew=on_renew,
            on_renewed=on_renewed,
        )
        return publication

    return append_events


def _fail_session_operation_checkpoint(
    *,
    idempotency_key: str,
    operation_id: str,
    attempt_id: str,
    failed_event_id: str,
    attempt_event_ids: list[str],
    error_type: str,
    clock: Callable[[], datetime],
    on_terminalize: Callable[[datetime], None],
) -> Callable[
    [Session, dict[str, Any] | None, dict[str, Any] | None],
    SessionOperationPublication,
]:
    def fail(
        _session: Session,
        checkpoint: dict[str, Any] | None,
        persisted_record: dict[str, Any] | None,
    ) -> SessionOperationPublication:
        completed_at = clock()
        if completed_at.tzinfo is None or completed_at.utcoffset() is None:
            raise ValueError("Session compaction failure time must be timezone-aware.")
        updated = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        operations = _session_operation_state(updated)
        record = operations["records"].get(idempotency_key)
        if type(record) is not dict or record.get("operation_id") != operation_id:
            if (
                persisted_record is not None
                and persisted_record.get("operation_id") == operation_id
            ):
                if (
                    persisted_record.get("status") in {"completed", "failed"}
                    and persisted_record.get("current_attempt_id") == attempt_id
                ):
                    raise SessionCompactionAttemptSuperseded(
                        "Session compaction attempt was already terminal before failure "
                        "publication."
                    )
                terminal_record = copy_json_value(persisted_record, "session_operation")
                existing_event_ids = terminal_record.get("event_ids", [])
                terminal_record["event_ids"] = [
                    *existing_event_ids,
                    *(
                        event_id
                        for event_id in [*attempt_event_ids, failed_event_id]
                        if event_id not in existing_event_ids
                    ),
                ]
                return SessionOperationPublication(
                    checkpoint=updated,
                    operation_records={idempotency_key: terminal_record},
                )
            raise RuntimeError("Session compaction operation claim was lost before failure.")
        existing_event_ids = record.get("event_ids", [])
        record["event_ids"] = [
            *existing_event_ids,
            *(
                event_id
                for event_id in [*attempt_event_ids, failed_event_id]
                if event_id not in existing_event_ids
            ),
        ]
        if (
            record.get("status") != "running"
            or record.get("current_attempt_id") != attempt_id
            or operations.get("active_operation_id") != operation_id
        ):
            updated[_SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
            return SessionOperationPublication(checkpoint=updated)
        claim_expires_at = _operation_claim_expiry(record)
        if claim_expires_at is None or claim_expires_at <= completed_at:
            updated[_SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
            return SessionOperationPublication(checkpoint=updated)
        on_terminalize(claim_expires_at)
        record["status"] = "failed"
        record["error_type"] = error_type
        record["completed_at"] = completed_at.isoformat()
        record["updated_at"] = completed_at.isoformat()
        record.pop("claim_expires_at", None)
        operations["active_operation_id"] = None
        terminal_record = operations["records"].pop(idempotency_key)
        _store_session_operation_state(updated, operations)
        return SessionOperationPublication(
            checkpoint=updated,
            operation_records={idempotency_key: terminal_record},
        )

    return fail


def _validate_run_request(request: RunRequest) -> RunRequest:
    return copy_run_request(request)


def _validate_resume_request(request: ResumeRequest) -> ResumeRequest:
    return copy_resume_request(request)


def _require_native_structured_output_support(
    structured_output: StructuredOutputSpec | None,
    *,
    registered_provider: runtime_records.RegisteredProvider,
) -> None:
    """Reject an unusable ``strategy=NATIVE`` spec at an entry point.

    Called by every entrance before it creates a session or transitions its
    status, so the typed error reaches the caller with no persisted state
    changed (model-pattern routing can select the provider by model name
    alone). Raises ``NativeStructuredOutputUnsupported`` when the resolved
    provider has no native mode, then lets the provider's own schema preflight
    reject schemas its native mode would refuse at request time
    (``NativeStructuredOutputSchemaInvalid``). One pre-existing mid-run call
    in ``_run_session`` reuses this helper as a typed backstop for missed
    entrances; a raise there fails the already-running session cleanly.
    """
    _require_provider_native_output_support(
        structured_output,
        provider_name=registered_provider.name,
        provider=registered_provider.provider,
    )


@dataclass(frozen=True, slots=True)
class _PreparedInitialRun:
    """Validated new-session material frozen before the store creation boundary."""

    request: RunRequest
    registered_agent: runtime_records.RegisteredAgentState
    registered_provider: runtime_records.RegisteredProvider
    registered_environment: runtime_records.RegisteredEnvironment | None
    workspace_instructions: WorkspaceInstructions | None
    rendered_system_prompt: str | None
    prompt_contributions: tuple[Any, ...]
    execution_profile: ExecutionProfileIdentity
    budget_policy: BudgetPolicy | None
    session_identity: SessionIdentity


def _session_identity(
    *,
    provider_name: str,
    model: str,
    execution_profile: ExecutionProfileIdentity | None = None,
) -> SessionIdentity:
    return SessionIdentity(
        provider_name=provider_name,
        model=model,
        runtime_name="cayu",
        runtime_version=_runtime_version(),
        execution_profile=execution_profile,
    )


def _runtime_version() -> str | None:
    try:
        return version("cayu")
    except PackageNotFoundError:
        return None


def _execution_profile_identity(
    *,
    registered_agent: runtime_records.RegisteredAgentState,
    provider_name: str,
    model: str,
    durable_system_prompt: str | None,
    redactor: SecretRedactor,
    registered_environment: runtime_records.RegisteredEnvironment | None = None,
    process_identity: str = "standalone-profile-builder",
    runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...] = (),
    loop_policies: tuple[LoopPolicy, ...] = (),
    loop_policy_execution_profile_identities: tuple[
        ExecutionProfileBehaviorIdentity | None, ...
    ] = (),
    request_loop_policies: tuple[LoopPolicy, ...] = (),
    request_loop_policy_instance_identities: tuple[str | None, ...] = (),
    registered_provider: runtime_records.RegisteredProvider | None = None,
    budget_policy: BudgetPolicy | None = None,
    request_budget_limits: tuple[BudgetLimit, ...] = (),
    causal_budget_id: str | None = None,
    structured_output: StructuredOutputSpec | None = None,
    thinking: ThinkingConfig | None = None,
    max_steps: int = 16,
    limits: RunLimits | None = None,
    retry_policy: RetryPolicy | None = None,
    finalization_material: dict[str, Any] | None = None,
) -> ExecutionProfileIdentity:
    provider_options, provider_options_process_local = _execution_profile_provider_options(
        registered_agent.spec.provider_options,
        provider=registered_provider.provider if registered_provider is not None else None,
        model=model,
        process_identity=process_identity,
    )
    effective_thinking = thinking if thinking is not None else registered_agent.spec.thinking
    app_limit_ids: tuple[str, ...] = ()
    request_limit_ids: tuple[str, ...] = ()
    if causal_budget_id is not None:
        app_limit_ids = tuple(
            limit.budget_limit_id
            for limit in budget_limits_for_session(
                policy=budget_policy,
                agent_name=registered_agent.spec.name,
                causal_budget_id=causal_budget_id,
            )
        )
        request_limit_ids = tuple(
            limit.budget_limit_id
            for limit in request_budget_limits_for_session(
                limits=request_budget_limits,
                agent_name=registered_agent.spec.name,
                causal_budget_id=causal_budget_id,
            )
        )
    return execution_profile_admission.resolve_execution_profile_identity(
        registered_agent=registered_agent,
        provider_name=provider_name,
        model=model,
        durable_system_prompt=durable_system_prompt,
        runtime_name="cayu",
        runtime_version=_runtime_version(),
        redactor=redactor,
        registered_environment=registered_environment,
        process_identity=process_identity,
        runtime_hooks=runtime_hooks,
        loop_policies=loop_policies,
        loop_policy_identities=loop_policy_execution_profile_identities,
        invocation_loop_policies=request_loop_policies,
        invocation_loop_policy_identities=tuple(
            copy_secret_free_execution_profile_behavior_identity(
                policy.execution_profile_identity,
                redactor=redactor,
                field_name=f"request.loop_policies[{index}].execution_profile_identity",
            )
            for index, policy in enumerate(request_loop_policies)
        ),
        invocation_loop_policy_instance_identities=(request_loop_policy_instance_identities),
        registered_provider=registered_provider,
        provider_options=provider_options,
        provider_options_process_local=provider_options_process_local,
        thinking=(
            None if effective_thinking is None else thinking_config_payload(effective_thinking)
        ),
        app_budget_limit_ids=app_limit_ids,
        request_budget_limit_ids=request_limit_ids,
        structured_output=_execution_profile_structured_output(structured_output),
        finalization=(
            {
                "kind": "cayu:model-finalization:v1",
                "max_steps": max_steps,
                "limits": copy_run_limits(limits).model_dump(mode="json"),
                "retry_policy": copy_retry_policy(retry_policy).model_dump(mode="json"),
            }
            if finalization_material is None
            else copy_json_value(finalization_material, "finalization_material")
        ),
    )


def _execution_profile_provider_options(
    options: dict[str, Any],
    *,
    provider: ModelProvider | None = None,
    model: str = "execution-profile-options",
    process_identity: str,
) -> tuple[dict[str, Any], bool]:
    """Project the selected adapter's effective request-option policy.

    Built-in adapters remove inactive namespaces, normalize controls, and add
    provider defaults through ``request_fingerprint_options``.  The optional
    provider preserves the conservative raw-options behavior for internal
    helper callers that do not yet have a resolved adapter.
    """

    effective_options = options
    cache_policy_material: dict[str, Any] | None = None
    private_options_material: dict[str, Any] = effective_options
    provider_defined_cache_policy = False
    if provider is not None:
        detached_request = ModelRequest(
            model=model,
            messages=[],
            options=options,
        )
        projected_options = provider.request_fingerprint_options(detached_request)
        if type(projected_options) is not dict:
            raise TypeError("ModelProvider.request_fingerprint_options() must return a dict.")
        copied_options = copy_durable_json_value(
            projected_options,
            "execution profile provider options",
        )
        if type(copied_options) is not dict:  # pragma: no cover - copier invariant.
            raise TypeError("ModelProvider.request_fingerprint_options() must return a dict.")
        effective_options = copied_options
        private_options_material = effective_options
        effective_cache_policy, cache_policy_is_authoritative = (
            _execution_profile_effective_cache_policy(
                provider,
                options=detached_request.options,
            )
        )
        cache_policy_material = _execution_profile_cache_policy(effective_cache_policy)
        raw_cache_policy = detached_request.options.get("cache_policy")
        if not cache_policy_is_authoritative and raw_cache_policy is not None:
            copied_cache_policy = copy_durable_json_value(
                raw_cache_policy,
                "execution profile provider-defined cache policy",
            )
            private_options_material = {
                "provider_options": effective_options,
                "provider_defined_cache_policy": copied_cache_policy,
            }
            provider_defined_cache_policy = True

    projected: dict[str, Any] = {}
    opaque = False
    for namespace, value in effective_options.items():
        if type(value) is dict:
            safe = privacy_safe_provider_option_projection(value)
            if safe:
                projected[namespace] = safe
            visible_source = {key: item for key, item in value.items() if item is not None}
            opaque |= safe != visible_source
        elif value is not None:
            opaque = True
    if cache_policy_material is not None:
        projected["cache_policy"] = cache_policy_material
    elif provider_defined_cache_policy:
        projected["cache_policy"] = {"configuration": "provider_defined"}
        opaque = True
    if not opaque:
        return projected, False
    private_digest = hmac.new(
        process_identity.encode("utf-8"),
        canonical_durable_json_bytes(private_options_material, "provider_options"),
        hashlib.sha256,
    ).hexdigest()
    return (
        {
            "public_projection": projected,
            "private_configuration_hmac_sha256": private_digest,
            "process_scope": hashlib.sha256(process_identity.encode("utf-8")).hexdigest(),
        },
        True,
    )


def _execution_profile_cache_policy(
    policy: CachePolicy | None,
) -> dict[str, Any] | None:
    """Canonicalize the effective provider cache policy without request content."""

    if policy is None:
        return None
    if type(policy) is not CachePolicy:
        raise TypeError("ModelProvider.request_cache_policy() must return CachePolicy or None.")
    copied = CachePolicy.model_validate(policy.model_dump(mode="python"))
    breakpoints = set(copied.breakpoints)
    if copied.conversation_prefix_strategy == "none":
        breakpoints.discard(CacheBreakpoint.CONVERSATION_PREFIX)
    if not breakpoints:
        return None
    ordered = tuple(sorted(breakpoints, key=lambda item: item.value))
    material: dict[str, Any] = {
        "breakpoints": [breakpoint.value for breakpoint in ordered],
        "ttl": "extended" if copied.ttl == "extended" else "standard",
    }
    if CacheBreakpoint.CONVERSATION_PREFIX in breakpoints:
        material["conversation_prefix_strategy"] = copied.conversation_prefix_strategy
        if copied.conversation_prefix_strategy == "all_but_last_n":
            material["conversation_prefix_n"] = copied.conversation_prefix_n
    return material


def _execution_profile_effective_cache_policy(
    provider: ModelProvider,
    *,
    options: dict[str, Any],
) -> tuple[CachePolicy | None, bool]:
    """Resolve cache configuration only for an adapter whose complete contract is known."""

    from cayu.providers.anthropic import AnthropicProvider
    from cayu.providers.cache import resolve_cache_policy

    if type(provider) is AnthropicProvider:
        return resolve_cache_policy(provider.cache_policy, options), True
    provider_type = type(provider)
    uses_default_cache_contract = (
        provider_type.request_cache_policy is ModelProvider.request_cache_policy
        and provider_type.request_cache_projection is ModelProvider.request_cache_projection
    )
    return None, uses_default_cache_contract


def _execution_profile_structured_output(
    spec: StructuredOutputSpec | None,
) -> dict[str, Any] | None:
    if spec is None:
        return None
    copied = copy_structured_output_spec(spec)
    if copied is None:
        return None
    schema_digest = hashlib.sha256(
        canonical_durable_json_bytes(copied.json_schema, "structured_output.json_schema")
    ).hexdigest()
    return {
        "kind": "structured_output",
        "version": 1,
        "schema_sha256": schema_digest,
        "strategy": copied.strategy.value,
        "max_retries": copied.max_retries,
        "name_sha256": (
            None if copied.name is None else hashlib.sha256(copied.name.encode("utf-8")).hexdigest()
        ),
        "repair_prompt_sha256": (
            None
            if copied.repair_prompt is None
            else hashlib.sha256(copied.repair_prompt.encode("utf-8")).hexdigest()
        ),
    }


_EXACT_PROFILE_POLICY_ID = "cayu:execution-profile-exact-reuse:v1"
_DEFAULT_PROFILE_POLICY_ID = "cayu:execution-profile-default-reject:v1"
_MODEL_TARGET_PROFILE_POLICY_ID = "cayu:model-target-adoption:v1"
_FORK_GROUP_INITIAL_PROFILE_POLICY_ID = "cayu:fork-group-initial-invocation:v1"
_MODEL_TARGET_BUILT_IN_COMPONENTS = frozenset(
    {
        ExecutionProfileComponentClass.PROVIDER_ADAPTER,
        ExecutionProfileComponentClass.PROVIDER_TARGET,
    }
)


def _model_target_profile_actor() -> ResolutionActor:
    return ResolutionActor(
        subject="cayu:model-target-adoption",
        source=ResolutionActorSource.SYSTEM,
    )


def _fork_group_initial_profile_actor() -> ResolutionActor:
    return ResolutionActor(
        subject="cayu:fork-group-initial-invocation",
        source=ResolutionActorSource.SYSTEM,
    )


def _execution_profile_decision_event_id(
    *,
    session_id: str,
    run_epoch: int,
    expected_profile: ExecutionProfileIdentity,
    candidate_profile: ExecutionProfileIdentity,
    intent: ExecutionProfileAdoptionIntent | None,
    policy_identity: str | None = None,
) -> tuple[str, str]:
    if intent is not None:
        idempotency_identity = intent.idempotency_key
        material = {
            "session_id": session_id,
            "idempotency_identity": idempotency_identity,
        }
    else:
        if policy_identity is None:
            raise ValueError(
                "A runtime-generated execution-profile decision requires policy authority."
            )
        policy_identity_digest = hashlib.sha256(policy_identity.encode("utf-8")).hexdigest()
        idempotency_identity = (
            f"run-epoch:{run_epoch}:{expected_profile.fingerprint}:"
            f"{candidate_profile.fingerprint}:{policy_identity_digest}"
        )
        material = {
            "session_id": session_id,
            "run_epoch": run_epoch,
            "expected_profile_fingerprint": expected_profile.fingerprint,
            "candidate_profile_fingerprint": candidate_profile.fingerprint,
            "policy_identity": policy_identity,
        }
    event_material = canonical_durable_json_bytes(material, "execution_profile_decision")
    return "epd_" + hashlib.sha256(event_material).hexdigest(), idempotency_identity


def _execution_profile_decision_event(
    *,
    session: Session,
    expected_profile: ExecutionProfileIdentity,
    candidate_profile: ExecutionProfileIdentity,
    changed_component_classes: tuple[ExecutionProfileComponentClass, ...],
    kind: ExecutionProfileDecisionKind,
    policy_identity: str,
    policy_reason: str,
    authority_decision: ExecutionProfileAuthorityDecision,
    intent: ExecutionProfileAdoptionIntent | None,
    adoption_request_fingerprint: str | None,
    fallback_actor: ResolutionActor | None,
    fallback_reason: str,
    clock: Callable[[], datetime],
) -> ExecutionProfileDecision:
    if (intent is None) != (adoption_request_fingerprint is None):
        raise ValueError(
            "Explicit execution-profile adoption intent and request fingerprint must "
            "be supplied together."
        )
    event_id, idempotency_identity = _execution_profile_decision_event_id(
        session_id=session.id,
        run_epoch=session.run_epoch,
        expected_profile=expected_profile,
        candidate_profile=candidate_profile,
        intent=intent,
        policy_identity=policy_identity,
    )
    actor = intent.requested_by if intent is not None else fallback_actor
    reason = intent.reason if intent is not None else fallback_reason
    event = Event(
        id=event_id,
        type=(
            EventType.SESSION_EXECUTION_PROFILE_REJECTED
            if kind is ExecutionProfileDecisionKind.REJECTED
            else EventType.SESSION_EXECUTION_PROFILE_DECIDED
        ),
        session_id=session.id,
        timestamp=clock(),
        agent_name=session.agent_name,
        environment_name=session.environment_name,
        payload=execution_profile_decision_payload(
            kind=kind,
            expected_profile=expected_profile,
            candidate_profile=candidate_profile,
            changed_component_classes=changed_component_classes,
            policy_identity=policy_identity,
            policy_reason=policy_reason,
            authority_decision=authority_decision,
            idempotency_identity=idempotency_identity,
            adoption_request_fingerprint=adoption_request_fingerprint,
            actor=actor,
            reason=reason,
        ),
    )
    authority_fields = [
        "decision",
        "policy_identity",
        "authority_decision",
        "idempotency_identity",
    ]
    if adoption_request_fingerprint is not None:
        authority_fields.append("adoption_request_fingerprint")
    event = event_with_runtime_payload_authority(
        event_with_runtime_envelope_authority(
            event_with_runtime_generated_id(event),
            "session_id",
        ),
        *authority_fields,
    )
    return ExecutionProfileDecision(
        kind=kind,
        expected_profile=expected_profile,
        candidate_profile=candidate_profile,
        changed_component_classes=changed_component_classes,
        policy_identity=policy_identity,
        policy_reason=policy_reason,
        authority_decision=authority_decision,
        idempotency_identity=idempotency_identity,
        adoption_request_fingerprint=adoption_request_fingerprint,
        actor=actor,
        reason=reason,
        event=event,
    )


def _with_environment_name(request: RunRequest, environment_name: str) -> RunRequest:
    # ``copy_run_request`` deliberately preserves the request's private runtime
    # authority provenance. Reconstructing a fresh model here would silently
    # downgrade trusted subagent/session lineage back to caller input.
    return copy_run_request(request).model_copy(update={"environment_name": environment_name})


def _environment_name(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> str | None:
    if registered_environment is None:
        return None
    return registered_environment.spec.name


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


async def _call_runtime_hook(
    *,
    hook: RuntimeHook,
    phase: RuntimeHookPhase,
    context: RuntimeHookContext,
) -> None:
    if phase == RuntimeHookPhase.AFTER_SESSION_COMPLETED:
        await hook.after_session_completed(context)
        return
    if phase == RuntimeHookPhase.AFTER_SESSION_FAILED:
        await hook.after_session_failed(context)
        return
    if phase == RuntimeHookPhase.AFTER_SESSION_INTERRUPTED:
        await hook.after_session_interrupted(context)
        return
    raise ValueError(f"Unsupported runtime hook phase: {phase}")


def _runtime_hook_actions_payload(context: RuntimeHookContext) -> dict[str, Any]:
    """Return portable action evidence without making hooks unterminalizable."""

    try:
        actions = copy_durable_json_value(context.actions, "hook_actions")
    except BaseException:
        return {"actions": [], "actions_omitted": True}
    if type(actions) is not list:
        return {"actions": [], "actions_omitted": True}
    return {"actions": actions}


def _loop_policy_supports_before_stop(policy: LoopPolicy) -> bool:
    policy_method = type(policy).before_stop
    default_method = LoopPolicy.before_stop
    return policy_method is not default_method


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


def _checkpoint_with_pending_session_interrupt(
    payload: dict[str, Any],
    *,
    include_interruption_cascade: bool = True,
    cascade_created_at: datetime | None = None,
):
    copied_payload = copy_json_value(payload, "interrupt_payload")
    if cascade_created_at is not None and (
        cascade_created_at.tzinfo is None or cascade_created_at.utcoffset() is None
    ):
        raise ValueError("cascade_created_at must be timezone-aware.")
    resolved_cascade_created_at = (
        datetime.now(UTC) if cascade_created_at is None else cascade_created_at.astimezone(UTC)
    )

    def transform(session: Session, checkpoint: dict[str, Any] | None) -> dict[str, Any]:
        copied_checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        active_profile = active_invocation_execution_profile_from_checkpoint(copied_checkpoint)
        if active_profile is not None:
            if not active_invocation_execution_profile_matches_session_epoch(
                active_profile,
                session_id=session.id,
                run_epoch=session.run_epoch,
            ):
                raise RuntimeError(
                    "Active invocation execution profile does not match the interrupted "
                    "session run epoch."
                )
            copied_checkpoint = checkpoint_with_active_invocation_execution_profile(
                copied_checkpoint,
                session_id=session.id,
                interaction_id=active_profile.interaction_id,
                run_epoch=session.run_epoch,
                profile=active_profile.profile,
                expected=active_profile,
            )
        copied_checkpoint[_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY] = copy_json_value(
            copied_payload,
            "interrupt_payload",
        )
        if (
            include_interruption_cascade
            and copied_payload.get("interruption_type") == _INTERRUPTION_TYPE_OPERATOR_REQUESTED
        ):
            copied_checkpoint[_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY] = {
                "attempt_id": str(uuid4()),
                "interrupt_payload": copy_json_value(
                    copied_payload,
                    "interrupt_payload",
                ),
                "created_at": resolved_cascade_created_at.isoformat(),
            }
        return copied_checkpoint

    return transform


def _runtime_interruption_event(
    event: Event,
    *,
    execution_profile: ExecutionProfileIdentity | None = None,
) -> Event:
    """Attest runtime-owned interruption identities copied through checkpoints."""

    fields = tuple(
        field_name
        for field_name in ("interruption_request_id", "retry_request_id", "attempt_id")
        if type(event.payload.get(field_name)) is str
    )
    event = event_with_runtime_payload_authority(event, *fields)
    profile_fingerprint = event.payload.get("execution_profile_fingerprint")
    if profile_fingerprint is None:
        return event
    if execution_profile is None:
        raise RuntimeError(
            "An interrupted event references an execution profile without validated authority."
        )
    if profile_fingerprint != execution_profile.fingerprint:
        raise RuntimeError(
            "An interrupted event references a different execution profile than recovery."
        )
    return event_with_execution_profile_authority(event, execution_profile)


def _replace_checkpoint_preserving_runtime_state(
    checkpoint: dict[str, Any],
):
    replacement = copy_json_value(checkpoint, "checkpoint")

    def transform(_session: Session, current: dict[str, Any] | None) -> dict[str, Any]:
        updated = copy_json_value(replacement, "checkpoint")
        updated[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION
        for key, field_name in (
            (
                ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
                "active_invocation_execution_profile",
            ),
            (
                INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY,
                "initial_transcript_pending",
            ),
            (_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY, "pending_session_interrupt"),
            (_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY, "pending_interruption_cascade"),
            (_SESSION_OPERATIONS_CHECKPOINT_KEY, "session_operations"),
            (_SESSION_RUN_OPERATION_CHECKPOINT_KEY, "session_run_operation"),
            (
                _QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY,
                "queued_dispatch_terminal_receipts",
            ),
            (
                _PROMPT_ANATOMY_TRANSITION_INTENTS_CHECKPOINT_KEY,
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
        ):
            updated.pop(key, None)
            if current is not None and key in current:
                updated[key] = copy_json_value(current[key], field_name)
        return updated

    return transform


def _limit_reached_payload(
    *,
    decision: StopDecision,
    usage_summary: SessionUsageSummary,
    cost_summary: SessionCostSummary | None,
) -> dict[str, Any]:
    payload = {
        "reason": "limit_reached",
        "limit": decision.limit.value,
        "maximum": _limit_value_for_payload(decision.maximum),
        "actual": _limit_value_for_payload(decision.actual),
        "message": decision.message,
        "usage_summary": session_usage_summary_payload(usage_summary),
    }
    if cost_summary is not None:
        payload["cost_summary"] = cost_summary.model_dump(mode="json")
    return payload


def _limit_value_for_payload(value: int | Decimal) -> int | str:
    if type(value) is Decimal:
        return str(value)
    if type(value) is int:
        if MIN_DURABLE_JSON_INTEGER <= value <= MAX_DURABLE_JSON_INTEGER:
            return value
        return str(value)
    raise TypeError("limit payload value must be an int or Decimal.")


def _limit_reached_tool_round_results(
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
    decision: StopDecision,
    tool_round_identity: ToolRoundIdentity,
) -> list[runtime_records.ToolCallOutcome]:
    identity = copy_tool_round_identity(tool_round_identity)
    outcomes: list[runtime_records.ToolCallOutcome] = []
    for tool_call in tool_calls:
        structured = {
            "skipped": True,
            "reason": "limit_reached",
            "limit": decision.limit.value,
            "maximum": _limit_value_for_payload(decision.maximum),
            "actual": _limit_value_for_payload(decision.actual),
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            **identity.payload(),
        }
        outcomes.append(
            runtime_records.ToolCallOutcome(
                call=tool_call,
                result=ToolResult(
                    content="Tool call skipped because a run limit was reached.",
                    structured=structured,
                    is_error=True,
                ),
            )
        )
    return outcomes


def _limit_reached_tool_call_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    tool_call_outcome: runtime_records.ToolCallOutcome,
    decision: StopDecision,
    tool_round_identity: ToolRoundIdentity,
    execution_profile: ExecutionProfileIdentity | None,
    approval_id: str | None = None,
) -> Event:
    identity = copy_tool_round_identity(tool_round_identity)
    payload = {
        "tool_call_id": tool_call_outcome.call.id,
        "idempotency_key": tool_execution.tool_idempotency_key(
            session_id=session.id,
            tool_round_id=identity.tool_round_id,
            tool_call_id=tool_call_outcome.call.id,
            approval_id=approval_id,
        ),
        "reason": "limit_reached",
        "limit": decision.limit.value,
        "result": tool_call_outcome.result.model_dump(),
        **tool_argument_publication.unavailable_argument_projection().payload_fields(),
        **identity.payload(),
    }
    if approval_id is not None:
        payload["approval_id"] = approval_id
    return event_with_execution_profile_authority(
        Event(
            type=EventType.TOOL_CALL_FAILED,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=_environment_name(registered_environment),
            tool_name=tool_call_outcome.call.name,
            payload=payload,
        ),
        execution_profile,
    )


async def _close_async_iterator(iterator: AsyncIterator[Any]) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()


async def _collect_through_event_type(
    iterator: AsyncIterator[Event],
    event_type: EventType,
    *,
    missing_message: str,
) -> tuple[list[Event], Event]:
    events: list[Event] = []
    async for event in iterator:
        events.append(event)
        if event.type == event_type:
            return events, event
    raise RuntimeError(missing_message)


def _runtime_hook_event(
    *,
    event_type: EventType,
    hook_name: str,
    scope: str,
    phase: RuntimeHookPhase,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    terminal_event: Event,
    payload: dict[str, Any],
    execution_profile: ExecutionProfileIdentity | None = None,
) -> Event:
    return event_with_execution_profile_authority(
        _build_runtime_hook_event(
            event_type=event_type,
            hook_name=hook_name,
            scope=scope,
            phase=phase,
            session=session,
            terminal_event=terminal_event,
            agent_name=registered_agent.spec.name,
            environment_name=_environment_name(registered_environment),
            payload=payload,
        ),
        execution_profile,
    )


def _task_event(
    *,
    event_type: EventType,
    task: Task,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> Event:
    event = Event(
        type=event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=_environment_name(registered_environment),
        payload={
            "task_id": task.id,
            "task_type": task.type,
            "task_status": task.status.value,
            "task_session_id": task.session_id,
            "assigned_agent_name": task.assigned_agent_name,
            "parent_task_id": task.parent_task_id,
        },
    )
    return event_with_runtime_payload_authority(
        event,
        *(
            field_name
            for field_name in ("task_id", "task_session_id", "parent_task_id")
            if event.payload.get(field_name) is not None
        ),
    )


def _task_terminalization_idempotency_key(
    *,
    task_id: str,
    session_id: str,
    kind: TaskTerminalKind,
) -> str:
    material = canonical_durable_json_bytes(
        {
            "schema": "cayu.runtime-task-terminalization.v1",
            "task_id": task_id,
            "session_id": session_id,
            "kind": kind.value,
        },
        "runtime_task_terminalization",
    )
    return f"runtime-task-terminal:v1:{hashlib.sha256(material).hexdigest()}"


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


def _raise_primary_with_secondary_failure(
    primary: BaseException,
    secondary: BaseException,
    *,
    group_message: str,
) -> NoReturn:
    existing_cause = exception_cause(primary)
    combined_cause: BaseException = secondary
    if existing_cause is not None:
        combined_cause = BaseExceptionGroup(
            group_message,
            [existing_cause, secondary],
        )
    if not set_exception_cause(primary, combined_cause):
        raise BaseExceptionGroup(
            group_message,
            [primary, secondary],
        )
    raise primary


def _failure_without_existing_exception_identities(
    error: BaseException,
    existing_ids: set[int],
) -> BaseException | None:
    """Remove already-owned failures while retaining non-overlapping subgroups."""

    pending: list[tuple[BaseException, bool]] = [(error, False)]
    children_by_group: dict[int, tuple[BaseException, ...]] = {}
    retained_by_identity: dict[int, BaseException | None] = {}
    while pending:
        candidate, expanded = pending.pop()
        candidate_id = id(candidate)
        if candidate_id in retained_by_identity:
            continue
        if candidate_id in existing_ids:
            retained_by_identity[candidate_id] = None
            continue
        if not isinstance(candidate, BaseExceptionGroup):
            retained_by_identity[candidate_id] = candidate
            continue
        if expanded:
            children = children_by_group.pop(candidate_id, ())
            retained_children = [
                retained
                for child in children
                if (retained := retained_by_identity.get(id(child))) is not None
            ]
            if not retained_children:
                retained_by_identity[candidate_id] = None
            elif len(retained_children) == len(children) and all(
                retained is child
                for retained, child in zip(retained_children, children, strict=True)
            ):
                retained_by_identity[candidate_id] = candidate
            else:
                retained_by_identity[candidate_id] = BaseExceptionGroup(
                    "Session interruption additional non-duplicate failures.",
                    retained_children,
                )
            continue
        children = exception_group_children(candidate)
        if children is None:
            retained_by_identity[candidate_id] = RuntimeError(
                "Session interruption received an unreadable exception group."
            )
            continue
        children_by_group[candidate_id] = children
        pending.append((candidate, True))
        pending.extend((child, False) for child in reversed(children))

    return retained_by_identity.get(id(error))


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


class SessionEngine:
    """Own durable session operations and one run's end-to-end orchestration."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        task_store: TaskStore | None,
        get_budget_policy: Callable[[], BudgetPolicy | None],
        event_writer: RuntimeEventWriter,
        environment_lifecycle: EnvironmentLifecycle,
        run_limit_controller: RunLimitController,
        session_control: SessionControl[SessionUsageTracker],
        model_step_executor: ModelStepExecutor,
        request_footprint: RequestFootprintConfig,
        tool_round_executor: ToolRoundExecutor,
        recovery_coordinator: RecoveryCoordinator,
        background_interruption_coordinator: BackgroundInterruptionCoordinator,
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
        runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
        loop_policies: tuple[LoopPolicy, ...],
        loop_policy_execution_profile_identities: tuple[
            ExecutionProfileBehaviorIdentity | None, ...
        ],
        hook_runtime: RuntimeHookRuntime,
        get_registered_agent: Callable[[str], runtime_records.RegisteredAgentState],
        get_registered_provider: Callable[[str | None], runtime_records.RegisteredProvider],
        route_registered_provider_for_model: Callable[
            [str], runtime_records.RegisteredProvider | None
        ],
        get_registered_environment: Callable[
            [str | None], runtime_records.RegisteredEnvironment | None
        ],
        get_registered_environment_for_session: Callable[
            [str | None], runtime_records.RegisteredEnvironment | None
        ],
        effective_retry_policy: Callable[[RetryPolicy | None], RetryPolicy],
        execution_profile_policy: ExecutionProfilePolicy | None,
        execution_profile_policy_identity: str | None,
        execution_profile_process_identity: str,
    ) -> None:
        self.session_store = session_store
        self.task_store = task_store
        self._get_budget_policy = get_budget_policy
        self._event_writer = event_writer
        self._environment_lifecycle = environment_lifecycle
        self._run_limit_controller = run_limit_controller
        self._session_control = session_control
        self._model_step_executor = model_step_executor
        self._request_footprint = copy_request_footprint_config(request_footprint)
        self._tool_round_executor = tool_round_executor
        self._recovery_coordinator = recovery_coordinator
        self._background_interruption_coordinator = background_interruption_coordinator
        self._secret_redactor = secret_redactor
        self._interaction_lifecycle_publication_authority = object()
        self._workflow_structured_output_handoff = WorkflowStructuredOutputHandoff()
        self._clock = clock
        self._runtime_hooks = runtime_hooks
        self._loop_policies = loop_policies
        if len(loop_policies) != len(loop_policy_execution_profile_identities):
            raise ValueError("Loop-policy execution-profile identities are inconsistent.")
        self._loop_policy_execution_profile_identities = loop_policy_execution_profile_identities
        self._hook_runtime = hook_runtime
        self._get_registered_agent = get_registered_agent

        def resolve_provider(
            name: str | None = None,
        ) -> runtime_records.RegisteredProvider:
            return get_registered_provider(name)

        self._get_registered_provider = resolve_provider

        def route_provider_for_model(*, model: str) -> runtime_records.RegisteredProvider | None:
            return route_registered_provider_for_model(model)

        self._route_registered_provider_for_model = route_provider_for_model
        self._get_registered_environment = get_registered_environment
        self._get_registered_environment_for_session = get_registered_environment_for_session
        self._effective_retry_policy = effective_retry_policy
        self._execution_profile_policy = execution_profile_policy
        self._execution_profile_policy_identity = execution_profile_policy_identity
        self._execution_profile_process_identity = require_clean_nonblank(
            execution_profile_process_identity,
            "execution_profile_process_identity",
        )
        self._invocation_loop_policy_identity_registry = (
            execution_profile_admission.ProcessLocalBehaviorIdentityRegistry()
        )
        self._detached_session_operation_tasks: set[asyncio.Task[Any]] = set()

    def _request_loop_policy_instance_identities(
        self,
        policies: tuple[LoopPolicy, ...],
    ) -> tuple[str, ...]:
        return tuple(
            self._invocation_loop_policy_identity_registry.identity_for(policy)
            for policy in policies
        )

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
        target = request.target or ModelTarget(
            provider_name=session.provider_name,
            model=session.model,
        )
        registered_agent = self._get_registered_agent(session.agent_name)
        registered_provider = self._get_registered_provider(target.provider_name)
        candidate = _execution_profile_identity(
            registered_agent=registered_agent,
            provider_name=registered_provider.name,
            registered_provider=registered_provider,
            model=target.model,
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
                self._request_loop_policy_instance_identities(request.loop_policies)
            ),
            budget_policy=self._get_budget_policy(),
            request_budget_limits=request.budget_limits,
            causal_budget_id=session.causal_budget_id,
            structured_output=request.structured_output,
            thinking=request.thinking,
            max_steps=request.max_steps,
            limits=request.limits,
            retry_policy=self._effective_retry_policy(request.retry_policy),
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

    async def validate_execution_profile_continuation(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        request_loop_policies: tuple[LoopPolicy, ...] | None = None,
        budget_policy: BudgetPolicy | None = None,
        request_budget_limits: tuple[BudgetLimit, ...] = (),
        structured_output: StructuredOutputSpec | None = None,
        thinking: ThinkingConfig | None = None,
        max_steps: int = 16,
        limits: RunLimits | None = None,
        retry_policy: RetryPolicy | None = None,
        invocation_semantics_available: bool = False,
        frozen_candidate_profile: ExecutionProfileIdentity | None = None,
        require_open_interaction: bool = True,
        additional_profile_fingerprints: tuple[str, ...] = (),
    ) -> ActiveInvocationExecutionProfile:
        """Resolve a recovery continuation against its durable invocation profile."""

        active_model_completion = await self.session_store.load_active_model_completion_stage(
            session.id
        )
        model_completion_context = (
            None
            if active_model_completion is None
            else model_completion_recovery_context_from_stage(active_model_completion.stage)
        )
        provider_options, provider_options_process_local = _execution_profile_provider_options(
            registered_agent.spec.provider_options,
            provider=registered_provider.provider,
            model=session.model,
            process_identity=self._execution_profile_process_identity,
        )
        effective_thinking = thinking if thinking is not None else registered_agent.spec.thinking
        app_limit_ids = tuple(
            limit.budget_limit_id
            for limit in budget_limits_for_session(
                policy=budget_policy,
                agent_name=registered_agent.spec.name,
                causal_budget_id=session.causal_budget_id,
            )
        )
        request_limit_ids: tuple[str, ...] = ()
        if invocation_semantics_available:
            request_limit_ids = tuple(
                limit.budget_limit_id
                for limit in request_budget_limits_for_session(
                    limits=request_budget_limits,
                    agent_name=registered_agent.spec.name,
                    causal_budget_id=session.causal_budget_id,
                )
            )
        plan = execution_profile_admission.prepare_execution_profile_continuation(
            session=session,
            checkpoint=checkpoint,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            runtime_version=_runtime_version(),
            redactor=self._secret_redactor,
            process_identity=self._execution_profile_process_identity,
            registered_environment=self._get_registered_environment_for_session(
                session.environment_name
            ),
            runtime_hooks=self._runtime_hooks,
            loop_policies=self._loop_policies,
            loop_policy_identities=self._loop_policy_execution_profile_identities,
            invocation_loop_policies=request_loop_policies,
            invocation_loop_policy_identities=(
                ()
                if request_loop_policies is None or frozen_candidate_profile is not None
                else tuple(
                    copy_secret_free_execution_profile_behavior_identity(
                        policy.execution_profile_identity,
                        redactor=self._secret_redactor,
                        field_name=(f"request.loop_policies[{index}].execution_profile_identity"),
                    )
                    for index, policy in enumerate(request_loop_policies)
                )
            ),
            invocation_loop_policy_instance_identities=(
                ()
                if request_loop_policies is None or frozen_candidate_profile is not None
                else self._request_loop_policy_instance_identities(request_loop_policies)
            ),
            additional_profile_fingerprints=(
                *additional_profile_fingerprints,
                *(
                    ()
                    if active_model_completion is None
                    else (
                        None
                        if model_completion_context is None
                        else model_completion_context.execution_profile_fingerprint,
                    )
                ),
            ),
            frozen_candidate_profile=frozen_candidate_profile,
            provider_options=provider_options,
            provider_options_process_local=provider_options_process_local,
            thinking=(
                None if effective_thinking is None else thinking_config_payload(effective_thinking)
            ),
            app_budget_limit_ids=app_limit_ids,
            request_budget_limit_ids=request_limit_ids,
            structured_output=_execution_profile_structured_output(structured_output),
            finalization={
                "kind": "cayu:model-finalization:v1",
                "max_steps": max_steps,
                "limits": copy_run_limits(limits).model_dump(mode="json"),
                "retry_policy": copy_retry_policy(retry_policy).model_dump(mode="json"),
            },
            invocation_semantics_available=invocation_semantics_available,
        )
        snapshot = plan.snapshot
        candidate = plan.candidate_profile
        changed = plan.changed_component_classes
        if not changed:
            if require_open_interaction:
                latest_interactions = await self.session_store.query_events(
                    EventQuery(
                        session_id=session.id,
                        event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                        order_by=EventOrder.SEQUENCE_DESC,
                        limit=1,
                    )
                )
                if not latest_interactions or (
                    latest_interactions[0].event.type in INTERACTION_TERMINAL_EVENT_TYPES
                ):
                    raise RuntimeError(
                        "Active invocation execution profile has no authoritative open interaction."
                    )
                open_interaction_id = latest_interactions[0].event.interaction_id
                if open_interaction_id is None:
                    raise RuntimeError("Interaction lifecycle event has no interaction identity.")
                if snapshot.interaction_id != open_interaction_id:
                    raise RuntimeError(
                        "Active invocation execution profile belongs to another interaction."
                    )
            if frozen_candidate_profile is not None:
                return snapshot.model_copy(update={"profile": frozen_candidate_profile})
            return snapshot

        policy_identity = "cayu:active-invocation-profile:v1"
        decision = _execution_profile_decision_event(
            session=session,
            expected_profile=snapshot.profile,
            candidate_profile=candidate,
            changed_component_classes=changed,
            kind=ExecutionProfileDecisionKind.REJECTED,
            policy_identity=policy_identity,
            policy_reason="Active invocation profiles require exact recovery reuse.",
            authority_decision=ExecutionProfileAuthorityDecision.NOT_REQUIRED,
            intent=None,
            adoption_request_fingerprint=None,
            fallback_actor=None,
            fallback_reason="The active invocation profile changed before continuation.",
            clock=self._clock,
        )
        rejection = await self.session_store.reject_active_invocation_execution_profile(
            session.id,
            expected_statuses={session.status},
            expected_run_epoch=session.run_epoch,
            expected_active_invocation_profile=snapshot,
            candidate_profile=candidate,
            event=decision.event,
            decision=decision,
        )
        if not rejection.replayed:
            await self._event_writer.fan_out_persisted([rejection.event])
        raise ExecutionProfileMismatchError(
            session_id=session.id,
            expected_profile_fingerprint=snapshot.profile.fingerprint,
            candidate_profile_fingerprint=candidate.fingerprint,
            changed_component_classes=changed,
        )

    async def _classify_execution_profile(
        self,
        *,
        session: Session,
        expected_profile: ExecutionProfileIdentity,
        candidate_profile: ExecutionProfileIdentity,
        changed_component_classes: tuple[ExecutionProfileComponentClass, ...],
        intent: ExecutionProfileAdoptionIntent | None,
        adoption_request_fingerprint: str | None,
        target_changed: bool,
        target_provider_name: str,
        target_model: str,
        built_in_model_target_transition: bool = False,
        decision_session: Session | None = None,
        source_provider_name: str | None = None,
        source_model: str | None = None,
        force_authority_review: bool = False,
    ) -> ExecutionProfileDecision:
        event_session = session if decision_session is None else decision_session
        if not changed_component_classes and not force_authority_review:
            return _execution_profile_decision_event(
                session=event_session,
                expected_profile=expected_profile,
                candidate_profile=candidate_profile,
                changed_component_classes=changed_component_classes,
                kind=ExecutionProfileDecisionKind.EXACT_REUSE,
                policy_identity=_EXACT_PROFILE_POLICY_ID,
                policy_reason="The execution profile is exactly equal.",
                authority_decision=ExecutionProfileAuthorityDecision.NOT_REQUIRED,
                intent=intent,
                adoption_request_fingerprint=adoption_request_fingerprint,
                fallback_actor=None,
                fallback_reason="The execution profile is exactly equal.",
                clock=self._clock,
            )

        changed_component_set = frozenset(changed_component_classes)
        model_target_only = (
            target_changed
            and built_in_model_target_transition
            and ExecutionProfileComponentClass.PROVIDER_TARGET in changed_component_set
            and changed_component_set <= _MODEL_TARGET_BUILT_IN_COMPONENTS
        )
        built_in_model_target_adoption = model_target_only and not force_authority_review
        fallback_actor = _model_target_profile_actor() if built_in_model_target_adoption else None
        fallback_reason = (
            "The caller explicitly requested a model-target transition."
            if built_in_model_target_adoption
            else "The execution profile changed without authorized adoption."
        )
        policy_identity = (
            _MODEL_TARGET_PROFILE_POLICY_ID
            if built_in_model_target_adoption
            else _DEFAULT_PROFILE_POLICY_ID
        )
        policy_reason = fallback_reason
        authority_decision = (
            ExecutionProfileAuthorityDecision.AUTHORIZED
            if built_in_model_target_adoption
            else ExecutionProfileAuthorityDecision.NOT_REQUIRED
        )
        action = (
            ExecutionProfilePolicyAction.ADOPT
            if built_in_model_target_adoption
            else ExecutionProfilePolicyAction.REJECT
        )

        policy = self._execution_profile_policy
        if policy is not None:
            stable_identity = self._execution_profile_policy_identity
            try:
                current_policy_identity = policy.identity
            except Exception as exc:
                raise ExecutionProfilePolicyError(
                    "The configured execution-profile policy identity could not be read."
                ) from exc
            if stable_identity is None or current_policy_identity != stable_identity:
                raise ExecutionProfilePolicyError(
                    "The configured execution-profile policy identity changed."
                )
            policy_request = ExecutionProfilePolicyRequest(
                session_id=event_session.id,
                expected_profile=expected_profile,
                candidate_profile=candidate_profile,
                changed_component_classes=changed_component_classes,
                intent=intent,
                authority_review_required=(
                    force_authority_review
                    or execution_profile_changes_authority(changed_component_classes)
                ),
                source_provider_name=(
                    session.provider_name if source_provider_name is None else source_provider_name
                ),
                source_model=session.model if source_model is None else source_model,
                target_provider_name=target_provider_name,
                target_model=target_model,
            )
            try:
                raw_result = await policy.decide(policy_request)
                result = copy_execution_profile_policy_result(raw_result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise ExecutionProfilePolicyError(
                    "The execution-profile policy failed before admission."
                ) from exc
            try:
                current_policy_identity = policy.identity
            except Exception as exc:
                raise ExecutionProfilePolicyError(
                    "The configured execution-profile policy identity could not be re-read."
                ) from exc
            if current_policy_identity != stable_identity:
                raise ExecutionProfilePolicyError(
                    "The configured execution-profile policy identity changed during review."
                )
            try:
                result = ExecutionProfilePolicyResult(
                    action=result.action,
                    reason=self._secret_redactor.redact_text(result.reason),
                    authority_decision=result.authority_decision,
                )
            except (TypeError, ValueError) as exc:
                raise ExecutionProfilePolicyError(
                    "The execution-profile policy result could not be made durable."
                ) from exc
            action = result.action
            policy_identity = stable_identity
            policy_reason = result.reason
            authority_decision = result.authority_decision

        broadens_authority = force_authority_review or execution_profile_changes_authority(
            changed_component_classes
        )
        if action is ExecutionProfilePolicyAction.COMPATIBLE_REUSE:
            changes_persistent_target = (
                ExecutionProfileComponentClass.PROVIDER_TARGET in changed_component_classes
            )
            kind = (
                ExecutionProfileDecisionKind.REJECTED
                if (
                    authority_decision is ExecutionProfileAuthorityDecision.DENIED
                    or broadens_authority
                    or changes_persistent_target
                )
                else ExecutionProfileDecisionKind.COMPATIBLE_REUSE
            )
            if (
                authority_decision is not ExecutionProfileAuthorityDecision.DENIED
                and broadens_authority
            ):
                policy_reason = (
                    "A compatibility decision cannot authorize broader execution authority."
                )
            elif (
                authority_decision is not ExecutionProfileAuthorityDecision.DENIED
                and changes_persistent_target
            ):
                policy_reason = (
                    "A provider-target change must be adopted because it changes the "
                    "session's persistent execution target."
                )
        elif action is ExecutionProfilePolicyAction.ADOPT:
            has_explicit_adoption = intent is not None or model_target_only
            authority_is_sufficient = (
                authority_decision is not ExecutionProfileAuthorityDecision.DENIED
                and (
                    not broadens_authority
                    or authority_decision is ExecutionProfileAuthorityDecision.AUTHORIZED
                )
            )
            kind = (
                ExecutionProfileDecisionKind.ADOPTED
                if has_explicit_adoption and authority_is_sufficient
                else ExecutionProfileDecisionKind.REJECTED
            )
            if not has_explicit_adoption:
                policy_reason = "Explicit execution-profile adoption intent is required."
            elif (
                not authority_is_sufficient
                and authority_decision is not ExecutionProfileAuthorityDecision.DENIED
            ):
                policy_reason = "Authority-broadening adoption requires explicit authorization."
        elif action is ExecutionProfilePolicyAction.MIGRATION_REQUIRED:
            # Migration is a non-admitting classification, not an adoption.
            # A policy may require migration for ordinary drift even when the
            # caller did not ask Cayu to adopt the candidate profile.
            kind = ExecutionProfileDecisionKind.MIGRATION_REQUIRED
        else:
            kind = ExecutionProfileDecisionKind.REJECTED

        return _execution_profile_decision_event(
            session=event_session,
            expected_profile=expected_profile,
            candidate_profile=candidate_profile,
            changed_component_classes=changed_component_classes,
            kind=kind,
            policy_identity=policy_identity,
            policy_reason=policy_reason,
            authority_decision=authority_decision,
            intent=intent,
            adoption_request_fingerprint=adoption_request_fingerprint,
            fallback_actor=fallback_actor,
            fallback_reason=fallback_reason,
            clock=self._clock,
        )

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
            )
        return copy_event(event)

    def discard_workflow_structured_output(self, session_id: str) -> None:
        self._workflow_structured_output_handoff.discard(session_id)

    def _record_workflow_structured_output(self, session_id: str, output: Any) -> None:
        self._workflow_structured_output_handoff.record(session_id, output)

    def _track_detached_session_operation_task(self, task: asyncio.Task[Any]) -> None:
        """Retain and observe best-effort work that cannot block its caller."""

        if task in self._detached_session_operation_tasks:
            return
        self._detached_session_operation_tasks.add(task)

        def observe(completed: asyncio.Task[Any]) -> None:
            self._detached_session_operation_tasks.discard(completed)
            _consume_detached_session_operation_task(completed)

        task.add_done_callback(observe)

    async def _await_session_operation_store_task(
        self,
        task: asyncio.Task[Any],
        *,
        cancellation: asyncio.CancelledError | None = None,
        claim_expires_at: datetime | None = None,
    ) -> ShieldedTaskOutcome[Any]:
        """Bound a shielded store wait and retain any write that outlives it."""

        timeout_s = _SESSION_OPERATION_STORE_WAIT_TIMEOUT_SECONDS
        if claim_expires_at is not None:
            now = self._clock()
            if now.tzinfo is None or now.utcoffset() is None:
                raise ValueError("Session operation store wait clock must be timezone-aware.")
            timeout_s = min(
                timeout_s,
                max(0.0, (claim_expires_at - now).total_seconds()),
            )
        outcome = await await_shielded_task_outcome(
            task,
            cancellation=cancellation,
            timeout_s=timeout_s,
        )
        if outcome.timed_out:
            task.cancel()
            self._track_detached_session_operation_task(task)
        return outcome

    async def _fan_out_reconciled_session_operation_events(
        self,
        events: list[Event],
        *,
        cancellation: asyncio.CancelledError | None = None,
    ) -> tuple[BaseException | None, asyncio.CancelledError | None]:
        """Retry durable side effects without letting an unavailable sink hang cleanup."""

        if not events:
            return None, cancellation

        async def acknowledge_and_fan_out() -> list[Event]:
            await self._run_limit_controller.acknowledge_budget_settlement_events(events)
            return await self._event_writer.fan_out_persisted(events)

        fan_out_task = asyncio.create_task(acknowledge_and_fan_out())
        outcome = await self._await_session_operation_store_task(
            fan_out_task,
            cancellation=cancellation,
        )
        if outcome.timed_out:
            return (
                TimeoutError(
                    "Session operation event side-effect delivery exceeded "
                    f"{_SESSION_OPERATION_STORE_WAIT_TIMEOUT_SECONDS:g} seconds."
                ),
                outcome.cancellation,
            )
        error = outcome.error
        if isinstance(error, asyncio.CancelledError):
            error = unexpected_child_cancellation_error(
                error,
                operation="Session operation event side-effect delivery",
            )
        return error, outcome.cancellation

    async def drain_background_interruptions(self, *, timeout_s: float = 10.0) -> bool:
        """Wait for accepted background interruption cascades to finish.

        Returns ``False`` when the bounded wait expires. In-memory coordinators
        and workers are then cancelled; their durable parent markers remain for
        the next process to recover.
        """
        return await self._background_interruption_coordinator.drain(timeout_s=timeout_s)

    async def resume_pending_interruption_cascades(
        self,
        *,
        interrupting_inactive_before: datetime | None = None,
    ) -> int:
        """Resume durable descendant interruption work left by an earlier process.

        Both ``interrupting`` and ``interrupted`` parents are inspected. An
        ``interrupting`` parent is finalized only when
        ``interrupting_inactive_before`` is supplied and the store can fence that
        inactive run. Work remains checkpointed until traversal succeeds, so
        another restart can retry it safely. Returns the number of roots scheduled.
        """

        scheduled = 0
        admitted_parent_ids: set[str] = set()
        for status in (SessionStatus.INTERRUPTING, SessionStatus.INTERRUPTED):
            if status == SessionStatus.INTERRUPTING and interrupting_inactive_before is None:
                continue
            cursor: str | None = None
            while True:
                result = await self.session_store.list_sessions_with_pending_interruption_cascade(
                    SessionQuery(
                        status=status,
                        last_activity_before=(
                            interrupting_inactive_before
                            if status == SessionStatus.INTERRUPTING
                            else None
                        ),
                        limit=1000,
                        cursor=cursor,
                        order_by=SessionOrder.CREATED_AT_ASC,
                    )
                )
                for session in result.sessions:
                    if session.id in admitted_parent_ids:
                        continue
                    try:
                        marker = await self._load_pending_interruption_cascade(session.id)
                    except (TypeError, ValueError) as exc:
                        logger.warning(
                            "Could not resume invalid interruption cascade checkpoint for %s: %s",
                            session.id,
                            exc,
                        )
                        continue
                    if marker is None:
                        continue
                    already_scheduled = self._background_interruption_coordinator.is_admitted(
                        session.id
                    )
                    if session.status == SessionStatus.INTERRUPTING:
                        if interrupting_inactive_before is None:
                            continue
                        with suppress_interruption_cascade():
                            recovery = await self._recovery_coordinator.recover_incomplete_session(
                                IncompleteSessionRecoveryRequest(
                                    session_id=session.id,
                                    inactive_before=interrupting_inactive_before,
                                    reason="interruption_cascade_startup_recovery",
                                    metadata={"source": "resume_pending_interruption_cascades"},
                                )
                            )
                        session = await self._require_session(session.id)
                        if session.status != SessionStatus.INTERRUPTED:
                            logger.warning(
                                "Could not finalize interruption cascade parent %s during "
                                "startup recovery: %s",
                                session.id,
                                recovery.message,
                            )
                            continue
                    admitted_parent_ids.add(session.id)
                    self._schedule_background_interruption_cascade(
                        parent_session_id=session.id,
                        interrupt_payload=marker["interrupt_payload"],
                        create_if_missing=False,
                    )
                    if not already_scheduled:
                        scheduled += 1
                cursor = result.next_cursor
                if cursor is None:
                    break
        return scheduled

    async def interruption_cascade_status(self, session_id: str) -> str:
        """Return the public control-plane state of a session's durable cascade."""

        marker = await self.session_store.load_interruption_cascade_marker(session_id)
        if marker is None:
            return "none"
        if type(marker) is not dict:
            return "failed"
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
            return "failed"
        failure_recorded = marker.get("failure_recorded", False)
        if type(failure_recorded) is not bool:
            return "failed"
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
                    return "failed"
            elif claim_expires_at is not None:
                return "failed"
            created_at = _interruption_cascade_marker_datetime(marker, "created_at")
        except ValueError:
            return "failed"
        if self._background_interruption_coordinator.is_pending(session_id):
            return "pending"
        if claim_id is not None:
            if claim_expires_at is None:
                return "failed"
            return "pending" if claim_expires_at > self._clock() else "failed"
        if failure_recorded:
            return "failed"
        if created_at is None:
            return "failed"
        unclaimed_grace = timedelta(seconds=interruption_cascade_lease_seconds())
        return "pending" if created_at + unclaimed_grace > self._clock() else "failed"

    def _schedule_background_interruption_cascade(
        self,
        *,
        parent_session_id: str,
        interrupt_payload: dict[str, Any],
        create_if_missing: bool,
        retry_request: dict[str, Any] | None = None,
        allow_during_drain: bool = False,
    ) -> asyncio.Task[None] | None:
        return self._background_interruption_coordinator.schedule(
            parent_session_id=parent_session_id,
            interrupt_payload=interrupt_payload,
            create_if_missing=create_if_missing,
            retry_request=retry_request,
            allow_during_drain=allow_during_drain,
        )

    def _defer_background_interruption_cascade(
        self,
        *,
        parent_session_id: str,
        interrupt_payload: dict[str, Any],
        retry_at: datetime,
        drain_required: bool,
        retry_request: dict[str, Any] | None,
    ) -> None:
        self._background_interruption_coordinator.defer(
            parent_session_id=parent_session_id,
            interrupt_payload=interrupt_payload,
            retry_at=retry_at,
            drain_required=drain_required,
            retry_request=retry_request,
        )

    async def recover_incomplete_session(
        self,
        request: IncompleteSessionRecoveryRequest,
    ) -> IncompleteSessionRecoveryResult:
        request = copy_incomplete_session_recovery_request(request)
        interaction_id = await self._activate_latest_open_interaction(request.session_id)
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
            result = await self._recovery_coordinator.recover_incomplete_session(request)
            if interaction_id is not None:
                _activate_session_interaction(request.session_id, interaction_id)
            return await self._reconcile_recovered_interaction(
                result,
                recovered_active_through=active_through,
            )
        finally:
            if interaction_id is not None:
                _deactivate_session_interaction(request.session_id)

    async def recover_incomplete_sessions(
        self,
        request: IncompleteSessionsRecoveryRequest,
    ) -> IncompleteSessionsRecoveryPage:
        request = copy_incomplete_sessions_recovery_request(request)
        active_through_by_session: dict[str, datetime] = {}

        async def activate_interaction(session_id: str) -> None:
            interaction_id = await self._activate_latest_open_interaction(session_id)
            if interaction_id is not None:
                active_through_by_session[session_id] = await self._latest_interaction_activity_at(
                    session_id,
                    interaction_id,
                )
                _set_session_interaction_recovered_active_through(
                    session_id,
                    active_through_by_session[session_id],
                )

        async def deactivate_interaction(session_id: str) -> None:
            _deactivate_session_interaction(session_id)

        async def reconcile_result(
            result: IncompleteSessionRecoveryResult,
        ) -> IncompleteSessionRecoveryResult:
            # Recovery cleanup can clear task-local interaction ownership. Restore
            # the still-open durable interaction before publishing its terminal
            # transition; the coordinator's after hook releases this ownership.
            await self._activate_latest_open_interaction(result.session_id)
            return await self._reconcile_recovered_interaction(
                result,
                recovered_active_through=active_through_by_session.get(result.session_id),
            )

        page = await self._recovery_coordinator.recover_incomplete_sessions(
            request,
            before_recovery=activate_interaction,
            after_recovery=deactivate_interaction,
            reconcile_result=reconcile_result,
        )
        return page

    async def _activate_latest_open_interaction(self, session_id: str) -> str | None:
        records = await self.session_store.query_events(
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
    ) -> tuple[Session, Event | None, bool]:
        """Publish terminal recovery evidence without mutable runtime registrations."""

        try:
            return await self._publish_interaction_transition(
                session=session,
                agent_name=session.agent_name,
                environment_name=session.environment_name,
                to_status=session.status,
                from_statuses={session.status},
                recovered_active_through=recovered_active_through,
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
                    committed = await self._recovery_coordinator._record_durable_interaction_transition_cancellation(
                        session=session,
                        transition=interaction_transition,
                        failures=tuple(interaction_transition_failures),
                        agent_name=session.agent_name,
                        environment_name=session.environment_name,
                    )
                    if committed:
                        return
                if transition_settled or _latest_session_invocation_interaction_is_settled(
                    session.id
                ):
                    return
                await self._publish_interaction_transition(
                    session=session,
                    agent_name=session.agent_name,
                    environment_name=session.environment_name,
                    to_status=session.status,
                    from_statuses={session.status},
                    recovered_active_through=recovered_active_through,
                )

            await _run_interaction_transition_cancellation_cleanup_steps(
                cancellation,
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
    ) -> IncompleteSessionRecoveryResult:
        if IncompleteSessionRecoveryAction.SKIPPED_UNREGISTERED_AGENT in result.actions:
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
        # Recovery already owns a terminal durable session. Interaction summary
        # publication needs only that session's durable authority; consulting the
        # mutable application registry here could silently switch runtime profiles
        # after recovery admission.
        _, interaction_event, _ = await self._publish_durable_recovery_interaction_transition(
            session=session,
            recovered_active_through=recovered_active_through,
        )
        if interaction_event is None:
            return result
        return result.model_copy(update={"events": (*result.events, interaction_event)})

    async def _emit_interaction_started(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
    ) -> Event:
        interaction_id = _current_session_interaction_id(session.id)
        if interaction_id is None:
            raise RuntimeError("Interaction identity is not active for the session.")
        return await self._event_writer.emit(
            self._interaction_started_event(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                interaction_id=interaction_id,
            )
        )

    def _interaction_started_event(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        interaction_id: str,
    ) -> Event:
        return self._interaction_started_event_from_identity(
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            interaction_id=interaction_id,
        )

    def _interaction_started_event_from_identity(
        self,
        *,
        session_id: str,
        agent_name: str,
        environment_name: str | None,
        interaction_id: str,
        event_id: str | None = None,
    ) -> Event:
        event_id = (
            str(uuid4()) if event_id is None else require_clean_nonblank(event_id, "event_id")
        )
        started_at = self._clock()
        evidence = InteractionSummaryEvidence(
            status=InteractionStatus.ACTIVE,
            start_event_id=event_id,
            started_at=started_at,
        )
        return event_with_runtime_payload_authority(
            event_with_runtime_envelope_authority(
                event_with_runtime_generated_id(
                    Event(
                        id=event_id,
                        type=EventType.INTERACTION_STARTED,
                        session_id=session_id,
                        interaction_id=interaction_id,
                        timestamp=started_at,
                        agent_name=agent_name,
                        environment_name=environment_name,
                        payload=evidence.model_dump(mode="json"),
                    )
                ),
                "session_id",
                "interaction_id",
            ),
            "start_event_id",
        )

    async def _emit_interaction_started_if_needed(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
    ) -> Event | None:
        interaction_id = _current_session_interaction_id(session.id)
        if interaction_id is None:
            raise RuntimeError("Interaction identity is not active for the session.")
        existing = await self.session_store.query_events(
            EventQuery(
                session_id=session.id,
                interaction_id=interaction_id,
                event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not existing:
            return await self._emit_interaction_started(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
            )
        latest = existing[0]
        if latest.event.type in INTERACTION_TERMINAL_EVENT_TYPES:
            raise RuntimeError(f"Interaction is already terminal: {interaction_id}")
        if latest.event.type in {EventType.INTERACTION_STARTED, EventType.INTERACTION_RESUMED}:
            return None
        return await self._emit_interaction_state(
            session=session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            event_type=EventType.INTERACTION_RESUMED,
            status=InteractionStatus.ACTIVE,
        )

    async def _prepare_interaction_state_event(
        self,
        *,
        session: Session,
        agent_name: str,
        environment_name: str | None,
        event_type: EventType,
        status: InteractionStatus,
        pending_action_kind: str | None = None,
        recovered_active_through: datetime | None = None,
        observed_at: datetime | None = None,
        event_id: str | None = None,
    ) -> Event | None:
        interaction_id = _current_session_interaction_id(session.id)
        if interaction_id is None:
            raise RuntimeError("Interaction identity is not active for the session.")
        start_records = await self.session_store.query_events(
            EventQuery(
                session_id=session.id,
                interaction_id=interaction_id,
                event_type=EventType.INTERACTION_STARTED,
                order_by=EventOrder.SEQUENCE_ASC,
                limit=1,
            )
        )
        if not start_records:
            raise RuntimeError(f"Interaction has no durable start: {interaction_id}")
        latest_records = await self.session_store.query_events(
            EventQuery(
                session_id=session.id,
                interaction_id=interaction_id,
                event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        latest = latest_records[0]
        if latest.event.type in INTERACTION_TERMINAL_EVENT_TYPES:
            return None
        start_record = start_records[0]
        start_evidence = InteractionSummaryEvidence.model_validate(start_record.event.payload)
        prior_evidence = InteractionSummaryEvidence.model_validate(latest.event.payload)
        if (
            event_type == EventType.INTERACTION_PAUSED
            and latest.event.type == EventType.INTERACTION_PAUSED
            and prior_evidence.pending_action_kind == pending_action_kind
        ):
            return None
        usage_pages: list[SessionUsageSummary] = []
        after_sequence: int | None = None
        while True:
            usage_records = await self.session_store.query_events(
                EventQuery(
                    session_id=session.id,
                    interaction_id=interaction_id,
                    event_types=INTERACTION_SUMMARY_EVENT_TYPES,
                    after_sequence=after_sequence,
                    order_by=EventOrder.SEQUENCE_ASC,
                    limit=5000,
                )
            )
            usage_pages.append(
                interaction_usage_summary(
                    session.id,
                    [record.event for record in usage_records],
                )
            )
            if len(usage_records) < 5000:
                break
            after_sequence = usage_records[-1].sequence
        usage_summary = combine_session_usage_summaries(session.id, usage_pages)

        async def transcript_range(role: MessageRole) -> tuple[int | None, int | None]:
            first_page = await self.session_store.query_transcript(
                TranscriptQuery(
                    session_id=session.id,
                    interaction_id=interaction_id,
                    role=role,
                    limit=1,
                )
            )
            if not first_page.records:
                return None, None
            first = first_page.records[0].index
            if first_page.total_records == 1:
                return first, first
            last_page = await self.session_store.query_transcript(
                TranscriptQuery(
                    session_id=session.id,
                    interaction_id=interaction_id,
                    role=role,
                    offset=first_page.total_records - 1,
                    limit=1,
                )
            )
            if not last_page.records:
                raise RuntimeError("Interaction transcript changed during summary projection.")
            return first, last_page.records[0].index

        source_start, source_end = await transcript_range(MessageRole.USER)
        result_start, result_end = await transcript_range(MessageRole.ASSISTANT)
        segment_started_at = _current_session_interaction_started_at(session.id)
        observed_at = self._clock() if observed_at is None else observed_at
        if recovered_active_through is None:
            recovered_active_through = _current_session_interaction_recovered_active_through(
                session.id
            )
        if recovered_active_through is not None:
            if (
                recovered_active_through.tzinfo is None
                or recovered_active_through.utcoffset() is None
            ):
                raise ValueError("recovered_active_through must be timezone-aware.")
            current_segment_ms = (
                max(
                    0,
                    int(
                        (
                            min(recovered_active_through, observed_at) - latest.event.timestamp
                        ).total_seconds()
                        * 1000
                    ),
                )
                if prior_evidence.status is InteractionStatus.ACTIVE
                else 0
            )
        else:
            current_segment_ms = (
                0
                if segment_started_at is None or event_type == EventType.INTERACTION_RESUMED
                else max(0, int((time.monotonic() - segment_started_at) * 1000))
            )
        terminal = event_type in INTERACTION_TERMINAL_EVENT_TYPES
        try:
            evidence = InteractionSummaryEvidence(
                status=status,
                start_event_id=start_record.event.id,
                start_event_sequence=start_record.sequence,
                source_transcript_start=source_start,
                source_transcript_end=source_end,
                result_transcript_start=result_start,
                result_transcript_end=result_end,
                started_at=start_evidence.started_at,
                completed_at=observed_at if terminal else None,
                active_duration_ms=prior_evidence.active_duration_ms + current_segment_ms,
                wall_duration_ms=(
                    max(
                        0,
                        int((observed_at - start_evidence.started_at).total_seconds() * 1000),
                    )
                    if terminal
                    else None
                ),
                model_step_count=usage_summary.model_steps,
                tool_call_count=usage_summary.tool_calls,
                token_usage=usage_summary.usage,
                provider_names=usage_summary.provider_names,
                models=usage_summary.models,
                pending_action_kind=pending_action_kind,
            )
        except ValidationError as exc:
            if terminal and observed_at < start_evidence.started_at:
                raise _runtime_interaction_lifecycle_publication_rejected(
                    session_id=session.id,
                    interaction_id=interaction_id,
                    runtime_authority=self._interaction_lifecycle_publication_authority,
                ) from exc
            raise
        event_payload = evidence.model_dump(mode="json")
        event = event_with_runtime_payload_authority(
            Event(
                type=event_type,
                session_id=session.id,
                interaction_id=interaction_id,
                timestamp=observed_at,
                agent_name=agent_name,
                environment_name=environment_name,
                payload=event_payload,
            )
            if event_id is None
            else Event(
                id=event_id,
                type=event_type,
                session_id=session.id,
                interaction_id=interaction_id,
                timestamp=observed_at,
                agent_name=agent_name,
                environment_name=environment_name,
                payload=event_payload,
            ),
            "start_event_id",
        )
        if event_type == EventType.INTERACTION_RESUMED:
            _clear_session_interaction_recovered_active_through(session.id)
        return event

    async def _failure_interaction_observed_at(
        self,
        session: Session,
        *,
        observed_at: datetime | None = None,
    ) -> datetime | None:
        interaction_id = _current_session_interaction_id(session.id)
        if interaction_id is None:
            return None
        records = await self.session_store.query_events(
            EventQuery(
                session_id=session.id,
                interaction_id=interaction_id,
                event_types=(EventType.INTERACTION_STARTED,),
                order_by=EventOrder.SEQUENCE_ASC,
                limit=1,
            )
        )
        if not records:
            return None
        start_evidence = InteractionSummaryEvidence.model_validate(records[0].event.payload)
        if observed_at is None:
            observed_at = self._clock()
        if observed_at < start_evidence.started_at:
            raise _runtime_interaction_lifecycle_publication_rejected(
                session_id=session.id,
                interaction_id=interaction_id,
                runtime_authority=self._interaction_lifecycle_publication_authority,
            )
        return observed_at

    async def _emit_interaction_state(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        event_type: EventType,
        status: InteractionStatus,
        pending_action_kind: str | None = None,
        recovered_active_through: datetime | None = None,
    ) -> Event | None:
        event = await self._prepare_interaction_state_event(
            session=session,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            event_type=event_type,
            status=status,
            pending_action_kind=pending_action_kind,
            recovered_active_through=recovered_active_through,
        )
        if event is None:
            return None
        return await self._event_writer.emit(event)

    async def _publish_interaction_transition(
        self,
        *,
        session: Session,
        agent_name: str,
        environment_name: str | None,
        to_status: SessionStatus,
        only_if_no_queued_messages: bool = False,
        from_statuses: set[SessionStatus] | None = None,
        recovered_active_through: datetime | None = None,
        observed_at: datetime | None = None,
        event_id: str | None = None,
    ) -> tuple[Session, Event | None, bool]:
        interaction_id = _current_session_interaction_id(session.id)
        if interaction_id is None:
            transitioned = (
                await self.session_store.transition_status_if_no_queued_messages(
                    session.id,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=to_status,
                )
                if only_if_no_queued_messages
                else await self.session_store.update_status(session.id, to_status)
            )
            return transitioned, None, True

        pending_action_kind: str | None = None
        if to_status is not SessionStatus.COMPLETED:
            checkpoint = await self.session_store.load_checkpoint(session.id)
            if approval_support.pending_approval_from_checkpoint(checkpoint) is not None:
                pending_action_kind = "tool_approval"
            elif pending_user_input_from_checkpoint(checkpoint) is not None:
                pending_action_kind = "user_input"
            elif tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint) is not None:
                pending_action_kind = "tool_recovery"
        if to_status is SessionStatus.COMPLETED:
            event_type = EventType.INTERACTION_COMPLETED
            interaction_status = InteractionStatus.COMPLETED
        elif pending_action_kind is not None:
            event_type = EventType.INTERACTION_PAUSED
            interaction_status = InteractionStatus.PAUSED
        elif to_status is SessionStatus.FAILED:
            event_type = EventType.INTERACTION_FAILED
            interaction_status = InteractionStatus.FAILED
        elif to_status is SessionStatus.INTERRUPTED:
            event_type = EventType.INTERACTION_INTERRUPTED
            interaction_status = InteractionStatus.INTERRUPTED
        else:
            raise ValueError(f"Unsupported interaction terminal session status: {to_status}")

        event = await self._prepare_interaction_state_event(
            session=session,
            agent_name=agent_name,
            environment_name=environment_name,
            event_type=event_type,
            status=interaction_status,
            pending_action_kind=pending_action_kind,
            recovered_active_through=recovered_active_through,
            observed_at=observed_at,
            event_id=event_id,
        )
        if event is None:
            loaded = await self.session_store.load(session.id)
            if loaded is None:
                raise KeyError(f"Session not found: {session.id}") from None
            if loaded.status is to_status:
                return loaded, None, True
            if (
                from_statuses is None
                and to_status is SessionStatus.FAILED
                and loaded.status in {SessionStatus.COMPLETED, SessionStatus.INTERRUPTED}
            ):
                # Environment finalization or a terminal hook can fail after the
                # response interaction and its original session outcome are
                # already durable. Keep that interaction terminal evidence
                # truthful while recording the later session-scoped failure.
                transitioned = await self.session_store.transition_status(
                    session.id,
                    from_statuses={loaded.status},
                    to_status=SessionStatus.FAILED,
                )
                return transitioned, None, True
            allowed_statuses = (
                {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}
                if from_statuses is None
                else from_statuses
            )
            transitioned = (
                await self.session_store.transition_status_if_no_queued_messages(
                    session.id,
                    from_statuses=allowed_statuses,
                    to_status=to_status,
                )
                if only_if_no_queued_messages
                else await self.session_store.transition_status(
                    session.id,
                    from_statuses=allowed_statuses,
                    to_status=to_status,
                )
            )
            return transitioned, None, True

        replay_event = copy_event(event)
        replay_from_statuses = frozenset(
            {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}
            if from_statuses is None
            else from_statuses
        )
        replay_transition = InteractionTransitionSpec(
            event=replay_event,
            from_statuses=tuple(replay_from_statuses),
            to_status=to_status,
            only_if_no_queued_messages=only_if_no_queued_messages,
        )
        result: InteractionTransitionResult | None = None
        pending_cancellation: asyncio.CancelledError | None = None
        replay_failures: list[Exception] = []
        replay_deadline: float | None = None
        loop = asyncio.get_running_loop()
        for attempt in range(_INTERACTION_TRANSITION_REPLAY_MAX_ATTEMPTS):
            transition_task = asyncio.create_task(
                self.session_store.publish_interaction_transition(
                    session.id,
                    event=copy_event(replay_transition.event),
                    from_statuses=set(replay_transition.from_statuses),
                    to_status=replay_transition.to_status,
                    only_if_no_queued_messages=(replay_transition.only_if_no_queued_messages),
                )
            )
            outcome = await await_shielded_task_outcome(transition_task)
            attempt_failure = _normalize_interaction_transition_child_cancellation(outcome.error)
            attempt_result: InteractionTransitionResult | None = None
            if attempt_failure is None and outcome.result is not None:
                try:
                    attempt_result = _validated_interaction_transition_result(
                        outcome.result,
                        transition=replay_transition,
                    )
                except (RuntimeError, TypeError) as result_failure:
                    attempt_failure = result_failure
            if outcome.cancellation is not None:
                if attempt_failure is not None:
                    cancellation_failures = [*replay_failures, attempt_failure]
                    _raise_interaction_transition_cancellation(
                        outcome.cancellation,
                        cancellation_failures,
                        failure_diagnostics=_interaction_transition_failure_diagnostics(
                            tuple(cancellation_failures),
                            redactor=self._secret_redactor,
                        ),
                        transition=replay_transition,
                    )
                if attempt_result is None:
                    cancellation_failures = [
                        *replay_failures,
                        RuntimeError("Interaction transition publication returned no result."),
                    ]
                    _raise_interaction_transition_cancellation(
                        outcome.cancellation,
                        cancellation_failures,
                        failure_diagnostics=_interaction_transition_failure_diagnostics(
                            tuple(cancellation_failures),
                            redactor=self._secret_redactor,
                        ),
                        transition=replay_transition,
                    )
                result = attempt_result
                pending_cancellation = outcome.cancellation
                break
            if attempt_failure is None:
                if attempt_result is None:
                    attempt_failure = RuntimeError(
                        "Interaction transition publication returned no result."
                    )
                else:
                    result = attempt_result
                    break
            if not isinstance(attempt_failure, Exception):
                if replay_failures:
                    raise BaseExceptionGroup(
                        "Interaction transition replay and fatal store failure.",
                        _unique_exception_identities([*replay_failures, attempt_failure]),
                    ) from None
                raise attempt_failure
            prior_failures = list(replay_failures)
            replay_failures.append(attempt_failure)
            if _interaction_transition_failure_contains_run_fence(attempt_failure):
                _mark_interaction_transition_run_fence(attempt_failure)
                if prior_failures:
                    _raise_primary_with_secondary_failure(
                        attempt_failure,
                        _interaction_transition_replay_failure(prior_failures),
                        group_message=("Interaction transition replay failure and run-fence loss."),
                    )
                raise attempt_failure
            terminal_failure = _interaction_transition_replay_failure(replay_failures)
            if _interaction_transition_replay_is_authoritatively_rejected(attempt_failure):
                raise terminal_failure from None
            if attempt + 1 >= _INTERACTION_TRANSITION_REPLAY_MAX_ATTEMPTS:
                raise terminal_failure from None
            if replay_deadline is None:
                replay_deadline = loop.time() + _INTERACTION_TRANSITION_REPLAY_WINDOW_SECONDS
            if loop.time() >= replay_deadline:
                raise terminal_failure from None
            # Give cancellation and a competing run-fence owner a boundary
            # before replay. The completed store task above is positively
            # settled before ownership cleanup or another attempt may begin.
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError as cancellation:
                _raise_interaction_transition_cancellation(
                    cancellation,
                    replay_failures,
                    failure_diagnostics=_interaction_transition_failure_diagnostics(
                        tuple(replay_failures),
                        redactor=self._secret_redactor,
                    ),
                    transition=replay_transition,
                )
            if loop.time() >= replay_deadline:
                raise terminal_failure from None
        if result is None:  # pragma: no cover - the bounded loop returns or raises
            raise AssertionError("Interaction transition replay produced no result.")
        if result.event.interaction_id is None:
            raise RuntimeError("Interaction transition result lost its interaction identity.")
        _mark_session_interaction_settled(
            session.id,
            result.event.interaction_id,
        )
        if result.event.type in INTERACTION_TERMINAL_EVENT_TYPES:
            _close_session_interaction(session.id)
        if pending_cancellation is not None:
            # The atomic store publication already owns a durable side-effect
            # handoff. Redeliver cancellation at this settled boundary instead
            # of beginning extension-controlled budget or sink work.
            _raise_interaction_transition_cancellation(
                pending_cancellation,
                replay_failures,
                transition_settled=True,
                failure_diagnostics=_interaction_transition_failure_diagnostics(
                    tuple(replay_failures),
                    redactor=self._secret_redactor,
                ),
                transition=replay_transition,
            )
        await self._event_writer.fan_out_persisted([result.event])
        return result.session, result.event, result.status_changed

    async def _reconcile_sibling_interaction_transition_cancellation(
        self,
        cancellation: asyncio.CancelledError,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        execution_profile: ExecutionProfileIdentity | None,
        finalize_unsettled: bool,
    ) -> None:
        """Consume transition handoff evidence outside the main run loop."""

        if _is_interaction_transition_run_fence(cancellation):
            pop_exception_state(
                cancellation,
                _INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE,
            )
            pop_exception_state(
                cancellation,
                _INTERACTION_TRANSITION_RUN_FENCE_ATTRIBUTE,
            )
            return

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

        if not interaction_transition_failures and (
            transition_settled or _latest_session_invocation_interaction_is_settled(session.id)
        ):
            return
        if not interaction_transition_failures and not finalize_unsettled:
            return

        recovery_request = RecoveryAbandonedSessionRequest(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            interaction_transition_failures=tuple(interaction_transition_failures),
            interaction_transition=interaction_transition,
            execution_profile=execution_profile,
        )
        if interaction_transition_failures and interaction_transition is not None:
            committed_transition_recorded = False

            async def record_committed_transition_cancellation() -> None:
                nonlocal committed_transition_recorded
                committed_transition_recorded = await self._recovery_coordinator._record_committed_interaction_transition_cancellation(
                    recovery_request
                )

            diagnostic_failures = await _run_interaction_transition_cancellation_cleanup_steps(
                cancellation,
                steps=(
                    (
                        "committed sibling interaction-transition cancellation diagnostics",
                        record_committed_transition_cancellation,
                    ),
                ),
            )
            if committed_transition_recorded or diagnostic_failures:
                return
        if transition_settled or _latest_session_invocation_interaction_is_settled(session.id):
            return
        if not finalize_unsettled:
            return
        cancellation_cleanup_steps = (
            (
                "cancelled sibling interaction-transition finalization",
                lambda: self._recovery_coordinator.finalize_abandoned_session_run(recovery_request),
            ),
        )
        if interaction_transition_failures:
            await _run_interaction_transition_cancellation_cleanup_steps(
                cancellation,
                steps=cancellation_cleanup_steps,
            )
        else:
            await _run_recovery_cleanup_steps(
                authoritative_failure=cancellation,
                steps=cancellation_cleanup_steps,
            )

    async def _publish_sibling_interaction_transition(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        to_status: SessionStatus,
        only_if_no_queued_messages: bool = False,
        from_statuses: set[SessionStatus] | None = None,
        recovered_active_through: datetime | None = None,
        observed_at: datetime | None = None,
        event_id: str | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
        finalize_unsettled_cancellation: bool = True,
    ) -> tuple[Session, Event | None, bool]:
        """Publish from a caller not enclosed by ``_run_session`` cleanup."""

        try:
            return await self._publish_interaction_transition(
                session=session,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                to_status=to_status,
                only_if_no_queued_messages=only_if_no_queued_messages,
                from_statuses=from_statuses,
                recovered_active_through=recovered_active_through,
                observed_at=observed_at,
                event_id=event_id,
            )
        except asyncio.CancelledError as cancellation:
            await self._reconcile_sibling_interaction_transition_cancellation(
                cancellation,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                execution_profile=execution_profile,
                finalize_unsettled=finalize_unsettled_cancellation,
            )
            raise

    async def resume_interaction(
        self,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> Event | None:
        return await self._emit_interaction_state(
            session=session,
            registered_agent=registered_agent,
            environment_name=_environment_name(registered_environment),
            event_type=EventType.INTERACTION_RESUMED,
            status=InteractionStatus.ACTIVE,
        )

    async def fail_provider_operation_resolution(
        self,
        *,
        resolution_event: Event,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity,
        legacy_resolution_without_profile: bool = False,
    ) -> AsyncIterator[Event]:
        """Terminalize one explicit provider failure through normal lifecycle hooks."""

        if resolution_event.type is not EventType.PROVIDER_OPERATION_RESOLVED:
            raise TypeError("Provider-operation failure requires a resolution event.")
        if resolution_event.session_id != session.id:
            raise ValueError("Provider-operation resolution belongs to another session.")
        resolution_id = resolution_event.payload.get("resolution_id")
        recovery_reason = resolution_event.payload.get("recovery_reason")
        if type(resolution_id) is not str or type(recovery_reason) is not str:
            raise ValueError("Provider-operation resolution evidence is malformed.")
        resolution_has_profile = "execution_profile_fingerprint" in resolution_event.payload
        resolution_profile = resolution_event.payload.get("execution_profile_fingerprint")
        if resolution_profile != execution_profile.fingerprint and not (
            legacy_resolution_without_profile and not resolution_has_profile
        ):
            raise ValueError("Provider-operation resolution belongs to another execution profile.")
        try:
            model_attempt_identity = ModelAttemptIdentity.model_validate(
                {
                    "model_step_id": resolution_event.payload.get("model_step_id"),
                    "model_attempt_id": resolution_event.payload.get("model_attempt_id"),
                }
            )
        except ValidationError:
            raise ValueError("Provider-operation resolution lost model identity.") from None

        environment_name = _environment_name(registered_environment)
        common_payload = {
            key: resolution_event.payload[key]
            for key in (
                "provider",
                "model",
                "step",
                "attempt",
                "max_attempts",
                "model_step_id",
                "model_attempt_id",
                "source_run_epoch",
                "run_epoch",
                "stage_id",
                "resolution_id",
                "resolution_action",
                "recovery_reason",
                "duplicate_request_risk",
                "reason",
                "metadata",
                "resolved_by",
            )
        }
        model_error_id = provider_operation_resolution_outcome_event_id(
            resolution_id,
            "model_error",
        )
        model_error_records = await self.session_store.query_events(
            EventQuery(session_id=session.id, event_id=model_error_id, limit=2)
        )
        if model_error_records:
            if len(model_error_records) != 1:
                raise ProviderOperationEvidenceError(
                    "Provider-operation failure has duplicate model evidence."
                )
            validate_provider_operation_resolution_outcome_event(
                model_error_records[0].event,
                resolution_event=resolution_event,
                outcome="model_error",
                expected_execution_profile_fingerprint=execution_profile.fingerprint,
            )
        else:
            model_error = event_with_execution_profile_authority(
                _event_with_model_identity_authority(
                    Event(
                        id=model_error_id,
                        type=EventType.MODEL_ERROR,
                        session_id=session.id,
                        interaction_id=resolution_event.interaction_id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        timestamp=resolution_event.timestamp,
                        payload={
                            **common_payload,
                            "error": "Provider operation was explicitly failed after recovery.",
                            "error_type": "provider_operation_unavailable",
                            "stage": "provider_operation_recovery",
                        },
                    ),
                    model_attempt_identity,
                ),
                execution_profile,
            )
            yield await self._event_writer.emit(model_error)

        interaction_failed_id = provider_operation_resolution_outcome_event_id(
            resolution_id,
            "interaction_failed",
        )
        interaction_failed_records = await self.session_store.query_events(
            EventQuery(session_id=session.id, event_id=interaction_failed_id, limit=2)
        )
        transitioned_session = await self.session_store.load(session.id)
        if transitioned_session is None:
            raise KeyError(f"Session not found: {session.id}")
        if interaction_failed_records:
            if len(interaction_failed_records) != 1:
                raise ProviderOperationEvidenceError(
                    "Provider-operation failure has duplicate interaction evidence."
                )
            validate_provider_operation_resolution_outcome_event(
                interaction_failed_records[0].event,
                resolution_event=resolution_event,
                outcome="interaction_failed",
                expected_execution_profile_fingerprint=execution_profile.fingerprint,
            )
            if transitioned_session.status is not SessionStatus.FAILED:
                raise ProviderOperationEvidenceError(
                    "Provider-operation interaction failure conflicts with session status."
                )
        else:
            interaction_id = await self._activate_latest_open_interaction(session.id)
            if interaction_id is None:
                raise RuntimeError("Provider-operation failure has no open durable interaction.")
            observed_at = await self._failure_interaction_observed_at(
                transitioned_session,
                observed_at=resolution_event.timestamp,
            )
            (
                transitioned_session,
                interaction_failed_event,
                _,
            ) = await self._publish_sibling_interaction_transition(
                session=transitioned_session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                to_status=SessionStatus.FAILED,
                from_statuses={SessionStatus.INTERRUPTED},
                observed_at=observed_at,
                event_id=interaction_failed_id,
                execution_profile=execution_profile,
            )
            if interaction_failed_event is not None:
                yield interaction_failed_event

        session_failed_id = provider_operation_resolution_outcome_event_id(
            resolution_id,
            "session_failed",
        )
        terminal_records = await self.session_store.query_events(
            EventQuery(session_id=session.id, event_id=session_failed_id, limit=2)
        )
        if terminal_records:
            if len(terminal_records) != 1:
                raise ProviderOperationEvidenceError(
                    "Provider-operation failure has duplicate session evidence."
                )
            validate_provider_operation_resolution_outcome_event(
                terminal_records[0].event,
                resolution_event=resolution_event,
                outcome="session_failed",
                expected_execution_profile_fingerprint=execution_profile.fingerprint,
            )
            if transitioned_session.status is not SessionStatus.FAILED:
                raise ProviderOperationEvidenceError(
                    "Provider-operation terminal failure conflicts with session status."
                )
            _deactivate_session_interaction(session.id)
            return
        try:
            async for terminal_event in self._emit_terminal_event_with_hooks(
                event=event_with_execution_profile_authority(
                    Event(
                        id=session_failed_id,
                        type=EventType.SESSION_FAILED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        timestamp=resolution_event.timestamp,
                        payload={
                            **common_payload,
                            "failure_type": "provider_operation_unavailable",
                            "error": "Provider operation was explicitly failed after recovery.",
                        },
                    ),
                    execution_profile,
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                session=transitioned_session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
            ):
                yield terminal_event
        finally:
            _deactivate_session_interaction(session.id)

    async def _prepare_initial_run(self, request: RunRequest) -> _PreparedInitialRun:
        """Resolve one new-session request without creating or admitting a session."""

        request = session_request_boundary.prepare_run_request(
            request,
            redactor=self._secret_redactor,
        )
        if request.session_id is None:
            request = request.model_copy(update={"session_id": str(uuid4())})
        if request.session_id is None:
            raise AssertionError("Run request session identity was not assigned.")
        if request.task_id is not None and self.task_store is not None:
            source_task = await self.task_store.load_invocation_snapshot(request.task_id)
            if source_task is not None and source_task.session_id in {None, request.session_id}:
                request = run_request_with_task_invocation(
                    request,
                    source_task,
                )
        # Freeze application-budget authority before any await can expose the
        # prepared invocation. The same snapshot must govern both the durable
        # execution profile and the model dispatch that follows it.
        budget_policy = copy_budget_policy(self._get_budget_policy())
        registered_agent = self._get_registered_agent(request.agent_name)
        # An explicit target is exact. Otherwise the agent model and optional
        # provider pin feed the existing routing/default selection.
        if request.target is not None:
            model = request.target.model
            registered_provider = self._get_registered_provider(request.target.provider_name)
        else:
            model = registered_agent.spec.model
            if registered_agent.spec.provider_name is not None:
                registered_provider = self._get_registered_provider(
                    registered_agent.spec.provider_name
                )
            else:
                registered_provider = (
                    self._route_registered_provider_for_model(model=model)
                    or self._get_registered_provider()
                )
        for field_name, value in (
            ("agent_name", registered_agent.spec.name),
            ("provider_name", registered_provider.name),
            ("model", model),
        ):
            session_request_boundary.require_secret_free_session_authority(
                value,
                field_name=field_name,
                redactor=self._secret_redactor,
            )
        registered_environment = self._get_registered_environment(request.environment_name)
        if registered_environment is not None:
            session_request_boundary.require_secret_free_session_authority(
                registered_environment.spec.name,
                field_name="environment_name",
                redactor=self._secret_redactor,
            )
        if request.environment_name is None and registered_environment is not None:
            request = _with_environment_name(request, registered_environment.spec.name)
        workspace_instructions = await self._environment_lifecycle.load_workspace_instructions(
            registered_environment,
        )
        (
            rendered_system_prompt,
            prompt_contributions,
        ) = render_initial_system_prompt_with_contributions(
            agent_system_prompt=registered_agent.spec.system_prompt,
            workspace_instructions=workspace_instructions,
        )
        execution_profile = _execution_profile_identity(
            registered_agent=registered_agent,
            provider_name=registered_provider.name,
            registered_provider=registered_provider,
            model=model,
            durable_system_prompt=rendered_system_prompt,
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
                self._request_loop_policy_instance_identities(request.loop_policies)
            ),
            budget_policy=budget_policy,
            request_budget_limits=request.budget_limits,
            causal_budget_id=(request.causal_budget_id or request.task_id or request.session_id),
            structured_output=request.structured_output,
            thinking=request.thinking,
            max_steps=request.max_steps,
            limits=request.limits,
            retry_policy=self._effective_retry_policy(request.retry_policy),
        )
        unavailable_profile_components = unavailable_execution_profile_components(execution_profile)
        if unavailable_profile_components:
            names = ", ".join(component.value for component in unavailable_profile_components)
            raise RuntimeError(
                "Invocation execution profile has unavailable required components: " + names
            )
        # Native schema validation is provider-owned code, but it is not an
        # execution attempt. Freeze the invocation profile first while retaining
        # the established no-session-on-invalid-schema entry-point contract.
        _require_native_structured_output_support(
            request.structured_output, registered_provider=registered_provider
        )
        session_identity = _session_identity(
            provider_name=registered_provider.name,
            model=model,
            execution_profile=execution_profile,
        )
        return _PreparedInitialRun(
            request=request,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            workspace_instructions=workspace_instructions,
            rendered_system_prompt=rendered_system_prompt,
            prompt_contributions=tuple(prompt_contributions),
            execution_profile=execution_profile,
            budget_policy=budget_policy,
            session_identity=session_identity,
        )

    async def run(self, request: RunRequest) -> AsyncGenerator[Event, None]:
        prepared = await self._prepare_initial_run(request)
        request = prepared.request
        registered_agent = prepared.registered_agent
        registered_provider = prepared.registered_provider
        registered_environment = prepared.registered_environment
        workspace_instructions = prepared.workspace_instructions
        rendered_system_prompt = prepared.rendered_system_prompt
        prompt_contributions = list(prepared.prompt_contributions)
        execution_profile = prepared.execution_profile
        budget_policy = prepared.budget_policy
        session_identity = prepared.session_identity
        # ``prepared`` also retains the registered provider, whose repr may contain
        # live credentials. Keep the deliberately scoped provider local below as the
        # sole remaining reference so the existing cancellation cleanup can drop it
        # before a traceback escapes this frame.
        del prepared
        session_id = request.session_id
        if session_id is None:
            raise AssertionError("Run request session identity was not assigned.")
        prepared_session_authority = runtime_prepared_session_authority(request)
        interaction_id = (
            str(uuid4())
            if prepared_session_authority is None
            else prepared_session_authority.interaction_id
        )
        interaction_started_event = self._interaction_started_event_from_identity(
            session_id=session_id,
            agent_name=registered_agent.spec.name,
            environment_name=_environment_name(registered_environment),
            interaction_id=interaction_id,
            event_id=(
                None
                if prepared_session_authority is None
                else prepared_session_authority.interaction_started_event_id
            ),
        )
        if prepared_session_authority is None:
            bind_runtime_session_create_claim(
                request,
                identity=session_identity,
                interaction_started_event=interaction_started_event,
            )

            def freeze_initial_invocation_profile(
                current_session: Session,
                checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any]:
                return checkpoint_with_active_invocation_execution_profile(
                    checkpoint,
                    session_id=current_session.id,
                    interaction_id=interaction_id,
                    run_epoch=current_session.run_epoch,
                    profile=execution_profile,
                )

            session = await self.session_store.create(
                request,
                identity=session_identity,
                interaction_started_event=interaction_started_event,
                interaction_source_messages=request.messages,
                checkpoint_transform=freeze_initial_invocation_profile,
            )
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

            session = await self.session_store.admit_session_invocation(
                session_id,
                admission=SessionInvocationAdmission(
                    from_statuses=frozenset({SessionStatus.PENDING}),
                    checkpoint_transform=validate_prepared_submission,
                    execution_profile=execution_profile,
                    interaction_source_messages=tuple(request.messages),
                    interaction_started_event=interaction_started_event,
                    defer_interaction_source=True,
                    allow_pending_initial_interaction=True,
                ),
            )
        _activate_session_interaction(session.id, interaction_id)
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
            await self._event_writer.fan_out_persisted([interaction_started_event])
            yield interaction_started_event
            factory_started_event = await self._environment_lifecycle.emit_factory_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
            )
            if factory_started_event is not None:
                yield factory_started_event
                async for queued_event in self._session_control.drain_out_of_band_events(
                    session.id
                ):
                    yield queued_event
                if request.task_id is not None:
                    task = await self._start_task(
                        task_id=request.task_id,
                        session=session,
                        worker_id=request.task_worker_id,
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
            )
            registered_environment = resolution.registered_environment
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
            messages = transcript_helpers.initial_messages(
                system_prompt=rendered_system_prompt,
                request_messages=request.messages,
            )
            await self.session_store.replace_initial_transcript_messages(
                session.id,
                request.messages,
                messages,
            )
            release_before_run = False
        except asyncio.CancelledError as cancellation:
            if await self._session_control.interrupt_requested(session.id):
                if interaction_started:
                    await self.session_store.materialize_deferred_interaction_input(session.id)
                async for event in self._handle_session_interrupted(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=_environment_name(registered_environment),
                    execution_profile=execution_profile,
                ):
                    yield event
                return
            await _run_recovery_cleanup_steps(
                authoritative_failure=cancellation,
                steps=(
                    (
                        "cancelled pre-run session finalization",
                        lambda: self._recovery_coordinator.finalize_abandoned_session_by_id(
                            session.id,
                            propagate_interaction_publication_rejection=True,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            execution_profile=execution_profile,
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
                runtime_authority=self._interaction_lifecycle_publication_authority,
            ):
                raise
            try:
                failure_interaction_observed_at = await self._failure_interaction_observed_at(
                    session
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
            task_failure_event, task_failure_error = await self._fail_task_for_run_setup_error(
                task_id=request.task_id,
                task_worker_id=request.task_worker_id,
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
            ) = await self._publish_sibling_interaction_transition(
                session=session,
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
            async for event in self._emit_terminal_event_with_hooks(
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
                    async for _ in self._handle_session_interrupted(
                        session=current_session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                        execution_profile=execution_profile,
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
            if (
                finalizer_failure is not None
                and workspace_observation_pending_cancellation_requests(finalizer_failure)
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
                            ),
                        ),
                        (
                            "Pre-run environment run-fence release",
                            lambda: (
                                self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                                    session_id=session.id,
                                    execution_profile=execution_profile,
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
                        )
                finally:
                    try:
                        if release_before_run:
                            await self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                                session_id=session.id,
                                execution_profile=execution_profile,
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
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                messages=messages,
                messages_to_append=[],
                max_steps=request.max_steps,
                limits=request.limits,
                budget_limits=request.budget_limits,
                budget_policy=budget_policy,
                retry_policy=self._effective_retry_policy(request.retry_policy),
                structured_output=request.structured_output,
                thinking=request.thinking,
                request_loop_policies=request.loop_policies,
                request_metadata=request.metadata,
                request_trace_metadata=request.metadata,
                task_id=request.task_id,
                task_worker_id=request.task_worker_id,
                start_event_type=EventType.SESSION_STARTED,
                start_event_payload={
                    "agent_name": registered_agent.spec.name,
                    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY: session_input_contract_evidence(
                        request,
                        message_start_index=len(messages) - len(request.messages),
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
            )
        except BaseException as exc:
            try:
                await self._environment_lifecycle.abort_environment_setup(
                    session_id=session.id,
                    original_error=exc,
                    execution_profile=execution_profile,
                )
            finally:
                try:
                    await self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                        session_id=session.id,
                        execution_profile=execution_profile,
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
            billing_identity_cancellation = detach_billing_identity_cancellation(exc)
            if billing_identity_cancellation is None:
                propagated_cancellation = exc
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

        # A registered provider can retain live credentials in its repr. Drop it
        # before any fallible interruption-store work and before raising a fresh
        # provider-free cancellation.
        del registered_provider

        if billing_identity_cancellation is not None:
            interruption_check_failed = False
            try:
                interruption_requested = await self._session_control.interrupt_requested(session.id)
            except Exception:
                interruption_check_failed = True
                interruption_requested = False
            if interruption_check_failed:
                raise RuntimeError(
                    "Session interruption state check failed after provider billing cancellation"
                ) from None
            if interruption_requested:
                interruption_transition_failed = False
                try:
                    async for event in self._handle_session_interrupted(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                        execution_profile=execution_profile,
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
            try:
                persisted_session = await self.session_store.load(session.id)
            except Exception:
                interruption_load_failed = True
                persisted_session = None
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
                    async for event in self._handle_session_interrupted(
                        session=persisted_session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                        execution_profile=execution_profile,
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
                async for event in self._handle_session_interrupted(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=_environment_name(registered_environment),
                    execution_profile=execution_profile,
                ):
                    yield event
                return
            raise propagated_cancellation

        if propagated_cancellation_group is not None:
            raise propagated_cancellation_group

    async def resume(
        self,
        request: ResumeRequest,
        *,
        store_resolved_session_id: str | None = None,
    ) -> AsyncGenerator[Event, None]:
        request = session_request_boundary.prepare_resume_request(
            request,
            redactor=self._secret_redactor,
            store_resolved_session_id=store_resolved_session_id,
        )
        task_id = await self._linked_running_task_id(request.session_id)
        session_stream = self._resume_session(
            request=request,
            task_id=task_id,
            start_event_payload_extra={},
            start_task_on_enter=False,
        )
        forwarded_stream = self._session_control.stream_with_out_of_band_events(
            request.session_id,
            session_stream,
        )
        billing_identity_cancellation: asyncio.CancelledError | None = None
        billing_identity_cancellation_group: BaseExceptionGroup | None = None
        try:
            async for event in forwarded_stream:
                yield event
        except asyncio.CancelledError as exc:
            billing_identity_cancellation = detach_billing_identity_cancellation(exc)
            if billing_identity_cancellation is None:
                raise
        except BaseExceptionGroup as exc:
            billing_identity_cancellation_group = detach_billing_identity_cancellation_group(exc)
            if billing_identity_cancellation_group is None:
                raise
        except GeneratorExit:
            await _close_async_iterator(forwarded_stream)
            raise
        if billing_identity_cancellation is not None:
            raise billing_identity_cancellation
        if billing_identity_cancellation_group is not None:
            raise billing_identity_cancellation_group

    async def compact_session(
        self,
        request: CompactSessionRequest,
        *,
        store_resolved_session_id: str | None = None,
    ) -> AsyncGenerator[Event, None]:
        request = session_request_boundary.prepare_compact_session_request(
            request,
            redactor=self._secret_redactor,
            store_resolved_session_id=store_resolved_session_id,
        )
        operation_stream = self._compact_session(request)
        forwarded_stream = self._session_control.stream_with_out_of_band_events(
            request.session_id,
            operation_stream,
        )
        billing_identity_cancellation: asyncio.CancelledError | None = None
        billing_identity_cancellation_group: BaseExceptionGroup | None = None
        try:
            async for event in forwarded_stream:
                yield event
        except asyncio.CancelledError as exc:
            billing_identity_cancellation = detach_billing_identity_cancellation(exc)
            if billing_identity_cancellation is None:
                raise
        except BaseExceptionGroup as exc:
            billing_identity_cancellation_group = detach_billing_identity_cancellation_group(exc)
            if billing_identity_cancellation_group is None:
                raise
        except GeneratorExit:
            await _close_async_iterator(forwarded_stream)
            raise
        if billing_identity_cancellation is not None:
            raise billing_identity_cancellation
        if billing_identity_cancellation_group is not None:
            raise billing_identity_cancellation_group

    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
        *,
        store_resolved_session_id: str | None = None,
    ) -> EnqueueSessionMessageResult:
        """Durably queue user steering for delivery by the active controller."""

        redacted_request = session_request_boundary.prepare_enqueue_message_request(
            request,
            redactor=self._secret_redactor,
            store_resolved_session_id=store_resolved_session_id,
        )
        result = await self.session_store.enqueue_session_message(redacted_request)
        if not result.replayed:
            await self._event_writer.fan_out_persisted([result.event])
        return result

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
    ) -> list[Event]:
        delivered_events: list[Event] = []
        eligible_through: int | None = None
        if continue_active_interaction:
            delivery_interaction_id = _current_session_interaction_id(session_id)
            if delivery_interaction_id is None:
                raise RuntimeError("Queued input has no active interaction to continue.")
            interaction_started = True
            interaction_started_event = None
        else:
            delivery_interaction_id = str(uuid4())
            interaction_started = False
            interaction_started_event = self._interaction_started_event(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                interaction_id=delivery_interaction_id,
            )
        delivery_batch_index = 0
        while True:
            delivery_id = (
                delivery_interaction_id
                if delivery_batch_index == 0
                else f"{delivery_interaction_id}:batch:{delivery_batch_index}"
            )
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
                Message.text(MessageRole.USER, queued_message.content)
                for queued_message in batch.messages
            )
            if batch.messages and not continue_active_interaction:
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
                return delivered_events
            delivery_batch_index += 1

    async def _compact_session(
        self,
        request: CompactSessionRequest,
    ) -> AsyncGenerator[Event, None]:
        """Compact model-facing context without appending a conversation turn."""

        request = copy_compact_session_request(request)
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
        request_digest = _compact_session_request_digest(request)
        checkpoint_before_claim = await self.session_store.load_checkpoint(loaded_session.id)
        checkpoint_before_claim = await _reconcile_committed_prompt_transition_intents(
            session_store=self.session_store,
            source_session_id=loaded_session.id,
            checkpoint=checkpoint_before_claim,
            clock=self._clock,
        )
        persisted_before_claim = await self.session_store.load_session_operation(
            loaded_session.id,
            request.idempotency_key,
        )
        if type(persisted_before_claim) is dict:
            if persisted_before_claim.get("request_digest") != request_digest:
                raise ValueError(
                    "Session compaction idempotency key was already used for a different request."
                )
            replay_event_ids = _session_compaction_replay_event_ids(persisted_before_claim)
            if replay_event_ids is not None:
                for event in await self._load_session_compaction_replay_events(
                    session_id=loaded_session.id,
                    event_ids=replay_event_ids,
                ):
                    yield event
                return
        if (
            checkpoint_before_claim is not None
            and _SESSION_OPERATIONS_CHECKPOINT_KEY in checkpoint_before_claim
        ):
            existing_before_claim = _session_operation_state(checkpoint_before_claim)[
                "records"
            ].get(request.idempotency_key)
            if type(existing_before_claim) is dict:
                if existing_before_claim.get("request_digest") != request_digest:
                    raise ValueError(
                        "Session compaction idempotency key was already used for a "
                        "different request."
                    )
                replay_event_ids = _session_compaction_replay_event_ids(existing_before_claim)
                if replay_event_ids is not None:
                    for event in await self._load_session_compaction_replay_events(
                        session_id=loaded_session.id,
                        event_ids=replay_event_ids,
                    ):
                        yield event
                    return
        _reject_unresumable_session_checkpoint(
            loaded_session,
            checkpoint_before_claim,
            redactor=self._secret_redactor,
            allow_active_operation=True,
        )
        if loaded_session.status not in _RESUMABLE_SESSION_STATUSES:
            raise ValueError(
                f"Session compaction requires a resumable session boundary: {loaded_session.status}"
            )
        if loaded_session.run_epoch != request.expected_run_epoch:
            raise ValueError(
                "Session compaction source run epoch is stale: expected "
                f"{request.expected_run_epoch}, current {loaded_session.run_epoch}."
            )

        registered_agent = self._get_registered_agent(loaded_session.agent_name)
        context_policy = registered_agent.context_policy
        if not isinstance(context_policy, CheckpointCompactionContextPolicy):
            raise ValueError(
                "Explicit session compaction requires a configured "
                "CheckpointCompactionContextPolicy."
            )
        compactor_name = _require_application_compaction_event_text(
            type(context_policy.compactor).__name__,
            "compactor",
        )
        registered_environment = self._get_registered_environment_for_session(
            loaded_session.environment_name
        )
        environment_name = _environment_name(registered_environment)
        transcript_snapshot = await self.session_store.load_transcript_snapshot(loaded_session.id)
        try:
            source_transcript_cursor = transcript_snapshot.cursor
            if source_transcript_cursor != request.expected_transcript_cursor:
                raise ValueError(
                    "Session compaction source transcript cursor is stale: expected "
                    f"{request.expected_transcript_cursor}, current {source_transcript_cursor}."
                )
            projection_cursor = session_model_projection_cursor(loaded_session)
            if projection_cursor:
                transcript = list(
                    _project_model_target_snapshot(
                        transcript_snapshot,
                        projection_cursor,
                    ).messages
                )
            else:
                transcript = _transcript_snapshot_messages(transcript_snapshot)
        finally:
            del transcript_snapshot
        transcript, transcript_is_valid = session_request_boundary.redact_transcript(
            transcript,
            redactor=self._secret_redactor,
            field_name="session.transcript",
            reject_secret_bearing_runtime_projection_authority=True,
        )
        if not transcript_is_valid:
            raise ValueError(
                "Session transcript contains a workload secret in execution authority "
                "and cannot be compacted."
            ) from None
        compaction_budget_policy = copy_budget_policy(self._get_budget_policy())
        candidate_app_policy_budget_limits = budget_limits_for_session(
            policy=compaction_budget_policy,
            agent_name=registered_agent.spec.name,
            causal_budget_id=loaded_session.causal_budget_id,
        )
        candidate_request_budget_limits = request_budget_limits_for_session(
            limits=request.budget_limits,
            agent_name=registered_agent.spec.name,
            causal_budget_id=loaded_session.causal_budget_id,
        )
        compactor_provider_name: str | None = None
        compactor_model: str | None = None
        uses_forced_dispatch_boundary = (
            context_policy.compactor._uses_runtime_provider_dispatch_runner_for_forced_compaction()
        )
        has_compaction_accounting_limits = bool(
            candidate_app_policy_budget_limits
            or candidate_request_budget_limits
            or has_run_limits(request.limits)
        )
        if (
            uses_forced_dispatch_boundary
            or has_compaction_accounting_limits
            or self._request_footprint.enabled
        ):
            try:
                provider_budget_identity = context_policy.compactor.provider_budget_identity(
                    loaded_session
                )
            except NotImplementedError as exc:
                if self._request_footprint.enabled:
                    raise RuntimeError(
                        "Explicit compaction with request footprints requires the "
                        "ContextCompactor to explicitly declare "
                        "provider_budget_identity(session), returning provider/model or None "
                        "for deterministic execution."
                    ) from exc
                if not has_compaction_accounting_limits:
                    provider_budget_identity = None
                else:
                    raise RuntimeError(
                        "Explicit provider-backed compaction under run or cost limits requires "
                        "the ContextCompactor to declare provider_budget_identity(session), "
                        "returning provider/model or None for deterministic execution."
                    ) from exc
            if provider_budget_identity is None and uses_forced_dispatch_boundary:
                raise RuntimeError(
                    "Provider-backed compaction cannot declare a deterministic budget identity."
                )
            if provider_budget_identity is not None:
                if (
                    type(provider_budget_identity) is not tuple
                    or len(provider_budget_identity) != 2
                ):
                    raise TypeError(
                        "ContextCompactor.provider_budget_identity must return a "
                        "(provider_name, model) tuple or None."
                    )
                compactor_provider_name = _require_application_compaction_event_text(
                    provider_budget_identity[0],
                    "compactor_provider_name",
                )
                compactor_model = _require_application_compaction_event_text(
                    provider_budget_identity[1],
                    "compactor_model",
                )
                if self._request_footprint.enabled and not uses_forced_dispatch_boundary:
                    raise RuntimeError(
                        "Explicit provider-backed compaction with request footprints cannot "
                        f"safely run opaque provider-backed compactor "
                        f"{type(context_policy.compactor).__name__}: Cayu cannot observe each "
                        "provider dispatch independently. Use an unmodified built-in provider "
                        "compactor or disable request footprints."
                    )
        if compactor_provider_name is None:
            app_policy_budget_limits: tuple[BudgetLimit, ...] = ()
            budget_limits: tuple[BudgetLimit, ...] = ()
        else:
            app_policy_budget_limits = candidate_app_policy_budget_limits
            budget_limits = (*app_policy_budget_limits, *candidate_request_budget_limits)
        dispatch_settlement_budget_limits = tuple(
            limit
            for limit in budget_limits
            if (
                limit.reservation is not None
                or has_deferred_contextual_price(
                    limit.pricing,
                    provider_name=compactor_provider_name,
                    model=compactor_model,
                )
            )
        )
        dispatch_settlement_budget_limit_ids = {
            _effective_budget_limit_id(limit) for limit in dispatch_settlement_budget_limits
        }
        outer_budget_limits = tuple(
            limit
            for limit in budget_limits
            if _effective_budget_limit_id(limit) not in dispatch_settlement_budget_limit_ids
        )
        outer_app_policy_budget_limits = tuple(
            limit
            for limit in app_policy_budget_limits
            if _effective_budget_limit_id(limit) not in dispatch_settlement_budget_limit_ids
        )
        needs_dispatch_admission = compactor_provider_name is not None and (
            has_run_limits(request.limits) or bool(budget_limits)
        )
        if needs_dispatch_admission and not uses_forced_dispatch_boundary:
            raise RuntimeError(
                "Explicit provider-backed compaction under run or cost limits requires "
                "an unmodified built-in provider compactor so Cayu can admit every "
                "provider dispatch before execution."
            )
        expected_profile = (
            None
            if EXECUTION_PROFILE_METADATA_KEY not in loaded_session.metadata
            else execution_profile_from_session_metadata(loaded_session.metadata)
        )
        try:
            registered_provider = self._get_registered_provider(loaded_session.provider_name)
        except KeyError:
            registered_provider = None

        def current_compaction_profile_candidate() -> ExecutionProfileIdentity:
            candidate = _execution_profile_identity(
                registered_agent=registered_agent,
                provider_name=loaded_session.provider_name,
                registered_provider=registered_provider,
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
                budget_policy=compaction_budget_policy,
                request_budget_limits=request.budget_limits,
                causal_budget_id=loaded_session.causal_budget_id,
                limits=request.limits,
                finalization_material={
                    "kind": "cayu:explicit-context-compaction:v1",
                    "reason": request.reason,
                    "instruction_present": request.instructions is not None,
                    "instruction_digest": _optional_text_digest(request.instructions),
                    "limits": request.limits.model_dump(mode="json"),
                    "provider_name": compactor_provider_name,
                    "model": compactor_model,
                },
            )
            if expected_profile is None:
                return execution_profile_with_durable_system_projection_digest(
                    candidate,
                    system_prompt_messages_sha256(transcript),
                )
            return execution_profile_with_component(
                candidate,
                expected_profile.component(
                    ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION
                ),
            )

        profile_candidate = current_compaction_profile_candidate()
        governed_registration_components = (
            ExecutionProfileComponentClass.CONTEXT_SELECTION,
            ExecutionProfileComponentClass.KNOWLEDGE_INJECTION,
            ExecutionProfileComponentClass.CONTEXT_COMPACTION,
            ExecutionProfileComponentClass.EXECUTION_ENVIRONMENT,
        )
        changed_registration_components = (
            ()
            if expected_profile is None or expected_profile.schema_version < 3
            else tuple(
                component
                for component in governed_registration_components
                if expected_profile.component(component) != profile_candidate.component(component)
            )
        )
        if changed_registration_components:
            assert expected_profile is not None
            raise ExecutionProfileMismatchError(
                session_id=loaded_session.id,
                expected_profile_fingerprint=expected_profile.fingerprint,
                candidate_profile_fingerprint=profile_candidate.fingerprint,
                changed_component_classes=changed_registration_components,
            )
        if expected_profile is None or expected_profile.schema_version < 3:
            compaction_execution_profile = profile_candidate
        else:
            compaction_execution_profile = expected_profile
            for component_class in (
                *governed_registration_components,
                ExecutionProfileComponentClass.APPLICATION_BUDGET_POLICY,
                ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY,
                ExecutionProfileComponentClass.FINALIZATION,
            ):
                compaction_execution_profile = execution_profile_with_component(
                    compaction_execution_profile,
                    profile_candidate.component(component_class),
                )

        def validate_live_compaction_semantics() -> None:
            candidate = current_compaction_profile_candidate()
            changed = tuple(
                component
                for component in governed_registration_components
                if candidate.component(component)
                != compaction_execution_profile.component(component)
            )
            if changed:
                raise ExecutionProfileMismatchError(
                    session_id=loaded_session.id,
                    expected_profile_fingerprint=compaction_execution_profile.fingerprint,
                    candidate_profile_fingerprint=candidate.fingerprint,
                    changed_component_classes=changed,
                )

        def governed_compaction_events(events: Iterable[Event]) -> list[Event]:
            return [
                event_with_execution_profile_authority(event, compaction_execution_profile)
                for event in events
            ]

        operation_started_at = time.monotonic()

        operation_id = str(uuid4())
        claim_probe_now = self._clock()
        attempt_id = str(uuid4())
        existing_before_claim: dict[str, Any] | None = None
        if (
            type(persisted_before_claim) is dict
            and persisted_before_claim.get("request_digest") == request_digest
            and persisted_before_claim.get("status") == "abandoned"
        ):
            persisted_operation_id = persisted_before_claim.get("operation_id")
            if type(persisted_operation_id) is not str:
                raise ValueError("Persisted session operation is missing its operation id.")
            operation_id = require_clean_nonblank(
                persisted_operation_id,
                "operation_id",
            )
        if (
            checkpoint_before_claim is not None
            and _SESSION_OPERATIONS_CHECKPOINT_KEY in checkpoint_before_claim
        ):
            existing_candidate = _session_operation_state(checkpoint_before_claim)["records"].get(
                request.idempotency_key
            )
            existing_before_claim = existing_candidate if type(existing_candidate) is dict else None
            if (
                type(existing_before_claim) is dict
                and existing_before_claim.get("request_digest") == request_digest
                and existing_before_claim.get("status") == "running"
                and (
                    (expiry := _operation_claim_expiry(existing_before_claim)) is not None
                    and expiry <= claim_probe_now
                )
            ):
                operation_id = require_clean_nonblank(
                    existing_before_claim.get("operation_id"),
                    "operation_id",
                )
        stored_model_step_identity: ModelStepIdentity | None = None
        for existing_record in (persisted_before_claim, existing_before_claim):
            if (
                type(existing_record) is not dict
                or existing_record.get("request_digest") != request_digest
                or existing_record.get("status") not in {"running", "abandoned"}
            ):
                continue
            raw_model_step_id = existing_record.get("model_step_id")
            if type(raw_model_step_id) is not str:
                raise RuntimeError("Existing session compaction lacks logical model-step identity.")
            candidate_identity = ModelStepIdentity(model_step_id=raw_model_step_id)
            if (
                stored_model_step_identity is not None
                and stored_model_step_identity != candidate_identity
            ):
                raise RuntimeError(
                    "Session compaction records disagree on logical model-step identity."
                )
            stored_model_step_identity = candidate_identity
            raw_profile = existing_record.get("execution_profile")
            if type(raw_profile) is not dict:
                raise RuntimeError(
                    "Existing session compaction lacks its governing execution profile."
                )
            stored_profile = ExecutionProfileIdentity.model_validate(raw_profile)
            if stored_profile != compaction_execution_profile:
                raise ExecutionProfileMismatchError(
                    session_id=loaded_session.id,
                    expected_profile_fingerprint=stored_profile.fingerprint,
                    candidate_profile_fingerprint=compaction_execution_profile.fingerprint,
                    changed_component_classes=changed_execution_profile_components(
                        stored_profile,
                        compaction_execution_profile,
                    ),
                )
        model_step_identity = stored_model_step_identity or new_model_step_identity()
        reservation_identity_guard = self._run_limit_controller.reservation_identity_guard()
        started_event = self._event_writer.prepare(
            event_with_execution_profile_authority(
                Event(
                    type=EventType.CONTEXT_COMPACTION_STARTED,
                    session_id=loaded_session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        **_application_compaction_causal_payload(
                            request=request,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            source_cursor=source_transcript_cursor,
                            compactor=compactor_name,
                        ),
                        **model_step_identity.payload(),
                    },
                ),
                compaction_execution_profile,
            )
        )
        claimed_checkpoint: dict[str, Any] | None = None
        claimed_operation_expires_at: datetime | None = None

        def claim_operation(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            persisted_record: dict[str, Any] | None,
        ) -> SessionOperationPublication:
            nonlocal operation_id, claimed_checkpoint, claimed_operation_expires_at
            claim_now = self._clock()
            claim_expires_at = claim_now + _SESSION_OPERATION_CLAIM_LEASE
            if current_session.run_epoch != request.expected_run_epoch:
                raise ValueError(
                    "Session compaction source run epoch is stale: expected "
                    f"{request.expected_run_epoch}, current {current_session.run_epoch}."
                )
            recovery_claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
            if recovery_claim is not None and recovery_claim[1] <= claim_now:
                raise _ExpiredIncompleteRecoveryClaim(recovery_claim[0])
            checkpoint = _checkpoint_without_active_incomplete_recovery_claim(
                checkpoint,
                now=claim_now,
            )
            _reject_unresumable_session_checkpoint(
                current_session,
                checkpoint,
                redactor=self._secret_redactor,
                allow_active_operation=True,
            )
            updated = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
            operations = _session_operation_state(updated)
            _abandon_expired_session_operation(operations, now=claim_now)
            records = operations["records"]
            existing = records.get(request.idempotency_key)
            if existing is None and persisted_record is not None:
                existing = copy_json_value(persisted_record, "session_operation")
                if existing.get("status") == "abandoned":
                    records[request.idempotency_key] = existing
            if existing is not None:
                if existing.get("request_digest") != request_digest:
                    raise ValueError(
                        "Session compaction idempotency key was already used for a "
                        "different request."
                    )
                status = existing.get("status")
                if status == "running":
                    raise RuntimeError(
                        "Equivalent session compaction operation is already running: "
                        f"{existing.get('operation_id')}"
                    )
                if status == "abandoned":
                    operation_id = require_clean_nonblank(
                        existing.get("operation_id"),
                        "operation_id",
                    )
                    if started_event.payload.get("operation_id") != operation_id:
                        raise RuntimeError(
                            "Abandoned session compaction changed during claim; retry it."
                        )
                    existing_model_step_identity = ModelStepIdentity(
                        model_step_id=existing.get("model_step_id")
                    )
                    if existing_model_step_identity != model_step_identity:
                        raise RuntimeError(
                            "Abandoned session compaction changed logical model-step identity."
                        )
                    if operations.get("active_operation_id") is not None:
                        raise RuntimeError(
                            "Session already has an active durable operation: "
                            f"{operations.get('active_operation_id')}"
                        )
                    operations["active_operation_id"] = operation_id
                    existing["status"] = "running"
                    existing["attempt_count"] = existing.get("attempt_count", 1) + 1
                    existing["current_attempt_id"] = attempt_id
                    existing["event_ids"] = [
                        *existing.get("event_ids", []),
                        started_event.id,
                    ]
                    existing["claim_expires_at"] = claim_expires_at.isoformat()
                    existing["updated_at"] = claim_now.isoformat()
                    existing.pop("abandoned_at", None)
                    updated[_SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
                    claimed_checkpoint = copy_json_value(updated, "checkpoint")
                    claimed_operation_expires_at = claim_expires_at
                    archived_records = _archive_inactive_session_operation_records(
                        updated,
                        except_idempotency_key=request.idempotency_key,
                    )
                    return SessionOperationPublication(
                        checkpoint=updated,
                        operation_records=archived_records,
                    )
                operation_id = require_clean_nonblank(
                    existing.get("operation_id"),
                    "operation_id",
                )
                stored_event_ids = _session_compaction_replay_event_ids(existing)
                if stored_event_ids is None:
                    raise RuntimeError("Running session compaction changed during claim.")
                raise _SessionCompactionReplay(stored_event_ids)
            active_operation_id = operations.get("active_operation_id")
            if active_operation_id is not None:
                raise RuntimeError(
                    f"Session already has an active durable operation: {active_operation_id}"
                )
            operations["active_operation_id"] = operation_id
            records[request.idempotency_key] = {
                "operation_id": operation_id,
                "model_step_id": model_step_identity.model_step_id,
                "kind": _CONTEXT_COMPACTION_OPERATION_KIND,
                "reason": request.reason,
                "request_digest": request_digest,
                "status": "running",
                "source_run_epoch": request.expected_run_epoch,
                "source_transcript_cursor": request.expected_transcript_cursor,
                "attempt_count": 1,
                "current_attempt_id": attempt_id,
                "event_ids": [started_event.id],
                "instruction_present": request.instructions is not None,
                "instruction_digest": _optional_text_digest(request.instructions),
                "execution_profile": compaction_execution_profile.model_dump(mode="json"),
                "claim_expires_at": claim_expires_at.isoformat(),
                "created_at": claim_now.isoformat(),
                "updated_at": claim_now.isoformat(),
            }
            updated[_SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
            claimed_checkpoint = copy_json_value(updated, "checkpoint")
            claimed_operation_expires_at = claim_expires_at
            archived_records = _archive_inactive_session_operation_records(
                updated,
                except_idempotency_key=request.idempotency_key,
            )
            return SessionOperationPublication(
                checkpoint=updated,
                operation_records=archived_records,
            )

        started_event = attribute_event_to_current_interaction(started_event)
        replay_event_ids: tuple[str, ...] | None = None

        def require_unexpired_initial_claim_commit() -> None:
            if claimed_operation_expires_at is None:
                raise AssertionError(
                    "Session compaction initial publication did not capture its claim."
                )
            _require_unexpired_session_operation_commit(
                clock=self._clock,
                claim_expires_at=claimed_operation_expires_at,
                operation="Session compaction initial publication",
            )

        try:
            await self.session_store.publish_session_operation_guarded(
                loaded_session.id,
                idempotency_key=request.idempotency_key,
                operation_transform=claim_operation,
                commit_guard=require_unexpired_initial_claim_commit,
                events=[started_event],
                expected_statuses=_RESUMABLE_SESSION_STATUSES,
                expected_run_epoch=request.expected_run_epoch,
                expected_transcript_cursor=request.expected_transcript_cursor,
            )
        except _ExpiredIncompleteRecoveryClaim as expired_claim:
            current = await self._require_session(loaded_session.id)
            fenced = await self._recovery_coordinator.fence_expired_incomplete_recovery_claim(
                session=current,
                claim_id=expired_claim.claim_id,
            )
            if not fenced:
                raise RuntimeError(
                    "Expired incomplete-session recovery ownership changed while compaction "
                    "was fencing it; retry with current session state."
                ) from None
            current = await self._require_session(loaded_session.id)
            raise ValueError(
                "Session compaction fenced an expired incomplete-session recovery owner; "
                f"retry with run epoch {current.run_epoch}."
            ) from None
        except _SessionCompactionReplay as replay:
            replay_event_ids = replay.event_ids
        if replay_event_ids is not None:
            for event in await self._load_session_compaction_replay_events(
                session_id=loaded_session.id,
                event_ids=replay_event_ids,
            ):
                yield event
            return
        if claimed_checkpoint is None:
            raise AssertionError("New session compaction did not persist its operation claim.")
        if claimed_operation_expires_at is None:
            raise AssertionError("New session compaction did not persist its claim expiry.")
        compaction_invocation_checkpoint = project_compaction_invocation_checkpoint(
            claimed_checkpoint,
            redactor=self._secret_redactor,
        )
        checkpoint_before_claim = None
        claimed_checkpoint = None
        _require_unexpired_session_operation_claim(
            clock=self._clock,
            claim_expires_at=claimed_operation_expires_at,
            operation="Session compaction initial publication",
        )

        operation_published = False
        operation_failure: BaseException | None = None
        stop_operation_heartbeat = asyncio.Event()
        operation_heartbeat_state = _SessionOperationClaimHeartbeatState()
        operation_heartbeat_state.confirm_claim(claimed_operation_expires_at)
        operation_heartbeat_task = asyncio.create_task(
            self._heartbeat_compaction_operation_claim(
                session=loaded_session,
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                claim_expires_at=claimed_operation_expires_at,
                stop=stop_operation_heartbeat,
                state=operation_heartbeat_state,
            )
        )

        async def stop_claim_heartbeat() -> asyncio.Task[Any] | None:
            stop_operation_heartbeat.set()
            if not operation_heartbeat_task.done():
                operation_heartbeat_task.cancel()
            outcome = await await_shielded_task_outcome(
                operation_heartbeat_task,
                timeout_s=_SESSION_OPERATION_CLAIM_HEARTBEAT_STOP_TIMEOUT_SECONDS,
            )
            pending_renewal = operation_heartbeat_state.pending_store_task
            if outcome.timed_out:
                if pending_renewal is not None and not pending_renewal.done():
                    pending_outcome = await await_shielded_task_outcome(
                        pending_renewal,
                        timeout_s=_SESSION_OPERATION_CLAIM_HEARTBEAT_STOP_TIMEOUT_SECONDS,
                    )
                    if pending_outcome.timed_out:
                        return pending_renewal
                    if pending_outcome.cancellation is not None:
                        raise pending_outcome.cancellation
                outcome = await await_shielded_task_outcome(
                    operation_heartbeat_task,
                    timeout_s=_SESSION_OPERATION_CLAIM_HEARTBEAT_STOP_TIMEOUT_SECONDS,
                )
                if outcome.timed_out:
                    return operation_heartbeat_task
            if pending_renewal is not None and not pending_renewal.done():
                pending_outcome = await await_shielded_task_outcome(
                    pending_renewal,
                    timeout_s=_SESSION_OPERATION_CLAIM_HEARTBEAT_STOP_TIMEOUT_SECONDS,
                )
                if pending_outcome.timed_out:
                    return pending_renewal
                if pending_outcome.cancellation is not None:
                    raise pending_outcome.cancellation
            if outcome.cancellation is not None:
                raise outcome.cancellation
            heartbeat_failure = outcome.error
            if heartbeat_failure is None:
                return None
            if isinstance(heartbeat_failure, asyncio.CancelledError):
                concurrent_failure = exception_cause(heartbeat_failure)
                if concurrent_failure is None:
                    return None
                heartbeat_failure = concurrent_failure
            if operation_published or heartbeat_failure is operation_failure:
                return None
            raise heartbeat_failure

        async def stop_claim_heartbeat_and_track() -> None:
            blocker = await stop_claim_heartbeat()
            if blocker is not None:
                self._track_detached_session_operation_task(blocker)

        initial_event_delivered = False
        initial_delivery_failure: BaseException | None = None
        try:
            await self._event_writer.fan_out_persisted([started_event])
            yield started_event
            initial_event_delivered = True
        except BaseException as exc:
            initial_delivery_failure = exc
            raise
        finally:
            if not initial_event_delivered:
                await _run_recovery_cleanup_steps(
                    authoritative_failure=initial_delivery_failure,
                    steps=(
                        (
                            "explicit compaction claim heartbeat stop",
                            stop_claim_heartbeat_and_track,
                        ),
                    ),
                )

        attempt_events: list[Event] = []
        prepublished_dispatch_events: list[Event] = []
        deferred_dispatch_settlement_events: list[Event] = []
        observed_dispatch_completion_events: list[Event] = []
        materialized_compaction_attempt_ids: set[str] = set()
        persisted_attempt_event_ids: set[str] = set()
        yielded_attempt_event_ids: set[str] = set()
        attempt_event_inventory: dict[str, Event] = {}
        completion_event_attempt_ids: dict[str, str] = {}
        unresolved_attempt_publication_tasks: set[asyncio.Task[Any]] = set()
        unresolved_completion_publication_tasks: set[asyncio.Task[Any]] = set()
        reached_budget_keys: set[str] = set()
        budget_reservations: list[BudgetStepReservation] = []
        budget_reservations_settled = False
        outer_model_attempt_identity = model_step_identity.new_attempt()
        try:
            limit_decision = await self._run_limit_controller.evaluate_operation_run_limit(
                session=loaded_session,
                limits=request.limits,
                operation_events=attempt_events,
                operation_started_at=operation_started_at,
            )
            if limit_decision is not None:
                raise RuntimeError(f"Compaction limit reached: {limit_decision.message}")
            budget_error = await self._enforce_compaction_budget_limits(
                session=loaded_session,
                budget_limits=outer_budget_limits,
                app_policy_budget_limits=outer_app_policy_budget_limits,
                attempt_events=attempt_events,
                reached_budget_keys=reached_budget_keys,
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                registered_agent=registered_agent,
                environment_name=environment_name,
                compactor=compactor_name,
                provider_name=compactor_provider_name,
                model=compactor_model,
                execution_identity=model_step_identity,
            )
            if attempt_events:
                persisted_attempt_events = governed_compaction_events(attempt_events)
                await self._persist_compaction_attempt_events(
                    session=loaded_session,
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    events=persisted_attempt_events,
                    heartbeat_state=operation_heartbeat_state,
                    persisted_event_ids=persisted_attempt_event_ids,
                    event_inventory=attempt_event_inventory,
                    unresolved_store_tasks=unresolved_attempt_publication_tasks,
                )
                attempt_events.clear()
                await self._run_limit_controller.acknowledge_budget_settlement_events(
                    persisted_attempt_events
                )
                await self._event_writer.fan_out_persisted(persisted_attempt_events)
                for event in persisted_attempt_events:
                    yielded_attempt_event_ids.add(event.id)
                    yield event
            if budget_error is not None:
                raise budget_error

            reservation_failure = await self._reserve_compaction_budget(
                session=loaded_session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                budget_limits=outer_budget_limits,
                provider_name=compactor_provider_name,
                model=compactor_model,
                model_attempt_identity=outer_model_attempt_identity,
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                compactor=compactor_name,
                reservations=budget_reservations,
                events=attempt_events,
                reservation_identity_guard=reservation_identity_guard,
                execution_profile_fingerprint=compaction_execution_profile.fingerprint,
            )
            reservation_error: RuntimeError | None = None
            if reservation_failure is not None:
                budget_reservations_settled = True
                attempt_events.append(
                    _application_compaction_ledger_event(
                        event_type=EventType.BUDGET_LIMIT_REACHED,
                        payload=budget_reservation_payload(reservation_failure),
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        session=loaded_session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        compactor=compactor_name,
                    )
                )
                reservation_error = RuntimeError(
                    f"Compaction budget reservation failed: {reservation_failure.message}"
                )
            if attempt_events:
                persisted_attempt_events = governed_compaction_events(attempt_events)
                await self._persist_compaction_attempt_events(
                    session=loaded_session,
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    events=persisted_attempt_events,
                    heartbeat_state=operation_heartbeat_state,
                    persisted_event_ids=persisted_attempt_event_ids,
                    event_inventory=attempt_event_inventory,
                    unresolved_store_tasks=unresolved_attempt_publication_tasks,
                )
                attempt_events.clear()
                await self._run_limit_controller.acknowledge_budget_settlement_events(
                    persisted_attempt_events
                )
                await self._event_writer.fan_out_persisted(persisted_attempt_events)
                for event in persisted_attempt_events:
                    yielded_attempt_event_ids.add(event.id)
                    yield event
            if reservation_error is not None:
                raise reservation_error

            async def publish_dispatch_budget_events() -> None:
                if not attempt_events:
                    return
                events = governed_compaction_events(attempt_events)
                await self._persist_compaction_attempt_events(
                    session=loaded_session,
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    events=events,
                    heartbeat_state=operation_heartbeat_state,
                    persisted_event_ids=persisted_attempt_event_ids,
                    event_inventory=attempt_event_inventory,
                    unresolved_store_tasks=unresolved_attempt_publication_tasks,
                )
                attempt_events.clear()
                await self._run_limit_controller.acknowledge_budget_settlement_events(events)
                await self._event_writer.fan_out_persisted(events)
                prepublished_dispatch_events.extend(events)

            active_compaction_model_attempt_identity: ModelAttemptIdentity | None = None
            model_attempts_by_compaction_id: dict[str, ModelAttemptIdentity] = {}
            compaction_ids_by_model_attempt_id: dict[str, str] = {}
            issued_compaction_model_attempts: dict[str, ModelAttemptIdentity] = {}

            async def record_compaction_footprint(
                *,
                provider: ModelProvider,
                provider_name: str,
                model_request: ModelRequest,
                dispatch_attempt: int,
                max_dispatch_attempts: int,
                model_attempt_identity: ModelAttemptIdentity,
            ) -> None:
                detached_request = _detach_model_request(model_request)
                if self._request_footprint.enabled:
                    footprint = analyze_request_footprint(
                        detached_request,
                        provider=provider,
                        provider_name=provider_name,
                        step=None,
                        attempt=dispatch_attempt,
                        max_attempts=max_dispatch_attempts,
                        request_variant=RequestVariant.CONTEXT_COMPACTION,
                        observation_id=str(uuid4()),
                        model_step_id=model_attempt_identity.model_step_id,
                        model_attempt_id=model_attempt_identity.model_attempt_id,
                        config=self._request_footprint,
                        operation_id=operation_id,
                        operation_attempt_id=attempt_id,
                        execution_profile_fingerprint=(compaction_execution_profile.fingerprint),
                    )
                    attempt_events.append(
                        event_with_execution_profile_authority(
                            event_with_runtime_payload_authority(
                                Event(
                                    type=EventType.REQUEST_FOOTPRINT_RECORDED,
                                    session_id=loaded_session.id,
                                    agent_name=registered_agent.spec.name,
                                    environment_name=environment_name,
                                    payload=footprint.model_dump(mode="json", exclude_none=True),
                                ),
                                "observation_id",
                                "model_step_id",
                                "model_attempt_id",
                                "operation_id",
                                "attempt_id",
                            ),
                            compaction_execution_profile,
                        )
                    )
                attempt_events.append(
                    event_with_execution_profile_authority(
                        event_with_runtime_payload_authority(
                            _application_compaction_ledger_event(
                                event_type=EventType.MODEL_STARTED,
                                payload={
                                    "model": detached_request.model,
                                    "provider": provider_name,
                                    "attempt": dispatch_attempt,
                                    "max_attempts": max_dispatch_attempts,
                                    "purpose": ModelCompletionPurpose.CONTEXT_COMPACTION.value,
                                    **model_attempt_identity.payload(),
                                },
                                request=request,
                                operation_id=operation_id,
                                attempt_id=attempt_id,
                                session=loaded_session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                compactor=compactor_name,
                            ),
                            "model_step_id",
                            "model_attempt_id",
                            "operation_id",
                            "attempt_id",
                        ),
                        compaction_execution_profile,
                    )
                )
                await publish_dispatch_budget_events()

            def identify_compaction_completion_payload(
                payload: dict[str, Any],
                *,
                expected_identity: ModelAttemptIdentity | None = None,
            ) -> dict[str, Any]:
                copied_payload = copy_json_value(
                    payload,
                    "compaction_model_completed_payload",
                )
                compaction_attempt_id = copied_payload.get(_COMPACTION_ATTEMPT_ID_KEY)
                if type(compaction_attempt_id) is not str:
                    raise RuntimeError("Compaction completion evidence lost its attempt identity.")
                identity = model_attempts_by_compaction_id.get(compaction_attempt_id)
                candidate = (
                    copy_model_attempt_identity(expected_identity)
                    if expected_identity is not None
                    else active_compaction_model_attempt_identity
                )
                payload_identity: ModelAttemptIdentity | None = None
                if "model_step_id" in copied_payload or "model_attempt_id" in copied_payload:
                    try:
                        payload_identity = ModelAttemptIdentity.model_validate(
                            {
                                "model_step_id": copied_payload.get("model_step_id"),
                                "model_attempt_id": copied_payload.get("model_attempt_id"),
                            }
                        )
                    except (TypeError, ValueError):
                        raise ValueError(
                            "Compaction completion carries an invalid model attempt identity."
                        ) from None
                    issued_identity = issued_compaction_model_attempts.get(
                        payload_identity.model_attempt_id
                    )
                    if issued_identity != payload_identity:
                        raise ValueError(
                            "Compaction completion carries a model attempt identity that "
                            "was not issued for this logical step."
                        )
                if (
                    candidate is not None
                    and payload_identity is not None
                    and candidate != payload_identity
                ):
                    raise ValueError(
                        "Compaction completion identity conflicts with its provider dispatch."
                    )
                candidate = candidate or payload_identity
                if identity is None:
                    if candidate is None:
                        raise RuntimeError(
                            "Compaction completion was observed outside its provider dispatch."
                        )
                    existing_compaction_id = compaction_ids_by_model_attempt_id.get(
                        candidate.model_attempt_id
                    )
                    if (
                        existing_compaction_id is not None
                        and existing_compaction_id != compaction_attempt_id
                    ):
                        raise ValueError(
                            "Compaction provider dispatch produced conflicting "
                            "completion identities."
                        )
                    identity = copy_model_attempt_identity(candidate)
                    model_attempts_by_compaction_id[compaction_attempt_id] = identity
                elif candidate is not None and identity != candidate:
                    raise ValueError(
                        "Compaction completion identity conflicts with its provider dispatch."
                    )
                existing_compaction_id = compaction_ids_by_model_attempt_id.setdefault(
                    identity.model_attempt_id,
                    compaction_attempt_id,
                )
                if existing_compaction_id != compaction_attempt_id:
                    raise ValueError(
                        "Compaction provider dispatch produced conflicting completion identities."
                    )
                strip_runtime_owned_execution_identity(copied_payload)
                copied_payload.update(identity.payload())
                return copied_payload

            def application_compaction_telemetry_event(
                telemetry: ContextCompactionTelemetry,
            ) -> Event:
                execution_identity: ModelStepIdentity | ModelAttemptIdentity = model_step_identity
                if telemetry.event_type == EventType.MODEL_COMPLETED:
                    identified_payload = identify_compaction_completion_payload(telemetry.payload)
                    telemetry = ContextCompactionTelemetry(
                        event_type=telemetry.event_type,
                        payload=identified_payload,
                    )
                    execution_identity = ModelAttemptIdentity.model_validate(
                        {
                            "model_step_id": identified_payload.get("model_step_id"),
                            "model_attempt_id": identified_payload.get("model_attempt_id"),
                        }
                    )
                return event_with_execution_profile_authority(
                    _application_compaction_event(
                        telemetry=telemetry,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        session=loaded_session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        compactor=compactor_name,
                        execution_identity=execution_identity,
                    ),
                    compaction_execution_profile,
                )

            def dispatch_completion_event(
                *,
                actual_pricing_provider_name: str,
                actual_model: str,
                actual_usage_dialect: UsageDialect,
                completed_metadata: dict[str, Any],
                model_attempt_identity: ModelAttemptIdentity,
            ) -> Event:
                provider_name = actual_pricing_provider_name
                compactor = compactor_name
                try:
                    payload = _compaction_model_completed_payload(
                        completed_payload=completed_metadata,
                        provider_name=provider_name,
                        fallback_model=actual_model,
                        compactor=compactor,
                        usage_dialect=actual_usage_dialect,
                    )
                except _CompactionAccountingUsageError as exc:
                    # A provider call still completed when normalized counters
                    # overflowed the durable int64 contract. Preserve the safe
                    # rejected-usage evidence so its reservation is settled at
                    # the reserved amount instead of being left uncertain.
                    payload = exc.payload
                durable_payload = _durable_compaction_completion_evidence(
                    payload,
                    provider_name=provider_name,
                    fallback_model=actual_model,
                    compactor=compactor,
                )
                durable_payload.update(
                    copy_model_attempt_identity(model_attempt_identity).payload()
                )
                return event_with_execution_profile_authority(
                    Event(
                        type=EventType.MODEL_COMPLETED,
                        session_id=loaded_session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload=durable_payload,
                    ),
                    compaction_execution_profile,
                )

            async def publish_compaction_completions(
                payloads: list[dict[str, Any]],
            ) -> None:
                pending: list[tuple[str, Event]] = []
                for raw_payload in payloads:
                    payload = identify_compaction_completion_payload(raw_payload)
                    execution_identity = ModelAttemptIdentity.model_validate(
                        {
                            "model_step_id": payload.get("model_step_id"),
                            "model_attempt_id": payload.get("model_attempt_id"),
                        }
                    )
                    compaction_attempt_id = payload.get(_COMPACTION_ATTEMPT_ID_KEY)
                    if type(compaction_attempt_id) is not str:
                        raise RuntimeError(
                            "Compaction completion evidence lost its attempt identity."
                        )
                    if compaction_attempt_id in materialized_compaction_attempt_ids:
                        continue
                    pending.append(
                        (
                            compaction_attempt_id,
                            _application_compaction_event(
                                telemetry=ContextCompactionTelemetry(
                                    event_type=EventType.MODEL_COMPLETED,
                                    payload=payload,
                                ),
                                request=request,
                                operation_id=operation_id,
                                attempt_id=attempt_id,
                                session=loaded_session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                compactor=compactor_name,
                                execution_identity=execution_identity,
                            ),
                        )
                    )
                if not pending and not deferred_dispatch_settlement_events:
                    return

                completion_events = governed_compaction_events(
                    event for _attempt_id, event in pending
                )
                completion_event_attempt_ids.update(
                    (event.id, compaction_attempt_id) for compaction_attempt_id, event in pending
                )
                events = [
                    *completion_events,
                    *deferred_dispatch_settlement_events,
                ]

                def record_durable_completion_batch() -> None:
                    materialized_compaction_attempt_ids.update(
                        compaction_attempt_id for compaction_attempt_id, _event in pending
                    )
                    observed_dispatch_completion_events.extend(completion_events)
                    prepublished_dispatch_events.extend(events)
                    deferred_dispatch_settlement_events.clear()

                try:
                    await self._persist_compaction_attempt_events(
                        session=loaded_session,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        events=events,
                        heartbeat_state=operation_heartbeat_state,
                        persisted_event_ids=persisted_attempt_event_ids,
                        event_inventory=attempt_event_inventory,
                        unresolved_store_tasks=unresolved_attempt_publication_tasks,
                        cost_bearing_store_tasks=(unresolved_completion_publication_tasks),
                    )
                except BaseException:
                    # The helper records IDs only after an ambiguous write is
                    # confirmed atomic and durable. Preserve that exact attempt
                    # identity before propagating the lost acknowledgement.
                    if all(event.id in persisted_attempt_event_ids for event in events):
                        record_durable_completion_batch()
                    raise
                record_durable_completion_batch()
                (
                    fan_out_error,
                    cancellation,
                ) = await self._fan_out_reconciled_session_operation_events(events)
                if cancellation is not None:
                    if fan_out_error is not None:
                        cancellation.add_note(
                            "Explicit compaction completion side-effect delivery also "
                            f"failed during cancellation: {type(fan_out_error).__name__}: "
                            f"{fan_out_error}"
                        )
                        raise cancellation from fan_out_error
                    raise cancellation
                if fan_out_error is not None:
                    raise fan_out_error

            async def settle_provider_dispatch(
                dispatch_reservations: list[BudgetStepReservation],
                *,
                completed_event: Event | None,
                uncertain_reason: str,
                release_reason: str | None = None,
            ) -> None:
                dispatch_reservation_ids = {
                    reservation.record.reservation_id for reservation in dispatch_reservations
                }
                if release_reason is not None:
                    settlement = self._release_compaction_budget_reservations(
                        dispatch_reservations,
                        session=loaded_session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        compactor=compactor_name,
                        reason=release_reason,
                    )
                elif completed_event is None:
                    settlement = self._reconcile_uncertain_compaction_budget_reservations(
                        dispatch_reservations,
                        session=loaded_session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        compactor=compactor_name,
                    )
                else:
                    settlement = self._reconcile_compaction_budget_reservations(
                        dispatch_reservations,
                        model_completed_events=[completed_event],
                        session=loaded_session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        compactor=compactor_name,
                    )
                settlement_events: list[Event] = []
                try:
                    async for event in settlement:
                        settlement_events.append(event)
                except Exception as exc:
                    exc.add_note(uncertain_reason)
                    raise
                finally:
                    # Provider completion telemetry is finalized by the context
                    # policy (including outcome annotations such as
                    # ``invalid_summary``). Keep reconciliation events in memory
                    # until that authoritative completion evidence can be
                    # published first.
                    deferred_dispatch_settlement_events.extend(settlement_events)
                    unsettled_ids = {
                        reservation.record.reservation_id for reservation in dispatch_reservations
                    }
                    settled_ids = dispatch_reservation_ids - unsettled_ids
                    budget_reservations[:] = [
                        reservation
                        for reservation in budget_reservations
                        if reservation.record.reservation_id not in settled_ids
                    ]

            async def _run_provider_dispatch(
                provider: ModelProvider,
                actual_provider_name: str,
                actual_pricing_provider_name: str,
                actual_model: str,
                actual_usage_dialect: UsageDialect,
                billing_identity: BillingIdentity | None,
                model_request: ModelRequest,
                dispatch_attempt: int,
                max_dispatch_attempts: int,
                dispatch: Callable[[], Awaitable[tuple[str, dict[str, Any]]]],
                *,
                model_attempt_identity: ModelAttemptIdentity,
            ) -> tuple[str, dict[str, Any]]:
                model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
                validate_live_compaction_semantics()
                actual_pricing_provider_name = _require_application_compaction_event_text(
                    actual_pricing_provider_name,
                    "compactor_provider_name",
                )
                actual_model = _require_application_compaction_event_text(
                    actual_model,
                    "compactor_model",
                )
                actual_usage_dialect = copy_usage_dialect(
                    actual_usage_dialect,
                    "provider.usage_dialect",
                )
                if actual_pricing_provider_name != compactor_provider_name:
                    raise RuntimeError(
                        "Compaction dispatch provider identity differs from its admitted identity."
                    )
                if actual_model != compactor_model:
                    raise RuntimeError(
                        "Compaction dispatch model identity differs from its admitted identity."
                    )
                limit_decision = await self._run_limit_controller.evaluate_operation_run_limit(
                    session=loaded_session,
                    limits=request.limits,
                    operation_events=observed_dispatch_completion_events,
                    operation_started_at=operation_started_at,
                )
                if limit_decision is not None:
                    raise RuntimeError(f"Compaction limit reached: {limit_decision.message}")
                budget_operation_events = [
                    *attempt_events,
                    *observed_dispatch_completion_events,
                ]
                budget_operation_event_count = len(budget_operation_events)
                budget_error = await self._enforce_compaction_budget_limits(
                    session=loaded_session,
                    budget_limits=budget_limits,
                    app_policy_budget_limits=app_policy_budget_limits,
                    attempt_events=budget_operation_events,
                    reached_budget_keys=reached_budget_keys,
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    compactor=compactor_name,
                    provider_name=actual_pricing_provider_name,
                    model=actual_model,
                    execution_identity=model_attempt_identity,
                    billing_identity_state=resolved_billing_identity(billing_identity),
                )
                attempt_events.extend(budget_operation_events[budget_operation_event_count:])
                if budget_error is not None:
                    await publish_dispatch_budget_events()
                    raise budget_error

                reservation_start = len(budget_reservations)
                reservation_failure = await self._reserve_compaction_budget(
                    session=loaded_session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    budget_limits=dispatch_settlement_budget_limits,
                    provider_name=actual_pricing_provider_name,
                    model=actual_model,
                    model_attempt_identity=model_attempt_identity,
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    compactor=compactor_name,
                    reservations=budget_reservations,
                    events=attempt_events,
                    billing_identity=billing_identity,
                    reservation_identity_guard=reservation_identity_guard,
                    execution_profile_fingerprint=(compaction_execution_profile.fingerprint),
                )
                dispatch_reservations = budget_reservations[reservation_start:]
                if reservation_failure is not None:
                    attempt_events.append(
                        _application_compaction_ledger_event(
                            event_type=EventType.BUDGET_LIMIT_REACHED,
                            payload=budget_reservation_payload(reservation_failure),
                            request=request,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            session=loaded_session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            compactor=compactor_name,
                        )
                    )
                    await publish_dispatch_budget_events()
                    raise RuntimeError(
                        f"Compaction budget reservation failed: {reservation_failure.message}"
                    )
                await publish_dispatch_budget_events()

                provider_dispatch_started = False

                async def tracked_dispatch() -> tuple[str, dict[str, Any]]:
                    nonlocal provider_dispatch_started
                    validate_live_compaction_semantics()
                    await record_compaction_footprint(
                        provider=provider,
                        provider_name=actual_provider_name,
                        model_request=model_request,
                        dispatch_attempt=dispatch_attempt,
                        max_dispatch_attempts=max_dispatch_attempts,
                        model_attempt_identity=model_attempt_identity,
                    )
                    validate_live_compaction_semantics()
                    deferred_dispatch_failure = (
                        await self._run_limit_controller.mark_reservations_dispatched(
                            dispatch_reservations,
                            dispatch_id=model_attempt_identity.model_attempt_id,
                        )
                    )
                    if deferred_dispatch_failure is not None:
                        provider_dispatch_started = True
                        raise deferred_dispatch_failure
                    validate_live_compaction_semantics()
                    provider_dispatch_started = True
                    with _compaction_model_attempt_identity_scope(model_attempt_identity):
                        return await dispatch()

                try:
                    (
                        dispatch_result,
                        lease_failure,
                    ) = await self._run_limit_controller.run_operation_with_reservation_heartbeat(
                        tracked_dispatch,
                        reservations=dispatch_reservations,
                        authoritative_failure_types=(ContextBuildError,),
                        lease_lost_before_dispatch_message=(
                            "Compaction budget reservation lease was lost before provider dispatch."
                        ),
                        authoritative_failure_note=(
                            "Budget reservation lease was also lost as compaction failed"
                        ),
                        concurrent_failure_note=(
                            "Compactor also failed while reservation lease loss was handled"
                        ),
                        completed_metadata_from_result=lambda completed: completed[1],
                    )
                except asyncio.CancelledError as exc:
                    raw_completed = getattr(exc, "completed_metadata", None)
                    completed_event = (
                        dispatch_completion_event(
                            actual_pricing_provider_name=actual_pricing_provider_name,
                            actual_model=actual_model,
                            actual_usage_dialect=actual_usage_dialect,
                            completed_metadata=raw_completed,
                            model_attempt_identity=model_attempt_identity,
                        )
                        if type(raw_completed) is dict
                        else None
                    )

                    async def settle_cancelled_dispatch() -> None:
                        await settle_provider_dispatch(
                            dispatch_reservations,
                            completed_event=completed_event,
                            uncertain_reason=(
                                "Compaction provider dispatch settlement failed "
                                "during cancellation."
                            ),
                            release_reason=(
                                None
                                if provider_dispatch_started
                                else "compaction cancelled before provider dispatch"
                            ),
                        )
                        await publish_dispatch_budget_events()

                    settlement_task = asyncio.create_task(settle_cancelled_dispatch())
                    settlement_failure: BaseException | None = None
                    while not settlement_task.done():
                        try:
                            await asyncio.shield(settlement_task)
                        except asyncio.CancelledError:
                            continue
                        except BaseException as candidate:
                            settlement_failure = candidate
                            break
                    if settlement_failure is None:
                        try:
                            settlement_task.result()
                        except BaseException as candidate:
                            settlement_failure = candidate
                    if settlement_failure is not None:
                        exc.add_note(
                            "Compaction provider cancellation settlement also failed: "
                            f"{type(settlement_failure).__name__}: {settlement_failure}"
                        )
                    raise
                except BaseException as exc:
                    raw_completed = getattr(exc, "completed_metadata", None)
                    completed_event = (
                        dispatch_completion_event(
                            actual_pricing_provider_name=actual_pricing_provider_name,
                            actual_model=actual_model,
                            actual_usage_dialect=actual_usage_dialect,
                            completed_metadata=raw_completed,
                            model_attempt_identity=model_attempt_identity,
                        )
                        if type(raw_completed) is dict
                        else None
                    )
                    try:
                        await settle_provider_dispatch(
                            dispatch_reservations,
                            completed_event=completed_event,
                            uncertain_reason=(
                                "Compaction provider dispatch settlement failed after "
                                "provider error."
                            ),
                            release_reason=(
                                "compaction reservation lease lost before provider dispatch"
                                if not provider_dispatch_started
                                else None
                            ),
                        )
                        await publish_dispatch_budget_events()
                    except BaseException as settlement_failure:
                        if isinstance(settlement_failure, Exception):
                            add_budget_failure_note(
                                exc,
                                operation="compaction provider dispatch settlement",
                                accounting_failure=settlement_failure,
                            )
                        else:
                            exc.add_note(
                                "Compaction provider dispatch settlement also ended with "
                                f"{type(settlement_failure).__name__}; "
                                "the original failure remains authoritative."
                            )
                    raise

                completed_event = dispatch_completion_event(
                    actual_pricing_provider_name=actual_pricing_provider_name,
                    actual_model=actual_model,
                    actual_usage_dialect=actual_usage_dialect,
                    completed_metadata=dispatch_result[1],
                    model_attempt_identity=model_attempt_identity,
                )

                async def settle_completed_dispatch() -> None:
                    await settle_provider_dispatch(
                        dispatch_reservations,
                        completed_event=completed_event,
                        uncertain_reason=(
                            "Compaction provider dispatch settlement failed after completion."
                        ),
                    )
                    await publish_dispatch_budget_events()

                settlement_task = asyncio.create_task(settle_completed_dispatch())
                settlement_outcome = await await_shielded_task_outcome(settlement_task)
                settlement_cancellation = settlement_outcome.cancellation
                if settlement_outcome.error is not None:
                    settlement_failure = settlement_outcome.error
                    if (
                        isinstance(settlement_failure, asyncio.CancelledError)
                        and settlement_cancellation is None
                    ):
                        settlement_failure = unexpected_child_cancellation_error(
                            settlement_failure,
                            operation="Completed compaction provider dispatch settlement",
                        )
                    if settlement_cancellation is not None:
                        settlement_cancellation.__dict__["completed_metadata"] = copy_json_value(
                            dispatch_result[1],
                            "completed_metadata",
                        )
                        settlement_cancellation.add_note(
                            "Completed compaction provider dispatch settlement also failed: "
                            f"{type(settlement_failure).__name__}: {settlement_failure}"
                        )
                        raise settlement_cancellation from settlement_failure
                    raise settlement_failure
                if settlement_outcome.result is not None:
                    raise RuntimeError(
                        "Completed compaction provider dispatch settlement returned an "
                        "unexpected result."
                    )
                if settlement_cancellation is not None:
                    settlement_cancellation.__dict__["completed_metadata"] = copy_json_value(
                        dispatch_result[1],
                        "completed_metadata",
                    )
                    raise settlement_cancellation
                if lease_failure is not None:
                    raise lease_failure
                return dispatch_result

            async def run_provider_dispatch(
                provider: ModelProvider,
                actual_provider_name: str,
                actual_pricing_provider_name: str,
                actual_model: str,
                actual_usage_dialect: UsageDialect,
                billing_identity: BillingIdentity | None,
                model_request: ModelRequest,
                dispatch_attempt: int,
                max_dispatch_attempts: int,
                dispatch: Callable[[], Awaitable[tuple[str, dict[str, Any]]]],
            ) -> tuple[str, dict[str, Any]]:
                nonlocal active_compaction_model_attempt_identity
                validate_live_compaction_semantics()
                if active_compaction_model_attempt_identity is not None:
                    raise RuntimeError("Compaction provider dispatches cannot overlap.")
                model_attempt_identity = model_step_identity.new_attempt()
                active_compaction_model_attempt_identity = model_attempt_identity
                issued_compaction_model_attempts[model_attempt_identity.model_attempt_id] = (
                    model_attempt_identity
                )
                try:
                    return await _run_provider_dispatch(
                        provider,
                        actual_provider_name,
                        actual_pricing_provider_name,
                        actual_model,
                        actual_usage_dialect,
                        billing_identity,
                        model_request,
                        dispatch_attempt,
                        max_dispatch_attempts,
                        dispatch,
                        model_attempt_identity=model_attempt_identity,
                    )
                finally:
                    if active_compaction_model_attempt_identity != model_attempt_identity:
                        raise RuntimeError(
                            "Compaction provider dispatch identity changed while active."
                        )
                    active_compaction_model_attempt_identity = None

            async def run_identity_only_dispatch(
                provider: ModelProvider,
                actual_provider_name: str,
                actual_pricing_provider_name: str,
                actual_model: str,
                actual_usage_dialect: UsageDialect,
                billing_identity: BillingIdentity | None,
                model_request: ModelRequest,
                dispatch_attempt: int,
                max_dispatch_attempts: int,
                dispatch: Callable[[], Awaitable[tuple[str, dict[str, Any]]]],
            ) -> tuple[str, dict[str, Any]]:
                """Identify a built-in dispatch reached through an opaque compactor."""

                del (
                    actual_pricing_provider_name,
                    actual_model,
                    actual_usage_dialect,
                    billing_identity,
                )
                nonlocal active_compaction_model_attempt_identity
                validate_live_compaction_semantics()
                if has_compaction_accounting_limits:
                    raise RuntimeError(
                        "Explicit compaction declared deterministic execution but "
                        "attempted provider work under run or cost limits."
                    )
                if active_compaction_model_attempt_identity is not None:
                    raise RuntimeError("Compaction provider dispatches cannot overlap.")
                model_attempt_identity = model_step_identity.new_attempt()
                active_compaction_model_attempt_identity = model_attempt_identity
                issued_compaction_model_attempts[model_attempt_identity.model_attempt_id] = (
                    model_attempt_identity
                )
                try:
                    await record_compaction_footprint(
                        provider=provider,
                        provider_name=actual_provider_name,
                        model_request=model_request,
                        dispatch_attempt=dispatch_attempt,
                        max_dispatch_attempts=max_dispatch_attempts,
                        model_attempt_identity=model_attempt_identity,
                    )
                    validate_live_compaction_semantics()
                    with _compaction_model_attempt_identity_scope(model_attempt_identity):
                        return await dispatch()
                finally:
                    if active_compaction_model_attempt_identity != model_attempt_identity:
                        raise RuntimeError(
                            "Compaction provider dispatch identity changed while active."
                        )
                    active_compaction_model_attempt_identity = None

            async def execute_compaction() -> ContextBuildResult:
                try:
                    with (
                        _compaction_completion_publisher_scope(publish_compaction_completions),
                        _context_secret_redactor_scope(self._secret_redactor),
                        _defer_billing_identity_cancellation_scope(),
                    ):
                        result = await context_policy.build_with_checkpoint(
                            ContextRequest(
                                session=loaded_session,
                                agent=_session_agent_spec(
                                    registered_agent=registered_agent,
                                    session=loaded_session,
                                ),
                                messages=transcript,
                                step=1,
                                environment_name=environment_name,
                                metadata={
                                    "operation_id": operation_id,
                                    "reason": request.reason,
                                },
                                force_compaction=True,
                                force_bounded_compaction=True,
                                compaction_instructions=request.instructions,
                            ),
                            checkpoint=compaction_invocation_checkpoint,
                        )
                except ContextBuildError as error:
                    sanitize_context_build_error_checkpoint(
                        error,
                        redactor=self._secret_redactor,
                    )
                    raise
                safe_checkpoint, safe_checkpoint_event_payload = (
                    sanitize_context_build_result_checkpoint(
                        result,
                        redactor=self._secret_redactor,
                    )
                )
                return result.model_copy(
                    update={
                        "checkpoint": safe_checkpoint,
                        "checkpoint_event_payload": safe_checkpoint_event_payload,
                    },
                    deep=True,
                )

            async def execute_compaction_with_limits() -> tuple[
                ContextBuildResult,
                BaseException | None,
            ]:
                dispatch_runner = (
                    run_provider_dispatch
                    if compactor_provider_name is not None
                    else run_identity_only_dispatch
                )
                with _automatic_compaction_dispatch_runner_scope(dispatch_runner):
                    if compactor_provider_name is not None:
                        return await execute_compaction(), None
                    return (
                        await self._run_limit_controller.run_operation_with_reservation_heartbeat(
                            execute_compaction,
                            reservations=budget_reservations,
                            authoritative_failure_types=(ContextBuildError,),
                            lease_lost_before_dispatch_message=(
                                "Compaction budget reservation lease was lost before provider "
                                "dispatch."
                            ),
                            authoritative_failure_note=(
                                "Budget reservation lease was also lost as compaction failed"
                            ),
                            concurrent_failure_note=(
                                "Compactor also failed while reservation lease loss was handled"
                            ),
                        )
                    )

            (
                result,
                reservation_lease_failure,
            ) = await _run_while_session_operation_claimed(
                execute_compaction_with_limits,
                heartbeat_task=operation_heartbeat_task,
            )
            if result.checkpoint is None or result.checkpoint_event_payload is None:
                raise ValueError("Session has no complete older context to compact.")

            telemetry_events = [
                application_compaction_telemetry_event(telemetry)
                for telemetry in result.compaction_telemetry
                if telemetry.event_type != EventType.CONTEXT_COMPACTION_STARTED
                and not (
                    telemetry.event_type == EventType.MODEL_COMPLETED
                    and telemetry.payload.get(_COMPACTION_ATTEMPT_ID_KEY)
                    in materialized_compaction_attempt_ids
                )
            ]
            attempt_events.extend(
                event for event in telemetry_events if event.type == EventType.MODEL_COMPLETED
            )
            materialized_compaction_attempt_ids.update(
                completion_attempt_id
                for telemetry in result.compaction_telemetry
                if telemetry.event_type == EventType.MODEL_COMPLETED
                and type(completion_attempt_id := telemetry.payload.get(_COMPACTION_ATTEMPT_ID_KEY))
                is str
            )
            attempt_events.extend(deferred_dispatch_settlement_events)
            deferred_dispatch_settlement_events.clear()
            async for event in self._reconcile_compaction_budget_reservations(
                budget_reservations,
                model_completed_events=[
                    event for event in attempt_events if event.type == EventType.MODEL_COMPLETED
                ],
                session=loaded_session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                compactor=compactor_name,
            ):
                attempt_events.append(event)
            budget_reservations_settled = True
            if reservation_lease_failure is not None:
                raise reservation_lease_failure
            limit_decision = await self._run_limit_controller.evaluate_operation_run_limit(
                session=loaded_session,
                limits=request.limits,
                operation_events=[
                    *attempt_events,
                    *observed_dispatch_completion_events,
                ],
                operation_started_at=operation_started_at,
            )
            if limit_decision is not None:
                raise RuntimeError(f"Compaction limit reached: {limit_decision.message}")
            final_budget_operation_events = [
                *attempt_events,
                *observed_dispatch_completion_events,
            ]
            final_budget_operation_event_count = len(final_budget_operation_events)
            budget_error = await self._enforce_compaction_budget_limits(
                session=loaded_session,
                budget_limits=budget_limits,
                app_policy_budget_limits=app_policy_budget_limits,
                attempt_events=final_budget_operation_events,
                reached_budget_keys=reached_budget_keys,
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                registered_agent=registered_agent,
                environment_name=environment_name,
                compactor=compactor_name,
                provider_name=compactor_provider_name,
                model=compactor_model,
                execution_identity=model_step_identity,
            )
            attempt_events.extend(
                final_budget_operation_events[final_budget_operation_event_count:]
            )
            if budget_error is not None:
                raise budget_error
            checkpoint_event = Event(
                type=EventType.SESSION_CHECKPOINTED,
                session_id=loaded_session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload={
                    **copy_json_value(
                        result.checkpoint_event_payload,
                        "checkpoint_event_payload",
                    ),
                    **model_step_identity.payload(),
                    **_application_compaction_causal_payload(
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        source_cursor=request.expected_transcript_cursor,
                        result_cursor=result.checkpoint_event_payload.get(
                            "compacted_transcript_cursor"
                        ),
                        compactor=compactor_name,
                    ),
                },
            )
            published_events = self._event_writer.prepare_many(
                attribute_events_to_current_interaction(
                    governed_compaction_events(
                        [
                            *attempt_events,
                            *[
                                event
                                for event in telemetry_events
                                if event.type != EventType.MODEL_COMPLETED
                            ],
                            checkpoint_event,
                        ]
                    )
                )
            )
            event_ids = [started_event.id, *[event.id for event in published_events]]
            try:
                heartbeat_blocker = await stop_claim_heartbeat()
            except BaseException as heartbeat_failure:
                _attach_context_build_termination_diagnostics(
                    heartbeat_failure,
                    compaction_telemetry=result.compaction_telemetry,
                )
                raise heartbeat_failure
            if heartbeat_blocker is not None:
                heartbeat_failure = SessionCompactionAttemptSuperseded(
                    "Session compaction operation heartbeat could not stop before "
                    "terminal publication."
                )
                _attach_context_build_termination_diagnostics(
                    heartbeat_failure,
                    compaction_telemetry=result.compaction_telemetry,
                )
                raise heartbeat_failure
            terminal_claim_expires_at: datetime | None = None

            def capture_terminal_claim_expiry(expires_at: datetime) -> None:
                nonlocal terminal_claim_expires_at
                terminal_claim_expires_at = expires_at

            def require_unexpired_terminal_commit() -> None:
                if terminal_claim_expires_at is None:
                    raise AssertionError(
                        "Session compaction terminal publication did not capture its claim."
                    )
                _require_unexpired_session_operation_commit(
                    clock=self._clock,
                    claim_expires_at=terminal_claim_expires_at,
                    operation="Session compaction terminal publication",
                )

            async def publish_terminal_success() -> None:
                await self.session_store.publish_session_operation_guarded(
                    loaded_session.id,
                    idempotency_key=request.idempotency_key,
                    operation_transform=lambda _session, checkpoint, persisted_record: (
                        _complete_session_operation_checkpoint(
                            checkpoint=checkpoint,
                            persisted_record=persisted_record,
                            compacted_checkpoint=result.checkpoint,
                            idempotency_key=request.idempotency_key,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            event_ids=event_ids,
                            result_cursor=result.checkpoint_event_payload.get(
                                "compacted_transcript_cursor"
                            ),
                            completed_at=self._clock(),
                            on_terminalize=capture_terminal_claim_expiry,
                        )
                    ),
                    commit_guard=require_unexpired_terminal_commit,
                    events=published_events,
                    expected_statuses=_RESUMABLE_SESSION_STATUSES,
                    expected_run_epoch=request.expected_run_epoch,
                    expected_transcript_cursor=request.expected_transcript_cursor,
                )

            terminal_publication_task = asyncio.create_task(publish_terminal_success())
            terminal_outcome = await self._await_session_operation_store_task(
                terminal_publication_task,
                claim_expires_at=operation_heartbeat_state.confirmed_claim_expires_at,
            )
            terminal_cancellation = terminal_outcome.cancellation
            terminal_publication_unknown = terminal_outcome.timed_out
            if terminal_publication_unknown:
                publication_failure: BaseException | None = TimeoutError(
                    "Session compaction terminal publication exceeded its bounded store wait."
                )
            else:
                publication_failure = terminal_outcome.error
            if isinstance(publication_failure, asyncio.CancelledError):
                publication_failure = unexpected_child_cancellation_error(
                    publication_failure,
                    operation="Session compaction terminal publication",
                )
            if publication_failure is not None:

                async def reconcile_terminal_success() -> tuple[
                    list[bool],
                    dict[str, Any] | None,
                    dict[str, Any] | None,
                ]:
                    commit_states = [
                        await self._event_writer.is_persisted(event) for event in published_events
                    ]
                    operation_record = await self.session_store.load_session_operation(
                        loaded_session.id,
                        request.idempotency_key,
                    )
                    checkpoint = await self.session_store.load_checkpoint(loaded_session.id)
                    return commit_states, operation_record, checkpoint

                reconciliation_task = asyncio.create_task(reconcile_terminal_success())
                reconciliation_outcome = await self._await_session_operation_store_task(
                    reconciliation_task,
                    cancellation=terminal_cancellation,
                    claim_expires_at=operation_heartbeat_state.confirmed_claim_expires_at,
                )
                terminal_cancellation = reconciliation_outcome.cancellation
                reconciliation_error: BaseException | None
                if reconciliation_outcome.timed_out:
                    reconciliation_error = TimeoutError(
                        "Session compaction terminal publication reconciliation exceeded "
                        "its bounded store wait."
                    )
                else:
                    reconciliation_error = reconciliation_outcome.error
                if isinstance(reconciliation_error, asyncio.CancelledError):
                    reconciliation_error = unexpected_child_cancellation_error(
                        reconciliation_error,
                        operation="Session compaction terminal publication reconciliation",
                    )
                if reconciliation_error is not None:
                    if terminal_publication_unknown:
                        # The guarded write can still resolve after this caller
                        # returns. Leave the durable claim for replay/recovery
                        # instead of racing it with a contradictory failure.
                        operation_published = True
                    publication_failure.add_note(
                        "Session compaction terminal publication reconciliation also "
                        f"failed: {type(reconciliation_error).__name__}: "
                        f"{reconciliation_error}"
                    )
                    if terminal_cancellation is not None:
                        terminal_cancellation.add_note(
                            "Terminal publication and reconciliation failed during cancellation."
                        )
                        raise terminal_cancellation from publication_failure
                    _attach_context_build_termination_diagnostics(
                        publication_failure,
                        compaction_telemetry=result.compaction_telemetry,
                    )
                    raise publication_failure from reconciliation_error
                reconciled = reconciliation_outcome.result
                if reconciled is None:
                    raise AssertionError(
                        "Session compaction terminal reconciliation lost its result."
                    ) from publication_failure
                commit_states, operation_record, durable_checkpoint = reconciled
                durable_event_ids = (
                    operation_record.get("event_ids", []) if type(operation_record) is dict else []
                )
                operation_completed = (
                    type(operation_record) is dict
                    and operation_record.get("status") == "completed"
                    and operation_record.get("operation_id") == operation_id
                    and operation_record.get("current_attempt_id") == attempt_id
                    and operation_record.get("result_transcript_cursor")
                    == result.checkpoint_event_payload.get("compacted_transcript_cursor")
                    and all(event_id in durable_event_ids for event_id in event_ids)
                )
                expected_compaction_checkpoint = result.checkpoint.get(
                    _CONTEXT_COMPACTION_OPERATION_KIND
                )
                checkpoint_completed = (
                    type(durable_checkpoint) is dict
                    and durable_checkpoint.get(_CONTEXT_COMPACTION_OPERATION_KIND)
                    == expected_compaction_checkpoint
                    and _active_session_operation_id(durable_checkpoint) != operation_id
                )
                events_completed = all(commit_states)
                fully_committed = operation_completed and checkpoint_completed and events_completed
                anything_committed = (
                    operation_completed or checkpoint_completed or any(commit_states)
                )
                if anything_committed and not fully_committed:
                    operation_published = True
                    committed_events = [
                        event
                        for event, committed in zip(
                            published_events,
                            commit_states,
                            strict=True,
                        )
                        if committed
                    ]
                    (
                        fan_out_error,
                        terminal_cancellation,
                    ) = await self._fan_out_reconciled_session_operation_events(
                        committed_events,
                        cancellation=terminal_cancellation,
                    )
                    atomicity_error = RuntimeError(
                        "The session store violated atomic terminal compaction publication."
                    )
                    if fan_out_error is not None:
                        atomicity_error.add_note(
                            "Committed terminal event side-effect delivery also failed: "
                            f"{type(fan_out_error).__name__}: {fan_out_error}"
                        )
                    if terminal_cancellation is not None:
                        terminal_cancellation.add_note(str(atomicity_error))
                        raise terminal_cancellation from publication_failure
                    raise atomicity_error from publication_failure
                if fully_committed:
                    operation_published = True
                    (
                        fan_out_error,
                        terminal_cancellation,
                    ) = await self._fan_out_reconciled_session_operation_events(
                        published_events,
                        cancellation=terminal_cancellation,
                    )
                    if fan_out_error is not None:
                        publication_failure.add_note(
                            "Committed terminal event side-effect delivery also failed: "
                            f"{type(fan_out_error).__name__}: {fan_out_error}"
                        )
                    publication_failure.add_note(
                        "Session compaction completed durably before its publication "
                        "acknowledgement failed; retry will replay the committed result."
                    )
                    if terminal_cancellation is not None:
                        raise terminal_cancellation from publication_failure
                    raise publication_failure
                _attach_context_build_termination_diagnostics(
                    publication_failure,
                    compaction_telemetry=result.compaction_telemetry,
                )
                if terminal_publication_unknown:
                    operation_published = True
                if terminal_cancellation is not None:
                    raise terminal_cancellation from publication_failure
                raise publication_failure
            operation_published = True
            (
                fan_out_error,
                terminal_cancellation,
            ) = await self._fan_out_reconciled_session_operation_events(
                published_events,
                cancellation=terminal_cancellation,
            )
            if terminal_cancellation is not None:
                if fan_out_error is not None:
                    terminal_cancellation.add_note(
                        "Terminal event side-effect delivery also failed: "
                        f"{type(fan_out_error).__name__}: {fan_out_error}"
                    )
                raise terminal_cancellation
            if fan_out_error is not None:
                raise fan_out_error
            for event in prepublished_dispatch_events:
                yield event
            prepublished_dispatch_events.clear()
            for event in published_events:
                yield event
        except GeneratorExit as exc:
            operation_failure = exc
            if operation_published:
                raise
            termination_telemetry = context_build_termination_compaction_telemetry(exc)
            if not termination_telemetry:
                if budget_reservations and not budget_reservations_settled:
                    release_events: list[Event] = []
                    try:
                        async for event in self._release_compaction_budget_reservations(
                            budget_reservations,
                            session=loaded_session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            request=request,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            compactor=compactor_name,
                            reason="compaction operation abandoned",
                        ):
                            release_events.append(event)
                    finally:
                        if release_events:
                            await self._persist_compaction_attempt_events(
                                session=loaded_session,
                                request=request,
                                operation_id=operation_id,
                                attempt_id=attempt_id,
                                events=release_events,
                                heartbeat_state=operation_heartbeat_state,
                                persisted_event_ids=persisted_attempt_event_ids,
                                event_inventory=attempt_event_inventory,
                                unresolved_store_tasks=(unresolved_attempt_publication_tasks),
                            )
                            await self._event_writer.fan_out_persisted(release_events)
                        budget_reservations_settled = not budget_reservations
                raise
            try:
                failed_model_events = [
                    application_compaction_telemetry_event(telemetry)
                    for telemetry in termination_telemetry
                    if telemetry.event_type == EventType.MODEL_COMPLETED
                    and telemetry.payload.get(_COMPACTION_ATTEMPT_ID_KEY)
                    not in materialized_compaction_attempt_ids
                ]
                attempt_events.extend(failed_model_events)
                attempt_events.extend(deferred_dispatch_settlement_events)
                deferred_dispatch_settlement_events.clear()
                if budget_reservations and not budget_reservations_settled:
                    model_completed_events = [
                        event for event in attempt_events if event.type == EventType.MODEL_COMPLETED
                    ]
                    settlement_stream = (
                        self._reconcile_compaction_budget_reservations(
                            budget_reservations,
                            model_completed_events=model_completed_events,
                            session=loaded_session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            request=request,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            compactor=compactor_name,
                        )
                        if model_completed_events
                        else self._release_compaction_budget_reservations(
                            budget_reservations,
                            session=loaded_session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            request=request,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            compactor=compactor_name,
                            reason="compaction operation abandoned",
                        )
                    )
                    async for event in settlement_stream:
                        attempt_events.append(event)
                    budget_reservations_settled = True
                failed_telemetry = next(
                    (
                        telemetry
                        for telemetry in reversed(termination_telemetry)
                        if telemetry.event_type == EventType.CONTEXT_COMPACTION_FAILED
                    ),
                    None,
                )
                failed_payload = _application_compaction_causal_payload(
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    source_cursor=request.expected_transcript_cursor,
                    compactor=compactor_name,
                )
                failed_payload.update(model_step_identity.payload())
                if failed_telemetry is not None:
                    failed_payload = application_compaction_telemetry_event(
                        failed_telemetry
                    ).payload
                safe_error_type = (
                    _application_compaction_event_text(type(exc).__name__) or "BaseException"
                )
                failed_payload["error_type"] = safe_error_type
                failed_event = Event(
                    type=EventType.CONTEXT_COMPACTION_FAILED,
                    session_id=loaded_session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=failed_payload,
                )
                unpublished_attempt_events = [
                    event for event in attempt_events if event.id not in persisted_attempt_event_ids
                ]
                failed_events = self._event_writer.prepare_many(
                    governed_compaction_events([*unpublished_attempt_events, failed_event])
                )
                failed_terminal_claim_expires_at: datetime | None = None

                def capture_failed_terminal_claim_expiry(expires_at: datetime) -> None:
                    nonlocal failed_terminal_claim_expires_at
                    failed_terminal_claim_expires_at = expires_at

                def require_unexpired_failed_terminal_commit() -> None:
                    if failed_terminal_claim_expires_at is None:
                        return
                    _require_unexpired_session_operation_commit(
                        clock=self._clock,
                        claim_expires_at=failed_terminal_claim_expires_at,
                        operation="Session compaction failure publication",
                    )

                await self.session_store.publish_session_operation_guarded(
                    loaded_session.id,
                    idempotency_key=request.idempotency_key,
                    operation_transform=_fail_session_operation_checkpoint(
                        idempotency_key=request.idempotency_key,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        failed_event_id=failed_event.id,
                        attempt_event_ids=[event.id for event in unpublished_attempt_events],
                        error_type=safe_error_type,
                        clock=self._clock,
                        on_terminalize=capture_failed_terminal_claim_expiry,
                    ),
                    commit_guard=require_unexpired_failed_terminal_commit,
                    events=failed_events,
                )
                operation_published = True
                await self._run_limit_controller.acknowledge_budget_settlement_events(failed_events)
                await self._event_writer.fan_out_persisted(failed_events)
            except BaseException as cleanup_error:
                cleanup_error.add_note(
                    "Compaction was abandoned by GeneratorExit, but its usage and "
                    "failure accounting could not be durably published. "
                    "The accounting failure is authoritative."
                )
                operation_failure = cleanup_error
                raise cleanup_error from exc
            raise
        except BaseException as exc:
            operation_failure = exc
            if operation_published:
                raise
            authoritative_failure = exc
            failed_events: list[Event] = []
            failure_accounting_published = False

            async def finalize_failed_operation() -> None:
                nonlocal budget_reservations_settled
                nonlocal failed_events
                nonlocal failure_accounting_published
                nonlocal operation_published

                failure_telemetry = (
                    authoritative_failure.compaction_telemetry
                    if isinstance(authoritative_failure, ContextBuildError)
                    else context_build_termination_compaction_telemetry(authoritative_failure)
                )
                if failure_telemetry:
                    failed_model_events = [
                        application_compaction_telemetry_event(telemetry)
                        for telemetry in failure_telemetry
                        if telemetry.event_type == EventType.MODEL_COMPLETED
                        and telemetry.payload.get(_COMPACTION_ATTEMPT_ID_KEY)
                        not in materialized_compaction_attempt_ids
                    ]
                    existing_completion_attempt_ids = {
                        completion_attempt_id
                        for event in attempt_events
                        if event.type == EventType.MODEL_COMPLETED
                        and type(
                            completion_attempt_id := event.payload.get(_COMPACTION_ATTEMPT_ID_KEY)
                        )
                        is str
                    }
                    for event in failed_model_events:
                        completion_attempt_id = event.payload.get(_COMPACTION_ATTEMPT_ID_KEY)
                        if (
                            type(completion_attempt_id) is str
                            and completion_attempt_id in existing_completion_attempt_ids
                        ):
                            continue
                        attempt_events.append(event)
                        if type(completion_attempt_id) is str:
                            existing_completion_attempt_ids.add(completion_attempt_id)
                attempt_events.extend(deferred_dispatch_settlement_events)
                deferred_dispatch_settlement_events.clear()
                if budget_reservations and not budget_reservations_settled:
                    model_completed_events = [
                        event for event in attempt_events if event.type == EventType.MODEL_COMPLETED
                    ]
                    if model_completed_events:
                        settlement_stream = self._reconcile_compaction_budget_reservations(
                            budget_reservations,
                            model_completed_events=model_completed_events,
                            session=loaded_session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            request=request,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            compactor=compactor_name,
                        )
                    elif isinstance(
                        authoritative_failure,
                        BudgetReservationLeaseLostBeforeModelDispatch,
                    ):
                        settlement_stream = self._release_compaction_budget_reservations(
                            budget_reservations,
                            session=loaded_session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            request=request,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            compactor=compactor_name,
                            reason="compaction reservation lease lost before provider dispatch",
                        )
                    elif isinstance(authoritative_failure, BudgetReservationLeaseLost):
                        settlement_stream = (
                            self._reconcile_uncertain_compaction_budget_reservations(
                                budget_reservations,
                                session=loaded_session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                request=request,
                                operation_id=operation_id,
                                attempt_id=attempt_id,
                                compactor=compactor_name,
                            )
                        )
                    else:
                        settlement_stream = self._release_compaction_budget_reservations(
                            budget_reservations,
                            session=loaded_session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            request=request,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            compactor=compactor_name,
                            reason="compaction provider step did not complete",
                        )
                    settlement_failure: BaseException | None = None
                    try:
                        async for event in settlement_stream:
                            attempt_events.append(event)
                    except BaseException as candidate:
                        settlement_failure = candidate
                    budget_reservations_settled = not budget_reservations
                    if settlement_failure is not None:
                        if isinstance(settlement_failure, Exception):
                            add_budget_failure_note(
                                authoritative_failure,
                                operation="compaction settlement",
                                accounting_failure=settlement_failure,
                            )
                        else:
                            authoritative_failure.add_note(
                                "Compaction settlement also ended with "
                                f"{type(settlement_failure).__name__}; "
                                "the original failure remains authoritative."
                            )
                failed_payload = _application_compaction_causal_payload(
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    source_cursor=request.expected_transcript_cursor,
                    compactor=compactor_name,
                )
                failed_payload.update(model_step_identity.payload())
                if failure_telemetry:
                    failed_telemetry = next(
                        (
                            telemetry
                            for telemetry in reversed(failure_telemetry)
                            if telemetry.event_type == EventType.CONTEXT_COMPACTION_FAILED
                        ),
                        None,
                    )
                    if failed_telemetry is not None:
                        failed_payload = application_compaction_telemetry_event(
                            failed_telemetry
                        ).payload
                safe_error_type = (
                    _application_compaction_event_text(type(authoritative_failure).__name__)
                    or "BaseException"
                )
                failed_payload["error_type"] = safe_error_type
                failed_event = Event(
                    type=EventType.CONTEXT_COMPACTION_FAILED,
                    session_id=loaded_session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=failed_payload,
                )
                unpublished_attempt_events = [
                    event for event in attempt_events if event.id not in persisted_attempt_event_ids
                ]
                failed_events = self._event_writer.prepare_many(
                    attribute_events_to_current_interaction(
                        governed_compaction_events([*unpublished_attempt_events, failed_event])
                    )
                )
                failed_terminal_claim_expires_at: datetime | None = None

                def capture_failed_terminal_claim_expiry(expires_at: datetime) -> None:
                    nonlocal failed_terminal_claim_expires_at
                    failed_terminal_claim_expires_at = expires_at

                def require_unexpired_failed_terminal_commit() -> None:
                    if failed_terminal_claim_expires_at is None:
                        return
                    _require_unexpired_session_operation_commit(
                        clock=self._clock,
                        claim_expires_at=failed_terminal_claim_expires_at,
                        operation="Session compaction failure publication",
                    )

                await self.session_store.publish_session_operation_guarded(
                    loaded_session.id,
                    idempotency_key=request.idempotency_key,
                    operation_transform=_fail_session_operation_checkpoint(
                        idempotency_key=request.idempotency_key,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        failed_event_id=failed_event.id,
                        attempt_event_ids=[event.id for event in unpublished_attempt_events],
                        error_type=safe_error_type,
                        clock=self._clock,
                        on_terminalize=capture_failed_terminal_claim_expiry,
                    ),
                    commit_guard=require_unexpired_failed_terminal_commit,
                    events=failed_events,
                )
                operation_published = True
                failure_accounting_published = True
                await self._run_limit_controller.acknowledge_budget_settlement_events(failed_events)
                await self._event_writer.fan_out_persisted(failed_events)

            async def stop_heartbeat_and_finalize_failed_operation() -> None:
                heartbeat_blocker = await stop_claim_heartbeat()
                uncertain_publication_tasks = set(unresolved_attempt_publication_tasks)
                publication_blockers = {
                    task for task in uncertain_publication_tasks if not task.done()
                }
                cost_bearing_publication_blockers = {
                    task for task in unresolved_completion_publication_tasks if not task.done()
                }
                if not uncertain_publication_tasks and heartbeat_blocker is None:
                    await finalize_failed_operation()
                    return

                store_blockers = set(publication_blockers)
                if heartbeat_blocker is not None:
                    store_blockers.add(heartbeat_blocker)

                async def finalize_after_pending_store_writes() -> None:
                    await asyncio.gather(*store_blockers, return_exceptions=True)
                    await asyncio.gather(operation_heartbeat_task, return_exceptions=True)
                    reconciled_events: list[Event] = []
                    existing_attempt_event_ids = {event.id for event in attempt_events}
                    for event in attempt_event_inventory.values():
                        durable = event.id in persisted_attempt_event_ids
                        if not durable:
                            durable = await self._event_writer.is_persisted(event)
                        if durable:
                            persisted_attempt_event_ids.add(event.id)
                            reconciled_events.append(event)
                            if (
                                all(
                                    published.id != event.id
                                    for published in prepublished_dispatch_events
                                )
                                and event.id not in yielded_attempt_event_ids
                            ):
                                prepublished_dispatch_events.append(event.model_copy(deep=True))
                            compaction_attempt_id = completion_event_attempt_ids.get(event.id)
                            if (
                                event.type == EventType.MODEL_COMPLETED
                                and type(compaction_attempt_id) is str
                            ):
                                materialized_compaction_attempt_ids.add(compaction_attempt_id)
                                if event.id not in existing_attempt_event_ids:
                                    attempt_events.append(event.model_copy(deep=True))
                                    existing_attempt_event_ids.add(event.id)
                                if all(
                                    observed.id != event.id
                                    for observed in observed_dispatch_completion_events
                                ):
                                    observed_dispatch_completion_events.append(
                                        event.model_copy(deep=True)
                                    )
                    fan_out_failure: BaseException | None = None
                    if reconciled_events:
                        try:
                            await self._event_writer.fan_out_persisted(reconciled_events)
                        except BaseException as exc:
                            fan_out_failure = exc
                    try:
                        await finalize_failed_operation()
                    except BaseException as finalization_failure:
                        if fan_out_failure is not None:
                            raise BaseExceptionGroup(
                                "Explicit compaction failure accounting and reconciled "
                                "event delivery both failed.",
                                [fan_out_failure, finalization_failure],
                            ) from None
                        raise
                    if fan_out_failure is not None:
                        raise fan_out_failure

                authoritative_failure.add_note(
                    "Explicit compaction failure accounting waited for an in-flight "
                    "operation write to release the session store."
                )
                if cost_bearing_publication_blockers:
                    # A completion publication can be the sole durable owner of
                    # already-incurred provider spend. Do not return while only
                    # volatile background work owns that evidence.
                    await finalize_after_pending_store_writes()
                    return

                if (
                    uncertain_publication_tasks
                    and not publication_blockers
                    and heartbeat_blocker is None
                ):
                    # A timed-out write can finish after its first reconciliation
                    # but before failure cleanup begins. Rescan the complete event
                    # inventory even when no task remains live so a late durable
                    # completion retains its original identity and accounting.
                    await finalize_after_pending_store_writes()
                    return

                deferred = asyncio.create_task(finalize_after_pending_store_writes())
                self._track_detached_session_operation_task(deferred)

            await _run_recovery_cleanup_steps(
                authoritative_failure=authoritative_failure,
                steps=(
                    (
                        "explicit compaction failure accounting",
                        stop_heartbeat_and_finalize_failed_operation,
                    ),
                ),
            )
            # Fatal BaseExceptions cannot safely yield from an async generator.
            # Ordinary exceptions retain the explicit API's durable replay stream.
            if isinstance(authoritative_failure, Exception) and failure_accounting_published:
                for event in prepublished_dispatch_events:
                    yield event
                prepublished_dispatch_events.clear()
                for event in failed_events:
                    yield event
            raise
        finally:
            await _run_recovery_cleanup_steps(
                authoritative_failure=operation_failure,
                steps=(
                    (
                        "explicit compaction claim heartbeat stop",
                        stop_claim_heartbeat_and_track,
                    ),
                ),
            )

    async def _load_session_compaction_replay_events(
        self,
        *,
        session_id: str,
        event_ids: tuple[str, ...],
    ) -> list[Event]:
        events: list[Event] = []
        for event_id in event_ids:
            records = await self.session_store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_id=event_id,
                    limit=1,
                )
            )
            if len(records) != 1:
                raise RuntimeError(
                    f"Session compaction replay event is missing from durable history: {event_id}"
                )
            events.append(copy_event(records[0].event))
        return events

    async def _complete_session_if_no_queued_messages(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
    ) -> tuple[Session, Event | None, bool]:
        try:
            return await self._publish_interaction_transition(
                session=session,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                to_status=SessionStatus.COMPLETED,
                only_if_no_queued_messages=True,
                from_statuses={SessionStatus.RUNNING},
            )
        except SessionStatusConflict:
            # An interrupt can win between the final provider response and the
            # atomic completion publication. Route that race through the normal
            # interrupt finalizer instead of the generic failure handler.
            await self._session_control.raise_if_interrupted(session.id)
            raise

    async def _handle_queued_messages_before_completion(
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
    ) -> tuple[bool, list[Event]]:
        if step < max_steps:
            return True, await self._deliver_queued_session_messages(
                session_id=session.id,
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                messages=messages,
                include_on_idle=True,
            )
        events = [
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
            )
        ]
        return False, events

    async def _enforce_compaction_budget_limits(
        self,
        *,
        session: Session,
        budget_limits: tuple[BudgetLimit, ...],
        app_policy_budget_limits: tuple[BudgetLimit, ...],
        attempt_events: list[Event],
        reached_budget_keys: set[str],
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        compactor: str,
        provider_name: str | None,
        model: str | None,
        execution_identity: ModelStepIdentity | ModelAttemptIdentity,
        billing_identity_state: BillingIdentityState = UNRESOLVED_BILLING_IDENTITY,
    ) -> RuntimeError | None:
        execution_payload = _model_execution_identity_payload(execution_identity)
        app_policy_budget_limit_ids = {
            _effective_budget_limit_id(limit) for limit in app_policy_budget_limits
        }
        checks = await self._run_limit_controller.evaluate_operation_budgets(
            session=session,
            budget_limits=budget_limits,
            operation_events=attempt_events,
            provider_name=provider_name,
            model=model,
            billing_identity_state=billing_identity_state,
        )
        interrupt_error: RuntimeError | None = None
        for outcome in checks:
            budget_limit, check = outcome.limit, outcome.check
            deferred_contextual_check = (
                not isinstance(billing_identity_state, ResolvedBillingIdentity)
                and not check.limit_reached
                and has_deferred_contextual_price(
                    budget_limit.pricing,
                    provider_name=provider_name,
                    model=model,
                )
            )
            if (
                check.budget_limit_id in app_policy_budget_limit_ids
                and not deferred_contextual_check
            ):
                attempt_events.append(
                    _application_compaction_ledger_event(
                        event_type=EventType.BUDGET_CHECKED,
                        payload={
                            **budget_check_payload(check),
                            **execution_payload,
                        },
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        compactor=compactor,
                    )
                )
            if not check.limit_reached:
                continue
            budget_key = _budget_check_identity(check)
            if budget_key not in reached_budget_keys:
                attempt_events.append(
                    _application_compaction_budget_event(
                        check=check,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        compactor=compactor,
                        execution_identity=execution_identity,
                    )
                )
                reached_budget_keys.add(budget_key)
            if budget_limit.action == "interrupt" and interrupt_error is None:
                interrupt_error = RuntimeError(f"Compaction budget limit reached: {check.message}")
        return interrupt_error

    async def _renew_compaction_operation_claim(
        self,
        *,
        session: Session,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        claim_expires_at: datetime,
        commit_started: threading.Event,
    ) -> datetime:
        renewed_until: datetime | None = None

        def require_unexpired_claim_commit() -> None:
            _require_unexpired_session_operation_commit(
                clock=self._clock,
                claim_expires_at=claim_expires_at,
                operation="Session compaction operation renewal",
            )
            commit_started.set()

        def renew(
            _session: Session,
            checkpoint: dict[str, Any] | None,
            persisted_record: dict[str, Any] | None,
        ) -> SessionOperationPublication:
            nonlocal renewed_until
            publication, renewed_until = _renew_session_operation_claim_publication(
                checkpoint=checkpoint,
                idempotency_key=request.idempotency_key,
                operation_id=operation_id,
                attempt_id=attempt_id,
                event_ids=[],
                renewed_at=self._clock(),
                persisted_record=persisted_record,
            )
            return publication

        await self.session_store.publish_session_operation_guarded(
            session.id,
            idempotency_key=request.idempotency_key,
            operation_transform=renew,
            commit_guard=require_unexpired_claim_commit,
            events=[],
            expected_statuses=_RESUMABLE_SESSION_STATUSES,
            expected_run_epoch=request.expected_run_epoch,
            expected_transcript_cursor=request.expected_transcript_cursor,
        )
        if renewed_until is None:
            raise AssertionError("Session compaction claim renewal did not update its lease.")
        return renewed_until

    async def _reconcile_compaction_operation_claim_expiry(
        self,
        *,
        session_id: str,
        idempotency_key: str,
        operation_id: str,
        attempt_id: str,
    ) -> datetime:
        checkpoint = await self.session_store.load_checkpoint(session_id)
        if checkpoint is None:
            raise SessionCompactionAttemptSuperseded(
                "Session compaction operation claim disappeared during renewal reconciliation."
            )
        operations = _session_operation_state(checkpoint)
        record = operations["records"].get(idempotency_key)
        if (
            type(record) is not dict
            or record.get("operation_id") != operation_id
            or record.get("status") != "running"
            or record.get("current_attempt_id") != attempt_id
            or operations.get("active_operation_id") != operation_id
        ):
            raise SessionCompactionAttemptSuperseded(
                "Session compaction operation ownership changed during renewal reconciliation."
            )
        claim_expires_at = _operation_claim_expiry(record)
        if claim_expires_at is None:
            raise RuntimeError(
                "Session compaction operation claim has no valid expiry after renewal."
            )
        return claim_expires_at

    async def _reconcile_compaction_operation_claim_before_deadline(
        self,
        *,
        session_id: str,
        idempotency_key: str,
        operation_id: str,
        attempt_id: str,
        local_claim_expires_at: datetime,
        state: _SessionOperationClaimHeartbeatState,
    ) -> datetime:
        confirmed_claim_expires_at = state.confirmed_claim_expires_at
        if (
            confirmed_claim_expires_at is not None
            and confirmed_claim_expires_at > local_claim_expires_at
        ):
            local_claim_expires_at = confirmed_claim_expires_at
        started_at = self._clock()
        if started_at.tzinfo is None or started_at.utcoffset() is None:
            raise ValueError(
                "Session compaction claim reconciliation clock must be timezone-aware."
            )
        remaining_seconds = (local_claim_expires_at - started_at).total_seconds()
        if remaining_seconds <= 0:
            raise SessionCompactionAttemptSuperseded(
                "Session compaction operation claim expired before reconciliation."
            )
        reconciliation_task = asyncio.create_task(
            self._reconcile_compaction_operation_claim_expiry(
                session_id=session_id,
                idempotency_key=idempotency_key,
                operation_id=operation_id,
                attempt_id=attempt_id,
            )
        )
        state.pending_store_task = reconciliation_task
        reconciliation_task.add_done_callback(state.observe_store_task)
        try:
            done, _pending = await asyncio.wait(
                {reconciliation_task},
                timeout=remaining_seconds,
            )
        except asyncio.CancelledError as cancellation:
            reconciliation_failure: BaseException | None = None
            if reconciliation_task.done():
                reconciliation_failure = _completed_session_operation_child_failure(
                    reconciliation_task,
                    operation="Session compaction claim reconciliation",
                )
            else:
                reconciliation_task.cancel()
            if reconciliation_failure is not None:
                cancellation.add_note(
                    "Session compaction claim reconciliation also failed during "
                    f"cancellation: {type(reconciliation_failure).__name__}: "
                    f"{reconciliation_failure}"
                )
                raise cancellation from reconciliation_failure
            raise
        if reconciliation_task not in done:
            reconciliation_task.cancel()
            confirmed_claim_expires_at = state.confirmed_claim_expires_at
            if (
                confirmed_claim_expires_at is not None
                and confirmed_claim_expires_at > local_claim_expires_at
            ):
                _require_unexpired_session_operation_claim(
                    clock=self._clock,
                    claim_expires_at=confirmed_claim_expires_at,
                    operation="Session compaction operation publication",
                )
                return confirmed_claim_expires_at
            raise SessionCompactionAttemptSuperseded(
                "Session compaction operation claim reconciliation was not confirmed "
                "before its lease deadline."
            )
        try:
            durable_claim_expires_at = reconciliation_task.result()
        except asyncio.CancelledError as child_cancellation:
            raise unexpected_child_cancellation_error(
                child_cancellation,
                operation="Session compaction claim reconciliation",
            ) from child_cancellation
        observed_at = self._clock()
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError(
                "Session compaction claim reconciliation clock must be timezone-aware."
            )
        if local_claim_expires_at <= observed_at:
            raise SessionCompactionAttemptSuperseded(
                "Session compaction operation claim expired during reconciliation."
            )
        _require_unexpired_session_operation_claim(
            clock=lambda: observed_at,
            claim_expires_at=durable_claim_expires_at,
            operation="Session compaction operation reconciliation",
        )
        state.confirm_claim(durable_claim_expires_at)
        return state.confirmed_claim_expires_at or durable_claim_expires_at

    async def _heartbeat_compaction_operation_claim(
        self,
        *,
        session: Session,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        claim_expires_at: datetime,
        stop: asyncio.Event,
        state: _SessionOperationClaimHeartbeatState,
    ) -> None:
        sleep_seconds = _SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=sleep_seconds)
            except TimeoutError:
                confirmed_claim_expires_at = state.confirmed_claim_expires_at
                if (
                    confirmed_claim_expires_at is not None
                    and confirmed_claim_expires_at > claim_expires_at
                ):
                    claim_expires_at = confirmed_claim_expires_at
                heartbeat_at = self._clock()
                if heartbeat_at.tzinfo is None or heartbeat_at.utcoffset() is None:
                    raise ValueError(
                        "Session compaction operation heartbeat clock must be timezone-aware."
                    ) from None
                try:
                    claim_expires_at = (
                        await self._reconcile_compaction_operation_claim_before_deadline(
                            session_id=session.id,
                            idempotency_key=request.idempotency_key,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            local_claim_expires_at=claim_expires_at,
                            state=state,
                        )
                    )
                except SessionCompactionAttemptSuperseded:
                    raise
                except Exception as reconciliation_failure:
                    failed_at = self._clock()
                    if failed_at.tzinfo is None or failed_at.utcoffset() is None:
                        raise ValueError(
                            "Session compaction operation heartbeat clock must be timezone-aware."
                        ) from reconciliation_failure
                    if claim_expires_at <= failed_at:
                        raise SessionCompactionAttemptSuperseded(
                            "Session compaction operation claim could not be reconciled "
                            "before its local lease deadline."
                        ) from reconciliation_failure
                renewal_started_at = self._clock()
                if renewal_started_at.tzinfo is None or renewal_started_at.utcoffset() is None:
                    raise ValueError(
                        "Session compaction operation heartbeat clock must be timezone-aware."
                    ) from None
                remaining_seconds = (claim_expires_at - renewal_started_at).total_seconds()
                if remaining_seconds <= 0:
                    raise SessionCompactionAttemptSuperseded(
                        "Session compaction operation claim expired before renewal started."
                    ) from None
                renewal_task = asyncio.create_task(
                    self._renew_compaction_operation_claim(
                        session=session,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        claim_expires_at=claim_expires_at,
                        commit_started=(renewal_commit_started := threading.Event()),
                    )
                )
                state.pending_store_task = renewal_task
                renewal_task.add_done_callback(state.observe_store_task)
                try:
                    done, _pending = await asyncio.wait(
                        {renewal_task},
                        timeout=remaining_seconds,
                    )
                    if renewal_task not in done:
                        if not renewal_commit_started.is_set():
                            renewal_task.cancel()
                        confirmed_claim_expires_at = state.confirmed_claim_expires_at
                        if (
                            confirmed_claim_expires_at is not None
                            and confirmed_claim_expires_at > claim_expires_at
                        ):
                            _require_unexpired_session_operation_claim(
                                clock=self._clock,
                                claim_expires_at=confirmed_claim_expires_at,
                                operation="Session compaction operation publication",
                            )
                            claim_expires_at = confirmed_claim_expires_at
                            now = self._clock()
                            if now.tzinfo is None or now.utcoffset() is None:
                                raise ValueError(
                                    "Session compaction operation heartbeat clock must be "
                                    "timezone-aware."
                                )
                            sleep_seconds = min(
                                _SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS,
                                max(0.0, (claim_expires_at - now).total_seconds()),
                            )
                            continue
                        raise SessionCompactionAttemptSuperseded(
                            "Session compaction operation claim renewal was not confirmed "
                            "before its lease deadline."
                        )
                    claim_expires_at = renewal_task.result()
                    _require_unexpired_session_operation_claim(
                        clock=self._clock,
                        claim_expires_at=claim_expires_at,
                        operation="Session compaction operation renewal",
                    )
                    state.confirm_claim(claim_expires_at)
                except asyncio.CancelledError as cancellation:
                    renewal_failure: BaseException | None = None
                    if renewal_task.done():
                        renewal_failure = _completed_session_operation_child_failure(
                            renewal_task,
                            operation="Session compaction claim renewal",
                        )
                    elif not renewal_commit_started.is_set():
                        renewal_task.cancel()
                    if renewal_failure is not None:
                        cancellation.add_note(
                            "Session compaction claim renewal also failed during "
                            f"cancellation: {type(renewal_failure).__name__}: "
                            f"{renewal_failure}"
                        )
                        raise cancellation from renewal_failure
                    raise
                except SessionCompactionAttemptSuperseded:
                    raise
                except Exception as exc:
                    try:
                        durable_claim_expires_at = (
                            await self._reconcile_compaction_operation_claim_before_deadline(
                                session_id=session.id,
                                idempotency_key=request.idempotency_key,
                                operation_id=operation_id,
                                attempt_id=attempt_id,
                                local_claim_expires_at=claim_expires_at,
                                state=state,
                            )
                        )
                    except SessionCompactionAttemptSuperseded as ownership_failure:
                        raise ownership_failure from exc
                    except Exception as reconciliation_failure:
                        exc.add_note(
                            "Session compaction claim renewal reconciliation also failed: "
                            f"{type(reconciliation_failure).__name__}: "
                            f"{reconciliation_failure}"
                        )
                    else:
                        if durable_claim_expires_at > claim_expires_at:
                            claim_expires_at = durable_claim_expires_at
                            now = self._clock()
                            if now.tzinfo is None or now.utcoffset() is None:
                                raise ValueError(
                                    "Session compaction operation heartbeat clock must be "
                                    "timezone-aware."
                                ) from exc
                            sleep_seconds = min(
                                _SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS,
                                max(0.0, (claim_expires_at - now).total_seconds()),
                            )
                            continue
                    now = self._clock()
                    if now.tzinfo is None or now.utcoffset() is None:
                        raise ValueError(
                            "Session compaction operation heartbeat clock must be timezone-aware."
                        ) from exc
                    if now >= claim_expires_at:
                        raise SessionCompactionAttemptSuperseded(
                            "Session compaction operation claim could not be renewed before expiry."
                        ) from exc
                    sleep_seconds = min(
                        _SESSION_OPERATION_CLAIM_HEARTBEAT_RETRY_SECONDS,
                        max(0.0, (claim_expires_at - now).total_seconds()),
                    )
                    continue
                sleep_seconds = _SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS

    async def _persist_compaction_attempt_events(
        self,
        *,
        session: Session,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        events: list[Event],
        heartbeat_state: _SessionOperationClaimHeartbeatState,
        persisted_event_ids: set[str],
        event_inventory: dict[str, Event],
        unresolved_store_tasks: set[asyncio.Task[Any]],
        cost_bearing_store_tasks: set[asyncio.Task[Any]] | None = None,
    ) -> None:
        if not events:
            return
        events[:] = self._event_writer.prepare_many(attribute_events_to_current_interaction(events))
        event_inventory.update((event.id, event.model_copy(deep=True)) for event in events)
        publication_claim_expires_at: datetime | None = None
        renewed_claim_expires_at: datetime | None = None

        def capture_publication_claim_expiry(expires_at: datetime) -> None:
            nonlocal publication_claim_expires_at
            publication_claim_expires_at = expires_at

        def capture_renewed_claim_expiry(expires_at: datetime) -> None:
            nonlocal renewed_claim_expires_at
            if renewed_claim_expires_at is None or expires_at > renewed_claim_expires_at:
                renewed_claim_expires_at = expires_at

        def require_unexpired_publication_commit() -> None:
            if publication_claim_expires_at is None:
                raise AssertionError(
                    "Session compaction event publication did not capture its claim."
                )
            _require_unexpired_session_operation_commit(
                clock=self._clock,
                claim_expires_at=publication_claim_expires_at,
                operation="Session compaction event publication",
            )

        event_ids = [event.id for event in events]

        async def publish() -> None:
            await self.session_store.publish_session_operation_guarded(
                session.id,
                idempotency_key=request.idempotency_key,
                operation_transform=_append_session_operation_attempt_events(
                    idempotency_key=request.idempotency_key,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    event_ids=event_ids,
                    clock=self._clock,
                    on_renew=capture_publication_claim_expiry,
                    on_renewed=capture_renewed_claim_expiry,
                ),
                commit_guard=require_unexpired_publication_commit,
                events=events,
                expected_statuses=_RESUMABLE_SESSION_STATUSES,
                expected_run_epoch=request.expected_run_epoch,
                expected_transcript_cursor=request.expected_transcript_cursor,
            )

        publication_task = asyncio.create_task(publish())
        publication_outcome = await self._await_session_operation_store_task(
            publication_task,
            claim_expires_at=heartbeat_state.confirmed_claim_expires_at,
        )
        if publication_outcome.timed_out:
            unresolved_store_tasks.add(publication_task)
            if cost_bearing_store_tasks is not None:
                cost_bearing_store_tasks.add(publication_task)
        cancellation = publication_outcome.cancellation
        if publication_outcome.timed_out:
            publication_error: BaseException | None = TimeoutError(
                "Session compaction event publication exceeded its bounded store wait."
            )
        else:
            publication_error = publication_outcome.error
        if isinstance(publication_error, asyncio.CancelledError):
            publication_error = unexpected_child_cancellation_error(
                publication_error,
                operation="Session compaction event publication",
            )

        async def confirm_current_claim(record: dict[str, Any] | None = None) -> None:
            nonlocal renewed_claim_expires_at
            if renewed_claim_expires_at is None and record is not None:
                renewed_claim_expires_at = _operation_claim_expiry(record)
            confirmed_expiry = heartbeat_state.confirmed_claim_expires_at
            if renewed_claim_expires_at is not None and (
                confirmed_expiry is None or renewed_claim_expires_at > confirmed_expiry
            ):
                heartbeat_state.confirm_claim(renewed_claim_expires_at)
                confirmed_expiry = renewed_claim_expires_at
            if confirmed_expiry is None:
                raise AssertionError(
                    "Session compaction event publication did not renew its claim."
                )
            _require_unexpired_session_operation_claim(
                clock=self._clock,
                claim_expires_at=confirmed_expiry,
                operation="Session compaction event publication acknowledgement",
            )

        if publication_error is not None:

            async def reconcile() -> tuple[list[bool], dict[str, Any] | None]:
                commit_states = [await self._event_writer.is_persisted(event) for event in events]
                checkpoint = await self.session_store.load_checkpoint(session.id)
                record = None
                if type(checkpoint) is dict:
                    record = _session_operation_state(checkpoint)["records"].get(
                        request.idempotency_key
                    )
                return commit_states, record

            reconciliation_task = asyncio.create_task(reconcile())
            reconciliation_outcome = await self._await_session_operation_store_task(
                reconciliation_task,
                cancellation=cancellation,
                claim_expires_at=heartbeat_state.confirmed_claim_expires_at,
            )
            cancellation = reconciliation_outcome.cancellation
            reconciliation_error: BaseException | None
            if reconciliation_outcome.timed_out:
                reconciliation_error = TimeoutError(
                    "Session compaction event publication reconciliation exceeded its "
                    "bounded store wait."
                )
            else:
                reconciliation_error = reconciliation_outcome.error
            if isinstance(reconciliation_error, asyncio.CancelledError):
                reconciliation_error = unexpected_child_cancellation_error(
                    reconciliation_error,
                    operation="Session compaction event publication reconciliation",
                )
            if reconciliation_error is not None:
                publication_error.add_note(
                    "Session compaction event publication reconciliation also failed: "
                    f"{type(reconciliation_error).__name__}: {reconciliation_error}"
                )
                if cancellation is not None:
                    cancellation.add_note(
                        "Session compaction event publication and reconciliation "
                        "failed during cancellation."
                    )
                    raise cancellation from publication_error
                raise publication_error from reconciliation_error
            reconciled = reconciliation_outcome.result
            if reconciled is None:
                raise AssertionError(
                    "Session compaction event publication reconciliation lost its result."
                ) from publication_error
            commit_states, record = reconciled
            record_event_ids = record.get("event_ids", []) if type(record) is dict else []
            record_has_all_events = all(event_id in record_event_ids for event_id in event_ids)
            # The guarded store commits the operation record and event batch
            # atomically. A per-event read can be stale when the physical write
            # lands between that read and the later checkpoint read, so the
            # operation record is the authoritative confirmation of the batch.
            fully_committed = record_has_all_events
            anything_committed = any(commit_states) or any(
                event_id in record_event_ids for event_id in event_ids
            )
            if anything_committed and not fully_committed:
                atomicity_error = RuntimeError(
                    "The session store violated atomic compaction event publication."
                )
                if cancellation is not None:
                    cancellation.add_note(str(atomicity_error))
                    raise cancellation from publication_error
                raise atomicity_error from publication_error
            if not fully_committed:
                if cancellation is not None:
                    raise cancellation from publication_error
                raise publication_error
            persisted_event_ids.update(event_ids)
            fan_out_error, cancellation = await self._fan_out_reconciled_session_operation_events(
                events,
                cancellation=cancellation,
            )
            if fan_out_error is not None:
                publication_error.add_note(
                    "Committed compaction event side-effect delivery also failed: "
                    f"{type(fan_out_error).__name__}: {fan_out_error}"
                )
            try:
                await confirm_current_claim(record)
            except BaseException as claim_error:
                if cancellation is not None:
                    cancellation.add_note(
                        "The compaction claim became unusable while publication "
                        "cancellation was being reconciled."
                    )
                    raise cancellation from claim_error
                raise claim_error from publication_error
            publication_error.add_note(
                "Session compaction events were durable; the operation will fail "
                "closed after the lost acknowledgement."
            )
            if cancellation is not None:
                raise cancellation from publication_error
            raise publication_error

        persisted_event_ids.update(event_ids)
        try:
            await confirm_current_claim()
        except BaseException as claim_error:
            authoritative_cancellation = cancellation
            (
                fan_out_error,
                later_cancellation,
            ) = await self._fan_out_reconciled_session_operation_events(
                events,
                cancellation=authoritative_cancellation,
            )
            cancellation = later_cancellation or authoritative_cancellation
            if fan_out_error is not None:
                claim_error.add_note(
                    "Committed compaction event side-effect delivery also failed "
                    "before claim rejection: "
                    f"{type(fan_out_error).__name__}: {fan_out_error}"
                )
            if cancellation is not None:
                cancellation.add_note(
                    "The compaction claim became unusable after event publication."
                )
                raise cancellation from claim_error
            raise
        if cancellation is not None:
            authoritative_cancellation = cancellation
            (
                fan_out_error,
                later_cancellation,
            ) = await self._fan_out_reconciled_session_operation_events(
                events,
                cancellation=authoritative_cancellation,
            )
            cancellation = later_cancellation or authoritative_cancellation
            if fan_out_error is not None:
                cancellation.add_note(
                    "Committed compaction event side-effect delivery also failed "
                    f"during cancellation: {type(fan_out_error).__name__}: "
                    f"{fan_out_error}"
                )
            raise cancellation

    async def _reserve_compaction_budget(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        budget_limits: tuple[BudgetLimit, ...],
        provider_name: str | None,
        model: str | None,
        model_attempt_identity: ModelAttemptIdentity,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        compactor: str,
        reservations: list[BudgetStepReservation],
        events: list[Event],
        reservation_identity_guard: BudgetReservationIdentityGuard,
        billing_identity: BillingIdentity | None = None,
        execution_profile_fingerprint: str | None = None,
    ) -> BudgetReservationResult | None:
        model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
        setup = await self._run_limit_controller.reserve_operation_budgets(
            budget_limits=budget_limits,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            provider_name=provider_name,
            model=model,
            model_attempt_identity=model_attempt_identity,
            environment_name=environment_name,
            settlement_event_payload=self._event_writer.prepare_budget_settlement_template(
                Event(
                    type=EventType.BUDGET_RECONCILED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=_application_compaction_causal_payload(
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        source_cursor=request.expected_transcript_cursor,
                        compactor=compactor,
                    ),
                )
            ).payload,
            billing_identity=billing_identity,
            execution_profile_fingerprint=execution_profile_fingerprint,
            reservation_identity_guard=reservation_identity_guard,
            rejection_release_reason="compaction budget reservation failed",
            accepted_record_error="Accepted compaction budget reservation has no record.",
            reservation_event_factory=lambda result: _application_compaction_ledger_event(
                event_type=(
                    EventType.BUDGET_RESERVED
                    if result.accepted
                    else EventType.BUDGET_RESERVATION_FAILED
                ),
                payload=budget_reservation_payload(result),
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                compactor=compactor,
            ),
        )
        reservations.extend(setup.reservations)
        events.extend(setup.events)
        events.extend(
            [
                await self._run_limit_controller.budget_settlement_event(reconciliation)
                for reconciliation in setup.releases
            ]
        )
        if setup.error is not None:
            raise setup.error
        return setup.failure

    async def _reconcile_compaction_budget_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        model_completed_events: list[Event],
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        compactor: str,
    ) -> AsyncGenerator[Event, None]:
        async for reconciliation in self._run_limit_controller.reconcile_operation_reservations(
            reservations,
            model_completed_events=model_completed_events,
            completed_reason="compaction model completed",
            missing_usage_reason=(
                "compaction completed without priced usage; charged reserved amount"
            ),
        ):
            yield await self._run_limit_controller.budget_settlement_event(reconciliation)

    async def _reconcile_uncertain_compaction_budget_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        compactor: str,
    ) -> AsyncGenerator[Event, None]:
        async for (
            reconciliation
        ) in self._run_limit_controller.reconcile_uncertain_operation_reservations(
            reservations,
            reason="compaction reservation lease lost; charged reserved amount",
        ):
            yield await self._run_limit_controller.budget_settlement_event(reconciliation)

    async def _release_compaction_budget_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        compactor: str,
        reason: str,
    ) -> AsyncGenerator[Event, None]:
        async for reconciliation in self._run_limit_controller.release_operation_reservations(
            reservations,
            reason=reason,
        ):
            yield await self._run_limit_controller.budget_settlement_event(reconciliation)

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
        if loaded_session.status == SessionStatus.INTERRUPTED:
            existing_interrupt_event = (
                await self._session_control.wait_for_active_interrupted_event(loaded_session.id)
            )
            if existing_interrupt_event is not None:
                retry_event: Event | None = None
                retry_request: dict[str, Any] | None = None
                if not interruption_cascade_suppressed():
                    marker = await self._load_pending_interruption_cascade(loaded_session.id)
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
                    self._schedule_background_interruption_cascade(
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

        if loaded_session.status == SessionStatus.INTERRUPTING:
            pending_interrupt_payload = await self._load_pending_session_interrupt_payload(
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
                if not interruption_cascade_suppressed():
                    self._schedule_background_interruption_cascade(
                        parent_session_id=loaded_session.id,
                        interrupt_payload=existing_interrupt_event.payload,
                        create_if_missing=False,
                    )
                yield existing_interrupt_event
                return
            raise TimeoutError(f"Session interruption is still finalizing: {loaded_session.id}")

        if loaded_session.status not in _INTERRUPTIBLE_SESSION_STATUSES:
            raise ValueError(f"Session cannot be interrupted from status: {loaded_session.status}")
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

        interrupt_payload = {
            "reason": request.reason,
            "metadata": request.metadata,
            "requested_by": resolution_actor_payload(request.requested_by),
            "interruption_type": _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
            "interruption_request_id": str(uuid4()),
        }
        cascade_suppressed = interruption_cascade_suppressed()
        self._session_control.begin_interruption_request(loaded_session.id)
        request_marker_active = True
        try:
            session = await self.session_store.transition_status_and_checkpoint(
                loaded_session.id,
                from_statuses=_INTERRUPTIBLE_SESSION_STATUSES,
                to_status=SessionStatus.INTERRUPTING,
                checkpoint_transform=_checkpoint_with_pending_session_interrupt(
                    interrupt_payload,
                    include_interruption_cascade=not cascade_suppressed,
                    cascade_created_at=self._clock(),
                ),
            )
            self._session_control.signal_interrupt(session.id)
            active_work_signalled = self._session_control.cancel_active_runs(session.id)
            if active_work_signalled:
                existing_interrupt_event = (
                    await self._session_control.wait_for_active_interrupted_event(
                        session.id,
                        interruption_request_id=interrupt_payload["interruption_request_id"],
                    )
                )
                if existing_interrupt_event is not None:
                    request_marker_active = False
                    self._session_control.end_interruption_request(loaded_session.id)
                    yield existing_interrupt_event
                    return
                raise TimeoutError(f"Session interruption is still finalizing: {session.id}")
            provider_operation_profile = (
                await self._recovery_coordinator.cancel_provider_operation_for_interruption(
                    session,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                )
            )
            provider_operation_addressed = provider_operation_profile is not None
            if loaded_session.status == SessionStatus.RUNNING and not provider_operation_addressed:
                existing_interrupt_event = (
                    await self._session_control.wait_for_active_interrupted_event(
                        session.id,
                        interruption_request_id=interrupt_payload["interruption_request_id"],
                    )
                )
                if existing_interrupt_event is not None:
                    request_marker_active = False
                    self._session_control.end_interruption_request(loaded_session.id)
                    yield existing_interrupt_event
                    return
                raise TimeoutError(f"Session interruption is still finalizing: {session.id}")
            if provider_operation_profile is not None:
                session = await self._recovery_coordinator.claim_provider_operation_interruption(
                    session,
                    provider_operation_profile,
                    interruption_request_id=interrupt_payload["interruption_request_id"],
                )
        except ValueError:
            reloaded_session = await self.session_store.load(loaded_session.id)
            if reloaded_session is None:
                raise KeyError(f"Session not found: {loaded_session.id}") from None
            if reloaded_session.status in INTERRUPT_REQUESTED_SESSION_STATUSES:
                self._session_control.signal_interrupt(reloaded_session.id)
                pending_interrupt_payload = await self._load_pending_session_interrupt_payload(
                    reloaded_session.id,
                    default={},
                )
                existing_interrupt_event = (
                    await self._session_control.wait_for_active_interrupted_event(
                        reloaded_session.id,
                        interruption_request_id=interruption_request_id_from_payload(
                            pending_interrupt_payload
                        ),
                    )
                )
                if existing_interrupt_event is not None:
                    request_marker_active = False
                    self._session_control.end_interruption_request(loaded_session.id)
                    if not interruption_cascade_suppressed():
                        self._schedule_background_interruption_cascade(
                            parent_session_id=reloaded_session.id,
                            interrupt_payload=existing_interrupt_event.payload,
                            create_if_missing=False,
                        )
                    yield existing_interrupt_event
                    return
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
            else:
                raise
        except BaseException:
            if request_marker_active:
                self._session_control.end_interruption_request(loaded_session.id)
            raise

        async def release_offline_provider_interruption(
            authoritative_failure: BaseException | None,
            *,
            operation: str,
        ) -> None:
            if provider_operation_profile is None:
                return
            await _run_recovery_cleanup_steps(
                authoritative_failure=authoritative_failure,
                steps=(
                    (
                        operation,
                        lambda: (
                            self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                                session_id=session.id,
                                execution_profile=provider_operation_profile.profile,
                            )
                        ),
                    ),
                ),
            )

        terminal_event_stream: AsyncIterator[Event] | None = None
        try:
            (
                session,
                _interaction_event,
                _,
            ) = await self._publish_sibling_interaction_transition(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=_environment_name(registered_environment),
                to_status=SessionStatus.INTERRUPTED,
                execution_profile=(
                    None
                    if provider_operation_profile is None
                    else provider_operation_profile.profile
                ),
            )
        except BaseException as transition_failure:
            try:
                await release_offline_provider_interruption(
                    transition_failure,
                    operation="failed offline interruption run-fence release",
                )
            finally:
                if request_marker_active:
                    request_marker_active = False
                    self._session_control.end_interruption_request(loaded_session.id)
            raise
        try:
            payload = await self._load_pending_session_interrupt_payload(
                session.id,
                default={
                    "reason": request.reason,
                    "metadata": request.metadata,
                    "requested_by": resolution_actor_payload(request.requested_by),
                    "interruption_type": _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
                    "interruption_request_id": interrupt_payload["interruption_request_id"],
                },
            )
        except BaseException as payload_failure:
            try:
                await release_offline_provider_interruption(
                    payload_failure,
                    operation="offline interruption payload run-fence release",
                )
            finally:
                if request_marker_active:
                    request_marker_active = False
                    self._session_control.end_interruption_request(loaded_session.id)
            raise
        try:
            existing_interrupt_event = await self._session_control.latest_interrupted_event(
                session.id,
                interruption_request_id=interruption_request_id_from_payload(payload),
            )
            if existing_interrupt_event is not None:
                await self._clear_pending_session_interrupt(session.id)
                if not cascade_suppressed:
                    self._schedule_background_interruption_cascade(
                        parent_session_id=session.id,
                        interrupt_payload=existing_interrupt_event.payload,
                        create_if_missing=False,
                    )
                yield existing_interrupt_event
                return
            await self._emit_active_turn_completed_if_needed(
                session=session,
                status=SessionStatus.INTERRUPTED,
            )
            terminal_event_stream = self._emit_terminal_event_with_hooks(
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
                    else provider_operation_profile.profile
                ),
            )
            terminal_prefix, interrupted_event = await _collect_through_event_type(
                terminal_event_stream,
                EventType.SESSION_INTERRUPTED,
                missing_message="Session interruption produced no terminal event.",
            )

            await self._clear_pending_session_interrupt(session.id)
            if not cascade_suppressed:
                self._schedule_background_interruption_cascade(
                    parent_session_id=session.id,
                    interrupt_payload=interrupted_event.payload,
                    create_if_missing=False,
                )
            for event in terminal_prefix:
                yield event
            async for event in terminal_event_stream:
                yield event
        except BaseException as terminal_failure:
            if terminal_event_stream is not None:
                await _run_recovery_cleanup_steps(
                    authoritative_failure=terminal_failure,
                    steps=(
                        (
                            "offline interruption terminal stream close",
                            lambda: _close_async_iterator(terminal_event_stream),
                        ),
                    ),
                )
            raise
        finally:
            try:
                await release_offline_provider_interruption(
                    sys.exception(),
                    operation="offline provider interruption run-fence release",
                )
            finally:
                if request_marker_active:
                    self._session_control.end_interruption_request(loaded_session.id)
        return

    async def _resume_session(
        self,
        *,
        request: ResumeRequest,
        task_id: str | None,
        start_event_payload_extra: dict[str, Any],
        start_task_on_enter: bool,
        source_execution_profile: ExecutionProfileIdentity | None = None,
        required_execution_profile: ExecutionProfileIdentity | None = None,
        required_session_instance_fingerprint: str | None = None,
        run_operation_id: str | None = None,
        terminal_event_id: str | None = None,
        queue_task_id: str | None = None,
    ) -> AsyncGenerator[Event, None]:
        # Resume profile admission and the resumed dispatch must observe one
        # application-budget snapshot even if the app configuration is changed
        # while the interaction-start event is being consumed.
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
        require_secret_free_structured_output_spec(
            request.structured_output,
            redactor=self._secret_redactor,
            field_name="ResumeRequest.structured_output",
        )
        adoption_request_fingerprint = (
            None
            if request.profile_adoption is None
            else execution_profile_adoption_request_fingerprint(
                request,
                redactor=self._secret_redactor,
            )
        )
        loaded_session = await self.session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")
        if (
            required_session_instance_fingerprint is not None
            and _queued_dispatch_session_instance_fingerprint(loaded_session)
            != required_session_instance_fingerprint
        ):
            raise RuntimeError("Queued dispatch target session instance changed.")
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

        registered_agent = self._get_registered_agent(loaded_session.agent_name)
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
        if (
            request.profile_adoption is not None
            and EXECUTION_PROFILE_METADATA_KEY in loaded_session.metadata
        ):
            replay_expected_profile = execution_profile_from_session_metadata(
                loaded_session.metadata
            )
            replay_candidate_profile = _execution_profile_identity(
                registered_agent=registered_agent,
                provider_name=registered_provider.name,
                registered_provider=registered_provider,
                model=(
                    requested_target.model
                    if target_changed and requested_target is not None
                    else loaded_session.model
                ),
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
                    self._request_loop_policy_instance_identities(request.loop_policies)
                ),
                budget_policy=budget_policy,
                request_budget_limits=request.budget_limits,
                causal_budget_id=loaded_session.causal_budget_id,
                structured_output=request.structured_output,
                thinking=request.thinking,
                max_steps=request.max_steps,
                limits=request.limits,
                retry_policy=self._effective_retry_policy(request.retry_policy),
            )
            replay_candidate_profile = execution_profile_with_component(
                replay_candidate_profile,
                replay_expected_profile.component(
                    ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION
                ),
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
        if loaded_session.status not in _RESUMABLE_SESSION_STATUSES:
            raise SessionStatusConflict(
                "Session status transition not allowed: "
                f"{loaded_session.status} -> {SessionStatus.RUNNING}"
            )
        if loaded_projection_cursor or self._secret_redactor.has_values:
            preflight_snapshot = await self.session_store.load_transcript_snapshot(
                loaded_session.id
            )
            preflight_transcript: list[Message] = []
            try:
                preflight_transcript = _transcript_snapshot_messages(preflight_snapshot)
                if loaded_projection_cursor:
                    _project_model_target_snapshot(
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
            str(uuid4())
            if run_operation_id is None
            else require_clean_nonblank(run_operation_id, "run_operation_id")
        )
        continuing_execution_profile_snapshot: ActiveInvocationExecutionProfile | None = None

        def validate_resumable_checkpoint(
            current_session: Session,
            current_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            if (
                required_session_instance_fingerprint is not None
                and _queued_dispatch_session_instance_fingerprint(current_session)
                != required_session_instance_fingerprint
            ):
                raise RuntimeError("Queued dispatch target session instance changed.")
            updated_checkpoint = (
                None
                if current_checkpoint is None
                else copy_json_value(current_checkpoint, "checkpoint")
            )
            updated_checkpoint = _checkpoint_without_active_incomplete_recovery_claim(
                updated_checkpoint,
                now=self._clock(),
            )
            _reject_pending_provider_operation_disposition(updated_checkpoint)
            if _initial_transcript_pending_interaction_id(updated_checkpoint) is not None:
                raise RuntimeError(
                    "Session setup did not publish its authoritative initial transcript; "
                    "resume fails closed. Start a new session."
                )
            if (
                updated_checkpoint is not None
                and _SESSION_OPERATIONS_CHECKPOINT_KEY in updated_checkpoint
            ):
                operations = _session_operation_state(updated_checkpoint)
                _abandon_expired_session_operation(operations, now=self._clock())
                updated_checkpoint[_SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
            active_operation_id = _active_session_operation_id(updated_checkpoint)
            if active_operation_id is not None:
                raise RuntimeError(
                    f"Session has an active durable operation: {active_operation_id}"
                )
            _reject_prepared_prompt_transition_intent(updated_checkpoint)
            if (
                approval_support.pending_approval_from_checkpoint(
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
            if (
                pending_user_input_from_checkpoint(
                    updated_checkpoint,
                    redactor=self._secret_redactor,
                    consume_on_rejection=True,
                )
                is not None
            ):
                raise RuntimeError(
                    "Session is awaiting user input. Answer it with "
                    "resolve_user_input(...) before resuming with new messages."
                )
            if target_changed or current_session.status == SessionStatus.COMPLETED:
                checkpoint_pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(
                    updated_checkpoint,
                    redactor=self._secret_redactor,
                    consume_on_rejection=True,
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
        checkpoint = await self.session_store.load_checkpoint(loaded_session.id)
        checkpoint = await _reconcile_committed_prompt_transition_intents(
            session_store=self.session_store,
            source_session_id=loaded_session.id,
            checkpoint=checkpoint,
            clock=self._clock,
        )
        validate_resumable_checkpoint(loaded_session, checkpoint)
        active_invocation_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
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
        active_model_completion_boundary = (
            await self._recovery_coordinator.load_model_completion_boundary(loaded_session)
        )
        if active_model_completion_boundary is None:
            pending_model_completion = False
        else:
            continuing_execution_profile_snapshot = (
                await self.validate_execution_profile_continuation(
                    session=loaded_session,
                    checkpoint=checkpoint,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
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
            )
            pending_model_completion = (
                await self._recovery_coordinator.preflight_model_completion_boundary(
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
            loaded_session,
            checkpoint,
        ) = await self._recovery_coordinator._reconcile_terminal_evidence_before_continuation(
            session=loaded_session,
            checkpoint=checkpoint,
        )
        validate_resumable_checkpoint(loaded_session, checkpoint)

        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
        continuing_recovery_boundary = pending_round is not None or pending_model_completion
        if continuing_recovery_boundary and request.profile_adoption is not None:
            raise RuntimeError(
                "An execution profile cannot be adopted while model or tool recovery is pending."
            )
        candidate_execution_profile: ExecutionProfileIdentity | None = None
        execution_profile_decision: ExecutionProfileDecision | None = None
        # #906 owns baseline selection for derived sessions. Until that contract
        # exists, preserve the pre-profile behavior only for fork records that do
        # not yet carry profile authority; a malformed or present profile still
        # enters the normal fail-closed path.
        legacy_unprofiled_fork = (
            loaded_session.parent_session_id is not None
            and EXECUTION_PROFILE_METADATA_KEY not in loaded_session.metadata
        )
        if legacy_unprofiled_fork and request.profile_adoption is not None:
            raise RuntimeError(
                "A legacy fork without a durable execution profile cannot adopt a new profile."
            )
        if not continuing_recovery_boundary:
            candidate_execution_profile = _execution_profile_identity(
                registered_agent=registered_agent,
                provider_name=registered_provider.name,
                registered_provider=registered_provider,
                model=(
                    requested_target.model
                    if target_changed and requested_target is not None
                    else loaded_session.model
                ),
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
                    self._request_loop_policy_instance_identities(request.loop_policies)
                ),
                budget_policy=budget_policy,
                request_budget_limits=request.budget_limits,
                causal_budget_id=loaded_session.causal_budget_id,
                structured_output=request.structured_output,
                thinking=request.thinking,
                max_steps=request.max_steps,
                limits=request.limits,
                retry_policy=self._effective_retry_policy(request.retry_policy),
            )
            if legacy_unprofiled_fork:
                fork_transcript = await self.session_store.load_transcript(loaded_session.id)
                try:
                    candidate_execution_profile = (
                        execution_profile_with_durable_system_projection_digest(
                            candidate_execution_profile,
                            system_prompt_messages_sha256(fork_transcript),
                        )
                    )
                finally:
                    fork_transcript.clear()
            else:
                expected_execution_profile = execution_profile_from_session_metadata(
                    loaded_session.metadata
                )
                # A resumed session executes the already-durable system projection.
                # The current AgentSpec prompt is not re-injected on resume.
                candidate_execution_profile = execution_profile_with_component(
                    candidate_execution_profile,
                    expected_execution_profile.component(
                        ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION
                    ),
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
                        source_registration_profile = _execution_profile_identity(
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
                                self._request_loop_policy_instance_identities(request.loop_policies)
                            ),
                            budget_policy=budget_policy,
                            request_budget_limits=request.budget_limits,
                            causal_budget_id=loaded_session.causal_budget_id,
                            structured_output=request.structured_output,
                            thinking=request.thinking,
                            max_steps=request.max_steps,
                            limits=request.limits,
                            retry_policy=self._effective_retry_policy(request.retry_policy),
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
                        )
            if not legacy_unprofiled_fork:
                changed_profile_components = changed_execution_profile_components(
                    expected_execution_profile,
                    candidate_execution_profile,
                )
                fork_relationship = session_fork_profile_relationship(loaded_session)
                exact_fork_group_initial_invocation = (
                    loaded_session.run_epoch == 0
                    and request.profile_adoption is None
                    and fork_relationship is not None
                    and fork_relationship.initial_invocation_request_sha256
                    == _fork_group_initial_invocation_request_sha256(request)
                    and fork_relationship.initial_invocation_profile == candidate_execution_profile
                )
                if exact_fork_group_initial_invocation and changed_profile_components:
                    execution_profile_decision = _execution_profile_decision_event(
                        session=loaded_session,
                        expected_profile=expected_execution_profile,
                        candidate_profile=candidate_execution_profile,
                        changed_component_classes=changed_profile_components,
                        kind=ExecutionProfileDecisionKind.ADOPTED,
                        policy_identity=_FORK_GROUP_INITIAL_PROFILE_POLICY_ID,
                        policy_reason=(
                            "The exact invocation was frozen with the durable fork-group "
                            "branch before child creation."
                        ),
                        authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
                        intent=None,
                        adoption_request_fingerprint=None,
                        fallback_actor=_fork_group_initial_profile_actor(),
                        fallback_reason=(
                            "Apply the runtime-owned initial fork-group branch invocation."
                        ),
                        clock=self._clock,
                    )
                else:
                    execution_profile_decision = await self._classify_execution_profile(
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
                    )
                if execution_profile_decision.kind in {
                    ExecutionProfileDecisionKind.MIGRATION_REQUIRED,
                    ExecutionProfileDecisionKind.REJECTED,
                }:
                    rejection = await self.session_store.reject_execution_profile_resume(
                        loaded_session.id,
                        expected_statuses=_RESUMABLE_SESSION_STATUSES,
                        expected_run_epoch=loaded_session.run_epoch,
                        expected_profile=expected_execution_profile,
                        candidate_profile=candidate_execution_profile,
                        event=execution_profile_decision.event,
                        decision=execution_profile_decision,
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
                    )
        if continuing_recovery_boundary:
            continuing_execution_profile_snapshot = (
                await self.validate_execution_profile_continuation(
                    session=loaded_session,
                    checkpoint=checkpoint,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
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
            )
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
                source_transcript = _transcript_snapshot_messages(source_snapshot)
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
                        tool_exposure=_all_registered_tool_exposure(registered_agent),
                        structured_output=request.structured_output,
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
            interaction_id = await self._activate_latest_open_interaction(loaded_session.id)
            if interaction_id is None:
                raise RuntimeError(
                    "Pending model or tool recovery state has no open interaction. "
                    "Pre-interaction prerelease recovery state is unsupported."
                )
            interaction_started_event = None
        else:
            interaction_id = str(uuid4())
            interaction_started_event = self._interaction_started_event(
                session=loaded_session,
                registered_agent=registered_agent,
                environment_name=_environment_name(registered_environment),
                interaction_id=interaction_id,
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
                            id=str(uuid4()),
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
        interaction_source_messages = [
            *(
                existing_deferred_input.source_messages
                if existing_deferred_input is not None
                else []
            ),
            *request.messages,
        ]
        invocation_profile = (
            continuing_execution_profile_snapshot.profile
            if continuing_execution_profile_snapshot is not None
            else candidate_execution_profile
        )
        if invocation_profile is None:
            raise AssertionError("Session invocation admission produced no execution profile.")
        try:
            session = await self.session_store.admit_session_invocation(
                loaded_session.id,
                admission=SessionInvocationAdmission(
                    from_statuses=frozenset(_RESUMABLE_SESSION_STATUSES),
                    checkpoint_transform=claim_resumable_checkpoint,
                    execution_profile=invocation_profile,
                    interaction_started_event=interaction_started_event,
                    continued_interaction_id=(
                        interaction_id if continuing_recovery_boundary else None
                    ),
                    interaction_source_messages=tuple(interaction_source_messages),
                    defer_interaction_source=continuing_recovery_boundary,
                    model_transition=model_transition,
                    execution_profile_decision=execution_profile_decision,
                    expected_active_invocation_profile=(continuing_execution_profile_snapshot),
                    allow_unprofiled_derived_session=legacy_unprofiled_fork,
                ),
            )
        except (SessionStatusConflict, ValueError):
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
        if not continuing_recovery_boundary:
            if interaction_id is None:
                raise RuntimeError("New interaction admission produced no interaction identity.")
            _activate_session_interaction(session.id, interaction_id)
        try:
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
            model_boundary = await self._recovery_coordinator.reconcile_model_completion_boundary(
                session,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
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
                    await self._clear_session_run_operation(
                        session_id=session.id,
                        operation_id=run_operation_id,
                    )
                await self.session_store.update_status(
                    session.id,
                    SessionStatus.INTERRUPTED,
                )
                await self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                    session_id=session.id,
                    execution_profile=invocation_profile,
                )
                _deactivate_session_interaction(session.id)
                return
            projection_cursor = session_model_projection_cursor(session)
            if projection_cursor:
                transcript_snapshot = await self.session_store.load_transcript_snapshot(session.id)
                try:
                    transcript = list(
                        _project_model_target_snapshot(
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
                await _run_recovery_cleanup_steps(
                    authoritative_failure=cancellation,
                    steps=(
                        (
                            "cancelled resume setup finalization",
                            lambda: self._recovery_coordinator.finalize_abandoned_session_by_id(
                                session.id,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                execution_profile=invocation_profile,
                            ),
                        ),
                        (
                            "cancelled resume run-fence release",
                            lambda: (
                                self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                                    session_id=session.id,
                                    execution_profile=invocation_profile,
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
                            failure_checkpoint = await self.session_store.load_checkpoint(
                                session.id
                            )
                            pending_failure_round = (
                                tool_round_recovery.pending_tool_round_from_checkpoint(
                                    failure_checkpoint,
                                    redactor=self._secret_redactor,
                                    consume_on_rejection=True,
                                )
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
                    ) = await self._publish_sibling_interaction_transition(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                        to_status=SessionStatus.FAILED,
                        execution_profile=invocation_profile,
                    )
                if interaction_failed_event is not None:
                    yield interaction_failed_event
                failure_event = await self._bind_event_to_session_run_operation(
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
                    await self._clear_session_run_operation(
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
                    )
                finally:
                    _deactivate_session_interaction(session.id)
            return
        except BaseException:
            try:
                await self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                    session_id=session.id,
                    execution_profile=invocation_profile,
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
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            execution_profile=(
                continuing_execution_profile_snapshot.profile
                if continuing_execution_profile_snapshot is not None
                else candidate_execution_profile
            ),
            messages=messages,
            messages_to_append=(
                interaction_source_messages if continuing_recovery_boundary else request.messages
            ),
            max_steps=request.max_steps,
            limits=request.limits,
            budget_limits=request.budget_limits,
            budget_policy=budget_policy,
            retry_policy=self._effective_retry_policy(request.retry_policy),
            structured_output=request.structured_output,
            thinking=request.thinking,
            request_loop_policies=request.loop_policies,
            request_metadata=request.metadata,
            request_trace_metadata=(
                _runtime_resume_transport_metadata(request) or request.metadata
            ),
            task_id=task_id,
            task_worker_id=None,
            start_event_type=EventType.SESSION_RESUMED,
            start_event_payload={
                "agent_name": registered_agent.spec.name,
                "appended_messages": len(request.messages),
                **copy_json_value(start_event_payload_extra, "start_event_payload_extra"),
            },
            start_task_on_enter=start_task_on_enter,
            messages_already_persisted=not continuing_recovery_boundary,
            messages_deferred=continuing_recovery_boundary,
            deliver_queued_input_before_first_step=not continuing_recovery_boundary,
            pending_tool_round_source_transcript_cursor=(
                None
                if reconciled_pending_round is None
                else reconciled_pending_round.source_transcript_cursor
            ),
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

    async def fork_session(
        self,
        request: ForkSessionRequest,
        *,
        request_sha256: str,
        accepted_request_sha256s: tuple[str, ...],
        store_resolved_source_session_id: str | None = None,
    ) -> AsyncGenerator[Event, None]:
        request = copy_fork_session_request(request)
        if (
            len(request_sha256) != 64
            or any(character not in "0123456789abcdef" for character in request_sha256)
            or not accepted_request_sha256s
            or accepted_request_sha256s[0] != request_sha256
            or any(
                len(candidate) != 64
                or any(character not in "0123456789abcdef" for character in candidate)
                for candidate in accepted_request_sha256s
            )
        ):
            raise ValueError("Fork request replay identity is invalid.")
        source_session = await self.session_store.load(request.source_session_id)
        if source_session is None:
            raise session_request_boundary.ForkSourceNotFoundError(
                "Fork source session was not found."
            ) from None
        try:
            source_session = session_request_boundary.prepare_fork_source_session(
                source_session,
                expected_source_session_id=request.source_session_id,
                store_resolved_source_session_id=store_resolved_source_session_id,
                redactor=self._secret_redactor,
            )
        except ValueError:
            del source_session
            raise
        fork_request_sha256 = request_sha256
        runtime_generated_session_id: str | None = None
        destination_session_id = request.session_id
        if destination_session_id is None:
            logical_fork_id = hashlib.sha256(
                canonical_durable_json_bytes(
                    request.model_dump(mode="json", warnings=False),
                    "generated_session_fork_identity",
                )
            ).hexdigest()
            runtime_generated_session_id = generate_child_session_id(
                kind=ChildSessionKind.SESSION_FORK,
                parent_session_id=source_session.id,
                logical_spawn_id=logical_fork_id,
            )
            destination_session_id = runtime_generated_session_id
        if destination_session_id is not None:
            existing_descendant = await self.session_store.load(destination_session_id)
            if existing_descendant is not None:
                relationship: SessionForkProfileRelationship | None = None
                malformed_relationship = False
                try:
                    relationship = session_fork_profile_relationship(existing_descendant)
                except Exception as exc:
                    if exc.__traceback__ is not None:
                        traceback_module.clear_frames(exc.__traceback__)
                    malformed_relationship = True
                if (
                    malformed_relationship
                    or relationship is None
                    or relationship.source_session_id != source_session.id
                    or relationship.request_sha256 not in accepted_request_sha256s
                ):
                    existing_descendant = None
                    relationship = None
                    raise RuntimeError(
                        "Existing fork destination conflicts with the exact profiled request."
                    ) from None
                evidence_ids = (
                    [relationship.decision.event_id] if relationship.decision is not None else []
                ) + [relationship.fork_event_id]
                persisted_events: list[Event] = []
                for evidence_id in evidence_ids:
                    records = await self.session_store.query_events(
                        EventQuery(
                            session_id=existing_descendant.id,
                            event_id=evidence_id,
                            limit=1,
                        )
                    )
                    if len(records) != 1:
                        raise RuntimeError(
                            "Existing profiled fork is missing its durable evidence."
                        )
                    persisted_events.append(records[0].event)
                validate_profiled_fork_evidence(
                    fork=existing_descendant,
                    relationship=relationship,
                    events=persisted_events,
                )
                if request.system_prompt_policy is ForkSystemPromptPolicy.CURRENT_AGENT:
                    succession = _PromptSuccessionWorkflow.recover(
                        request=request,
                        accepted_request_sha256s=accepted_request_sha256s,
                        descendant=existing_descendant,
                    )

                    def reconcile_surviving_intent(
                        _current_source: Session,
                        source_checkpoint: dict[str, Any] | None,
                    ) -> dict[str, Any] | None:
                        if source_checkpoint is None:
                            return None
                        updated = copy_json_value(source_checkpoint, "checkpoint")
                        intent_ledger = _PromptTransitionIntentLedger.from_checkpoint(
                            updated,
                            required=False,
                        )
                        if intent_ledger.complete_from_receipt(
                            succession.receipt,
                            descendant_created_at=existing_descendant.created_at,
                            completed_at=self._clock(),
                        ):
                            intent_ledger.store_in(updated)
                        return updated

                    await self.session_store.publish_checkpoint_and_events(
                        source_session.id,
                        checkpoint_transform=reconcile_surviving_intent,
                        events=[],
                    )
                delivered = await self._event_writer.fan_out_persisted(persisted_events)
                yield delivered[-1]
                return
        expected_source_snapshot = request._expected_source_snapshot
        if expected_source_snapshot is not None and (
            source_session.run_epoch != expected_source_snapshot.run_epoch
        ):
            raise SessionRunFenced(
                "Fork source changed after its coordinator snapshot was frozen: "
                f"expected run epoch {expected_source_snapshot.run_epoch}, "
                f"current {source_session.run_epoch}."
            )
        if source_session.status not in _FORKABLE_SESSION_STATUSES:
            raise ValueError(
                "Only completed, failed, or interrupted sessions can be forked: "
                f"{source_session.status}"
            )
        if request.transcript_cursor is not None and request.copy_checkpoint:
            raise ValueError(
                "ForkSessionRequest.copy_checkpoint must be false when transcript_cursor is set."
            )
        if source_session.status == SessionStatus.INTERRUPTED and not request.copy_checkpoint:
            raise ValueError("Interrupted sessions cannot be forked without checkpoint state.")
        if not self.session_store.supports_profiled_forks:
            raise RuntimeError("Session store does not support atomic profiled session forks.")

        source_checkpoint_for_profile = await self.session_store.load_checkpoint(source_session.id)
        try:
            (
                source_profile_source,
                source_execution_profile,
                source_active_profile,
            ) = session_request_boundary.prepare_fork_source_execution_profile(
                source_session,
                source_checkpoint_for_profile,
            )
        finally:
            source_checkpoint_for_profile = None
        if expected_source_snapshot is not None and (
            source_execution_profile.fingerprint != expected_source_snapshot.profile_fingerprint
        ):
            raise SessionRunFenced(
                "Fork source execution profile changed after its coordinator snapshot was frozen."
            )
        try:
            source_registered_agent = self._get_registered_agent(source_session.agent_name)
        except KeyError as exc:
            raise KeyError(
                "Source agent must be registered to derive inherited taint before forking: "
                f"{source_session.agent_name}"
            ) from exc
        agent_name = request.agent_name or source_session.agent_name
        registered_agent = self._get_registered_agent(agent_name)
        model = request.model or (
            registered_agent.spec.model if request.agent_name is not None else source_session.model
        )
        environment_name = (
            request.environment_name
            if request.environment_name is not None
            else source_session.environment_name
        )
        try:
            (
                agent_name,
                registered_agent_provider_name,
                model,
                environment_name,
            ) = session_request_boundary.prepare_fork_registered_authority(
                agent_name=agent_name,
                agent_provider_name=(
                    registered_agent.spec.provider_name if request.agent_name is not None else None
                ),
                model=model,
                environment_name=environment_name,
                redactor=self._secret_redactor,
            )
        except ValueError:
            del registered_agent
            agent_name = model = environment_name = ""
            raise
        registered_provider = self._get_registered_provider(
            registered_agent_provider_name or source_session.provider_name
        )
        registered_environment = self._get_registered_environment_for_session(environment_name)
        prompt_workflow = await _ForkPromptWorkflow.prepare(
            request=request,
            request_sha256=fork_request_sha256,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_lifecycle=self._environment_lifecycle,
        )

        initial_invocation = request._fork_group_initial_invocation
        if initial_invocation is not None and (
            initial_invocation.request.session_id != destination_session_id
            or _fork_group_initial_invocation_request_sha256(initial_invocation.request)
            != initial_invocation.request_sha256
        ):
            raise RuntimeError("Fork-group initial invocation authority is inconsistent.")

        def current_child_execution_profile() -> ExecutionProfileIdentity:
            candidate = _execution_profile_identity(
                registered_agent=registered_agent,
                provider_name=registered_provider.name,
                registered_provider=registered_provider,
                model=model,
                durable_system_prompt=prompt_workflow.rendered_child_prompt,
                redactor=self._secret_redactor,
                registered_environment=registered_environment,
                process_identity=self._execution_profile_process_identity,
                runtime_hooks=self._runtime_hooks,
                loop_policies=self._loop_policies,
                loop_policy_execution_profile_identities=(
                    self._loop_policy_execution_profile_identities
                ),
                request_loop_policies=(
                    () if initial_invocation is None else initial_invocation.request.loop_policies
                ),
                request_loop_policy_instance_identities=(
                    ()
                    if initial_invocation is None
                    else self._request_loop_policy_instance_identities(
                        initial_invocation.request.loop_policies
                    )
                ),
                budget_policy=self._get_budget_policy(),
                request_budget_limits=(
                    () if initial_invocation is None else initial_invocation.request.budget_limits
                ),
                causal_budget_id=source_session.causal_budget_id,
                structured_output=(
                    None
                    if initial_invocation is None
                    else initial_invocation.request.structured_output
                ),
                thinking=(
                    None if initial_invocation is None else initial_invocation.request.thinking
                ),
                max_steps=(
                    16 if initial_invocation is None else initial_invocation.request.max_steps
                ),
                limits=(None if initial_invocation is None else initial_invocation.request.limits),
                retry_policy=(
                    None
                    if initial_invocation is None
                    else self._effective_retry_policy(initial_invocation.request.retry_policy)
                ),
            )
            if (
                prompt_workflow.rendered_child_prompt is None
                and request.system_prompt_policy is ForkSystemPromptPolicy.INHERIT_SOURCE
            ):
                candidate = execution_profile_with_component(
                    candidate,
                    source_execution_profile.component(
                        ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION
                    ),
                )
            return candidate

        initial_invocation_profile: ExecutionProfileIdentity | None = None
        selected_execution_profile = source_execution_profile
        execution_profile_decision: ExecutionProfileDecision | None = None
        if request.execution_profile_selection is ForkExecutionProfileSelection.CURRENT_CHILD:
            candidate_execution_profile = current_child_execution_profile()
            unavailable_child_components = unavailable_execution_profile_components(
                candidate_execution_profile
            )
            if unavailable_child_components:
                raise RuntimeError(
                    "Current-child profile selection has unavailable required components: "
                    + ", ".join(component.value for component in unavailable_child_components)
                )
            decision_session = source_session.model_copy(
                update={
                    "id": destination_session_id,
                    "agent_name": agent_name,
                    "provider_name": registered_provider.name,
                    "model": model,
                    "parent_session_id": source_session.id,
                    "environment_name": environment_name,
                    "run_epoch": 0,
                },
                deep=True,
            )
            changed_profile_components = changed_execution_profile_components(
                source_execution_profile,
                candidate_execution_profile,
            )
            execution_profile_decision = await self._classify_execution_profile(
                session=source_session,
                decision_session=decision_session,
                expected_profile=source_execution_profile,
                candidate_profile=candidate_execution_profile,
                changed_component_classes=changed_profile_components,
                intent=request.profile_adoption,
                adoption_request_fingerprint=fork_request_sha256,
                target_changed=(
                    registered_provider.name != source_session.provider_name
                    or model != source_session.model
                ),
                target_provider_name=registered_provider.name,
                target_model=model,
                source_provider_name=source_session.provider_name,
                source_model=source_session.model,
                force_authority_review=True,
            )
            if execution_profile_decision.kind in {
                ExecutionProfileDecisionKind.MIGRATION_REQUIRED,
                ExecutionProfileDecisionKind.REJECTED,
            }:
                error_type = (
                    ExecutionProfileMigrationRequired
                    if execution_profile_decision.kind
                    is ExecutionProfileDecisionKind.MIGRATION_REQUIRED
                    else ExecutionProfileAdoptionRejected
                )
                raise error_type(
                    session_id=destination_session_id,
                    expected_profile_fingerprint=source_execution_profile.fingerprint,
                    candidate_profile_fingerprint=candidate_execution_profile.fingerprint,
                    changed_component_classes=changed_profile_components,
                )
            selected_execution_profile = candidate_execution_profile
            initial_invocation_profile = (
                None if initial_invocation is None else candidate_execution_profile
            )
        elif initial_invocation is not None:
            candidate_execution_profile = current_child_execution_profile()
            changed_profile_components = changed_execution_profile_components(
                source_execution_profile,
                candidate_execution_profile,
            )
            allowed_initial_changes = {
                ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY,
                ExecutionProfileComponentClass.STRUCTURED_OUTPUT,
                ExecutionProfileComponentClass.FINALIZATION,
            }
            unsupported_changes = tuple(
                component
                for component in changed_profile_components
                if component not in allowed_initial_changes
            )
            if unsupported_changes:
                raise RuntimeError(
                    "An inherited fork-group branch changed execution authority outside its "
                    "declared initial budget, structured-output, or finalization semantics: "
                    + ", ".join(component.value for component in unsupported_changes)
                    + ". Use an explicitly authorized current-child branch."
                )
            unavailable_child_components = unavailable_execution_profile_components(
                candidate_execution_profile
            )
            if unavailable_child_components:
                raise RuntimeError(
                    "Fork-group initial invocation has unavailable required components: "
                    + ", ".join(component.value for component in unavailable_child_components)
                )
            initial_invocation_profile = candidate_execution_profile

        if (
            prompt_workflow.prompt_replacement is not None
            and registered_provider.name != source_session.provider_name
        ):
            raise ValueError(
                "Prompt-anatomy succession to an agent with a different provider is not supported."
            )

        source_projection_cursor = session_model_projection_cursor(source_session)
        model_changed = model != source_session.model
        provider_changed = registered_provider.name != source_session.provider_name
        target_changed = model_changed or provider_changed
        expected_fork_transcript: tuple[Message, ...] | None = None
        fork_projection_cursor = 0
        selected_source_cursor = 0
        source_snapshot_cursor = 0
        source_prompt_sha256: str | None = None
        child_prompt_sha256: str | None = None
        if prompt_workflow.requires_transcript_snapshot(
            target_changed=target_changed,
            source_projection_cursor=source_projection_cursor,
        ):
            source_snapshot = await self.session_store.load_transcript_snapshot(source_session.id)
            source_snapshot_cursor = source_snapshot.cursor
            if source_projection_cursor > source_snapshot.cursor:
                del source_snapshot
                raise ValueError(
                    "Source session model-target projection cursor exceeds its transcript."
                )
            if request.transcript_cursor is not None:
                if request.transcript_cursor > source_snapshot.cursor:
                    del source_snapshot
                    raise ValueError("transcript_cursor is greater than source transcript length.")
                selected_source_records = [
                    record
                    for record in source_snapshot.records
                    if record.index < request.transcript_cursor
                ]
            else:
                selected_source_records = list(source_snapshot.records)
            selected_source_cursor = (
                source_snapshot.cursor
                if request.transcript_cursor is None
                else request.transcript_cursor
            )
            selected_messages = [
                detach_message(record.message) for record in selected_source_records
            ]
            source_messages = [detach_message(record.message) for record in source_snapshot.records]
            try:
                prompt_workflow.require_inherited_system_projection(
                    source_messages=source_messages,
                    selected_messages=selected_messages,
                )
            except BaseException:
                selected_source_records.clear()
                selected_messages.clear()
                del source_snapshot
                raise
            finally:
                source_messages.clear()
            source_prompt_sha256 = system_prompt_messages_sha256(selected_messages)
            selected_messages, child_prompt_sha256 = prompt_workflow.project_selected_messages(
                selected_messages,
                [record.interaction_id for record in selected_source_records],
            )
            expected_fork_transcript = tuple(selected_messages)
            if (
                self._secret_redactor.has_values
                and not session_request_boundary.fork_transcript_is_secret_free(
                    expected_fork_transcript,
                    redactor=self._secret_redactor,
                )
            ):
                selected_source_records.clear()
                selected_messages.clear()
                expected_fork_transcript = None
                del source_snapshot
                raise ValueError(FORK_TRANSCRIPT_VALIDATION_ERROR) from None
            if prompt_workflow.requires_portability_validation(target_changed=target_changed):
                portable_projection = None
                portable_messages: list[Message] = []
                try:
                    if target_changed:
                        portable_projection = model_target.project_portable_transcript(
                            list(expected_fork_transcript)
                        )
                        portable_messages = [
                            detach_message(message) for message in portable_projection.messages
                        ]
                    else:
                        portable_messages = [
                            detach_message(message) for message in expected_fork_transcript
                        ]
                    if portable_messages:
                        validate_context_messages(portable_messages)
                    if target_changed:
                        hook_tools: list[dict[str, Any]] = []
                        try:
                            hook_tools = _model_request_tools(
                                tool_exposure=_all_registered_tool_exposure(registered_agent),
                                structured_output=None,
                            )
                            registered_provider.provider.preflight_portable_messages(
                                model=model,
                                messages=portable_messages,
                                tools=hook_tools,
                            )
                        finally:
                            hook_tools.clear()
                    fork_projection_cursor = len(expected_fork_transcript)
                except BaseException:
                    expected_fork_transcript = None
                    portable_projection = None
                    portable_messages.clear()
                    selected_source_records.clear()
                    del source_snapshot
                    raise
                finally:
                    portable_messages.clear()
            else:
                fork_projection_cursor = sum(
                    record.index < source_projection_cursor for record in selected_source_records
                )
            selected_source_records.clear()
            del source_snapshot

        def reject_active_or_expired_recovery_claim(
            source_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            claim_now = self._clock()
            recovery_claim = _incomplete_recovery_claim_from_checkpoint(source_checkpoint)
            if recovery_claim is not None and recovery_claim[1] <= claim_now:
                raise _ExpiredIncompleteRecoveryClaim(recovery_claim[0])
            return _checkpoint_without_active_incomplete_recovery_claim(
                source_checkpoint,
                now=claim_now,
            )

        def validate_source_checkpoint(
            current_source: Session,
            source_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            source_checkpoint = reject_active_or_expired_recovery_claim(source_checkpoint)
            if _initial_transcript_pending_interaction_id(source_checkpoint) is not None:
                raise RuntimeError(
                    "Source session setup did not publish its authoritative initial "
                    "transcript; fork fails closed. Start a new session."
                )
            if (
                source_checkpoint is not None
                and _SESSION_OPERATIONS_CHECKPOINT_KEY in source_checkpoint
            ):
                operations = _session_operation_state(source_checkpoint)
                _abandon_expired_session_operation(operations, now=self._clock())
                source_checkpoint[_SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
            active_operation_id = _active_session_operation_id(source_checkpoint)
            if active_operation_id is not None:
                raise RuntimeError(
                    f"Session has an active durable operation: {active_operation_id}"
                )
            if current_source.status == SessionStatus.INTERRUPTED and not request.copy_checkpoint:
                raise ValueError("Interrupted sessions cannot be forked without checkpoint state.")
            if current_source.status == SessionStatus.INTERRUPTED and source_checkpoint is None:
                raise RuntimeError(
                    "Interrupted session cannot be forked because checkpoint state is missing."
                )
            if (
                pending_user_input_from_checkpoint(
                    source_checkpoint,
                    redactor=self._secret_redactor,
                    consume_on_rejection=True,
                )
                is not None
            ):
                raise RuntimeError(
                    "Session awaiting user input cannot be forked; answer it with "
                    "resolve_user_input(...) first."
                )
            if (
                approval_support.pending_approval_from_checkpoint(
                    source_checkpoint,
                    redactor=self._secret_redactor,
                    consume_on_rejection=True,
                )
                is not None
            ):
                raise RuntimeError(
                    "Session awaiting tool approval cannot be forked; resolve it with "
                    "resolve_tool_approval(...) first."
                )
            if (
                tool_round_recovery.pending_tool_round_from_checkpoint(
                    source_checkpoint,
                    redactor=self._secret_redactor,
                    consume_on_rejection=True,
                )
                is not None
            ):
                raise RuntimeError(
                    "Session with a pending tool round cannot be forked; resume or "
                    "recover the session first."
                )
            if (
                source_checkpoint is not None
                and _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY in source_checkpoint
            ):
                raise RuntimeError(
                    "Session has an incomplete background interruption cascade. "
                    "Retry the interruption before forking it."
                )
            return source_checkpoint

        def checkpoint_transform(
            current_source: Session,
            source_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            """Validate live checkpoint state even when the fork will discard it."""

            if expected_source_snapshot is not None:
                _, live_source_profile, _ = (
                    session_request_boundary.prepare_fork_source_execution_profile(
                        current_source,
                        source_checkpoint,
                    )
                )
                if live_source_profile.fingerprint != expected_source_snapshot.profile_fingerprint:
                    raise SessionRunFenced(
                        "Fork source execution profile changed after its coordinator "
                        "snapshot was frozen."
                    )
            if expected_source_snapshot is not None and (
                fork_source_checkpoint_sha256(source_checkpoint)
                != expected_source_snapshot.checkpoint_sha256
            ):
                raise SessionRunFenced(
                    "Fork source checkpoint changed after its coordinator snapshot was frozen."
                )
            source_checkpoint = validate_source_checkpoint(current_source, source_checkpoint)
            prompt_workflow.require_prepared_intent(source_checkpoint)
            fork_checkpoint = approval_support.checkpoint_for_fork(
                checkpoint=source_checkpoint,
            )
            if not request.copy_checkpoint:
                return None
            if fork_checkpoint is not None:
                fork_checkpoint.pop(_SESSION_OPERATIONS_CHECKPOINT_KEY, None)
                fork_checkpoint.pop(_SESSION_RUN_OPERATION_CHECKPOINT_KEY, None)
                fork_checkpoint.pop(
                    _QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY,
                    None,
                )
                # Active invocation authority is bound to the source session's
                # interaction and run epoch, never to the derived child.
                fork_checkpoint.pop(
                    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
                    None,
                )
                # A model-step publication receipt belongs to the source session.
                # The child inherits transcript state, not its reconstruction pointer.
                fork_checkpoint.pop(
                    model_completion_publication.LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
                    None,
                )
            if not session_request_boundary.fork_checkpoint_is_secret_free(
                fork_checkpoint,
                redactor=self._secret_redactor,
            ):
                if source_checkpoint is not None:
                    source_checkpoint.clear()
                if fork_checkpoint is not None:
                    fork_checkpoint.clear()
                source_checkpoint = fork_checkpoint = None
                raise ValueError(
                    "source_session.checkpoint contains a workload secret and cannot "
                    "be copied without changing resumable execution state."
                ) from None
            return fork_checkpoint

        def transcript_validator(messages: tuple[Message, ...]) -> bool:
            return (
                (expected_fork_transcript is None or messages == expected_fork_transcript)
                and (
                    expected_source_snapshot is None
                    or session_input_messages_sha256(messages)
                    == expected_source_snapshot.transcript_sha256
                )
                and session_request_boundary.fork_transcript_is_secret_free(
                    messages,
                    redactor=self._secret_redactor,
                )
            )

        inherited_taint_labels = await self._tool_round_executor.prior_taint_labels_for_policy(
            session_id=source_session.id,
            policy=source_registered_agent.tool_policy,
            request_metadata=source_session.metadata,
        )
        fork_metadata = copy_json_value(request.metadata, "metadata")
        if inherited_taint_labels:
            fork_metadata = metadata_with_taint_labels(
                fork_metadata,
                inherited_taint_labels | taint_labels_from_metadata(request.metadata),
            )
        fork_metadata = _session_metadata_with_model_projection(
            fork_metadata,
            target=ModelTarget(
                provider_name=registered_provider.name,
                model=model,
            ),
            transcript_cursor=fork_projection_cursor,
        )
        prompt_workflow.attach_receipt(
            fork_metadata,
            source_session=source_session,
            destination_session_id=destination_session_id,
            selected_source_cursor=selected_source_cursor,
            source_prompt_sha256=source_prompt_sha256,
            child_prompt_sha256=child_prompt_sha256,
            child_agent_name=agent_name,
            provider_name=registered_provider.name,
            model=model,
            model_changed=model_changed,
            inherited_taint_labels=inherited_taint_labels,
        )
        fork_session = Session(
            id=destination_session_id,
            agent_name=agent_name,
            provider_name=registered_provider.name,
            model=model,
            parent_session_id=source_session.id,
            causal_budget_id=source_session.causal_budget_id,
            runtime_name=source_session.runtime_name,
            runtime_version=source_session.runtime_version,
            environment_name=environment_name,
            status=source_session.status,
            invocation=fork_session_invocation(source_session),
            labels=source_session.labels,
            metadata=copy_json_value(fork_metadata, "metadata"),
        )
        try:
            fork_session = session_request_boundary.prepare_derived_fork_session(
                fork_session,
                source_session=source_session,
                runtime_generated_session_id=runtime_generated_session_id,
                store_resolved_source_session_id=store_resolved_source_session_id,
                redactor=self._secret_redactor,
            )
        except ValueError:
            del (
                fork_session,
                source_session,
                registered_agent,
                source_registered_agent,
                registered_environment,
                registered_provider,
            )
            if type(fork_metadata) is dict:
                fork_metadata.clear()
            inherited_taint_labels = frozenset()
            agent_name = model = environment_name = destination_session_id = ""
            runtime_generated_session_id = None
            raise
        if expected_source_snapshot is not None:
            authoritative_metadata = copy_json_value(fork_session.metadata, "metadata")
            authoritative_metadata[FORK_GROUP_SOURCE_SNAPSHOT_METADATA_KEY] = {
                "source_session_id": source_session.id,
                "status": source_session.status.value,
                "run_epoch": expected_source_snapshot.run_epoch,
                "transcript_cursor": expected_source_snapshot.transcript_cursor,
                "transcript_sha256": expected_source_snapshot.transcript_sha256,
                "checkpoint_sha256": expected_source_snapshot.checkpoint_sha256,
                "execution_profile_fingerprint": (expected_source_snapshot.profile_fingerprint),
                "causal_budget_id": source_session.causal_budget_id,
            }
            fork_session = fork_session.model_copy(
                update={"metadata": authoritative_metadata},
                deep=True,
            )
        fork_session = await prompt_workflow.prepare_effect(
            session_store=self.session_store,
            source_session=source_session,
            source_snapshot_cursor=source_snapshot_cursor,
            fork_session=fork_session,
            validate_source_checkpoint=validate_source_checkpoint,
            clock=self._clock,
        )
        shared_live_workspace = (
            registered_environment is not None
            and registered_environment.environment.workspace is not None
            and source_session.environment_name is not None
            and source_session.environment_name == fork_session.environment_name
        )
        fork_event_payload: dict[str, Any] = {
            "source_session_id": source_session.id,
            "source_status": source_session.status.value,
            "parent_session_id": fork_session.parent_session_id,
            "causal_budget_id": fork_session.causal_budget_id,
            "transcript_cursor": request.transcript_cursor,
            "copy_checkpoint": request.copy_checkpoint,
            "system_prompt_policy": request.system_prompt_policy.value,
            "agent_name": fork_session.agent_name,
            "provider_name": fork_session.provider_name,
            "model": fork_session.model,
            "environment_name": fork_session.environment_name,
            "inherited_taint_labels": sorted(inherited_taint_labels),
            "execution_profile_selection": request.execution_profile_selection.value,
            "selected_profile_fingerprint": selected_execution_profile.fingerprint,
            "source_profile_fingerprint": source_execution_profile.fingerprint,
            "fork_request_sha256": fork_request_sha256,
            "workspace_lineage": WorkspaceForkLineage(
                status=(
                    WorkspaceForkLineageStatus.SHARED_OR_AMBIGUOUS
                    if shared_live_workspace
                    else WorkspaceForkLineageStatus.UNPROVEN
                ),
                detail_code=(
                    "shared_live_workspace_not_isolated"
                    if shared_live_workspace
                    else "child_workspace_derivation_unproven"
                ),
            ).model_dump(mode="json"),
        }
        if initial_invocation is not None:
            if initial_invocation_profile is None:
                raise AssertionError("Fork-group initial invocation lost its frozen profile.")
            fork_event_payload.update(
                {
                    "initial_invocation_request_sha256": (initial_invocation.request_sha256),
                    "initial_invocation_profile_fingerprint": (
                        initial_invocation_profile.fingerprint
                    ),
                }
            )
        ordinary_fork_event = event_with_runtime_generated_id(
            Event(
                id=(
                    "fork_"
                    + hashlib.sha256(
                        canonical_durable_json_bytes(
                            {
                                "child_session_id": fork_session.id,
                                "request_sha256": fork_request_sha256,
                            },
                            "profiled_fork_event",
                        )
                    ).hexdigest()
                ),
                type=EventType.SESSION_FORKED,
                session_id=fork_session.id,
                timestamp=fork_session.created_at,
                agent_name=registered_agent.spec.name,
                environment_name=_environment_name(registered_environment),
                payload=fork_event_payload,
            )
        )
        raw_fork_event = prompt_workflow.raw_fork_event(
            created=fork_session,
            ordinary_event=ordinary_fork_event,
        )
        raw_fork_event = raw_fork_event.model_copy(
            update={"payload": {**raw_fork_event.payload, **fork_event_payload}},
            deep=True,
        )
        fork_event = self._event_writer.prepare(_fork_event_with_runtime_authority(raw_fork_event))
        decision_record: ForkExecutionProfileDecisionRecord | None = None
        prepared_events: list[Event] = []
        if execution_profile_decision is not None:
            if execution_profile_decision.actor is None:
                raise AssertionError("Current-child profile selection lost its actor.")
            decision_event = self._event_writer.prepare(execution_profile_decision.event)
            decision_payload = decision_event.payload
            decision_record = ForkExecutionProfileDecisionRecord.model_validate(
                {
                    "kind": execution_profile_decision.kind,
                    "policy_identity": decision_payload.get("policy_identity"),
                    "policy_reason": decision_payload.get("policy_reason"),
                    "authority_decision": execution_profile_decision.authority_decision,
                    "actor": decision_payload.get("actor"),
                    "reason": decision_payload.get("reason"),
                    "idempotency_identity": decision_payload.get("idempotency_identity"),
                    "adoption_request_fingerprint": fork_request_sha256,
                    "event_id": decision_event.id,
                }
            )
            prepared_events.append(decision_event)
        relationship = SessionForkProfileRelationship(
            request_sha256=fork_request_sha256,
            source_session_id=source_session.id,
            child_session_id=fork_session.id,
            child_agent_name=fork_session.agent_name,
            child_provider_name=fork_session.provider_name,
            child_model=fork_session.model,
            child_environment_name=fork_session.environment_name,
            source_status=source_session.status,
            source_run_epoch=source_session.run_epoch,
            source_profile_source=source_profile_source,
            source_profile=source_execution_profile,
            source_active_interaction_id=(
                None if source_active_profile is None else source_active_profile.interaction_id
            ),
            source_active_run_epoch=(
                None if source_active_profile is None else source_active_profile.run_epoch
            ),
            transcript_cursor=request.transcript_cursor,
            copy_checkpoint=request.copy_checkpoint,
            system_prompt_policy=request.system_prompt_policy,
            selection=request.execution_profile_selection,
            selected_profile=selected_execution_profile,
            initial_invocation_request_sha256=(
                None if initial_invocation is None else initial_invocation.request_sha256
            ),
            initial_invocation_profile=initial_invocation_profile,
            decision=decision_record,
            fork_event_id=fork_event.id,
        )
        authoritative_metadata = copy_json_value(fork_session.metadata, "metadata")
        authoritative_metadata[EXECUTION_PROFILE_METADATA_KEY] = execution_profile_session_metadata(
            selected_execution_profile
        )
        authoritative_metadata[FORK_EXECUTION_PROFILE_METADATA_KEY] = relationship.model_dump(
            mode="json"
        )
        fork_session = fork_session.model_copy(
            update={"metadata": authoritative_metadata},
            deep=True,
        )
        prepared_events.append(fork_event)

        async def reconcile_profiled_fork(
            initial_error: BaseException | None,
        ) -> tuple[ProfiledSessionForkResult, BaseException | None]:
            try:
                existing = await self.session_store.load(fork_session.id)
                if existing is None or not _prompt_fork_matches_prepared(
                    existing,
                    fork_session,
                ):
                    if initial_error is not None:
                        raise initial_error
                    raise RuntimeError(
                        "Profiled fork publication returned no authoritative result."
                    )
                existing_relationship = session_fork_profile_relationship(existing)
                if existing_relationship != relationship:
                    raise RuntimeError("Existing profiled fork conflicts with the exact request.")
                recovered_events: list[Event] = []
                for prepared_event in prepared_events:
                    records = await self.session_store.query_events(
                        EventQuery(
                            session_id=existing.id,
                            event_id=prepared_event.id,
                            limit=1,
                        )
                    )
                    if len(records) != 1 or records[0].event.model_dump(
                        mode="json",
                        exclude={"timestamp"},
                    ) != prepared_event.model_dump(
                        mode="json",
                        exclude={"timestamp"},
                    ):
                        raise RuntimeError(
                            "Existing profiled fork is missing exact durable evidence."
                        )
                    recovered_events.append(records[0].event)
                validate_profiled_fork_evidence(
                    fork=existing,
                    relationship=relationship,
                    events=recovered_events,
                )
                recovered = ProfiledSessionForkResult(
                    session=existing,
                    events=tuple(recovered_events),
                )
                # Lost acknowledgements and malformed ordinary acknowledgements
                # are repaired by the immutable receipt. Process-control outcomes
                # remain authoritative even when their durable mutation committed.
                retained_error = (
                    initial_error
                    if initial_error is not None and not isinstance(initial_error, Exception)
                    else None
                )
                return recovered, retained_error
            except BaseException as reconciliation_error:
                if initial_error is None or reconciliation_error is initial_error:
                    raise
                raise BaseExceptionGroup(
                    "Profiled fork publication and exact reconciliation failed.",
                    [initial_error, reconciliation_error],
                ) from None

        async def publish_or_reconcile_profiled_fork() -> tuple[
            ProfiledSessionForkResult,
            BaseException | None,
        ]:
            try:
                raw_result = await self.session_store.create_profiled_fork(
                    source_session_id=source_session.id,
                    fork=fork_session.model_copy(deep=True),
                    source_statuses=set(_FORKABLE_SESSION_STATUSES),
                    transcript_cursor=request.transcript_cursor,
                    checkpoint_transform=checkpoint_transform,
                    system_prompt_replacement=prompt_workflow.prompt_replacement,
                    expected_source_run_epoch=source_session.run_epoch,
                    relationship=SessionForkProfileRelationship.model_validate(
                        relationship.model_dump(mode="json")
                    ),
                    events=[copy_event(event) for event in prepared_events],
                    transcript_validator=transcript_validator,
                )
                try:
                    result = copy_profiled_session_fork_result(raw_result)
                    if not _prompt_fork_matches_prepared(
                        result.session,
                        fork_session,
                    ):
                        raise RuntimeError(
                            "Profiled fork publication returned a different session snapshot."
                        )
                    validate_profiled_fork_evidence(
                        fork=result.session,
                        relationship=relationship,
                        events=result.events,
                    )
                    return result, None
                except Exception as invalid_result:
                    return await reconcile_profiled_fork(invalid_result)
            except asyncio.CancelledError as child_cancellation:
                return await reconcile_profiled_fork(
                    unexpected_child_cancellation_error(
                        child_cancellation,
                        operation="Profiled fork durable publication",
                    )
                )
            except BaseException as initial_error:
                return await reconcile_profiled_fork(initial_error)

        async def await_owned_fork_task(
            task: asyncio.Task[Any],
        ) -> tuple[ShieldedTaskOutcome[Any], list[BaseException]]:
            """Settle one dispatched fork mutation without losing supervisor control."""

            supervisory_controls: list[BaseException] = []
            while True:
                try:
                    return await await_shielded_task_outcome(task), supervisory_controls
                except asyncio.CancelledError:
                    # New Task.cancel() delivery is owned by the shield helper. An
                    # unowned cancellation must retain its normal caller meaning.
                    raise
                except BaseException as control:
                    # Generator/stream abandonment and process control cannot
                    # orphan a durable mutation. Retain the exact signal until
                    # the owned task reaches a terminal outcome.
                    supervisory_controls.append(control)

        publication_task = asyncio.create_task(
            capture_awaitable_outcome(publish_or_reconcile_profiled_fork),
            name="cayu-profiled-session-fork-publication",
        )
        publication_outcome, publication_controls = await await_owned_fork_task(publication_task)
        publication_error = publication_outcome.error
        publication_result = publication_outcome.result
        if publication_error is None and publication_result is not None:
            publication_error = publication_result.error
            publication_value = publication_result.result
        else:
            publication_value = None
        if (
            isinstance(publication_error, asyncio.CancelledError)
            and publication_outcome.cancellation is None
        ):
            publication_error = unexpected_child_cancellation_error(
                publication_error,
                operation="Profiled fork durable publication",
            )
        if publication_error is not None:
            if (
                publication_outcome.cancellation is not None
                and publication_outcome.cancellation_requests_consumed
            ):
                retain_workspace_observation_pending_cancellation_requests(
                    publication_outcome.cancellation,
                    publication_outcome.cancellation_requests_consumed,
                )
            restore_task_cancellation_requests(publication_outcome.cancellation_requests_consumed)
            if publication_outcome.cancellation is not None:
                if isinstance(publication_error, Exception) and not publication_controls:
                    raise publication_outcome.cancellation from publication_error
                concurrent_control = BaseExceptionGroup(
                    "Profiled fork publication failed concurrently with caller cancellation.",
                    [
                        publication_error,
                        *publication_controls,
                        publication_outcome.cancellation,
                    ],
                )
                retain_workspace_observation_pending_cancellation_requests(
                    concurrent_control,
                    publication_outcome.cancellation_requests_consumed,
                )
                raise concurrent_control from None
            if publication_controls:
                raise BaseExceptionGroup(
                    "Profiled fork publication failed with concurrent process control.",
                    [publication_error, *publication_controls],
                ) from None
            if isinstance(publication_error, SessionForkSourceNotFound):
                raise session_request_boundary.ForkSourceNotFoundError(
                    "Fork source session was not found."
                ) from None
            if isinstance(publication_error, SessionForkActiveModelStageConflict):
                raise session_request_boundary.ForkActiveModelStageError(
                    "Fork source session has an active model-completion stage."
                ) from None
            if isinstance(publication_error, _ExpiredIncompleteRecoveryClaim):
                current = await self._require_session(source_session.id)
                fenced = await self._recovery_coordinator.fence_expired_incomplete_recovery_claim(
                    session=current,
                    claim_id=publication_error.claim_id,
                )
                if not fenced:
                    raise RuntimeError(
                        "Expired incomplete-session recovery ownership changed while the "
                        "fork was fencing it; retry with current session state."
                    ) from None
                raise ValueError(
                    "Session fork fenced an expired incomplete-session recovery owner; "
                    "retry with current session state."
                ) from None
            raise publication_error
        if publication_value is None:
            if (
                publication_outcome.cancellation is not None
                and publication_outcome.cancellation_requests_consumed
            ):
                retain_workspace_observation_pending_cancellation_requests(
                    publication_outcome.cancellation,
                    publication_outcome.cancellation_requests_consumed,
                )
            restore_task_cancellation_requests(publication_outcome.cancellation_requests_consumed)
            missing_result = RuntimeError(
                "Profiled fork publication returned no authoritative result."
            )
            if publication_outcome.cancellation is not None:
                if not publication_controls:
                    raise publication_outcome.cancellation from missing_result
                concurrent_control = BaseExceptionGroup(
                    "Profiled fork publication returned no result with concurrent control.",
                    [
                        missing_result,
                        *publication_controls,
                        publication_outcome.cancellation,
                    ],
                )
                if publication_outcome.cancellation_requests_consumed:
                    retain_workspace_observation_pending_cancellation_requests(
                        concurrent_control,
                        publication_outcome.cancellation_requests_consumed,
                    )
                raise concurrent_control from None
            if publication_controls:
                raise BaseExceptionGroup(
                    "Profiled fork publication returned no result with process control.",
                    [missing_result, *publication_controls],
                ) from None
            raise missing_result
        profiled_result, post_commit_control = publication_value
        created = profiled_result.session
        if (
            created.id != fork_session.id
            or created.parent_session_id != fork_session.parent_session_id
            or created.causal_budget_id != fork_session.causal_budget_id
            or session_fork_profile_relationship(created) != relationship
            or len(profiled_result.events) != len(prepared_events)
            or any(
                stored.model_dump(mode="json", exclude={"timestamp"})
                != prepared.model_dump(mode="json", exclude={"timestamp"})
                for stored, prepared in zip(
                    profiled_result.events,
                    prepared_events,
                    strict=True,
                )
            )
        ):
            raise RuntimeError("Session store changed prepared fork identity authority.")
        completion_task = asyncio.create_task(
            capture_awaitable_outcome(
                lambda: prompt_workflow.complete_effect(
                    session_store=self.session_store,
                    source_session_id=source_session.id,
                    created=created,
                    clock=self._clock,
                )
            ),
            name="cayu-profiled-session-fork-effect-completion",
        )
        completion_outcome, completion_controls = await await_owned_fork_task(completion_task)
        completion_error = completion_outcome.error
        if completion_error is None and completion_outcome.result is not None:
            completion_error = completion_outcome.result.error
        if (
            isinstance(completion_error, asyncio.CancelledError)
            and completion_outcome.cancellation is None
        ):
            completion_error = unexpected_child_cancellation_error(
                completion_error,
                operation="Profiled fork effect completion",
            )
        authoritative_cancellation = (
            publication_outcome.cancellation or completion_outcome.cancellation
        )
        cancellation_requests_consumed = (
            publication_outcome.cancellation_requests_consumed
            + completion_outcome.cancellation_requests_consumed
        )
        concurrent_failures: list[BaseException] = []
        for failure in (
            post_commit_control,
            completion_error,
            *publication_controls,
            *completion_controls,
        ):
            if isinstance(failure, BaseException):
                concurrent_failures.append(failure)
            elif failure is not None:
                concurrent_failures.append(
                    RuntimeError("Profiled fork completion returned invalid failure evidence.")
                )
        if authoritative_cancellation is not None:
            if cancellation_requests_consumed:
                retain_workspace_observation_pending_cancellation_requests(
                    authoritative_cancellation,
                    cancellation_requests_consumed,
                )
            restore_task_cancellation_requests(cancellation_requests_consumed)
            ordinary_failures = [
                failure for failure in concurrent_failures if isinstance(failure, Exception)
            ]
            if concurrent_failures and len(ordinary_failures) == len(concurrent_failures):
                cleanup_failure = (
                    ordinary_failures[0]
                    if len(ordinary_failures) == 1
                    else ExceptionGroup(
                        "Profiled fork post-commit completion failed.",
                        ordinary_failures,
                    )
                )
                raise authoritative_cancellation from cleanup_failure
            if concurrent_failures:
                concurrent_control = BaseExceptionGroup(
                    "Profiled fork committed with concurrent process control and cancellation.",
                    [*concurrent_failures, authoritative_cancellation],
                )
                if cancellation_requests_consumed:
                    retain_workspace_observation_pending_cancellation_requests(
                        concurrent_control,
                        cancellation_requests_consumed,
                    )
                raise concurrent_control from None
            raise authoritative_cancellation
        if concurrent_failures:
            if len(concurrent_failures) == 1:
                raise concurrent_failures[0]
            raise BaseExceptionGroup(
                "Profiled fork post-commit completion failed.",
                concurrent_failures,
            ) from None
        delivered_events = await self._event_writer.fan_out_persisted(list(profiled_result.events))
        yield delivered_events[-1]

    async def _publish_assistant_model_completion(
        self,
        publication: ModelCompletionPublicationRequest,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        task_id: str | None,
        request_metadata: dict[str, Any],
        structured_output: StructuredOutputSpec | None,
        thinking: ThinkingConfig | None,
        max_steps: int,
        limits: RunLimits,
        budget_limits: tuple[BudgetLimit, ...],
        retry_policy: RetryPolicy,
        structured_output_attempt: int | None,
        structured_output_retries: int,
        run_limit_accounting: RunLimitAccountingContext | None,
    ) -> ModelCompletionPublicationResult:
        """Commit one staged model completion and its next durable action atomically."""

        if publication.dispatch.stage.session_id != session.id:
            raise RuntimeError("Model completion publication belongs to a different session.")
        assistant_step_result = publication.assistant_step_result
        assistant_message = publication.authoritative_assistant_message
        tool_calls = (
            assistant_step_result.tool_calls
            if assistant_message is not None and assistant_step_result is not None
            else []
        )
        tool_round_identity = (
            assistant_step_result.tool_round_identity
            if assistant_message is not None and assistant_step_result is not None
            else None
        )
        if bool(tool_calls) != (tool_round_identity is not None):
            raise RuntimeError(
                "Model completion tool calls and tool-round identity must be published together."
            )
        source_checkpoint = await self.session_store.load_checkpoint(session.id)
        target_checkpoint = source_checkpoint
        tool_round_id = None if tool_round_identity is None else tool_round_identity.tool_round_id
        if tool_calls:
            if tool_round_identity is None or assistant_step_result is None:
                raise RuntimeError("Model completion lost its tool-round execution material.")
            if publication.tool_exposure is None:
                raise RuntimeError("Model completion lost its frozen tool exposure.")
            tool_exposure = validate_resolved_tool_exposure_authority(
                publication.tool_exposure,
                registered_agent.tool_capabilities,
            )
            tool_redactor = self._tool_round_executor.redactor_for_tool_calls(
                registered_agent=registered_agent,
                tool_calls=tool_calls,
            )
            target_checkpoint, _pending_round = (
                tool_round_recovery.checkpoint_with_pending_tool_round(
                    source_checkpoint,
                    agent_name=registered_agent.spec.name,
                    environment_name=_environment_name(registered_environment),
                    task_id=task_id,
                    tool_calls=tool_calls,
                    policy_outcomes=None,
                    tool_exposure=tool_exposure,
                    policy_context_version=1,
                    request_metadata=request_metadata,
                    assistant_message_state=(
                        "quarantined" if publication.defer_assistant_message else "published"
                    ),
                    quarantined_assistant_message=(
                        assistant_message if publication.defer_assistant_message else None
                    ),
                    secret_resolution_scope=invocation_secrets.registered_environment_secret_resolution_scope(
                        registered_environment
                    ),
                    structured_output=structured_output,
                    thinking=thinking,
                    max_steps=max_steps,
                    limits=limits,
                    run_limit_accounting=run_limit_accounting,
                    budget_limits=budget_limits,
                    retry_policy=retry_policy,
                    tool_round_identity=tool_round_identity,
                    redactor=tool_redactor,
                    source_model_step_id=tool_round_identity.model_step_id,
                    source_transcript_cursor=(publication.dispatch.stage.source_transcript_cursor),
                    model_step=assistant_step_result.step,
                    structured_output_attempt=structured_output_attempt,
                    structured_output_retries=structured_output_retries,
                    structured_output_validation=(publication.structured_output_validation),
                )
            )
        target_checkpoint = (
            {}
            if target_checkpoint is None
            else copy_json_value(target_checkpoint, "model_completion_checkpoint")
        )
        classification = publication.completion_event.payload.get("step_classification")
        if type(classification) is not dict:
            raise RuntimeError(
                "Model completion publication requires a durable step classification."
            )
        pointer = model_completion_publication.ModelStepPublicationCheckpoint(
            logical_step_id=publication.dispatch.logical_step_id,
            stage_id=publication.dispatch.stage_id,
            source_transcript_cursor=(publication.dispatch.stage.source_transcript_cursor),
            transcript_end_cursor=(
                publication.dispatch.stage.source_transcript_cursor
                + int(assistant_message is not None and not publication.defer_assistant_message)
            ),
            completion_event_id=publication.completion_event.id,
            classification=classification,
            assistant_message_published=(
                assistant_message is not None and not publication.defer_assistant_message
            ),
            assistant_message_deferred=publication.defer_assistant_message,
            tool_round_id=tool_round_id,
        )
        target_checkpoint[
            model_completion_publication.LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY
        ] = pointer.model_dump(mode="json")
        runtime_publication = RuntimePublicationRequest(
            publication_id=publication.dispatch.logical_step_id,
            kind="model-step",
            interaction_id=publication.completion_event.interaction_id,
            intent=publication.dispatch.intent,
            mutation=runtime_publication_checkpoint_mutation(
                source_checkpoint,
                target_checkpoint,
            ),
            transcript_messages=(
                ()
                if assistant_message is None or publication.defer_assistant_message
                else (assistant_message,)
            ),
            events=(publication.completion_event,),
        )

        async def commit_and_fan_out_once() -> ModelCompletionPublicationResult:
            completion = await self.session_store.complete_model_completion_stage(
                session.id,
                stage_id=publication.dispatch.stage_id,
                publication=runtime_publication,
            )
            promoted = await self.session_store.promote_model_completion_stage(
                session.id,
                stage_id=publication.dispatch.stage_id,
                expected_run_epoch=session.run_epoch,
            )
            await self._event_writer.fan_out_persisted([publication.completion_event])
            return ModelCompletionPublicationResult(
                completion=completion,
                publication=promoted,
            )

        async def commit_and_fan_out() -> ModelCompletionPublicationResult:
            try:
                return await commit_and_fan_out_once()
            except Exception as first_error:
                try:
                    return await commit_and_fan_out_once()
                except Exception as replay_error:
                    replay_error.add_note(
                        "Exact model-completion publication replay also failed after "
                        f"{type(first_error).__name__}: {first_error}"
                    )
                    raise replay_error from first_error

        commit_task = asyncio.create_task(commit_and_fan_out())
        outcome = await await_shielded_task_outcome(commit_task)
        cancellation = outcome.cancellation
        error = outcome.error
        if isinstance(error, asyncio.CancelledError) and cancellation is None:
            error = unexpected_child_cancellation_error(
                error,
                operation="Model completion durable publication",
            )
        if error is not None:
            if cancellation is not None:
                cancellation.add_note(
                    "Model completion durable publication also failed: "
                    f"{type(error).__name__}: {error}"
                )
                raise cancellation from error
            raise error
        if outcome.result is None:
            result_error = RuntimeError(
                "Model completion durable publication returned no acknowledgement."
            )
            if cancellation is not None:
                cancellation.add_note(str(result_error))
                raise cancellation from result_error
            raise result_error
        if cancellation is not None:
            raise cancellation
        return outcome.result

    async def _run_session(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity | None,
        messages: list[Message],
        messages_to_append: list[Message],
        max_steps: int,
        limits: RunLimits,
        budget_limits: tuple[BudgetLimit, ...],
        budget_policy: BudgetPolicy | None,
        retry_policy: RetryPolicy,
        structured_output: StructuredOutputSpec | None,
        thinking: ThinkingConfig | None,
        request_loop_policies: tuple[LoopPolicy, ...],
        request_metadata: dict[str, Any],
        request_trace_metadata: dict[str, Any],
        task_id: str | None,
        task_worker_id: str | None,
        start_event_type: EventType | None,
        start_event_payload: dict[str, Any],
        start_task_on_enter: bool = True,
        release_run_fence_on_exit: bool = True,
        messages_already_persisted: bool = False,
        messages_deferred: bool = False,
        deliver_queued_input_before_first_step: bool = True,
        pending_tool_round_source_transcript_cursor: int | None = None,
        run_limit_accounting: RunLimitAccountingContext | None = None,
        initial_model_step_identity: ModelStepIdentity | None = None,
        initial_model_step_number: int | None = None,
        initial_model_step_tool_exposure: ResolvedToolExposure | None = None,
        previous_tool_exposure_profile_id: str | None = None,
        preserve_failure_until_initial_provider_dispatch: bool = False,
    ) -> AsyncGenerator[Event, None]:
        # Deep defense for internal recovery callers. Public entry points
        # validate before claiming mutable session ownership.
        require_secret_free_structured_output_spec(
            structured_output,
            redactor=self._secret_redactor,
            field_name="structured_output",
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

        async def materialize_deferred_messages() -> None:
            nonlocal messages_deferred
            if not messages_deferred:
                return
            await self.session_store.materialize_deferred_interaction_input(session.id)
            messages_deferred = False

        async def materialize_deferred_messages_after_failure() -> None:
            if not messages_deferred:
                return
            checkpoint = await self.session_store.load_checkpoint(session.id)
            if (
                tool_round_recovery.pending_tool_round_from_checkpoint(
                    checkpoint,
                    redactor=self._secret_redactor,
                    consume_on_rejection=True,
                )
                is not None
            ):
                return
            await materialize_deferred_messages()

        provider = registered_provider.provider
        model_projection_cursor = session_model_projection_cursor(session)
        model_projection_source_prefix_count = 0
        model_projection_prefix_count = 0
        if model_projection_cursor:
            model_projection_snapshot = await self.session_store.load_transcript_snapshot(
                session.id
            )
            try:
                if model_projection_cursor > model_projection_snapshot.cursor:
                    raise ValueError(
                        "Model-target projection cursor exceeds the session transcript."
                    )
                deferred_tail_count = len(messages_to_append) if messages_deferred else 0
                retained_message_count = len(messages) - deferred_tail_count
                model_projection = _project_model_target_snapshot(
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
        task_started = task_id is not None and not start_task_on_enter
        task_start_attempted = task_started
        task_finished = False
        current_task = asyncio.current_task()
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None
        run_started_at = time.monotonic()
        # A fresh run means any earlier interrupt was fully handled before the
        # session transitioned back to RUNNING; drop a stale signal so it does
        # not force per-delta store polling for the whole resumed run.
        self._session_control.discard_interrupt_signal(session.id)
        limits = copy_run_limits(limits)
        budget_policy = copy_budget_policy(budget_policy)
        budget_limits = request_budget_limits_for_session(
            limits=budget_limits,
            agent_name=registered_agent.spec.name,
            causal_budget_id=session.causal_budget_id,
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
                    session_id=session.id,
                    budget_limits=budget_limits,
                    now=self._clock(),
                )
            )
        turn_usage_tracker = self._run_limit_controller.usage_tracker(session.id)
        baseline_events: list[Event] = []
        request_budget_notify_events: list[Event] = []
        close_new_pending_round_on_interrupt = False
        initial_provider_dispatch_started = False

        async def start_linked_task_if_needed(*, only_if_exists: bool = False) -> Event | None:
            nonlocal task_start_attempted, task_started
            if task_id is None or task_started:
                return None
            if self.task_store is None:
                raise RuntimeError("task_store is required when RunRequest.task_id is set.")
            if only_if_exists and await self.task_store.load_task(task_id) is None:
                return None
            task_start_attempted = True
            task = await self._start_task(
                task_id=task_id,
                session=session,
                worker_id=task_worker_id,
            )
            task_started = True
            if active_run is not None:
                active_run.task_started = True
            return await self._event_writer.emit(
                _task_event(
                    event_type=EventType.TASK_STARTED,
                    task=task,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                )
            )

        try:
            if structured_output is not None and (
                STRUCTURED_OUTPUT_TOOL_NAME in registered_agent.tools
            ):
                raise ValueError(
                    f"Tool name is reserved for structured output: {STRUCTURED_OUTPUT_TOOL_NAME}"
                )
            if current_task is not None:
                active_run = self._session_control.register_active_task(
                    session.id,
                    current_task,
                    task_id=task_id,
                    task_started=task_started,
                    task_finished=task_finished,
                    turn_registered_agent=registered_agent,
                    turn_environment_name=environment_name,
                    turn_started_at=run_started_at,
                    turn_usage_tracker=turn_usage_tracker,
                )
            await turn_usage_tracker.mark_current_position()
            if has_run_limit_accounting_authority(limits, budget_limits):
                baseline_events = await self._run_limit_controller.session_usage_events(session.id)
            if limits.scope == "run" and has_run_limits(limits) and run_limit_accounting is None:
                run_baseline = session_usage_summary(session.id, baseline_events)

            model_boundary = await self._recovery_coordinator.reconcile_model_completion_boundary(
                session,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
            )
            session = model_boundary.session
            if model_boundary.blocks_provider_dispatch and not messages_to_append:
                raise ModelCompletionManualRecoveryRequired(
                    "The latest model completion is already durable at the transcript tail. "
                    "Provider redispatch requires an explicit new message."
                )
            if model_boundary.state == "promoted" and model_boundary.completion_event is not None:
                yield copy_event(model_boundary.completion_event)
            factory_started_event = await self._environment_lifecycle.emit_factory_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
            )
            if factory_started_event is not None:
                yield factory_started_event
                task_start_event = await start_linked_task_if_needed()
                if task_start_event is not None:
                    yield task_start_event
            factory_resolution = await self._environment_lifecycle.resolve_factory(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=factory_started_event,
                operation=(
                    EnvironmentFactoryOperation.CREATE
                    if start_event_type is EventType.SESSION_STARTED
                    else EnvironmentFactoryOperation.RECONNECT
                ),
                execution_profile=execution_profile,
            )
            registered_environment = factory_resolution.registered_environment
            environment_name = _environment_name(registered_environment)
            if active_run is not None:
                active_run.turn_environment_name = environment_name
            for event in factory_resolution.events:
                yield event
            if factory_resolution.error is not None:
                raise factory_resolution.error
            binding_started_event = await self._environment_lifecycle.emit_binding_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
            )
            if binding_started_event is not None:
                yield binding_started_event
                task_start_event = await start_linked_task_if_needed()
                if task_start_event is not None:
                    yield task_start_event
            binding_result = await self._environment_lifecycle.bind(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=binding_started_event,
                execution_profile=execution_profile,
            )
            registered_environment = binding_result.registered_environment
            environment_name = _environment_name(registered_environment)
            if active_run is not None:
                active_run.turn_environment_name = environment_name
            for event in binding_result.events:
                yield event
            if binding_result.error is not None:
                raise binding_result.error
            async for event in self._tool_round_executor.emit_mcp_manifest_checks(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
            ):
                yield event
            if start_event_type is not None:
                interaction_started_event = await self._emit_interaction_started_if_needed(
                    session=session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                )
                if interaction_started_event is not None:
                    yield interaction_started_event
                start_event = Event(
                    type=start_event_type,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        **start_event_payload,
                        **_session_trace_event_fields(session, request_trace_metadata),
                    },
                )
                lineage_fields = tuple(
                    field_name
                    for field_name, expected in (
                        ("parent_session_id", session.parent_session_id),
                        ("causal_budget_id", session.causal_budget_id),
                    )
                    if expected is not None and start_event.payload.get(field_name) == expected
                )
                if (
                    start_event_type is EventType.SESSION_STARTED
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
            recovered_structured_outcome: Event | None = None
            recovered_structured_retry = False
            recovery_tail_message_count = len(messages_to_append)
            recovery_expected_transcript_cursor: int | None = None
            if pending_tool_round_source_transcript_cursor is not None:
                checkpoint = await self.session_store.load_checkpoint(session.id)
                entrance_pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(
                    checkpoint
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
                pending_snapshot = await self.session_store.load_transcript_snapshot(session.id)
                try:
                    pending_result_position = pending_snapshot.retained_position(
                        pending_result_cursor
                    )
                    if model_projection_cursor:
                        pending_projection = _project_model_target_snapshot(
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
            async for event in self._recovery_coordinator.recover_pending_tool_round(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                messages=messages,
                execution_profile=execution_profile,
                tail_message_count=recovery_tail_message_count,
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
            task_start_event = await start_linked_task_if_needed()
            if task_start_event is not None:
                yield task_start_event
            if messages_deferred:
                await materialize_deferred_messages()
            elif not messages_already_persisted:
                await self.session_store.append_transcript_messages(
                    session.id,
                    messages_to_append,
                )
            skip_model_steps = False
            if (
                recovered_structured_outcome is not None
                and recovered_structured_outcome.type == EventType.STRUCTURED_OUTPUT_VALIDATED
                and not messages_to_append
            ):
                (
                    session,
                    interaction_completed_event,
                    session_completed,
                ) = await self._complete_session_if_no_queued_messages(
                    session=session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                )
                if interaction_completed_event is not None:
                    yield interaction_completed_event
                if not session_completed:
                    recovered_step = recovered_structured_outcome.payload.get("step")
                    if type(recovered_step) is not int:
                        raise RuntimeError(
                            "Recovered structured-output result has no valid model step."
                        ) from None
                    (
                        should_continue,
                        queued_events,
                    ) = await self._handle_queued_messages_before_completion(
                        session=session,
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
                    )
                    for event in queued_events:
                        yield event
                    if not should_continue:
                        return
                else:
                    skip_model_steps = True
            if (
                has_run_limit_accounting_authority(limits, budget_limits)
                and run_limit_accounting is None
            ):
                run_limit_accounting = capture_run_limit_accounting_context(
                    session_id=session.id,
                    run_started_at=run_started_at,
                    run_baseline=run_baseline,
                    budget_limits=budget_limits,
                    now=self._clock(),
                )
                run_budget_authorities = run_budget_authorities_from_context(
                    run_limit_accounting,
                    budget_limits=budget_limits,
                )
            limit_gate = RunLimitGate(
                self._run_limit_controller,
                session=session,
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
                session=session,
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
            )

            def model_completion_recovery_context(
                billing_identity: BillingIdentity | None,
                reservations: tuple[BudgetStepReservation, ...],
            ) -> ModelCompletionRecoveryContext | None:
                execution_profile_fingerprint = (
                    None if execution_profile is None else execution_profile.fingerprint
                )
                if (
                    registered_provider.provider.provider_operation_mode
                    is not ProviderOperationMode.BACKGROUND
                ):
                    return ModelCompletionRecoveryContext(
                        execution_profile_fingerprint=execution_profile_fingerprint
                    )
                context = ModelCompletionRecoveryContext(
                    execution_profile_fingerprint=execution_profile_fingerprint,
                    task_id=task_id,
                    request_metadata=request_metadata,
                    structured_output=structured_output,
                    thinking=effective_thinking,
                    max_steps=max_steps,
                    limits=limits,
                    run_limit_accounting=run_limit_accounting,
                    budget_limits=budget_limits,
                    budget_reservations=tuple(
                        BudgetReservationRecoveryContext(
                            reservation_id=reservation.record.reservation_id,
                            budget_limit_id=reservation.record.budget_limit_id,
                            limit=reservation.limit,
                        )
                        for reservation in reservations
                    ),
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
                )
                payload = context.model_dump(mode="json")
                if self._secret_redactor.redact_json(payload) != payload:
                    raise ValueError(
                        "Background provider-operation recovery semantics contain a workload "
                        "secret and cannot be persisted exactly; the provider operation was "
                        "not started."
                    )
                return context

            async def publish_model_completion(
                publication: ModelCompletionPublicationRequest,
            ) -> ModelCompletionPublicationResult:
                return await self._publish_assistant_model_completion(
                    publication,
                    session=session,
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
                ExecutionProfileComponentClass.KNOWLEDGE_INJECTION,
                ExecutionProfileComponentClass.CONTEXT_COMPACTION,
                ExecutionProfileComponentClass.PROVIDER_ADAPTER,
                ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY,
            )
            request_loop_policy_instance_identities = self._request_loop_policy_instance_identities(
                request_loop_policies
            )

            def validate_live_model_semantics() -> None:
                """Fail before mutable registered model semantics can drift."""

                if execution_profile is None:
                    return
                candidate = _execution_profile_identity(
                    registered_agent=registered_agent,
                    provider_name=registered_provider.name,
                    registered_provider=registered_provider,
                    model=session.model,
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
                    causal_budget_id=session.causal_budget_id,
                    structured_output=structured_output,
                    thinking=effective_thinking,
                    max_steps=max_steps,
                    limits=limits,
                    retry_policy=retry_policy,
                )
                candidate = execution_profile_with_component(
                    candidate,
                    execution_profile.component(
                        ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION
                    ),
                )
                changed = tuple(
                    component_class
                    for component_class in live_model_semantic_components
                    if candidate.component(component_class)
                    != execution_profile.component(component_class)
                )
                if changed:
                    raise ExecutionProfileMismatchError(
                        session_id=session.id,
                        expected_profile_fingerprint=execution_profile.fingerprint,
                        candidate_profile_fingerprint=candidate.fingerprint,
                        changed_component_classes=changed,
                    )

            model_step_run = self._model_step_executor.create_run(
                provider=provider,
                session=session,
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
                validate_live_model_semantics=validate_live_model_semantics,
                initial_tool_exposure=initial_model_step_tool_exposure,
                previous_tool_exposure_profile_id=previous_tool_exposure_profile_id,
                model_completion_recovery_context_factory=(model_completion_recovery_context),
                model_completion_publisher=publish_model_completion,
            )
            first_model_step = initial_model_step_number or 1
            model_steps = () if skip_model_steps else range(first_model_step, max_steps + 1)
            for step in model_steps:
                model_step_identity = (
                    initial_model_step_identity
                    if step == first_model_step and initial_model_step_identity is not None
                    else new_model_step_identity()
                )
                await self._session_control.raise_if_interrupted(session.id)
                if step == first_model_step and deliver_queued_input_before_first_step:
                    for event in await self._deliver_queued_session_messages(
                        session_id=session.id,
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        messages=messages,
                        include_on_idle=False,
                        continue_active_interaction=True,
                    ):
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
                    session.id
                )
                budget_evaluation = await limit_gate.evaluate_budget(
                    budget_policy,
                    execution_identity=model_step_identity,
                )
                async for event in self._apply_budget_evaluation(
                    evaluation=budget_evaluation,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    messages=messages,
                    run_started_at=run_started_at,
                    turn_usage_tracker=turn_usage_tracker,
                    active_run=active_run,
                    execution_profile=execution_profile,
                ):
                    yield event
                if budget_evaluation.check is not None:
                    return
                limit_evaluation = await limit_gate.evaluate_limits(
                    execution_identity=model_step_identity,
                )
                async for event in self._apply_limit_evaluation(
                    evaluation=limit_evaluation,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    messages=messages,
                    run_started_at=run_started_at,
                    turn_usage_tracker=turn_usage_tracker,
                    active_run=active_run,
                    execution_profile=execution_profile,
                ):
                    yield event
                if limit_evaluation.decision is not None:
                    return
                model_step_flow_outcome: ModelStepFlowOutcome | None = None
                close_new_pending_round_on_interrupt = True
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
                                # The replacement model-completion stage is durable before
                                # this event is published. From this boundary onward the
                                # ordinary model/session failure contract owns recovery.
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
                    await self._session_control.raise_if_interrupted(session.id)
                    raise RuntimeError("Model step finished without a terminal flow outcome.")
                if model_step_flow_outcome.stop_session:
                    return
                await self._session_control.raise_if_interrupted(session.id)
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
                    checkpoint = await self.session_store.load_checkpoint(session.id)
                    pending_tool_round = tool_round_recovery.pending_tool_round_from_checkpoint(
                        checkpoint
                    )
                    if pending_tool_round is None:
                        raise RuntimeError(
                            "Model completion with tool calls lost its durable pending marker."
                        )
                    if (
                        tool_round_recovery.pending_tool_round_identity(pending_tool_round)
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
                            expected_durable_tool_calls.append(
                                runtime_records.ToolCallRequest(
                                    id=call.id,
                                    name=call.name,
                                    arguments=arguments,
                                )
                            )
                    if durable_tool_calls != expected_durable_tool_calls:
                        raise RuntimeError(
                            "Model completion tool calls conflict with its durable pending marker."
                        )
                if assistant_message is not None and (
                    pending_tool_round is None
                    or pending_tool_round.assistant_message_state == "published"
                ):
                    messages.append(assistant_message)
                limit_evaluation = await limit_gate.evaluate_limits(
                    pending_tool_calls=_user_tool_call_count(tool_calls),
                    execution_identity=completed_model_attempt_identity,
                )
                async for event in self._apply_limit_evaluation(
                    evaluation=limit_evaluation,
                    session=session,
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
                ):
                    yield event
                if limit_evaluation.decision is not None:
                    return

                budget_evaluation = await limit_gate.evaluate_budget(
                    budget_policy,
                    execution_identity=completed_model_attempt_identity,
                )
                async for event in self._apply_budget_evaluation(
                    evaluation=budget_evaluation,
                    session=session,
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
                    structured_attempt = structured_output_retries + 1
                    validation = _validate_structured_output_tool_round(
                        tool_calls=tool_calls,
                        spec=structured_output,
                    )
                    durable_validation = pending_tool_round.structured_output_validation
                    if durable_validation is None:
                        raise RuntimeError(
                            "Structured-output tool round lost its authoritative validation."
                        )
                    if durable_validation != _redact_structured_output_validation(
                        validation,
                        self._secret_redactor,
                    ):
                        raise RuntimeError(
                            "Structured-output validation conflicts with its durable "
                            "model-completion evidence."
                        )
                    structured_tool_outcomes = _structured_output_tool_round_outcomes(
                        tool_calls=tool_calls,
                        spec=structured_output,
                        validation=validation,
                    )
                    structured_tool_outcomes = tool_results.redact_tool_call_outcomes(
                        structured_tool_outcomes,
                        self._secret_redactor,
                    )
                    structured_round_redactor = self._tool_round_executor.redactor_for_tool_calls(
                        registered_agent=registered_agent,
                        tool_calls=tool_calls,
                    )
                    terminal_events: list[Event] = []
                    for outcome in structured_tool_outcomes:
                        await self.session_store.transform_checkpoint(
                            session.id,
                            tool_round_recovery.assistant_publication_snapshot_transform(
                                tool_round_identity=tool_round_identity,
                                tool_call_id=outcome.call.id,
                                redactor=structured_round_redactor,
                                unsafe_output=False,
                            ),
                        )
                        terminal_event = await self._event_writer.emit(
                            event_with_execution_profile_authority(
                                _structured_output_tool_terminal_event(
                                    session=session,
                                    registered_agent=registered_agent,
                                    environment_name=environment_name,
                                    tool_round_identity=tool_round_identity,
                                    outcome=outcome,
                                ),
                                execution_profile,
                            )
                        )
                        terminal_events.append(terminal_event)
                        yield terminal_event

                    retry_scheduled = (
                        not validation.valid
                        and structured_output_retries < structured_output.max_retries
                        and step < max_steps
                    )
                    validating_event = _structured_output_validating_event(
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        spec=structured_output,
                        step=step,
                        attempt=structured_attempt,
                        tool_round_identity=tool_round_identity,
                    )
                    outcome_event = _structured_output_event(
                        event_type=(
                            EventType.STRUCTURED_OUTPUT_VALIDATED
                            if validation.valid
                            else EventType.STRUCTURED_OUTPUT_FAILED
                        ),
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        spec=structured_output,
                        validation=validation,
                        step=step,
                        attempt=structured_attempt,
                        redactor=self._secret_redactor,
                        tool_round_identity=tool_round_identity,
                    )
                    auxiliary_events = [validating_event, outcome_event]
                    if retry_scheduled:
                        auxiliary_events.append(
                            _structured_output_event(
                                event_type=EventType.STRUCTURED_OUTPUT_RETRY,
                                session=session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                spec=structured_output,
                                validation=validation,
                                step=step,
                                attempt=structured_attempt,
                                redactor=self._secret_redactor,
                                tool_round_identity=tool_round_identity,
                            )
                        )
                    auxiliary_events = self._event_writer.prepare_many(
                        [
                            event_with_execution_profile_authority(event, execution_profile)
                            for event in auxiliary_events
                        ]
                    )

                    source_checkpoint = await self.session_store.load_checkpoint(session.id)
                    durable_pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(
                        source_checkpoint
                    )
                    if (
                        durable_pending_round is None
                        or tool_round_recovery.pending_tool_round_identity(durable_pending_round)
                        != tool_round_identity
                    ):
                        raise RuntimeError(
                            "Structured-output tool round marker changed before publication."
                        )
                    lifecycle_events = (
                        await self.session_store.load_tool_round_lifecycle_events_for_round(
                            session.id,
                            [call.tool_call_id for call in durable_pending_round.tool_calls],
                            tool_round_identity=tool_round_recovery.pending_tool_round_identity(
                                durable_pending_round
                            ),
                        )
                    )
                    extension = _StructuredOutputToolRoundPublicationExtension(
                        intent={
                            "schema_version": 1,
                            "kind": "structured-output-validation",
                            "step": step,
                            "attempt": structured_attempt,
                            "valid": validation.valid,
                            "retry_scheduled": retry_scheduled,
                            "event_ids": [event.id for event in auxiliary_events],
                        },
                        events=tuple(auxiliary_events),
                    )
                    prepared_publication = tool_round_publication.prepare_tool_round_publication(
                        session_id=session.id,
                        pending_round=durable_pending_round,
                        source_checkpoint=source_checkpoint,
                        durable_events=lifecycle_events,
                        expected_statuses={
                            SessionStatus.RUNNING,
                            SessionStatus.INTERRUPTING,
                        },
                        expected_run_epoch=session.run_epoch,
                        expected_transcript_cursor=(
                            await self.session_store.load_transcript_cursor(session.id)
                        ),
                        extension=extension,
                    )
                    cancellation = (
                        await tool_round_publication.publish_tool_round_with_exact_replay(
                            prepared_publication,
                            session_store=self.session_store,
                            event_writer=self._event_writer,
                        )
                    )
                    messages.extend(prepared_publication.request.transcript_messages)
                    close_new_pending_round_on_interrupt = False
                    if cancellation is not None:
                        raise cancellation
                    await self._session_control.raise_if_interrupted(session.id)
                    for event in auxiliary_events:
                        yield copy_event(event)

                    if validation.valid:
                        self._record_workflow_structured_output(
                            session.id,
                            validation.output,
                        )
                        (
                            session,
                            interaction_completed_event,
                            session_completed,
                        ) = await self._complete_session_if_no_queued_messages(
                            session=session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                        )
                        if interaction_completed_event is not None:
                            yield interaction_completed_event
                        if not session_completed:
                            (
                                should_continue,
                                queued_events,
                            ) = await self._handle_queued_messages_before_completion(
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
                            )
                            for event in queued_events:
                                yield event
                            if not should_continue:
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
                    if structured_output is not None:
                        yield await self._event_writer.emit(
                            event_with_execution_profile_authority(
                                _structured_output_validating_event(
                                    session=session,
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
                                    session.id,
                                    validation.output,
                                )
                                yield await self._event_writer.emit(
                                    event_with_execution_profile_authority(
                                        _structured_output_event(
                                            event_type=EventType.STRUCTURED_OUTPUT_VALIDATED,
                                            session=session,
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
                                (
                                    session,
                                    interaction_completed_event,
                                    session_completed,
                                ) = await self._complete_session_if_no_queued_messages(
                                    session=session,
                                    registered_agent=registered_agent,
                                    environment_name=environment_name,
                                )
                                if interaction_completed_event is not None:
                                    yield interaction_completed_event
                                if not session_completed:
                                    (
                                        should_continue,
                                        queued_events,
                                    ) = await self._handle_queued_messages_before_completion(
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
                                    )
                                    for event in queued_events:
                                        yield event
                                    if not should_continue:
                                        return
                                    continue
                                break
                        else:
                            validation = structured_output_tool_required_validation()
                        yield await self._event_writer.emit(
                            event_with_execution_profile_authority(
                                _structured_output_event(
                                    event_type=EventType.STRUCTURED_OUTPUT_FAILED,
                                    session=session,
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
                                session.id,
                                [repair_message],
                                runtime_message_transform,
                            )
                        )
                        yield await self._event_writer.emit(
                            event_with_execution_profile_authority(
                                _structured_output_event(
                                    event_type=EventType.STRUCTURED_OUTPUT_RETRY,
                                    session=session,
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
                    async for event, policy_decision in self._run_before_stop_policies(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        step_result=assistant_step_result,
                        step=step,
                        max_steps=max_steps,
                        request_metadata=request_metadata,
                        request_loop_policies=request_loop_policies,
                    ):
                        yield event
                        if policy_decision is not None:
                            before_stop_decision = policy_decision
                    if before_stop_decision is not None:
                        if before_stop_decision.action == BeforeStopAction.CONTINUE:
                            if step >= max_steps:
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
                                ):
                                    yield event
                                return
                            if before_stop_decision.message is None:
                                raise RuntimeError(
                                    "Before-stop continue decision requires a message."
                                )
                            repair_message = before_stop_decision.message
                            messages.append(repair_message)
                            runtime_message_transform = (
                                _runtime_authored_user_message_checkpoint_transform(
                                    anchor_index=len(messages) - 1,
                                    message=repair_message,
                                )
                            )
                            await self.session_store.append_transcript_messages_and_transform_checkpoint(
                                session.id,
                                [repair_message],
                                runtime_message_transform,
                            )
                            continue
                        if before_stop_decision.action == BeforeStopAction.INTERRUPT:
                            (
                                session,
                                interaction_interrupted_event,
                                _,
                            ) = await self._publish_interaction_transition(
                                session=session,
                                agent_name=registered_agent.spec.name,
                                environment_name=environment_name,
                                to_status=SessionStatus.INTERRUPTED,
                            )
                            if interaction_interrupted_event is not None:
                                yield interaction_interrupted_event
                            for event in await self._emit_turn_completed_once(
                                session=session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                status=SessionStatus.INTERRUPTED,
                                run_started_at=run_started_at,
                                usage_tracker=turn_usage_tracker,
                                active_run=active_run,
                            ):
                                yield event
                            async for event in self._emit_terminal_event_with_hooks(
                                event=Event(
                                    type=EventType.SESSION_INTERRUPTED,
                                    session_id=session.id,
                                    agent_name=registered_agent.spec.name,
                                    environment_name=environment_name,
                                    payload={
                                        "interruption_type": (
                                            _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED
                                        ),
                                        "reason": before_stop_decision.reason,
                                        "policy_metadata": copy_json_value(
                                            before_stop_decision.metadata,
                                            "policy_metadata",
                                        ),
                                    },
                                ),
                                phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                                session=session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                execution_profile=execution_profile,
                            ):
                                yield event
                            return
                        if before_stop_decision.action == BeforeStopAction.FAIL:
                            raise RuntimeError(
                                f"Before-stop policy failed session: {before_stop_decision.reason}"
                            )
                    (
                        session,
                        interaction_completed_event,
                        session_completed,
                    ) = await self._complete_session_if_no_queued_messages(
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                    )
                    if interaction_completed_event is not None:
                        yield interaction_completed_event
                    if not session_completed:
                        (
                            should_continue,
                            queued_events,
                        ) = await self._handle_queued_messages_before_completion(
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
                        )
                        for event in queued_events:
                            yield event
                        if not should_continue:
                            return
                        continue
                    break

                if tool_round_identity is None:
                    raise RuntimeError("Ordinary tool calls require a tool-round identity.")
                async for event in tool_round_runner.run(
                    messages=messages,
                    tool_calls=tool_calls,
                    tool_round_identity=tool_round_identity,
                    model_step=step,
                ):
                    yield event
                close_new_pending_round_on_interrupt = False
                if tool_round_runner.stopped_for_limit:
                    return
            else:
                if not skip_model_steps:
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
                    ):
                        yield event
                    return

            if task_id is not None:
                await self._session_control.raise_if_interrupted(session.id)
                task = await self._complete_task(
                    task_id=task_id,
                    task_worker_id=task_worker_id,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                )
                task_finished = True
                if active_run is not None:
                    active_run.task_finished = True
                yield await self._event_writer.emit(
                    _task_event(
                        event_type=EventType.TASK_COMPLETED,
                        task=task,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                    )
                )
            for event in await self._emit_turn_completed_once(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                status=SessionStatus.COMPLETED,
                run_started_at=run_started_at,
                usage_tracker=turn_usage_tracker,
                active_run=active_run,
            ):
                yield event
            async for event in self._emit_terminal_event_with_hooks(
                event=Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_COMPLETED,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
            ):
                yield event
        except ToolApprovalRequired as exc:
            await materialize_deferred_messages_after_failure()
            (
                session,
                interaction_paused_event,
                _,
            ) = await self._publish_sibling_interaction_transition(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                to_status=SessionStatus.INTERRUPTED,
                execution_profile=execution_profile,
            )
            if interaction_paused_event is not None:
                yield interaction_paused_event
            for event in await self._emit_turn_completed_once(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                status=SessionStatus.INTERRUPTED,
                run_started_at=run_started_at,
                usage_tracker=turn_usage_tracker,
                active_run=active_run,
            ):
                yield event
            async for event in self._emit_terminal_event_with_hooks(
                event=approval_support.event_with_pending_approval_authority(
                    Event(
                        type=EventType.SESSION_INTERRUPTED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload={
                            "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                            "model_step_id": exc.approval.model_step_id,
                            "model_attempt_id": exc.approval.model_attempt_id,
                            "tool_round_id": exc.approval.tool_round_id,
                            **approval_support.bounded_pending_approval_event_payload(
                                exc.approval,
                                redactor=self._secret_redactor,
                            ),
                        },
                    ),
                    exc.approval,
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
            ):
                yield event
        except UserInputRequired as exc:
            await materialize_deferred_messages_after_failure()
            (
                session,
                interaction_paused_event,
                _,
            ) = await self._publish_sibling_interaction_transition(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                to_status=SessionStatus.INTERRUPTED,
                execution_profile=execution_profile,
            )
            if interaction_paused_event is not None:
                yield interaction_paused_event
            for event in await self._emit_turn_completed_once(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                status=SessionStatus.INTERRUPTED,
                run_started_at=run_started_at,
                usage_tracker=turn_usage_tracker,
                active_run=active_run,
            ):
                yield event
            async for event in self._emit_terminal_event_with_hooks(
                event=event_with_pending_user_input_authority(
                    Event(
                        type=EventType.SESSION_INTERRUPTED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload={
                            "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                            "model_step_id": exc.pending.model_step_id,
                            "model_attempt_id": exc.pending.model_attempt_id,
                            "tool_round_id": exc.pending.tool_round_id,
                            **pending_user_input_interruption_payload(exc.pending),
                        },
                    ),
                    exc.pending,
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
            ):
                yield event
        except SessionInterruptedByRequest:
            await materialize_deferred_messages_after_failure()
            interruption_events: list[Event] = []
            if close_new_pending_round_on_interrupt:
                async for event in self._close_durable_pending_tool_round_after_interrupt(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile,
                ):
                    interruption_events.append(event)
            async for event in self._handle_session_interrupted(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                execution_profile=execution_profile,
                run_started_at=run_started_at,
                turn_usage_tracker=turn_usage_tracker,
                active_run=active_run,
            ):
                interruption_events.append(event)
            for event in interruption_events:
                yield event
            return
        except asyncio.CancelledError as cancellation:
            if _is_interaction_transition_run_fence(cancellation):
                # The owned transition attempt proved that this worker lost
                # authority. Preserve caller cancellation without attempting
                # abandoned-run writes under the stale epoch. The transition
                # handoff is runtime-only evidence for cleanup that this branch
                # must not perform, so do not let it escape on the public error.
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
                    committed_transition_recorded = await self._recovery_coordinator._record_committed_interaction_transition_cancellation(
                        RecoveryAbandonedSessionRequest(
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            environment_name=environment_name,
                            run_started_at=run_started_at,
                            turn_usage_tracker=turn_usage_tracker,
                            active_run=active_run,
                            interaction_transition_failures=tuple(interaction_transition_failures),
                            interaction_transition=interaction_transition,
                            execution_profile=execution_profile,
                        )
                    )

                diagnostic_failures = await _run_interaction_transition_cancellation_cleanup_steps(
                    cancellation,
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
            if transition_settled or _latest_session_invocation_interaction_is_settled(session.id):
                # The authoritative interaction/session decision already
                # committed. Its durable side-effect handoff remains
                # recoverable, so abandoned-run cleanup must not rewrite it.
                raise
            billing_identity_cancellation = detach_billing_identity_cancellation(cancellation)
            if billing_identity_cancellation is not None:
                if not preserve_failure_until_initial_provider_dispatch:
                    # The ordinary outer run boundary owns credential-safe
                    # detachment and environment finalization.
                    raise
                # Provider-operation fallback recovery has no equivalent outer
                # run boundary. Drop provider-bearing locals before fallible
                # interruption work and surface only a detached cancellation.
                del provider, registered_provider
                try:
                    interruption_requested = await self._session_control.interrupt_requested(
                        session.id
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
                                session=session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                execution_profile=execution_profile,
                            ):
                                interruption_events.append(event)
                        async for event in self._handle_session_interrupted(
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            environment_name=environment_name,
                            execution_profile=execution_profile,
                            run_started_at=run_started_at,
                            turn_usage_tracker=turn_usage_tracker,
                            active_run=active_run,
                            interaction_transition_failures=tuple(interaction_transition_failures),
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
            if await self._session_control.interrupt_requested(session.id):
                clear_current_task_cancellation()
                await materialize_deferred_messages_after_failure()
                interruption_events = []
                if close_new_pending_round_on_interrupt:
                    async for event in self._close_durable_pending_tool_round_after_interrupt(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile,
                    ):
                        interruption_events.append(event)
                async for event in self._handle_session_interrupted(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    execution_profile=execution_profile,
                    run_started_at=run_started_at,
                    turn_usage_tracker=turn_usage_tracker,
                    active_run=active_run,
                    interaction_transition_failures=tuple(interaction_transition_failures),
                ):
                    interruption_events.append(event)
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
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile,
                ):
                    pass

            cancellation_cleanup_steps = (
                (
                    "cancelled tool-round finalization",
                    close_cancelled_tool_round,
                ),
                (
                    "cancelled session finalization",
                    lambda: self._recovery_coordinator.finalize_abandoned_session_run(
                        RecoveryAbandonedSessionRequest(
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            environment_name=environment_name,
                            run_started_at=run_started_at,
                            turn_usage_tracker=turn_usage_tracker,
                            active_run=active_run,
                            interaction_transition_failures=tuple(interaction_transition_failures),
                            interaction_transition=interaction_transition,
                            execution_profile=execution_profile,
                        )
                    ),
                ),
            )
            if interaction_transition_failures:
                await _run_interaction_transition_cancellation_cleanup_steps(
                    cancellation,
                    steps=cancellation_cleanup_steps,
                )
            else:
                await _run_recovery_cleanup_steps(
                    authoritative_failure=cancellation,
                    steps=cancellation_cleanup_steps,
                )
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
            if _latest_session_invocation_interaction_is_settled(session.id):
                # Closing after an authoritative transition was exposed must
                # not rewrite the completed/paused interaction decision.
                raise
            await materialize_deferred_messages_after_failure()
            try:
                await self._recovery_coordinator.finalize_abandoned_session_run(
                    RecoveryAbandonedSessionRequest(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=environment_name,
                        run_started_at=run_started_at,
                        turn_usage_tracker=turn_usage_tracker,
                        active_run=active_run,
                        execution_profile=execution_profile,
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
            ambiguous_provider_start = is_ambiguous_provider_operation_start_error(exc)
            if not ambiguous_provider_start:
                provider_operation = await inspect_provider_operation(
                    self.session_store,
                    session.id,
                )
                ambiguous_provider_start = (
                    provider_operation.status
                    is ProviderOperationInspectionStatus.AMBIGUOUS_SUBMISSION
                )
            if ambiguous_provider_start:
                interrupted_session = await self.session_store.transition_status(
                    session.id,
                    from_statuses={SessionStatus.RUNNING, SessionStatus.INTERRUPTING},
                    to_status=SessionStatus.INTERRUPTED,
                )
                async for event in self._emit_terminal_event_with_hooks(
                    event=Event(
                        type=EventType.SESSION_INTERRUPTED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload={
                            "interruption_type": "provider_operation_unavailable",
                            "recovery_reason": "ambiguous_submission",
                            "duplicate_request_risk": True,
                        },
                    ),
                    phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                    session=interrupted_session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile,
                ):
                    yield event
                interaction_id = _current_session_interaction_id(session.id)
                if interaction_id is not None:
                    _mark_session_interaction_settled(session.id, interaction_id)
                return
            if is_durable_subagent_submission_unsettled(
                exc,
                parent_session_id=session.id,
            ):
                raise
            if _is_interaction_transition_run_fence(exc):
                raise
            if _is_runtime_interaction_lifecycle_publication_rejection(
                exc,
                session_id=session.id,
                interaction_id=_current_session_interaction_id(session.id),
                runtime_authority=self._interaction_lifecycle_publication_authority,
            ):
                # A terminal interaction cannot be represented using the observed
                # wall clock. Preserve the live session, linked task, and durable
                # run-operation marker so normal incomplete-session recovery owns
                # the eventual terminal classification.
                raise
            try:
                failure_interaction_observed_at = await self._failure_interaction_observed_at(
                    session
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
            await materialize_deferred_messages_after_failure()
            task_failure_error: Exception | None = None
            if (
                not task_started
                and not task_start_attempted
                and task_id is not None
                and self.task_store is not None
            ):
                try:
                    task_start_event = await start_linked_task_if_needed(only_if_exists=True)
                    if task_start_event is not None:
                        yield task_start_event
                except Exception as task_exc:
                    task_failure_error = task_exc
            if (
                task_started
                and not task_finished
                and task_id is not None
                and self.task_store is not None
            ):
                try:
                    task = await self._fail_task(
                        task_id=task_id,
                        task_worker_id=task_worker_id,
                        session=session,
                        error=task_failure_payload_from_diagnostic(
                            failure_diagnostic,
                            session_id=session.id,
                        ),
                    )
                    task_finished = True
                    if active_run is not None:
                        active_run.task_finished = True
                    yield await self._event_writer.emit(
                        _task_event(
                            event_type=EventType.TASK_FAILED,
                            task=task,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                        )
                    )
                except Exception as task_exc:
                    task_failure_error = task_exc
            (
                session,
                interaction_failed_event,
                _,
            ) = await self._publish_sibling_interaction_transition(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                to_status=SessionStatus.FAILED,
                observed_at=failure_interaction_observed_at,
                execution_profile=execution_profile,
            )
            if interaction_failed_event is not None:
                yield interaction_failed_event
            payload = exception_failure_payload(
                exc,
                diagnostic=failure_diagnostic,
                redactor=self._secret_redactor,
            )
            provider_cleanup_failure = budget_provider_cleanup_failure(exc)
            if provider_cleanup_failure is not None:
                provider_cleanup_diagnostic = exception_diagnostic(
                    provider_cleanup_failure,
                    empty_message="provider iterator cleanup failed",
                    nonportable_message=(
                        "Provider iterator cleanup failed with a non-portable diagnostic."
                    ),
                    redactor=self._secret_redactor,
                )
                payload["provider_cleanup_failure"] = {
                    **provider_cleanup_diagnostic.payload_fields(),
                    "phase": "provider_iterator_cleanup",
                }
            transition_replay_failures = _interaction_transition_replay_failures(exc)
            if transition_replay_failures is not None:
                payload["interaction_transition_failures"] = (
                    _interaction_transition_failure_diagnostics(
                        transition_replay_failures,
                        redactor=self._secret_redactor,
                    )
                )
            if isinstance(exc, resume_ledger.ToolCallEvidenceConflict):
                payload[resume_ledger.TOOL_EVIDENCE_CONFLICT_PAYLOAD_KEY] = True
            if task_failure_error is not None:
                payload.update(
                    task_update_error_payload(
                        task_failure_error,
                        redactor=self._secret_redactor,
                    )
                )
            for event in await self._emit_turn_completed_once(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                status=SessionStatus.FAILED,
                run_started_at=run_started_at,
                usage_tracker=turn_usage_tracker,
                active_run=active_run,
            ):
                yield event
            async for event in self._emit_terminal_event_with_hooks(
                event=Event(
                    type=EventType.SESSION_FAILED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=payload,
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
            ):
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
                            session.id
                        )
                    except Exception:
                        raise _FallbackBillingCancellationStateCheckFailed from None
                    if not interruption_requested:
                        raise
                elif not await self._session_control.interrupt_requested(session.id):
                    raise
            # ``ExceptionGroup`` is also an ``Exception`` and is handled by
            # the ordinary failure branch above. Only groups carrying a true
            # ``BaseException`` control leaf reach this interruption path.
            if binding_finalize_explicit_cancellation(failure) is None:
                raise
            if _latest_session_invocation_interaction_is_settled(session.id):
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
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile,
                    ):
                        prepared_events.append(event)
                return prepared_events

            (
                _interruption_events,
                propagated_failure,
            ) = await self._handle_session_interrupted_preserving_failure(
                authoritative_failure=failure,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                execution_profile=execution_profile,
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
            if (
                finalizer_failure is not None
                and workspace_observation_pending_cancellation_requests(finalizer_failure)
            ):

                async def discard_interrupt_signal() -> None:
                    self._session_control.discard_interrupt_signal(session.id)

                finalizer_steps: list[tuple[str, Callable[[], Awaitable[None]]]] = [
                    (
                        "Session environment setup abort",
                        lambda: self._environment_lifecycle.abort_environment_setup(
                            session_id=session.id,
                            original_error=finalizer_failure,
                            execution_profile=execution_profile,
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
                                    session_id=session.id,
                                    execution_profile=execution_profile,
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
                    _deactivate_session_run_fence(session.id)
                try:
                    if current_task is not None:
                        self._session_control.unregister_active_task(
                            session.id,
                            current_task,
                        )
                finally:
                    _clear_session_interaction_settlement(session.id)
                    _deactivate_session_interaction(session.id)
                if propagated_finalizer_failure is not finalizer_failure:
                    raise propagated_finalizer_failure from None
            else:
                try:
                    await self._environment_lifecycle.abort_environment_setup(
                        session_id=session.id,
                        original_error=finalizer_failure,
                        execution_profile=execution_profile,
                    )
                finally:
                    self._session_control.discard_interrupt_signal(session.id)
                    try:
                        if release_run_fence_on_exit:
                            await self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                                session_id=session.id,
                                execution_profile=execution_profile,
                            )
                    finally:
                        try:
                            if current_task is not None:
                                self._session_control.unregister_active_task(
                                    session.id,
                                    current_task,
                                )
                        finally:
                            _clear_session_interaction_settlement(session.id)
                            _deactivate_session_interaction(session.id)

    async def _emit_turn_completed(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        status: SessionStatus,
        run_started_at: float,
        usage_tracker: SessionUsageTracker,
    ) -> tuple[Event, ...]:
        interaction_id = _current_session_interaction_id(session.id)
        if interaction_id is not None and status != SessionStatus.COMPLETED:
            _, interaction_event, _ = await self._publish_sibling_interaction_transition(
                session=session,
                registered_agent=registered_agent,
                registered_environment=None,
                environment_name=environment_name,
                to_status=status,
                from_statuses={status},
                finalize_unsettled_cancellation=False,
            )
        else:
            interaction_event = None
        usage_events = await usage_tracker.usage_events()
        summary = session_usage_summary(session.id, usage_events)
        duration_ms = max(0, int((time.monotonic() - run_started_at) * 1000))
        interaction_ids = _current_session_invocation_interaction_ids(session.id)
        turn_completed = event_with_runtime_nested_payload_authority(
            Event(
                type=EventType.TURN_COMPLETED,
                session_id=session.id,
                interaction_id=None,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload={
                    "status": status.value,
                    "duration_ms": duration_ms,
                    "step_count": summary.model_steps,
                    "tool_call_count": summary.tool_calls,
                    "token_usage": aggregate_usage_metrics_payload(summary.usage),
                    "provider_names": summary.provider_names,
                    "models": summary.models,
                    "interaction_ids": list(interaction_ids),
                },
            ),
            ("interaction_ids", "*"),
        )
        turn_completed_event = await self._event_writer.emit(turn_completed)
        if interaction_event is None:
            return (turn_completed_event,)
        return (interaction_event, turn_completed_event)

    async def _emit_turn_completed_once(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        status: SessionStatus,
        run_started_at: float,
        usage_tracker: SessionUsageTracker,
        active_run: ActiveSessionRun[SessionUsageTracker] | None,
    ) -> tuple[Event, ...]:
        if active_run is None:
            return await self._emit_turn_completed(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                status=status,
                run_started_at=run_started_at,
                usage_tracker=usage_tracker,
            )
        async with active_run.turn_completed_lock:
            if active_run.turn_completed_event is not None:
                return (active_run.turn_completed_event,)
            events = await self._emit_turn_completed(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                status=status,
                run_started_at=run_started_at,
                usage_tracker=usage_tracker,
            )
            active_run.turn_completed_event = events[-1]
            return events

    async def _emit_active_turn_completed_if_needed(
        self,
        *,
        session: Session,
        status: SessionStatus,
    ) -> Event | None:
        for active_run in self._session_control.active_runs(session.id):
            if (
                active_run.turn_registered_agent is None
                or active_run.turn_started_at is None
                or active_run.turn_usage_tracker is None
            ):
                continue
            events = await self._emit_turn_completed_once(
                session=session,
                registered_agent=active_run.turn_registered_agent,
                environment_name=active_run.turn_environment_name,
                status=status,
                run_started_at=active_run.turn_started_at,
                usage_tracker=active_run.turn_usage_tracker,
                active_run=active_run,
            )
            return events[-1]
        return None

    async def _start_task(
        self,
        *,
        task_id: str,
        session: Session,
        worker_id: str | None = None,
    ) -> Task:
        if self.task_store is None:
            raise RuntimeError("task_store is required when RunRequest.task_id is set.")
        session_invocation = SessionInvocationBinding(
            id=session.id,
            invocation=session.invocation,
        )
        existing = await self.task_store.load_task(task_id)
        if (
            worker_id is None
            and existing is not None
            and existing.status is TaskStatus.RUNNING
            and existing.session_id == session.id
            and existing.worker_id is None
        ):
            _task_invocation_for_attachment(
                existing.invocation,
                session_id=session.id,
                session_binding=session_invocation,
            )
            return existing
        if worker_id is not None:
            return await self.task_store.attach_task(
                task_id,
                session_id=session.id,
                session_invocation=session_invocation,
                worker_id=worker_id,
            )
        return await self.task_store.start_task(
            task_id,
            session_id=session.id,
            session_invocation=session_invocation,
        )

    async def _fail_task_for_run_setup_error(
        self,
        *,
        task_id: str | None,
        task_worker_id: str | None,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        diagnostic: ExceptionDiagnostic,
    ) -> tuple[Event | None, Exception | None]:
        """Fail only a task this run can prove it owns before `_run_session`."""
        if task_id is None:
            return None, None
        if self.task_store is None:
            return None, RuntimeError("task_store is required when RunRequest.task_id is set.")
        try:
            task = await self.task_store.load_task(task_id)
            if task is None:
                return None, KeyError(f"Task not found: {task_id}")
            worker_matches = task.worker_id == task_worker_id
            attached_to_session = task.session_id == session.id
            owned_claim = (
                task_worker_id is not None
                and task.status is TaskStatus.CLAIMED
                and task.session_id is None
                and worker_matches
            )
            unclaimed_pending = (
                task_worker_id is None
                and task.status is TaskStatus.PENDING
                and task.session_id is None
                and task.worker_id is None
            )
            if not ((attached_to_session and worker_matches) or owned_claim or unclaimed_pending):
                return None, None
            task = await self._fail_task(
                task_id=task_id,
                task_worker_id=task_worker_id,
                session=session,
                error=task_failure_payload_from_diagnostic(
                    diagnostic,
                    session_id=session.id,
                ),
            )
            return (
                await self._event_writer.emit(
                    _task_event(
                        event_type=EventType.TASK_FAILED,
                        task=task,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                    )
                ),
                None,
            )
        except Exception as task_error:
            return None, task_error

    async def _linked_running_task_id(self, session_id: str) -> str | None:
        """Find the one running task already attached to a resumed session.

        Task attachment is durable in ``TaskStore`` while ``ResumeRequest`` carries
        no task id. Re-associate that task before entering the resumed loop so a
        post-crash completion or failure terminalizes the original work item.
        """
        if self.task_store is None:
            return None
        tasks = await self.task_store.list_tasks(
            TaskQuery(
                status=TaskStatus.RUNNING,
                session_id=session_id,
                limit=2,
            )
        )
        if len(tasks) > 1:
            raise RuntimeError(f"Session has multiple running tasks attached: {session_id}")
        return tasks[0].id if tasks else None

    async def _complete_task(
        self,
        *,
        task_id: str,
        task_worker_id: str | None,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> Task:
        if self.task_store is None:
            raise RuntimeError("task_store is required when RunRequest.task_id is set.")
        result = {
            "session_id": session.id,
            "agent_name": registered_agent.spec.name,
            "environment_name": _environment_name(registered_environment),
        }
        if task_worker_id is not None:
            return await _terminalize_claimed_task(
                self.task_store,
                TaskTerminalizationRequest(
                    task_id=task_id,
                    worker_id=task_worker_id,
                    kind=TaskTerminalKind.COMPLETED,
                    result=result,
                    idempotency_key=_task_terminalization_idempotency_key(
                        task_id=task_id,
                        session_id=session.id,
                        kind=TaskTerminalKind.COMPLETED,
                    ),
                ),
            )
        return await self.task_store.complete_task(
            task_id,
            result,
            worker_id=task_worker_id,
        )

    async def _fail_task(
        self,
        *,
        task_id: str,
        task_worker_id: str | None,
        session: Session,
        error: dict[str, Any],
    ) -> Task:
        if self.task_store is None:
            raise RuntimeError("task_store is required when RunRequest.task_id is set.")
        if task_worker_id is not None:
            return await _terminalize_claimed_task(
                self.task_store,
                TaskTerminalizationRequest(
                    task_id=task_id,
                    worker_id=task_worker_id,
                    kind=TaskTerminalKind.FAILED,
                    error=error,
                    idempotency_key=_task_terminalization_idempotency_key(
                        task_id=task_id,
                        session_id=session.id,
                        kind=TaskTerminalKind.FAILED,
                    ),
                ),
            )
        return await self.task_store.fail_task(
            task_id,
            error,
            worker_id=task_worker_id,
        )

    async def _apply_model_step_budget_evaluation(
        self,
        request: ModelStepBudgetEvaluationRequest,
    ) -> AsyncGenerator[Event, None]:
        events = self._apply_budget_evaluation(
            evaluation=request.evaluation,
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            environment_name=request.environment_name,
            messages=request.messages,
            run_started_at=request.run_started_at,
            turn_usage_tracker=request.turn_usage_tracker,
            active_run=request.active_run,
            execution_profile=request.execution_profile,
        )
        try:
            async for event in events:
                yield event
        finally:
            await _close_async_iterator(events)

    async def _apply_model_step_limit_evaluation(
        self,
        request: ModelStepLimitEvaluationRequest,
    ) -> AsyncGenerator[Event, None]:
        events = self._apply_limit_evaluation(
            evaluation=request.evaluation,
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            environment_name=request.environment_name,
            messages=request.messages,
            run_started_at=request.run_started_at,
            turn_usage_tracker=request.turn_usage_tracker,
            active_run=request.active_run,
            execution_profile=request.execution_profile,
        )
        try:
            async for event in events:
                yield event
        finally:
            await _close_async_iterator(events)

    async def _stop_for_model_step_budget_reservation_failure(
        self,
        request: ModelStepBudgetReservationFailureRequest,
    ) -> AsyncGenerator[Event, None]:
        events = self._stop_session_for_budget_reservation_failed(
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            environment_name=request.environment_name,
            result=request.result,
            messages=request.messages,
            run_started_at=request.run_started_at,
            turn_usage_tracker=request.turn_usage_tracker,
            active_run=request.active_run,
            execution_profile=request.execution_profile,
        )
        try:
            async for event in events:
                yield event
        finally:
            await _close_async_iterator(events)

    async def _apply_tool_round_limit(
        self,
        request: ToolRoundLimitRequest,
    ) -> AsyncGenerator[Event, None]:
        async for event in self._apply_limit_evaluation(
            evaluation=request.evaluation,
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            environment_name=request.environment_name,
            messages=request.messages,
            tool_calls=request.tool_calls,
            completed_tool_outcomes=request.completed_tool_outcomes,
            tool_round_identity=request.tool_round_identity,
            run_started_at=request.run_started_at,
            turn_usage_tracker=request.turn_usage_tracker,
            active_run=request.active_run,
            execution_profile=request.execution_profile,
        ):
            yield event

    async def _apply_limit_evaluation(
        self,
        *,
        evaluation: LimitEvaluation,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest] | None = None,
        completed_tool_outcomes: list[runtime_records.ToolCallOutcome] | None = None,
        tool_round_identity: ToolRoundIdentity | None = None,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
        execution_profile: ExecutionProfileIdentity | None,
    ) -> AsyncGenerator[Event, None]:
        for event in evaluation.events:
            yield event
        if evaluation.decision is None:
            return
        async for event in self._stop_session_for_limit_reached(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            decision=evaluation.decision,
            usage_summary=evaluation.usage_summary,
            cost_summary=evaluation.cost_summary,
            messages=messages,
            tool_calls=tool_calls if tool_calls is not None else [],
            completed_tool_outcomes=(
                completed_tool_outcomes if completed_tool_outcomes is not None else []
            ),
            tool_round_identity=tool_round_identity,
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
            execution_profile=execution_profile,
            reconcile_transition_cancellation=False,
        ):
            yield event

    async def _apply_budget_evaluation(
        self,
        *,
        evaluation: BudgetEvaluation,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest] | None = None,
        tool_round_identity: ToolRoundIdentity | None = None,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
        execution_profile: ExecutionProfileIdentity | None,
    ) -> AsyncGenerator[Event, None]:
        for event in evaluation.events:
            yield event
        if evaluation.check is None:
            return
        async for event in self._stop_session_for_budget_limit_reached(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            check=evaluation.check,
            messages=messages,
            tool_calls=tool_calls if tool_calls is not None else [],
            completed_tool_outcomes=[],
            tool_round_identity=tool_round_identity,
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
            execution_profile=execution_profile,
        ):
            yield event

    async def _stop_session_for_limit_reached(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        decision: StopDecision,
        usage_summary: SessionUsageSummary,
        cost_summary: SessionCostSummary | None,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        completed_tool_outcomes: list[runtime_records.ToolCallOutcome],
        pending_approval_to_clear: PendingToolApproval | None = None,
        deferred_messages: list[Message] | None = None,
        requested_approval_decision: ToolApprovalDecision | None = None,
        approval_resolution_request_digest: str | None = None,
        tool_round_identity: ToolRoundIdentity | None = None,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
        execution_profile: ExecutionProfileIdentity | None,
        reconcile_transition_cancellation: bool,
    ) -> AsyncGenerator[Event, None]:
        if type(reconcile_transition_cancellation) is not bool:
            raise TypeError("reconcile_transition_cancellation must be a bool.")
        if tool_round_identity is None and pending_approval_to_clear is not None:
            tool_round_identity = ToolRoundIdentity(
                tool_round_id=pending_approval_to_clear.tool_round_id,
                model_step_id=pending_approval_to_clear.model_step_id,
                model_attempt_id=pending_approval_to_clear.model_attempt_id,
            )
        limit_payload = _limit_reached_payload(
            decision=decision,
            usage_summary=usage_summary,
            cost_summary=cost_summary,
        )
        yield await self._event_writer.emit(
            Event(
                type=EventType.SESSION_LIMIT_REACHED,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload=limit_payload,
            )
        )
        if tool_calls or completed_tool_outcomes or pending_approval_to_clear is not None:
            async for event in self._close_limited_tool_round(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                messages=messages,
                tool_calls=tool_calls,
                completed_tool_outcomes=completed_tool_outcomes,
                decision=decision,
                pending_approval_to_clear=pending_approval_to_clear,
                deferred_messages=deferred_messages,
                requested_approval_decision=requested_approval_decision,
                approval_resolution_request_digest=approval_resolution_request_digest,
                tool_round_identity=tool_round_identity,
                execution_profile=execution_profile,
            ):
                yield event

        if reconcile_transition_cancellation:
            (
                interrupted_session,
                interaction_interrupted_event,
                _,
            ) = await self._publish_sibling_interaction_transition(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                to_status=SessionStatus.INTERRUPTED,
                execution_profile=execution_profile,
            )
        else:
            (
                interrupted_session,
                interaction_interrupted_event,
                _,
            ) = await self._publish_interaction_transition(
                session=session,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                to_status=SessionStatus.INTERRUPTED,
            )
        if interaction_interrupted_event is not None:
            yield interaction_interrupted_event
        terminal_payload = {
            "interruption_type": _INTERRUPTION_TYPE_LIMIT_REACHED,
            **limit_payload,
        }
        if tool_round_identity is not None:
            terminal_payload.update(tool_round_identity.payload())
        if run_started_at is not None and turn_usage_tracker is not None:
            for event in await self._emit_turn_completed_once(
                session=interrupted_session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                status=SessionStatus.INTERRUPTED,
                run_started_at=run_started_at,
                usage_tracker=turn_usage_tracker,
                active_run=active_run,
            ):
                yield event
        async for event in self._emit_terminal_event_with_hooks(
            event=Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id=interrupted_session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload=terminal_payload,
            ),
            phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
            session=interrupted_session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            execution_profile=execution_profile,
        ):
            yield event

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
    ) -> AsyncGenerator[Event, None]:
        usage_summary = session_usage_summary(
            session.id,
            await self._run_limit_controller.session_usage_events(session.id),
        )
        decision = StopDecision(
            limit=StopLimit.MODEL_STEPS,
            maximum=max_steps,
            actual=step,
            message=(
                "Run limit reached: the configured model-step allowance was "
                f"exhausted ({step} >= {max_steps})."
            ),
        )
        async for event in self._stop_session_for_limit_reached(
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
            reconcile_transition_cancellation=False,
        ):
            yield event

    async def _stop_session_for_budget_reservation_failed(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        result: BudgetReservationResult,
        messages: list[Message],
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
    ) -> AsyncGenerator[Event, None]:
        payload = budget_reservation_payload(result)
        yield await self._event_writer.emit(
            event_with_execution_profile_authority(
                Event(
                    type=EventType.BUDGET_LIMIT_REACHED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=payload,
                ),
                execution_profile,
            )
        )
        session_events = await self._run_limit_controller.session_usage_events(session.id)
        usage_summary = session_usage_summary(session.id, session_events)
        decision = StopDecision(
            limit=StopLimit.ESTIMATED_COST,
            maximum=result.maximum,
            actual=result.actual,
            message=result.message,
        )
        async for event in self._stop_session_for_limit_reached(
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
            reconcile_transition_cancellation=False,
        ):
            yield event

    async def _stop_session_for_budget_limit_reached(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        check: BudgetCheck,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        completed_tool_outcomes: list[runtime_records.ToolCallOutcome],
        tool_round_identity: ToolRoundIdentity | None = None,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
    ) -> AsyncGenerator[Event, None]:
        payload = budget_limit_reached_payload(check)
        yield await self._event_writer.emit(
            event_with_execution_profile_authority(
                Event(
                    type=EventType.BUDGET_LIMIT_REACHED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=payload,
                ),
                execution_profile,
            )
        )
        decision = StopDecision(
            limit=StopLimit.ESTIMATED_COST,
            maximum=check.maximum,
            actual=check.actual,
            message=check.message,
        )
        session_events = await self._run_limit_controller.session_usage_events(session.id)
        usage_summary = session_usage_summary(session.id, session_events)
        async for event in self._stop_session_for_limit_reached(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            decision=decision,
            usage_summary=usage_summary,
            cost_summary=check.cost_summary,
            messages=messages,
            tool_calls=tool_calls,
            completed_tool_outcomes=completed_tool_outcomes,
            tool_round_identity=tool_round_identity,
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
            execution_profile=execution_profile,
            reconcile_transition_cancellation=False,
        ):
            yield event

    async def _close_limited_tool_round(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        completed_tool_outcomes: list[runtime_records.ToolCallOutcome],
        decision: StopDecision,
        pending_approval_to_clear: PendingToolApproval | None = None,
        deferred_messages: list[Message] | None = None,
        requested_approval_decision: ToolApprovalDecision | None = None,
        approval_resolution_request_digest: str | None = None,
        tool_round_identity: ToolRoundIdentity | None = None,
        execution_profile: ExecutionProfileIdentity | None,
    ) -> AsyncGenerator[Event, None]:
        if tool_round_identity is None and pending_approval_to_clear is not None:
            tool_round_identity = ToolRoundIdentity(
                tool_round_id=pending_approval_to_clear.tool_round_id,
                model_step_id=pending_approval_to_clear.model_step_id,
                model_attempt_id=pending_approval_to_clear.model_attempt_id,
            )
        if tool_round_identity is None:
            raise RuntimeError("Closing a tool round requires its durable identity.")
        if pending_approval_to_clear is not None:
            if (
                type(requested_approval_decision) is not ToolApprovalDecision
                or type(approval_resolution_request_digest) is not str
                or not approval_resolution_request_digest
            ):
                raise RuntimeError(
                    "Closing a limited approval requires its original resolution request identity."
                )
            deferred_messages = (
                []
                if deferred_messages is None
                else [detach_message(message) for message in deferred_messages]
            )
            # The existing transcript may be a legacy, pre-quarantine assistant
            # message. Validate that boundary against the private durable pause
            # descriptor, not against terminal outcomes whose argument projection
            # can intentionally be unavailable.
            expected_tool_calls = [
                runtime_records.ToolCallRequest(
                    id=call.tool_call_id,
                    name=call.tool_name,
                    arguments=call.arguments,
                )
                for call in pending_approval_to_clear.tool_calls
            ]
            if await transcript_helpers.tool_round_has_result_messages(
                self.session_store,
                session.id,
                expected_tool_calls,
                tool_round_identity=tool_round_identity,
            ):

                def clear_exact_approval_round(
                    _current_session: Session,
                    checkpoint: dict[str, Any] | None,
                ) -> dict[str, Any]:
                    return approval_support.checkpoint_without_exact_pending_approval_round(
                        checkpoint,
                        approval=pending_approval_to_clear,
                        redactor=self._secret_redactor,
                    )

                await self.session_store.transform_checkpoint(
                    session.id,
                    clear_exact_approval_round,
                )
                materialized = await self._recovery_coordinator.materialize_expected_deferred_input(
                    session.id,
                    deferred_messages,
                )
                messages[:] = materialized.messages
                yield await self._event_writer.emit(
                    approval_support.cleared_event(
                        session=session,
                        agent_name=registered_agent.spec.name,
                        environment_name=_environment_name(registered_environment),
                        approval=pending_approval_to_clear,
                    )
                )
                if materialized.cancellation is not None:
                    raise materialized.cancellation
                return
            async for event in self._close_limited_approval_tool_round(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                messages=messages,
                tool_calls=tool_calls,
                completed_tool_outcomes=completed_tool_outcomes,
                decision=decision,
                pending_approval_to_clear=pending_approval_to_clear,
                deferred_messages=deferred_messages,
                requested_approval_decision=requested_approval_decision,
                approval_resolution_request_digest=approval_resolution_request_digest,
                tool_round_identity=tool_round_identity,
                execution_profile=execution_profile,
            ):
                yield event
            return

        tool_round_id = tool_round_identity.tool_round_id
        publication_id = f"tool-round:{tool_round_id}"
        if (
            await self.session_store.load_runtime_publication_receipt(
                session.id,
                publication_id,
            )
            is not None
        ):
            await self._recovery_coordinator.materialize_deferred_input_if_present(session.id)
            messages[:] = await self.session_store.load_transcript(session.id)
            return

        completed_ids = {outcome.call.id for outcome in completed_tool_outcomes}
        remaining_tool_calls = [
            tool_call for tool_call in tool_calls if tool_call.id not in completed_ids
        ]
        skipped_outcomes = _limit_reached_tool_round_results(
            tool_calls=remaining_tool_calls,
            decision=decision,
            tool_round_identity=tool_round_identity,
        )
        completed_tool_outcomes = tool_results.redact_tool_call_outcomes(
            completed_tool_outcomes,
            self._secret_redactor,
        )
        skipped_outcomes = tool_results.redact_tool_call_outcomes(
            skipped_outcomes,
            self._secret_redactor,
        )
        base_round_redactor = self._tool_round_executor.redactor_for_tool_calls(
            registered_agent=registered_agent,
            tool_calls=tool_calls,
        )
        for skipped_outcome in skipped_outcomes:
            await self.session_store.transform_checkpoint(
                session.id,
                lambda _current_session, current_checkpoint, call_id=skipped_outcome.call.id: (
                    tool_round_recovery.checkpoint_with_assistant_publication_snapshot(
                        current_checkpoint,
                        tool_round_identity=tool_round_identity,
                        tool_call_id=call_id,
                        redactor=base_round_redactor,
                        unsafe_output=False,
                    )
                ),
            )
            yield await self._event_writer.emit(
                _limit_reached_tool_call_event(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call_outcome=skipped_outcome,
                    decision=decision,
                    tool_round_identity=tool_round_identity,
                    execution_profile=execution_profile,
                )
            )

        source_checkpoint = await self.session_store.load_checkpoint(session.id)
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(source_checkpoint)
        if (
            pending_round is None
            or tool_round_recovery.pending_tool_round_identity(pending_round) != tool_round_identity
        ):
            raise RuntimeError("Limited tool round lost its durable pending marker.")
        lifecycle_events = await self.session_store.load_tool_round_lifecycle_events_for_round(
            session.id,
            [call.tool_call_id for call in pending_round.tool_calls],
            tool_round_identity=tool_round_recovery.pending_tool_round_identity(pending_round),
        )
        durable_transcript_cursor = await self.session_store.load_transcript_cursor(session.id)
        prepared = tool_round_publication.prepare_tool_round_publication(
            session_id=session.id,
            pending_round=pending_round,
            source_checkpoint=source_checkpoint,
            durable_events=lifecycle_events,
            expected_statuses={
                SessionStatus.RUNNING,
                SessionStatus.INTERRUPTING,
            },
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=durable_transcript_cursor,
        )
        cancellation = await tool_round_publication.publish_tool_round_with_exact_replay(
            prepared,
            session_store=self.session_store,
            event_writer=self._event_writer,
        )
        materialized = await self._recovery_coordinator.materialize_expected_deferred_input(
            session.id,
            pending_round.deferred_messages,
            cancellation=cancellation,
        )
        messages[:] = materialized.messages
        cancellation = materialized.cancellation
        if cancellation is not None:
            raise cancellation
        await self._session_control.raise_if_interrupted(session.id)

    async def _close_limited_approval_tool_round(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        completed_tool_outcomes: list[runtime_records.ToolCallOutcome],
        decision: StopDecision,
        pending_approval_to_clear: PendingToolApproval,
        deferred_messages: list[Message],
        requested_approval_decision: ToolApprovalDecision,
        approval_resolution_request_digest: str,
        tool_round_identity: ToolRoundIdentity,
        execution_profile: ExecutionProfileIdentity | None,
    ) -> AsyncGenerator[Event, None]:
        completed_ids = {outcome.call.id for outcome in completed_tool_outcomes}
        remaining_tool_calls = [
            tool_call for tool_call in tool_calls if tool_call.id not in completed_ids
        ]
        skipped_outcomes = _limit_reached_tool_round_results(
            tool_calls=remaining_tool_calls,
            decision=decision,
            tool_round_identity=tool_round_identity,
        )
        completed_tool_outcomes = tool_results.redact_tool_call_outcomes(
            completed_tool_outcomes,
            self._secret_redactor,
        )
        skipped_outcomes = tool_results.redact_tool_call_outcomes(
            skipped_outcomes,
            self._secret_redactor,
        )
        base_round_redactor = self._tool_round_executor.redactor_for_tool_calls(
            registered_agent=registered_agent,
            tool_calls=tool_calls,
        )
        for skipped_outcome in skipped_outcomes:
            await self.session_store.transform_checkpoint(
                session.id,
                lambda _current_session, current_checkpoint, call_id=skipped_outcome.call.id: (
                    tool_round_recovery.checkpoint_with_assistant_publication_snapshot(
                        current_checkpoint,
                        tool_round_identity=tool_round_identity,
                        tool_call_id=call_id,
                        redactor=base_round_redactor,
                        unsafe_output=False,
                    )
                ),
            )
            yield await self._event_writer.emit(
                _limit_reached_tool_call_event(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call_outcome=skipped_outcome,
                    decision=decision,
                    tool_round_identity=tool_round_identity,
                    execution_profile=execution_profile,
                    approval_id=pending_approval_to_clear.approval_id,
                )
            )
        skipped_outcomes = [
            runtime_records.ToolCallOutcome(
                call=runtime_records.ToolCallRequest(
                    id=outcome.call.id,
                    name=outcome.call.name,
                    arguments={},
                ),
                result=outcome.result,
            )
            for outcome in skipped_outcomes
        ]
        source_checkpoint = await self.session_store.load_checkpoint(session.id)
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(
            source_checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        if (
            pending_round is None
            or pending_round.tool_round_id != tool_round_identity.tool_round_id
        ):
            raise RuntimeError("Pending approval round changed before limit closure.")
        lifecycle_events = await self.session_store.load_tool_round_lifecycle_events_for_round(
            session.id,
            [call.tool_call_id for call in pending_round.tool_calls],
            tool_round_identity=tool_round_identity,
        )
        tool_result_messages = ordered_tool_result_messages(
            tool_calls,
            [*completed_tool_outcomes, *skipped_outcomes],
            parallel=True,
            tool_round_identity=tool_round_identity,
        )
        transcript_messages = list(tool_result_messages)
        if pending_round.assistant_message_state == "quarantined":
            transcript_messages.insert(
                0,
                transcript_helpers.assistant_message_with_projected_tool_arguments(
                    tool_round_recovery.ready_assistant_publication_message(pending_round),
                    [*completed_tool_outcomes, *skipped_outcomes],
                ),
            )
        target_checkpoint = approval_support.checkpoint_without_exact_pending_approval_round(
            source_checkpoint,
            approval=pending_approval_to_clear,
            redactor=self._secret_redactor,
        )
        clear_event = approval_support.cleared_event(
            session=session,
            agent_name=registered_agent.spec.name,
            environment_name=_environment_name(registered_environment),
            approval=pending_approval_to_clear,
        )
        prepared_close = approval_publication.prepare_approval_publication(
            session_id=session.id,
            publication_id=f"approval-close:{pending_approval_to_clear.approval_id}",
            kind="approval-close",
            intent={
                "schema_version": 1,
                "approval_id": pending_approval_to_clear.approval_id,
                "tool_call_id": pending_approval_to_clear.tool_call_id,
                **tool_round_identity.payload(),
                "decision": "limit_reached",
                "requested_decision": requested_approval_decision.value,
                "resolution_request_digest": approval_resolution_request_digest,
                "tool_call_ids": [call.tool_call_id for call in pending_round.tool_calls],
                "approval_digest": runtime_publication_checkpoint_value_digest(
                    pending_approval_to_clear.model_dump(mode="json")
                ),
                "pending_round_digest": runtime_publication_checkpoint_value_digest(
                    pending_round.model_dump(mode="json")
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
            expected_transcript_cursor=(
                await self.session_store.load_transcript_cursor(session.id)
            ),
        )
        prepared_events = prepared_close.request.events
        if len(prepared_events) != 1:
            raise AssertionError("Approval limit closure must publish one checkpoint event.")
        clear_event = prepared_events[0]
        cancellation = await approval_publication.publish_approval_with_exact_replay(
            prepared_close,
            session_store=self.session_store,
            event_writer=self._event_writer,
            fan_out=False,
        )
        materialized = await self._recovery_coordinator.materialize_expected_deferred_input(
            session.id,
            deferred_messages,
            cancellation=cancellation,
        )
        messages[:] = materialized.messages
        cancellation = materialized.cancellation
        await self._event_writer.fan_out_persisted([clear_event])
        yield clear_event
        if cancellation is not None:
            raise cancellation

    async def _clear_pending_tool_round_if_matches(
        self,
        session_id: str,
        pending_round: tool_round_recovery.PendingToolRound,
    ) -> None:
        checkpoint = await self.session_store.load_checkpoint(session_id)
        if checkpoint is None:
            return
        copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
        current = tool_round_recovery.pending_tool_round_from_checkpoint(
            copied_checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        if current is None or current.tool_round_id != pending_round.tool_round_id:
            return
        copied_checkpoint.pop(tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY, None)
        await self.session_store.transform_checkpoint(
            session_id,
            _replace_checkpoint_preserving_runtime_state(copied_checkpoint),
        )

    async def _load_pending_session_interrupt_payload(
        self,
        session_id: str,
        *,
        default: dict[str, Any],
    ) -> dict[str, Any]:
        checkpoint = await self.session_store.load_checkpoint(session_id)
        if checkpoint is None:
            return copy_json_value(default, "interrupt_payload")
        copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
        value = copied_checkpoint.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
        if value is None:
            return copy_json_value(default, "interrupt_payload")
        if type(value) is not dict:
            raise ValueError("Pending session interrupt checkpoint must be an object.")
        return copy_json_value(value, "interrupt_payload")

    async def _clear_pending_session_interrupt(self, session_id: str) -> None:
        def transform(_session: Session, checkpoint: dict[str, Any] | None) -> dict[str, Any]:
            copied = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
            copied.pop(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY, None)
            return copied

        await self.session_store.transform_checkpoint(session_id, transform)

    async def _load_pending_interruption_cascade(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
        checkpoint = await self.session_store.load_checkpoint(session_id)
        if checkpoint is None:
            return None
        marker = checkpoint.get(_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY)
        if marker is None:
            return None
        if type(marker) is not dict:
            raise ValueError("Pending interruption cascade checkpoint must be an object.")
        copied_marker = copy_json_value(marker, "pending_interruption_cascade")
        attempt_id = copied_marker.get("attempt_id")
        interrupt_payload = copied_marker.get("interrupt_payload")
        if type(attempt_id) is not str or not attempt_id.strip():
            raise ValueError("Pending interruption cascade attempt_id must be a non-blank string.")
        if type(interrupt_payload) is not dict:
            raise ValueError("Pending interruption cascade payload must be an object.")
        failure_recorded = copied_marker.get("failure_recorded", False)
        if type(failure_recorded) is not bool:
            raise ValueError("Pending interruption cascade failure_recorded must be a boolean.")
        generation = copied_marker.get("generation", 0)
        if type(generation) is not int or generation < 0:
            raise ValueError("Pending interruption cascade generation must be non-negative.")
        claim_id = copied_marker.get("claim_id")
        claim_expires_at = _interruption_cascade_marker_datetime(
            copied_marker,
            "claim_expires_at",
        )
        if claim_id is not None:
            if type(claim_id) is not str or not claim_id.strip() or claim_expires_at is None:
                raise ValueError("Pending interruption cascade claim is invalid.")
        elif claim_expires_at is not None:
            raise ValueError("Pending interruption cascade claim is invalid.")
        retry_request = _copy_interruption_cascade_retry_request(copied_marker.get("retry_request"))
        if retry_request is not None:
            copied_marker["retry_request"] = retry_request
        _interruption_cascade_marker_datetime(copied_marker, "created_at")
        return copied_marker

    async def _claim_pending_interruption_cascade(
        self,
        session_id: str,
        interrupt_payload: dict[str, Any],
        *,
        create_if_missing: bool = True,
        retry_request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        retry_request = _copy_interruption_cascade_retry_request(retry_request)
        resolved_marker: dict[str, Any] | None = None

        def transform(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            nonlocal resolved_marker
            copied_checkpoint = (
                {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
            )
            existing = copied_checkpoint.get(_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY)
            if existing is None:
                if not create_if_missing:
                    return None
                marker = {
                    "attempt_id": str(uuid4()),
                    "interrupt_payload": copy_json_value(interrupt_payload, "interrupt_payload"),
                    "created_at": self._clock().isoformat(),
                }
            elif type(existing) is not dict:
                raise ValueError("Pending interruption cascade checkpoint must be an object.")
            else:
                marker = copy_json_value(existing, "pending_interruption_cascade")
            attempt_id = marker.get("attempt_id")
            if type(attempt_id) is not str or not attempt_id.strip():
                raise ValueError(
                    "Pending interruption cascade attempt_id must be a non-blank string."
                )
            if type(marker.get("interrupt_payload")) is not dict:
                raise ValueError("Pending interruption cascade payload must be an object.")
            existing_retry_request = _copy_interruption_cascade_retry_request(
                marker.get("retry_request")
            )
            if retry_request is not None:
                marker["retry_request"] = copy_json_value(retry_request, "retry_request")
            elif existing_retry_request is not None:
                marker["retry_request"] = existing_retry_request
            failure_recorded = marker.get("failure_recorded", False)
            if type(failure_recorded) is not bool:
                raise ValueError("Pending interruption cascade failure_recorded must be a boolean.")
            generation = marker.get("generation", 0)
            if type(generation) is not int or generation < 0:
                raise ValueError("Pending interruption cascade generation must be non-negative.")
            now = self._clock()
            claim_id = marker.get("claim_id")
            claim_expires_at = _interruption_cascade_marker_datetime(
                marker,
                "claim_expires_at",
            )
            if claim_id is not None:
                if type(claim_id) is not str or not claim_id.strip() or claim_expires_at is None:
                    raise ValueError("Pending interruption cascade claim is invalid.")
                if claim_expires_at > now:
                    return None
            elif claim_expires_at is not None:
                raise ValueError("Pending interruption cascade claim is invalid.")
            marker.setdefault("created_at", now.isoformat())
            marker["generation"] = generation + 1
            marker["claim_id"] = str(uuid4())
            marker["claim_expires_at"] = (
                now + timedelta(seconds=interruption_cascade_lease_seconds())
            ).isoformat()
            copied_checkpoint[_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY] = marker
            resolved_marker = copy_json_value(marker, "pending_interruption_cascade")
            return copied_checkpoint

        await self.session_store.transform_checkpoint(session_id, transform)
        return resolved_marker

    async def _mark_pending_interruption_cascade_failed(
        self,
        session_id: str,
        attempt_id: str,
        generation: int,
        claim_id: str,
    ) -> bool:
        recorded = False

        def transform(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            nonlocal recorded
            if checkpoint is None:
                return None
            copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
            marker = copied_checkpoint.get(_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY)
            if (
                type(marker) is not dict
                or marker.get("attempt_id") != attempt_id
                or marker.get("generation") != generation
                or marker.get("claim_id") != claim_id
            ):
                return None
            marker["failure_recorded"] = True
            marker.pop("claim_id", None)
            marker.pop("claim_expires_at", None)
            recorded = True
            return copied_checkpoint

        await self.session_store.transform_checkpoint(session_id, transform)
        return recorded

    async def _complete_pending_interruption_cascade(
        self,
        session_id: str,
        attempt_id: str,
        generation: int,
        claim_id: str,
    ) -> tuple[bool, bool]:
        cleared = False
        failure_recorded = False

        def transform(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            nonlocal cleared, failure_recorded
            if checkpoint is None:
                return None
            copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
            marker = copied_checkpoint.get(_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY)
            if (
                type(marker) is not dict
                or marker.get("attempt_id") != attempt_id
                or marker.get("generation") != generation
                or marker.get("claim_id") != claim_id
            ):
                return None
            current_failure_recorded = marker.get("failure_recorded", False)
            if type(current_failure_recorded) is not bool:
                raise ValueError("Pending interruption cascade failure_recorded must be a boolean.")
            failure_recorded = current_failure_recorded
            copied_checkpoint.pop(_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY)
            cleared = True
            return copied_checkpoint

        await self.session_store.transform_checkpoint(session_id, transform)
        return cleared, failure_recorded

    async def _renew_pending_interruption_cascade_claim(
        self,
        session_id: str,
        attempt_id: str,
        generation: int,
        claim_id: str,
    ) -> bool:
        renewed = False

        def transform(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            nonlocal renewed
            if checkpoint is None:
                return None
            copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
            marker = copied_checkpoint.get(_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY)
            if (
                type(marker) is not dict
                or marker.get("attempt_id") != attempt_id
                or marker.get("generation") != generation
                or marker.get("claim_id") != claim_id
            ):
                return None
            marker["claim_expires_at"] = (
                self._clock() + timedelta(seconds=interruption_cascade_lease_seconds())
            ).isoformat()
            renewed = True
            return copied_checkpoint

        await self.session_store.transform_checkpoint(session_id, transform)
        return renewed

    async def _release_pending_interruption_cascade_claim(
        self,
        session_id: str,
        attempt_id: str,
        generation: int,
        claim_id: str,
    ) -> None:
        def transform(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            if checkpoint is None:
                return None
            copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
            marker = copied_checkpoint.get(_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY)
            if (
                type(marker) is not dict
                or marker.get("attempt_id") != attempt_id
                or marker.get("generation") != generation
                or marker.get("claim_id") != claim_id
            ):
                return None
            marker.pop("claim_id", None)
            marker.pop("claim_expires_at", None)
            return copied_checkpoint

        await self.session_store.transform_checkpoint(session_id, transform)

    async def _clear_pending_interruption_cascade(self, session_id: str) -> str | None:
        cleared_attempt_id: str | None = None

        def transform(_session: Session, checkpoint: dict[str, Any] | None) -> dict[str, Any]:
            nonlocal cleared_attempt_id
            if checkpoint is None:
                return {}
            copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
            marker = copied_checkpoint.pop(_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY, None)
            if marker is None:
                return copied_checkpoint
            if type(marker) is not dict or type(marker.get("attempt_id")) is not str:
                raise ValueError("Pending interruption cascade checkpoint is invalid.")
            cleared_attempt_id = marker["attempt_id"]
            return copied_checkpoint

        await self.session_store.transform_checkpoint(session_id, transform)
        return cleared_attempt_id

    async def _require_session(self, session_id: str) -> Session:
        loaded = await self.session_store.load(session_id)
        if loaded is None:
            raise KeyError(f"Session not found: {session_id}") from None
        return loaded

    async def _handle_session_interrupted_preserving_failure(
        self,
        *,
        authoritative_failure: BaseException,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        execution_profile: ExecutionProfileIdentity | None = None,
        prepare_interruption: Callable[[], Awaitable[Iterable[Event] | None]] | None = None,
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
            if (
                persisted_session is None
                or persisted_session.status not in _INTERRUPTIBLE_SESSION_STATUSES
            ):
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
            async for event in self._handle_session_interrupted(
                session=persisted_session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                execution_profile=execution_profile,
                run_started_at=run_started_at,
                turn_usage_tracker=turn_usage_tracker,
                active_run=active_run,
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
        """Settle finalizer awaits without replacing an interruption aggregate."""

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

    async def _handle_session_interrupted(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        execution_profile: ExecutionProfileIdentity | None = None,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
        interaction_transition_failures: tuple[dict[str, Any], ...] = (),
    ) -> AsyncGenerator[Event, None]:
        clear_current_task_cancellation()
        current_task = asyncio.current_task()
        if current_task is not None:
            self._session_control.unregister_active_task(session.id, current_task)
        self._session_control.begin_emitting_interrupted(session.id)
        try:
            loaded_interrupted = await self.session_store.load(session.id)
            if loaded_interrupted is None:
                raise KeyError(f"Session not found: {session.id}") from None
            interaction_event: Event | None = None
            if loaded_interrupted.status != SessionStatus.INTERRUPTED:
                (
                    loaded_interrupted,
                    interaction_event,
                    _,
                ) = await self._publish_sibling_interaction_transition(
                    session=loaded_interrupted,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    to_status=SessionStatus.INTERRUPTED,
                    execution_profile=execution_profile,
                    finalize_unsettled_cancellation=False,
                )
            payload = await self._load_pending_session_interrupt_payload(session.id, default={})
            if interaction_transition_failures:
                copied_failures = copy_durable_json_value(
                    list(interaction_transition_failures),
                    "interaction transition cancellation diagnostics",
                )
                if type(copied_failures) is not list:
                    raise TypeError(
                        "Interaction transition cancellation diagnostics must be a list."
                    )
                payload["interaction_transition_failures"] = copied_failures
            if (
                interaction_event is None
                and _current_session_interaction_id(session.id) is not None
            ):
                _, interaction_event, _ = await self._publish_sibling_interaction_transition(
                    session=loaded_interrupted,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    to_status=SessionStatus.INTERRUPTED,
                    from_statuses={SessionStatus.INTERRUPTED},
                    execution_profile=execution_profile,
                    finalize_unsettled_cancellation=False,
                )
            if interaction_event is not None:
                yield interaction_event
            interruption_request_id = interruption_request_id_from_payload(payload)
            existing_interrupt_event = await self._session_control.wait_for_interrupted_event(
                session.id,
                interruption_request_id=interruption_request_id,
            )
            if existing_interrupt_event is not None:
                await self._clear_pending_session_interrupt(session.id)
                if not interruption_cascade_suppressed():
                    self._schedule_background_interruption_cascade(
                        parent_session_id=session.id,
                        interrupt_payload=existing_interrupt_event.payload,
                        create_if_missing=False,
                    )
                turn_completed_event = (
                    active_run.turn_completed_event
                    if active_run is not None and active_run.turn_completed_event is not None
                    else self._session_control.active_turn_completed_event(session.id)
                )
                if turn_completed_event is not None:
                    yield turn_completed_event
                yield existing_interrupt_event
                return
            payload.setdefault("interruption_type", _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED)
            payload.setdefault("interruption_request_id", str(uuid4()))
            if run_started_at is not None and turn_usage_tracker is not None:
                for event in await self._emit_turn_completed_once(
                    session=loaded_interrupted,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    status=SessionStatus.INTERRUPTED,
                    run_started_at=run_started_at,
                    usage_tracker=turn_usage_tracker,
                    active_run=active_run,
                ):
                    yield event
            terminal_event_stream = self._emit_terminal_event_with_hooks(
                event=_runtime_interruption_event(
                    Event(
                        type=EventType.SESSION_INTERRUPTED,
                        session_id=loaded_interrupted.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload=payload,
                    ),
                    execution_profile=execution_profile,
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                session=loaded_interrupted,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
            )
            terminal_prefix, interrupted_event = await _collect_through_event_type(
                terminal_event_stream,
                EventType.SESSION_INTERRUPTED,
                missing_message="Session interruption produced no terminal event.",
            )

            await self._clear_pending_session_interrupt(session.id)
            if not interruption_cascade_suppressed():
                self._schedule_background_interruption_cascade(
                    parent_session_id=session.id,
                    interrupt_payload=interrupted_event.payload,
                    create_if_missing=False,
                )
            for event in terminal_prefix:
                yield event
            async for event in terminal_event_stream:
                yield event
        finally:
            self._session_control.end_emitting_interrupted(session.id)

    async def _close_durable_pending_tool_round_after_interrupt(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Close a round when interruption precedes creation of its live runner."""
        checkpoint = await self.session_store.load_checkpoint(session.id)
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
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
            tool_round_identity=tool_round_recovery.pending_tool_round_identity(pending_round),
            cancellation_artifacts=None,
            cancellation_artifacts_by_id=None,
            execution_profile=execution_profile,
        )
        async for event in self._recovery_coordinator.close_interrupted_tool_round(request):
            yield event

    async def _close_tool_round_after_interrupt(
        self,
        request: InterruptedToolRoundRequest,
    ) -> AsyncGenerator[Event, None]:
        async for event in self._recovery_coordinator.close_interrupted_tool_round(request):
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

    async def _bind_event_to_session_run_operation(
        self,
        event: Event,
        *,
        session: Session,
    ) -> Event:
        checkpoint = await self.session_store.load_checkpoint(session.id)
        run_operation = _session_run_operation_from_checkpoint(checkpoint)
        if run_operation is None:
            return event
        if run_operation.run_epoch != session.run_epoch:
            raise RuntimeError(
                "Terminal event session run operation does not match the active run epoch."
            )
        return _event_with_session_run_operation(event, run_operation)

    async def _clear_session_run_operation(
        self,
        *,
        session_id: str,
        operation_id: str,
        terminal_evidence_durable: bool = False,
    ) -> None:
        def clear_operation(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            run_operation = _session_run_operation_from_checkpoint(checkpoint)
            if run_operation is None:
                return checkpoint
            if run_operation.operation_id != operation_id:
                raise RuntimeError(
                    "Session run operation changed before terminal evidence cleanup."
                )
            return _checkpoint_after_session_run_operation_cleanup(
                checkpoint,
                operation=run_operation,
                retain_terminal_receipt=terminal_evidence_durable,
            )

        await self.session_store.transform_checkpoint(
            session_id,
            clear_operation,
        )

    async def _reconcile_persisted_terminal_event(
        self,
        event: Event,
    ) -> Event | None:
        """Return the exact durable event after an ambiguous publication failure."""
        records = await self.session_store.query_events(
            EventQuery(
                session_id=event.session_id,
                event_id=event.id,
                limit=1,
            )
        )
        return _reconcile_exact_persisted_event(
            event,
            records,
            conflict_message=(
                "Terminal event identity is already used by different durable evidence."
            ),
        )

    async def _emit_terminal_event_with_hooks(
        self,
        *,
        event: Event,
        phase: RuntimeHookPhase,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity | None = None,
    ) -> AsyncGenerator[Event, None]:
        event = await self._bind_event_to_session_run_operation(
            event,
            session=session,
        )
        finalize_result = await self._environment_lifecycle.finalize_terminal_event(
            event=event,
            session=session,
            registered_environment=registered_environment,
            execution_profile=execution_profile,
        )
        for binding_event in finalize_result.events:
            yield binding_event
        prepared_terminal_event = self._event_writer.prepare(finalize_result.event)
        try:
            terminal_event = await self._event_writer.emit(prepared_terminal_event)
        except Exception as publication_failure:
            try:
                reconciled_terminal_event = await self._reconcile_persisted_terminal_event(
                    prepared_terminal_event
                )
            except Exception as reconciliation_failure:
                uncertainty = TerminalEventPublicationUncertain(
                    event=prepared_terminal_event,
                    publication_failure=publication_failure,
                    reconciliation_failure=reconciliation_failure,
                )
                raise uncertainty from uncertainty.failures
            if reconciled_terminal_event is None:
                raise
            terminal_event = reconciled_terminal_event
            logger.warning(
                "Terminal event is durable but its publication acknowledgement or "
                "side-effect delivery failed: session_id=%s event_id=%s error_type=%s",
                session.id,
                terminal_event.id,
                type(publication_failure).__name__,
            )
        if terminal_event.model_dump(mode="json") != prepared_terminal_event.model_dump(
            mode="json"
        ):
            raise RuntimeError("Terminal event publication returned different durable evidence.")
        if event_id_is_runtime_generated(prepared_terminal_event):
            terminal_event = event_with_runtime_generated_id(terminal_event)
        run_operation_id = terminal_event.payload.get(_SESSION_RUN_OPERATION_ID_PAYLOAD_KEY)
        if run_operation_id is not None:
            try:
                await self._clear_session_run_operation(
                    session_id=session.id,
                    operation_id=require_clean_nonblank(
                        run_operation_id,
                        "terminal event session_run_operation_id",
                    ),
                    terminal_evidence_durable=True,
                )
            except Exception as cleanup_failure:
                logger.warning(
                    "Terminal evidence is durable but session run operation cleanup "
                    "remains pending: session_id=%s event_id=%s error_type=%s",
                    session.id,
                    terminal_event.id,
                    type(cleanup_failure).__name__,
                )
        yield terminal_event
        async for hook_event in self._run_runtime_hooks(
            phase=phase,
            session=session,
            terminal_event=terminal_event,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            hooks=self._runtime_hooks,
            scope="app",
            execution_profile=execution_profile,
        ):
            yield hook_event
        async for hook_event in self._run_runtime_hooks(
            phase=phase,
            session=session,
            terminal_event=terminal_event,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            hooks=registered_agent.runtime_hooks,
            scope="agent",
            execution_profile=execution_profile,
        ):
            yield hook_event

    async def _run_runtime_hooks(
        self,
        *,
        phase: RuntimeHookPhase,
        session: Session,
        terminal_event: Event,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
        scope: str,
        execution_profile: ExecutionProfileIdentity | None = None,
    ) -> AsyncGenerator[Event, None]:
        for registered_hook in hooks:
            hook = registered_hook.hook
            if not _runtime_hook_supports_phase(
                hook=hook,
                phase=phase,
            ):
                continue
            hook_name = registered_hook.name
            yield await self._event_writer.emit(
                _runtime_hook_event(
                    event_type=EventType.HOOK_STARTED,
                    hook_name=hook_name,
                    scope=scope,
                    phase=phase,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    terminal_event=terminal_event,
                    execution_profile=execution_profile,
                    payload={},
                )
            )
            context = RuntimeHookContext(
                runtime=self._hook_runtime,
                hook_name=hook_name,
                phase=phase,
                session=session,
                terminal_event=terminal_event,
                execution_profile=execution_profile,
            )
            try:
                await _call_runtime_hook(hook=hook, phase=phase, context=context)
            except Exception as exc:
                diagnostic = exception_diagnostic(
                    exc,
                    empty_message="runtime hook failed",
                    nonportable_message="Runtime hook failed with a non-portable diagnostic.",
                    redactor=self._secret_redactor,
                )
                yield await self._event_writer.emit(
                    _runtime_hook_event(
                        event_type=EventType.HOOK_FAILED,
                        hook_name=hook_name,
                        scope=scope,
                        phase=phase,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        terminal_event=terminal_event,
                        execution_profile=execution_profile,
                        payload={
                            **diagnostic.payload_fields(),
                            **_runtime_hook_actions_payload(context),
                        },
                    )
                )
                continue
            yield await self._event_writer.emit(
                _runtime_hook_event(
                    event_type=EventType.HOOK_COMPLETED,
                    hook_name=hook_name,
                    scope=scope,
                    phase=phase,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    terminal_event=terminal_event,
                    execution_profile=execution_profile,
                    payload=_runtime_hook_actions_payload(context),
                )
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
    ) -> AsyncGenerator[tuple[Event, BeforeStopDecision | None], None]:
        classification = classify_assistant_step(step_result)
        policy_groups = (
            ("app", self._loop_policies),
            ("agent", registered_agent.loop_policies),
            (
                "request",
                validate_loop_policies(
                    request_loop_policies,
                    field_name="request_loop_policies",
                ),
            ),
        )
        for scope, policies in policy_groups:
            for policy in policies:
                if not _loop_policy_supports_before_stop(policy):
                    continue
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
                        decision,
                    )
                    return
