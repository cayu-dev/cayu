from __future__ import annotations

import asyncio
import contextvars
import hmac
import json
import math
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Executor
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeVar, cast
from uuid import uuid4

from pydantic import ValidationError

from cayu._clock import normalize_utc_datetime, utc_clock, utc_duration_cutoff
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    JsonUtf8SizeCounter,
    copy_durable_json_object,
    copy_label_map,
    require_nonblank,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu.core.events import (
    EVENT_ID_MAX_CHARS,
    Event,
    EventType,
    event_with_runtime_payload_authority,
)
from cayu.core.messages import Message, MessageRole
from cayu.core.runtime_authority import CheckpointValueAuthority
from cayu.core.workflows import WORKFLOW_ATTEMPT_EVENT_TYPE
from cayu.memory_evidence import (
    MAX_RECALL_RECEIPT_ITEMS,
    ContextExposure,
    ContextExposurePage,
    ContextExposureTransitionConflict,
    ContextExposureTransitionRequest,
    RecallEvidenceConflict,
    RecallEvidenceQuery,
    RecallItemExposure,
    RecallReceipt,
    RecallReceiptPage,
    append_context_exposure_transition,
    context_exposure_creation_matches,
    context_exposure_transition_replays,
    copy_context_exposure,
    copy_recall_item_exposure,
    copy_recall_receipt,
    decode_recall_evidence_cursor,
    encode_recall_evidence_cursor,
    memory_evidence_document_bytes,
    recall_item_exposure_matches_receipt_item,
    require_memory_evidence_id,
    require_memory_evidence_session_id,
    validate_context_exposure_receipt_scope,
    validate_new_context_exposure,
)
from cayu.runtime._child_session_notifications import (
    ChildSessionLifecycleOccurrence,
    ChildSessionLifecycleOccurrenceSource,
    ChildSessionLifecyclePage,
    ChildSessionLifecycleQuery,
    child_session_notification_stage_binding,
    child_session_notification_storage_key,
)
from cayu.runtime._provider_operation_cancellation_claim import (
    active_provider_operation_cancellation_claim_from_checkpoint,
)
from cayu.runtime._task_lease_authority import managed_task_lease_mutation
from cayu.runtime.aggregates import EXACT_AGGREGATE, UsageRollupStoreResult
from cayu.runtime.approvals import ResolutionActor, resolution_actor_payload
from cayu.runtime.completion_verifier_profiles import (
    CompletionVerifierProfilePreparationRequest,
    CompletionVerifierProfileRecord,
    completion_verifier_profile_preparation_request_sha256,
    completion_verifier_profile_record_from_document,
    completion_verifier_profile_record_from_preparation,
    copy_completion_verifier_profile_preparation_request,
    copy_completion_verifier_profile_record,
    require_completion_verifier_profile_transition,
)
from cayu.runtime.execution_profiles import (
    ActiveInvocationExecutionProfile,
    ExecutionProfileDecision,
    ExecutionProfileIdentity,
    ExecutionProfileRejectionResult,
)
from cayu.runtime.execution_units import ToolRoundIdentity, copy_tool_round_identity
from cayu.runtime.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
    INTERACTION_TERMINAL_EVENT_TYPES,
)
from cayu.runtime.invocation import SessionInvocation, SessionInvocationBinding, TaskInvocation
from cayu.runtime.local_execution_attempts import (
    LocalExecutionAttemptAuthority,
    LocalExecutionAttemptConflict,
    LocalExecutionAttemptListCursor,
    LocalExecutionAttemptRecord,
    LocalExecutionAttemptRecoveryClaim,
    LocalExecutionAttemptSettlement,
    LocalExecutionAttemptStart,
    _copy_authenticated_local_execution_attempt_settlement,
    _copy_local_execution_attempt_authority,
    _copy_local_execution_attempt_list_cursor,
    _copy_local_execution_attempt_recovery_claim,
    _copy_local_execution_attempt_start,
    advance_local_execution_attempt_start,
    claim_local_execution_attempt_recovery_record,
    prepare_local_execution_attempt_record,
    require_local_execution_recovery_eligible,
    require_local_execution_task_authority,
    settle_local_execution_attempt_record,
)
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, parse_public_authority_alias
from cayu.runtime.service_manifest import RuntimeStoreDurability
from cayu.runtime.sessions import (
    _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES,
    _TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT,
    _TOOL_ROUND_LIFECYCLE_EVENT_TYPES,
    CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS,
    DELETE_BLOCKED_SESSION_STATUSES,
    FORK_EXECUTION_PROFILE_METADATA_KEY,
    FORK_TRANSCRIPT_VALIDATION_ERROR,
    INHERIT_INTERACTION,
    LATEST_TRANSCRIPT_TEXT_MAX_CHARS,
    LATEST_TRANSCRIPT_TEXT_MAX_PARTS,
    LATEST_TRANSCRIPT_TEXT_MAX_SOURCE_BYTES,
    MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL,
    MAX_PENDING_ACTION_TOOL_CALLS,
    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
    MODEL_TARGET_PROJECTION_METADATA_KEY,
    PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY,
    RUNTIME_BUILD_PROVENANCE_METADATA_KEY,
    RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS,
    RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX,
    SESSION_INSPECTION_LABEL_LIMIT,
    SESSION_LINEAGE_MAX_EVENT_ID_BYTES,
    SESSION_LINEAGE_MAX_IDENTIFIER_BYTES,
    SESSION_LINEAGE_MAX_ORIGIN_EVENTS,
    SESSION_LINEAGE_MAX_TIMESTAMP_BYTES,
    SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES,
    BudgetReservationIdentityConflict,
    CheckpointRootFieldGuard,
    CheckpointTransform,
    DeferredInteractionInput,
    EnqueueSessionMessageRequest,
    EnqueueSessionMessageResult,
    EventQuery,
    EventQueryResultTooLarge,
    EventRecord,
    EventSummary,
    ForkCheckpointAuthorityDecoder,
    ForkSystemPromptReplacement,
    ForkTranscriptValidator,
    InteractionAttribution,
    InteractionTransitionReceiptResult,
    InteractionTransitionResult,
    InteractionTransitionSpec,
    McpManifestBaseline,
    McpManifestBaselineLoadResult,
    McpManifestPublicationResult,
    ModelCompletionStage,
    ModelCompletionStageAbandonmentResult,
    ModelCompletionStageDispatch,
    ModelCompletionStageResult,
    ModelCompletionStageSettlementRequest,
    PendingActionIssue,
    PendingActionKind,
    PendingActionListResult,
    PendingActionQuery,
    PendingActionSession,
    PersistedEventSideEffectClaim,
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectDelivery,
    PersistedEventSideEffectStatus,
    ProfiledSessionForkResult,
    QueuedDispatchTerminalReceipt,
    QueuedDispatchTerminalReceiptQuery,
    RunnerObservedEventIdentity,
    RunRequest,
    RuntimePublicationMutation,
    RuntimePublicationReceipt,
    RuntimePublicationResult,
    Session,
    SessionAggregateFilter,
    SessionForkActiveModelStageConflict,
    SessionForkProfileRelationship,
    SessionIdentity,
    SessionInspectionIdentity,
    SessionInvocationSnapshot,
    SessionLineageNode,
    SessionLineageOrigin,
    SessionLineageQuery,
    SessionLineageResult,
    SessionListResult,
    SessionMessageDeliveryBatch,
    SessionMessageQueueStatus,
    SessionModelCompletionStageConflict,
    SessionModelTransition,
    SessionOperationalSnapshot,
    SessionOperationInitializer,
    SessionOperationPublication,
    SessionOperationTransform,
    SessionOrder,
    SessionOutcome,
    SessionQuery,
    SessionQueuedMessage,
    SessionQueuedMessagesPending,
    SessionRunFenced,
    SessionRuntimeIdentity,
    SessionRuntimePublicationConflict,
    SessionStateSnapshot,
    SessionStatus,
    SessionStatusConflict,
    SessionStatusCounts,
    SessionStore,
    SessionTopologyCycle,
    SessionTopologyDepthExceeded,
    SessionTopologyNode,
    SessionTopologyQuery,
    SessionTopologyStoreResult,
    StoreTimeCheckpointTransform,
    StoreTimeSessionOperationTransform,
    TerminalPublicationMarker,
    TerminalSessionEvidence,
    TerminalSessionEvidenceError,
    TerminalSessionEvidenceErrorCode,
    TerminalSessionEvidenceLimits,
    TranscriptPage,
    TranscriptQuery,
    TranscriptRecord,
    TranscriptSearchHit,
    TranscriptSearchQuery,
    TranscriptSearchResult,
    TranscriptSnapshot,
    TranscriptTextReadLimitExceeded,
    UsageRollupQuery,
    _activate_session_run_fence,
    _active_model_completion_stage_record,
    _active_unexpired_incomplete_recovery_claim_id,
    _active_unexpired_session_operation_id,
    _apply_runtime_publication_checkpoint_mutation,
    _apply_runtime_publication_operation_record_mutations,
    _assemble_terminal_session_evidence,
    _assert_session_run_epoch,
    _assert_session_run_epoch_value,
    _authenticated_public_authority_alias_private_value,
    _build_runtime_publication_receipt,
    _checkpoint_after_initial_transcript_publication,
    _checkpoint_transform_result_preserving_completion_result_event_publications,
    _child_session_lifecycle_entry,
    _child_session_lifecycle_entry_sort_key,
    _child_session_lifecycle_occurrence,
    _child_session_notification_consumption_record,
    _child_session_notification_consumption_replays,
    _classify_terminal_session_evidence_records,
    _completion_result_event_publication_delete_block_reason,
    _copy_checkpoint_for_transform,
    _copy_mcp_manifest_publication,
    _copy_optional_execution_profile,
    _copy_optional_execution_profile_decision,
    _copy_optional_interaction_admission,
    _copy_optional_tool_capability_ceiling,
    _copy_profiled_fork_authority,
    _copy_queued_interaction_started_event,
    _copy_runner_owned_interruption_proof,
    _copy_session_event_batch,
    _copy_session_model_transition,
    _copy_terminal_session_evidence_limits,
    _copy_transition_interaction_admission,
    _copy_workflow_step_reservation,
    _current_session_run_epoch,
    _deactivate_session_interaction,
    _deactivate_session_run_fence,
    _durable_subagent_parent_delete_block_reason,
    _event_file_attachment_attestations_are_runtime_owned,
    _event_input_contract_is_runtime_owned,
    _execution_profile_rejection_events_equivalent,
    _incomplete_recovery_claim_from_checkpoint,
    _initial_transcript_pending_checkpoint,
    _initial_transcript_prefix_count,
    _interaction_transition_receipt_record,
    _interaction_transition_spec_from_receipt,
    _interaction_transition_storage_key,
    _invocation_terminal_event_receipt_record,
    _invocation_terminal_event_storage_key,
    _load_interaction_transition_receipt,
    _model_completion_retry_settlement_request,
    _model_completion_stage_abandonment_record,
    _model_completion_stage_dispatch_record,
    _model_completion_stage_dispatch_storage_key,
    _model_completion_stage_preparation_record,
    _model_completion_stage_settlement_record,
    _model_completion_stage_settlement_storage_key,
    _model_completion_stage_storage_identity,
    _model_completion_stage_terminal_record,
    _model_completion_stage_winner_record,
    _model_completion_stage_winner_storage_key,
    _model_completion_terminal_advances_last_activity,
    _ModelCompletionStagePromotionContext,
    _next_runtime_publication_timestamp,
    _prepare_execution_profile_rejection,
    _prepare_initial_session_operation_records,
    _prepare_interaction_transition,
    _prepare_interaction_transition_receipt_lookup,
    _prepare_model_completion_stage_promotion,
    _prepare_session_fork_request,
    _PreparedModelCompletionStage,
    _PreparedModelCompletionStageAbandonment,
    _PreparedModelCompletionStageTerminal,
    _PreparedRuntimePublication,
    _profiled_fork_authority_validation_error,
    _project_interruption_cascade_marker_fields,
    _public_authority_alias_store_key,
    _queued_dispatch_terminal_receipts_from_checkpoint,
    _queued_session_message_event_payload,
    _reconstruct_active_model_completion_stage,
    _reconstruct_active_model_completion_stage_record,
    _reconstruct_interaction_transition_receipt,
    _reconstruct_model_completion_stage,
    _reconstruct_model_completion_stage_abandonment,
    _reconstruct_model_completion_stage_dispatch,
    _reconstruct_runtime_publication_receipt,
    _reject_reserved_runtime_publication_key,
    _reject_settled_model_completion_stage,
    _replace_checkpoint_preserving_completion_result_event_publications,
    _replay_model_completion_stage_abandonment,
    _replay_promoted_model_completion_stage,
    _require_invocation_release_recovery_claim,
    _require_invocation_release_settlement_record,
    _require_invocation_release_terminal_session_event,
    _require_live_incomplete_recovery_claim_for_run_epoch_transfer,
    _runtime_publication_json_equal,
    _runtime_publication_receipt_record,
    _runtime_publication_referenced_event_ids,
    _runtime_publication_storage_key,
    _session_metadata_after_model_transition,
    _session_metadata_after_runtime_identity_adoption,
    _session_metadata_after_tool_capability_ceiling_admission,
    _session_run_operation_from_checkpoint,
    _stored_mcp_manifest_baseline_json,
    _terminal_publication_delete_block_reason,
    _terminal_session_evidence_expected_event_type,
    _tool_lifecycle_publication_identity,
    _validate_equivalent_queued_session_message,
    _validate_execution_profile_admission,
    _validate_execution_profile_rejection_session,
    _validate_inactive_for_seconds,
    _validate_interaction_page,
    _validate_interaction_transition_invocation_authority_parameters,
    _validate_interaction_transition_receipt_authority,
    _validate_interaction_transition_receipt_recovery_authority,
    _validate_interaction_transition_recovery_claim_id,
    _validate_invocation_release_settlement_receipt_authority,
    _validate_mcp_manifest_history_keys,
    _validate_mcp_manifest_publication_state,
    _validate_message_delivery_eligible_through,
    _validate_model_completion_active_marker_for_preparation,
    _validate_model_completion_active_marker_for_promotion,
    _validate_model_completion_preparation_replay_state,
    _validate_model_completion_promotion_replay_active_marker,
    _validate_model_completion_stage_dispatch,
    _validate_model_completion_stage_for_abandonment,
    _validate_model_completion_stage_for_dispatch,
    _validate_model_completion_stage_for_settlement,
    _validate_model_completion_stage_preparation_replay,
    _validate_model_completion_stage_publication,
    _validate_model_completion_stage_release,
    _validate_model_completion_stage_repreparation,
    _validate_model_completion_stage_terminal_replay,
    _validate_profiled_fork_authority,
    _validate_profiled_fork_checkpoint_result,
    _validate_runner_observed_event_identity_snapshot,
    _validate_runtime_publication_durable_material,
    _validate_runtime_publication_event_references,
    _validate_runtime_publication_replay_receipt,
    _validate_session_fork_source,
    _validate_session_model_transition,
    _validate_session_operation_record_keys,
    _validate_status_set,
    _validate_tool_round_call_ids,
    _validate_tool_round_checkpoint_mutation,
    _validate_tool_round_publication,
    _validate_user_input_checkpoint_mutation,
    apply_fork_system_prompt_replacement,
    build_session_topology_result,
    checkpoint_root_field_projection_from_storage,
    copy_enqueue_session_message_request,
    copy_event_query,
    copy_run_request,
    copy_session_aggregate_filter,
    copy_session_identity,
    copy_session_lineage_query,
    copy_session_query,
    copy_session_runtime_identity,
    copy_session_user_metadata,
    copy_transcript_messages,
    copy_transcript_query,
    copy_transcript_search_query,
    copy_usage_rollup_query,
    decode_session_cursor,
    decode_session_lineage_cursor,
    decode_session_topology_cursor,
    decode_transcript_search_cursor,
    deferred_interaction_input_for_run_request,
    deferred_interaction_input_from_storage_payload,
    deferred_interaction_input_storage_payload,
    encode_session_cursor,
    encode_session_lineage_cursor,
    encode_transcript_search_cursor,
    enforce_pending_action_result_size,
    enqueue_session_message_input,
    filter_transcript_records,
    fork_transcript_is_accepted,
    queued_session_message_input,
    replace_session_user_metadata,
    require_deferred_initial_transcript_replacement,
    resolve_interaction_attribution,
    restore_persisted_event_authority,
    runtime_build_provenance_from_session_metadata,
    session_messages_input_contract_evidence,
    session_next_cursor,
    session_outcome,
    session_query_from_aggregate_filter,
    transcript_search_document,
    transcript_search_document_score,
    transcript_search_hit_from_message,
    transcript_search_position_after_cursor,
    transcript_search_query_document,
    transcript_search_session_token,
    transform_fork_checkpoint,
    validate_persisted_event_side_effect_error,
)
from cayu.runtime.tasks import (
    _TASK_CANCELLATION_REQUESTED_REASON,
    _TASK_INTERRUPTED_HANDOFF_RECOVERY_MAX_PAGE_SIZE,
    _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
    TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    CompletionDecisionApplicationReceipt,
    InterruptedTaskContinuationClaimPage,
    Task,
    TaskAggregateFilter,
    TaskCancellationReconciliationRequest,
    TaskCancellationReconciliationResult,
    TaskClaimLost,
    TaskCreate,
    TaskInterruptedHandoffConflict,
    TaskInterruptedHandoffReceipt,
    TaskInterruptedHandoffRequest,
    TaskInvocationSnapshot,
    TaskOperationalSnapshot,
    TaskOrder,
    TaskQuery,
    TaskRetryCancellationReconciliationRequest,
    TaskRetrySeriesDisposition,
    TaskRetrySettlementRequest,
    TaskRetrySettlementResult,
    TaskStatus,
    TaskStatusCounts,
    TaskStore,
    TaskTerminalizationConflict,
    TaskTerminalizationReceipt,
    TaskTerminalizationRequest,
    TaskTopologyInconsistent,
    TaskTopologyNode,
    TaskTopologyQuery,
    TaskTopologyStoreResult,
    _allocate_task_topology_branch_limits,
    _bounded_optional_task_topology_parent_id,
    _can_attach_claimed_task_state,
    _cancelled_task_retry_settlement,
    _claimed_task_retry_attempt_elapsed,
    _copy_optional_session_binding,
    _copy_optional_status_payload,
    _copy_optional_status_reason,
    _copy_required_session_binding,
    _copy_task_cancellation_reconciliation_result,
    _elapsed_claimed_task_retry_settlement,
    _ensure_can_hold_task,
    _ensure_can_resume_task,
    _ensure_can_transition,
    _ensure_claim_query_supported,
    _ensure_exact_owned_active_task_lease,
    _ensure_owned_active_task_lease,
    _ensure_recovered_attached_task_failure_authority,
    _ensure_recovered_attached_task_session,
    _ensure_retry_series_queue_attempt,
    _ensure_task_handoff_authority,
    _ensure_task_terminalization_lease_authority,
    _expired_dispatched_task_cancellation,
    _expired_task_retry_settlement,
    _interrupted_task_continuation_handoff_id_sha256,
    _raise_task_claim_attach_error,
    _reconciled_task_cancellation,
    _reconciled_task_retry_cancellation,
    _rejected_task_cancellation_reconciliation,
    _rejected_task_retry_cancellation_reconciliation,
    _replay_interrupted_task_handoff_receipt,
    _replay_task_cancellation_reconciliation,
    _replay_task_cancellation_reconciliation_rejection,
    _replay_task_retry_cancellation_reconciliation,
    _replay_task_retry_cancellation_reconciliation_rejection,
    _replay_task_retry_settlement,
    _replay_task_terminalization_receipt,
    _require_active_attached_task_worker,
    _require_direct_attached_task_resume,
    _require_interrupted_task_handoff_authority,
    _running_task_from_create,
    _settled_task_retry_attempt,
    _task_cancellation_reconciliation_conflict,
    _task_cancellation_reconciliation_rejection_record,
    _task_cancellation_requested,
    _task_cancellation_requested_task,
    _task_from_create,
    _task_invocation_for_attachment,
    _task_matches_claim_filter,
    _task_retry_cancellation_reconciliation_conflict,
    _task_retry_cancellation_reconciliation_rejection_record,
    _task_retry_cancellation_requested_task,
    _task_retry_events,
    _task_retry_reconciliation_identity_is_bounded,
    _task_session_id_for_start,
    _task_session_instance_for_attachment,
    _TaskCancellationReconciliationRejectionRecord,
    _TaskRetryCancellationReconciliationRejectionRecord,
    _validate_ordinary_task_terminalization_against_cancellation,
    _validate_task_topology_ancestry,
    _validated_task_cancellation,
    _validated_task_retry_cancellation,
    _validated_task_retry_terminal_accounting,
    build_task_topology_result,
    copy_task,
    copy_task_aggregate_filter,
    copy_task_create,
    copy_task_query,
    decode_task_topology_cursor,
    prepare_interrupted_task_continuation_claim_page,
    prepare_interrupted_task_handoff,
    prepare_interrupted_task_handoff_candidate_page,
    prepare_interrupted_task_handoff_receipt_lookup,
    prepare_task_cancellation_reconciliation,
    prepare_task_retry_cancellation_reconciliation,
    prepare_task_retry_settlement,
    prepare_task_terminalization,
    prepare_task_terminalization_receipt_lookup,
    task_query_from_aggregate_filter,
)
from cayu.runtime.tool_exposure import ToolCapabilityCeiling
from cayu.runtime.tool_grants import (
    TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS,
    TARGETED_TOOL_GRANT_MAX_REQUESTS,
    TARGETED_TOOL_REFERENCE_FIELD_NAME,
    TargetedToolGrantIssueOutcome,
    TargetedToolGrantIssueResult,
    TargetedToolGrantReconstructionResult,
    TargetedToolGrantRecord,
    TargetedToolGrantStateSnapshot,
    TargetedToolUseBinding,
    TargetedToolUseDisposition,
    TargetedToolUseRejectionReason,
    TargetedToolUseRequest,
    TargetedToolUseResult,
    copy_targeted_tool_grant_record,
    targeted_tool_grant_event,
    targeted_tool_grant_reconstruction_rejection_reason,
    targeted_tool_grant_with_active_reference,
    targeted_tool_unresolved_rejection_event,
    targeted_tool_use_binding,
    targeted_tool_use_rejection_event,
    targeted_tool_use_rejection_reason,
    targeted_tool_use_scope_rejection_reason,
    validate_targeted_tool_grant_batch_evidence,
    validate_targeted_tool_grant_issuance_evidence,
    validate_targeted_tool_grant_lifecycle_event,
    validate_targeted_tool_grant_reference,
    validate_targeted_tool_grant_revocation_evidence,
    validate_targeted_tool_grant_revocation_reason,
    validate_targeted_tool_unresolved_rejection_evidence,
    validate_targeted_tool_use_rejection_evidence,
)
from cayu.runtime.work_attempt_admission import (
    AdmittedCompletionProposalRequest,
    WorkAttemptAdmission,
    WorkAttemptAdmissionActivate,
    WorkAttemptAdmissionConflict,
    WorkAttemptAdmissionPrepare,
    WorkAttemptAdmissionState,
    WorkAttemptContinuationContext,
    WorkAttemptExecutionClaim,
    WorkAttemptExecutionClaimLost,
    WorkAttemptExecutionClaimRequest,
    WorkAttemptRecoveryActivate,
    copy_admitted_completion_proposal_request,
    copy_work_attempt_admission_activate,
    copy_work_attempt_admission_prepare,
    copy_work_attempt_execution_claim_request,
    copy_work_attempt_recovery_activate,
    work_attempt_admission_prepare_matches_sha256,
    work_attempt_admission_prepare_sha256,
    work_attempt_execution_claim_request_sha256,
)
from cayu.runtime.work_contracts import (
    CompletionDecision,
    CompletionDecisionApplicationRequest,
    CompletionDecisionCreate,
    CompletionProposal,
    CompletionProposalCreate,
    CompletionRejectionAction,
    CompletionVerdict,
    CompletionVerificationClaim,
    CompletionVerificationClaimLost,
    CompletionVerificationClaimRequest,
    TaskCompletionDecisionRequired,
    WorkAttempt,
    WorkAttemptCreate,
    WorkCompletionConflict,
    WorkContract,
    WorkContractConflict,
    WorkContractRef,
    completion_decision_application_request_sha256,
    completion_decision_request_sha256,
    completion_gap_fingerprint,
    completion_proposal_request_sha256,
    completion_verification_claim_authority_sha256,
    completion_verification_claim_request_sha256,
    copy_completion_decision_application_request,
    copy_completion_decision_create,
    copy_completion_proposal_create,
    copy_completion_verification_claim_request,
    copy_work_attempt_create,
    copy_work_contract,
    copy_work_contract_ref,
    validate_completion_decision_contract,
    validate_work_completion_idempotency_key,
    work_attempt_request_sha256,
)
from cayu.storage import _session_store_sql as session_store_sql
from cayu.storage import _sqlite_aggregates as sqlite_aggregates
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import _verified_work_support as verified_work_support
from cayu.storage import migrations as schema

_EVENT_QUERY_SESSION_IDS_BATCH_SIZE = 500
_SQLITE_NON_SESSION_MIN_REQUIRED_REVISION = 18
_SQLITE_SESSION_MIN_REQUIRED_REVISION = 62
_SQLITE_TASK_MIN_REQUIRED_REVISION = 76
_SQL_DIALECT = session_store_sql.SessionStoreSqlDialect(
    placeholder="?",
    contains_style="sqlite_nocase_like",
    datetime_param=sqlite_support.format_datetime,
)
_T = TypeVar("_T")


def _sqlite_recall_receipt(row: sqlite3.Row) -> RecallReceipt:
    try:
        receipt = RecallReceipt.model_validate(json.loads(row["receipt_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("SQLite recall receipt contains invalid durable material.") from exc
    document = memory_evidence_document_bytes(receipt, "stored recall receipt")
    if (
        receipt.receipt_id != row["receipt_id"]
        or receipt.session_id != row["session_id"]
        or receipt.interaction_id != row["interaction_id"]
        or receipt.model_step_id != row["model_step_id"]
        or receipt.created_at != sqlite_support.parse_datetime(row["created_at"])
        or len(document) != row["document_bytes"]
    ):
        raise RuntimeError("SQLite recall receipt index columns conflict with its document.")
    return receipt


def _sqlite_context_exposure(row: sqlite3.Row) -> ContextExposure:
    try:
        exposure = ContextExposure.model_validate(json.loads(row["exposure_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("SQLite context exposure contains invalid durable material.") from exc
    document = memory_evidence_document_bytes(exposure, "stored context exposure")
    if (
        exposure.exposure_id != row["exposure_id"]
        or exposure.session_id != row["session_id"]
        or exposure.interaction_id != row["interaction_id"]
        or exposure.model_step_id != row["model_step_id"]
        or exposure.model_attempt_id != row["model_attempt_id"]
        or exposure.provider_attempt_id != row["provider_attempt_id"]
        or str(exposure.state) != row["state"]
        or exposure.state_revision != row["state_revision"]
        or exposure.created_at != sqlite_support.parse_datetime(row["created_at"])
        or exposure.updated_at != sqlite_support.parse_datetime(row["updated_at"])
        or len(document) != row["document_bytes"]
    ):
        raise RuntimeError("SQLite context exposure index columns conflict with its document.")
    return exposure


def _sqlite_recall_item_exposures(
    rows: Sequence[sqlite3.Row],
) -> tuple[RecallItemExposure, ...]:
    items: list[RecallItemExposure] = []
    for row in rows:
        try:
            item = RecallItemExposure.model_validate(json.loads(row["item_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "SQLite recall item exposure contains invalid durable material."
            ) from exc
        document = memory_evidence_document_bytes(item, "stored recall item exposure")
        if (
            item.exposure_id != row["exposure_id"]
            or item.ordinal != row["ordinal"]
            or item.receipt_id != row["receipt_id"]
            or item.receipt_item_ordinal != row["receipt_item_ordinal"]
            or len(document) != row["document_bytes"]
        ):
            raise RuntimeError(
                "SQLite recall item exposure index columns conflict with its document."
            )
        items.append(item)
    if tuple(item.ordinal for item in items) != tuple(range(len(items))):
        raise RuntimeError("SQLite recall item exposure ordinals are incomplete.")
    return tuple(items)


def _sqlite_transcript_search_expression(query: TranscriptSearchQuery) -> str:
    session_terms = " OR ".join(
        f'"{transcript_search_session_token(session_id)}"' for session_id in query.session_ids
    )
    text_terms = " OR ".join(
        f'"{token}"' for token in transcript_search_query_document(query.text).split()
    )
    return f"session_token:({session_terms}) AND message_text:({text_terms})"


def _alias_key_fingerprint_matches(value: object, expected: str) -> bool:
    if type(value) is not str:
        return False
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(encoded, expected.encode("ascii"))


async def _run_off_thread_with_connection_ownership(
    lock: asyncio.Lock,
    connection: sqlite3.Connection,
    operation: Callable[[sqlite3.Connection], _T],
    *,
    executor: Executor | None = None,
    interrupt_on_cancellation: bool = False,
) -> _T:
    """Keep a SQLite connection owned until its off-thread operation terminates.

    Cancelling an ``asyncio.to_thread`` await does not stop the worker thread.
    For an interruptible read, request ``sqlite3_interrupt()`` after cancellation;
    in every case defer the signal while holding the connection lock so no
    subsequent operation or shutdown can reuse the connection before the worker
    has left it in a terminal transaction state.
    """

    if type(interrupt_on_cancellation) is not bool:
        raise TypeError("interrupt_on_cancellation must be a bool.")

    async with lock:

        def capture_outcome() -> tuple[bool, object]:
            try:
                return True, operation(connection)
            except BaseException as worker_failure:
                # The executor future must complete normally even when the
                # operation raises CancelledError. That makes every cancellation
                # from shield() unambiguously caller-owned and keeps ownership
                # tied to the executor's physical completion.
                return False, worker_failure

        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        worker = loop.run_in_executor(executor, context.run, capture_outcome)
        cancellation: asyncio.CancelledError | None = None

        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
                    if interrupt_on_cancellation:
                        connection.interrupt()
            except BaseException:
                if worker.done():
                    break
                raise

        succeeded, outcome = worker.result()
        if not succeeded:
            if not isinstance(outcome, BaseException):
                raise RuntimeError("SQLite worker returned an invalid failure outcome.")
            if cancellation is None:
                raise outcome
            cancellation.add_note(
                "SQLite worker failed while caller cancellation was pending: "
                f"{type(outcome).__name__}: {outcome}"
            )
            raise cancellation from outcome
        if cancellation is not None:
            raise cancellation
        return cast("_T", outcome)


def _like_contains_pattern(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _session_exists(connection: sqlite3.Connection, session_id: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM cayu_sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    return row is not None


def _transcript_cursor(connection: sqlite3.Connection, session_id: str) -> int:
    """Return the permanent next transcript position, independent of retention."""

    row = connection.execute(
        "SELECT transcript_seq FROM cayu_sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Session not found: {session_id}")
    return int(row["transcript_seq"])


def _claim_budget_reservation_identity(
    connection: sqlite3.Connection,
    *,
    reservation_id: str,
    publication_session_id: str,
    publication_id: str,
) -> None:
    inserted = connection.execute(
        """
        INSERT OR IGNORE INTO cayu_budget_reservation_identities (
            reservation_id,
            publication_session_id,
            publication_id,
            published
        )
        VALUES (?, ?, ?, 0)
        """,
        (reservation_id, publication_session_id, publication_id),
    )
    if inserted.rowcount == 1:
        return
    existing = connection.execute(
        """
        SELECT publication_session_id, publication_id
        FROM cayu_budget_reservation_identities
        WHERE reservation_id = ?
        """,
        (reservation_id,),
    ).fetchone()
    assert existing is not None
    if (
        existing["publication_session_id"],
        existing["publication_id"],
    ) != (publication_session_id, publication_id):
        raise BudgetReservationIdentityConflict("Budget ledger reused a reservation identity.")


def _publish_budget_reservation_identities(
    connection: sqlite3.Connection,
    events: list[Event],
) -> None:
    for event in events:
        if event.type != EventType.BUDGET_RESERVED:
            continue
        raw_reservation_id = event.payload.get("reservation_id")
        if type(raw_reservation_id) is not str:
            continue
        updated = connection.execute(
            """
            UPDATE cayu_budget_reservation_identities
            SET published = 1
            WHERE reservation_id = ?
              AND publication_session_id = ?
              AND publication_id = ?
              AND published = 0
            """,
            (raw_reservation_id, event.session_id, event.id),
        )
        if updated.rowcount == 1:
            continue
        try:
            connection.execute(
                """
                INSERT INTO cayu_budget_reservation_identities (
                    reservation_id,
                    publication_session_id,
                    publication_id,
                    published
                )
                VALUES (?, ?, ?, 1)
                """,
                (raw_reservation_id, event.session_id, event.id),
            )
        except sqlite3.IntegrityError:
            existing = connection.execute(
                """
                SELECT publication_session_id, publication_id, published
                FROM cayu_budget_reservation_identities
                WHERE reservation_id = ?
                """,
                (raw_reservation_id,),
            ).fetchone()
            if (
                existing is not None
                and (
                    existing["publication_session_id"],
                    existing["publication_id"],
                    existing["published"],
                )
                == (event.session_id, event.id, 1)
                and connection.execute(
                    """
                    SELECT 1
                    FROM cayu_events
                    WHERE session_id = ? AND event_id = ?
                    """,
                    (event.session_id, event.id),
                ).fetchone()
                is not None
            ):
                # The reservation belongs to this exact persisted event. Let the
                # event insert below classify the replay as a duplicate event.
                continue
            raise BudgetReservationIdentityConflict(
                "Budget ledger reused a reservation identity."
            ) from None


def _raise_session_write_conflict(
    connection: sqlite3.Connection,
    session_id: str,
    expected_run_epoch: int,
) -> None:
    row = connection.execute(
        "SELECT run_epoch FROM cayu_sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Session not found: {session_id}")
    raise SessionRunFenced(
        f"Session run epoch no longer owns {session_id}: expected {expected_run_epoch}, "
        f"current {row['run_epoch']}."
    )


def _touch_session_activity(
    connection: sqlite3.Connection,
    session_id: str,
    activity_at: datetime,
) -> None:
    expected_run_epoch = _current_session_run_epoch(session_id)
    if expected_run_epoch is None:
        cursor = connection.execute(
            "UPDATE cayu_sessions SET last_activity_at = ? WHERE id = ?",
            (sqlite_support.format_datetime(activity_at), session_id),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"Session not found: {session_id}")
        return
    cursor = connection.execute(
        "UPDATE cayu_sessions SET last_activity_at = ? WHERE id = ? AND run_epoch = ?",
        (sqlite_support.format_datetime(activity_at), session_id, expected_run_epoch),
    )
    if cursor.rowcount != 1:
        _raise_session_write_conflict(connection, session_id, expected_run_epoch)


def _load_labels(connection: sqlite3.Connection, session_id: str) -> dict[str, str]:
    rows = connection.execute(
        """
        SELECT key, value
        FROM cayu_session_labels
        WHERE session_id = ?
        ORDER BY key ASC
        """,
        (session_id,),
    ).fetchall()
    return {row["key"]: row["value"] for row in rows}


def _load_session(connection: sqlite3.Connection, session_id: str) -> Session | None:
    row = connection.execute(
        """
        SELECT id, instance_id, agent_name, provider_name, model, parent_session_id,
               causal_budget_id, runtime_name, runtime_version, environment_name,
               status, created_at, updated_at, last_activity_at, run_epoch,
               invocation_json, metadata_json
        FROM cayu_sessions
        WHERE id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return sqlite_support.session_from_row(
        row,
        labels=_load_labels(connection, session_id),
    )


_SESSION_TOPOLOGY_COLUMNS = """
    id, agent_name, provider_name, model, parent_session_id,
    causal_budget_id, runtime_name, runtime_version, environment_name,
    status, created_at, updated_at, last_activity_at,
    json_extract(metadata_json, '$."cayu:runtime_build_provenance"')
        AS runtime_build_provenance_json
"""

_SESSION_TOPOLOGY_PROJECTED_COLUMNS = """
    id, agent_name, provider_name, model, parent_session_id,
    causal_budget_id, runtime_name, runtime_version, environment_name,
    status, created_at, updated_at, last_activity_at,
    runtime_build_provenance_json
"""


def _session_topology_node_from_sqlite_row(row: sqlite3.Row) -> SessionTopologyNode:
    return SessionTopologyNode(
        id=row["id"],
        agent_name=row["agent_name"],
        provider_name=row["provider_name"],
        model=row["model"],
        parent_session_id=row["parent_session_id"],
        causal_budget_id=row["causal_budget_id"],
        runtime_name=row["runtime_name"],
        runtime_version=row["runtime_version"],
        runtime_build_provenance=runtime_build_provenance_from_session_metadata(
            {}
            if row["runtime_build_provenance_json"] is None
            else {
                RUNTIME_BUILD_PROVENANCE_METADATA_KEY: json.loads(
                    row["runtime_build_provenance_json"]
                )
            }
        ),
        environment_name=row["environment_name"],
        status=SessionStatus(row["status"]),
        created_at=sqlite_support.parse_datetime(row["created_at"]),
        updated_at=sqlite_support.parse_datetime(row["updated_at"]),
        last_activity_at=sqlite_support.parse_datetime(row["last_activity_at"]),
    )


def _load_checkpoint_state(
    connection: sqlite3.Connection,
    session_id: str,
) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT state_json
        FROM cayu_checkpoints
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return copy_durable_json_object(json.loads(row["state_json"]), "checkpoint")


def _load_interruption_cascade_marker(
    connection: sqlite3.Connection,
    session_id: str,
    checkpoint_root_guard: CheckpointRootFieldGuard | None,
) -> dict[str, Any] | None:
    checkpoint_root_key = (
        "__cayu_no_checkpoint_root_guard__"
        if checkpoint_root_guard is None
        else checkpoint_root_guard.key
    )
    checkpoint_root_path = f"$.{checkpoint_root_key}"
    row = connection.execute(
        f"""
        SELECT
            json_type(state_json, '{checkpoint_root_path}')
                AS checkpoint_root_field_type,
            CASE
                WHEN json_type(state_json, '{checkpoint_root_path}') = 'integer'
                THEN substr(
                    CAST(json_extract(
                        state_json,
                        '{checkpoint_root_path}'
                    ) AS TEXT),
                    1,
                    {CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS + 1}
                )
            END AS checkpoint_root_field_scalar,
            json_type(state_json, '$.pending_interruption_cascade') AS marker_type,
            json_type(state_json, '$.pending_interruption_cascade.attempt_id') AS attempt_id_type,
            substr(
                CAST(json_extract(
                    state_json,
                    '$.pending_interruption_cascade.attempt_id'
                ) AS TEXT),
                1,
                129
            ) AS attempt_id,
            json_type(
                state_json,
                '$.pending_interruption_cascade.interrupt_payload'
            ) AS interrupt_payload_type,
            json_type(state_json, '$.pending_interruption_cascade.generation') AS generation_type,
            substr(
                CAST(json_extract(
                    state_json,
                    '$.pending_interruption_cascade.generation'
                ) AS TEXT),
                1,
                33
            ) AS generation,
            json_type(
                state_json,
                '$.pending_interruption_cascade.failure_recorded'
            ) AS failure_recorded_type,
            json_extract(
                state_json,
                '$.pending_interruption_cascade.failure_recorded'
            ) AS failure_recorded,
            json_type(state_json, '$.pending_interruption_cascade.claim_id') AS claim_id_type,
            substr(
                CAST(json_extract(
                    state_json,
                    '$.pending_interruption_cascade.claim_id'
                ) AS TEXT),
                1,
                129
            ) AS claim_id,
            json_type(
                state_json,
                '$.pending_interruption_cascade.claim_expires_at'
            ) AS claim_expires_at_type,
            substr(
                CAST(json_extract(
                    state_json,
                    '$.pending_interruption_cascade.claim_expires_at'
                ) AS TEXT),
                1,
                65
            ) AS claim_expires_at,
            json_type(state_json, '$.pending_interruption_cascade.created_at') AS created_at_type,
            substr(
                CAST(json_extract(
                    state_json,
                    '$.pending_interruption_cascade.created_at'
                ) AS TEXT),
                1,
                65
            ) AS created_at
        FROM cayu_checkpoints
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return None

    def sqlite_json_type(value: str | None) -> str | None:
        if value == "text":
            return "string"
        if value == "real":
            return "number"
        if value in {"true", "false"}:
            return "boolean"
        return value

    checkpoint_root_field_type = row["checkpoint_root_field_type"]
    scalar_text = row["checkpoint_root_field_scalar"]
    if checkpoint_root_guard is not None:
        checkpoint_root_guard.validate(
            session_id,
            checkpoint_root_field_projection_from_storage(
                json_type=checkpoint_root_field_type,
                scalar_text=scalar_text,
            ),
        )

    field_names = (
        "attempt_id",
        "interrupt_payload",
        "generation",
        "failure_recorded",
        "claim_id",
        "claim_expires_at",
        "created_at",
    )
    field_types = {field: sqlite_json_type(row[f"{field}_type"]) for field in field_names}
    field_values = {
        "attempt_id": row["attempt_id"],
        "generation": row["generation"],
        "failure_recorded": (
            bool(row["failure_recorded"])
            if field_types["failure_recorded"] == "boolean"
            else row["failure_recorded"]
        ),
        "claim_id": row["claim_id"],
        "claim_expires_at": row["claim_expires_at"],
        "created_at": row["created_at"],
    }
    return _project_interruption_cascade_marker_fields(
        sqlite_json_type(row["marker_type"]),
        field_types,
        field_values,
    )


def _first_existing_event_id(
    connection: sqlite3.Connection,
    session_id: str,
    event_ids: list[str],
) -> str | None:
    for event_id in event_ids:
        row = connection.execute(
            "SELECT 1 FROM cayu_events WHERE session_id = ? AND event_id = ?",
            (session_id, event_id),
        ).fetchone()
        if row is not None:
            return event_id
    return None


def _decode_runtime_publication_record(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication receipt is malformed or conflicts with its key."
        ) from exc


def _decode_model_completion_stage_record(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SessionModelCompletionStageConflict(
            "The durable model-completion stage record is malformed."
        ) from exc
    if type(decoded) is not dict:
        raise SessionModelCompletionStageConflict(
            "The durable model-completion stage record is malformed."
        )
    return decoded


def _event_query_session_id_batches(
    session_ids: tuple[str, ...],
) -> list[tuple[str, ...]]:
    return [
        session_ids[index : index + _EVENT_QUERY_SESSION_IDS_BATCH_SIZE]
        for index in range(0, len(session_ids), _EVENT_QUERY_SESSION_IDS_BATCH_SIZE)
    ]


# Columns needed to reconstruct an Event, in a stable order. The formerly-stored
# event_json blob duplicated exactly these (plus payload_json), so the store now
# rebuilds Events from the individual columns instead of parsing a redundant copy.
_EVENT_COLUMN_NAMES: tuple[str, ...] = (
    "session_id",
    "event_id",
    "interaction_id",
    "event_type",
    "timestamp",
    "agent_name",
    "environment_name",
    "workflow_name",
    "tool_name",
    "payload_json",
    "input_contract_runtime_owned",
    "file_attachment_attestations_runtime_owned",
)

# Keep this predicate text aligned with the revision-17 partial index. SQLite
# can prove a parameterized lifecycle subset is covered by that index only when
# the query also carries the index's literal predicate.
_PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL = """
    event_type IN (
        'tool.call.approval_requested',
        'session.awaiting_user_input',
        'session.interrupted',
        'tool.call.started',
        'tool.call.completed',
        'tool.call.failed',
        'tool.call.blocked',
        'tool.call.approval_denied'
    )
    AND pending_action_lookup_key IS NOT NULL
"""


def _event_from_row(row: sqlite3.Row) -> Event:
    """Reconstruct an :class:`Event` from its individual cayu_events columns."""
    input_contract_runtime_owned = row["input_contract_runtime_owned"]
    if type(input_contract_runtime_owned) is not int or input_contract_runtime_owned not in {
        0,
        1,
    }:
        raise ValueError("Stored input-contract authority proof is malformed.")
    file_attachment_attestations_runtime_owned = row["file_attachment_attestations_runtime_owned"]
    if type(
        file_attachment_attestations_runtime_owned
    ) is not int or file_attachment_attestations_runtime_owned not in {0, 1}:
        raise ValueError("Stored file-attachment attestation proof is malformed.")
    return restore_persisted_event_authority(
        Event(
            type=row["event_type"],
            session_id=row["session_id"],
            interaction_id=row["interaction_id"],
            id=row["event_id"],
            timestamp=row["timestamp"],
            agent_name=row["agent_name"],
            environment_name=row["environment_name"],
            workflow_name=row["workflow_name"],
            tool_name=row["tool_name"],
            payload=json.loads(row["payload_json"]),
        ),
        input_contract_runtime_owned=input_contract_runtime_owned == 1,
        file_attachment_attestations_runtime_owned=(
            file_attachment_attestations_runtime_owned == 1
        ),
    )


def _event_record_from_row(row: sqlite3.Row | None) -> EventRecord | None:
    if row is None:
        return None
    return EventRecord(
        sequence=row["sequence"],
        event=_event_from_row(row),
    )


def _targeted_tool_grant_from_json(value: object) -> TargetedToolGrantRecord:
    if type(value) is not str:
        raise ValueError("Stored targeted tool grant is malformed.")
    try:
        return copy_targeted_tool_grant_record(TargetedToolGrantRecord.model_validate_json(value))
    except (TypeError, ValueError):
        raise ValueError("Stored targeted tool grant is malformed.") from None


def _targeted_tool_use_from_json(value: object) -> TargetedToolUseBinding:
    if type(value) is not str:
        raise ValueError("Stored targeted tool use is malformed.")
    try:
        return TargetedToolUseBinding.model_validate_json(value)
    except (TypeError, ValueError):
        raise ValueError("Stored targeted tool use is malformed.") from None


def _targeted_tool_grant_from_row(row: sqlite3.Row) -> TargetedToolGrantRecord:
    record = _targeted_tool_grant_from_json(row["record_json"])
    indexed = (
        ("grant_id", record.grant_id),
        ("session_id", record.session_id),
        ("interaction_id", record.interaction_id),
        ("request_id", record.request_id),
        ("tool_ref", record.tool_ref),
        ("generation_id", record.generation_id),
        ("tool_id", record.tool_id),
        ("tool_name", record.tool_name),
        ("catalogue_revision", record.catalogue_revision),
        ("descriptor_version", record.descriptor_version),
        ("issued_at", sqlite_support.format_datetime(record.issued_at)),
        ("expires_at", sqlite_support.format_datetime(record.expires_at)),
        ("max_calls", record.max_calls),
        ("used_calls", record.used_calls),
        ("revoked_at", sqlite_support.format_optional_datetime(record.revoked_at)),
    )
    if any(row[field_name] != expected for field_name, expected in indexed):
        raise ValueError("Stored targeted tool grant conflicts with indexed authority.")
    return record


def _targeted_tool_use_from_row(row: sqlite3.Row) -> TargetedToolUseBinding:
    binding = _targeted_tool_use_from_json(row["record_json"])
    indexed = (
        ("use_id", binding.use_id),
        ("grant_id", binding.grant_id),
        ("session_id", binding.session_id),
        ("interaction_id", binding.interaction_id),
        ("model_step_id", binding.model_step_id),
        ("outer_tool_call_id", binding.outer_tool_call_id),
        ("arguments_sha256", binding.arguments_sha256),
        ("invocation_id", binding.invocation_id),
        ("bound_at", sqlite_support.format_datetime(binding.bound_at)),
    )
    if any(row[field_name] != expected for field_name, expected in indexed):
        raise ValueError("Stored targeted tool use conflicts with indexed authority.")
    return binding


def _validate_targeted_tool_use_counts(
    connection: sqlite3.Connection,
    records: Iterable[TargetedToolGrantRecord],
) -> None:
    expected = {record.grant_id: record.used_calls for record in records}
    if not expected:
        return
    placeholders = ", ".join("?" for _ in expected)
    actual = dict.fromkeys(expected, 0)
    for row in connection.execute(
        "SELECT grant_id, COUNT(*) AS use_count "
        "FROM cayu_targeted_tool_grant_uses "
        f"WHERE grant_id IN ({placeholders}) GROUP BY grant_id",
        tuple(expected),
    ):
        actual[str(row["grant_id"])] = int(row["use_count"])
    if actual != expected:
        raise ValueError("Targeted grant call counter conflicts with durable uses.")


def _persisted_event_side_effect_delivery_from_row(
    row: sqlite3.Row,
) -> PersistedEventSideEffectDelivery:
    return PersistedEventSideEffectDelivery(
        session_id=row["session_id"],
        event_id=row["event_id"],
        event_sequence=row["event_sequence"],
        status=PersistedEventSideEffectStatus(row["status"]),
        attempts=row["attempts"],
        claim_id=row["claim_id"],
        lease_expires_at=(
            None
            if row["lease_expires_at"] is None
            else sqlite_support.parse_datetime(row["lease_expires_at"])
        ),
        next_attempt_at=(
            None
            if row["next_attempt_at"] is None
            else sqlite_support.parse_datetime(row["next_attempt_at"])
        ),
        last_error=row["last_error"],
        updated_at=sqlite_support.parse_datetime(row["updated_at"]),
    )


def _enqueue_persisted_event_side_effects(
    connection: sqlite3.Connection,
    session_id: str,
    events: Sequence[Event],
) -> None:
    if not events:
        return
    event_ids: list[str] = []
    runtime_owned_input_contract_event_ids: list[str] = []
    runtime_owned_file_attestation_event_ids: list[str] = []
    for event in events:
        event_ids.append(event.id)
        if _event_input_contract_is_runtime_owned(event):
            runtime_owned_input_contract_event_ids.append(event.id)
        if _event_file_attachment_attestations_are_runtime_owned(event):
            runtime_owned_file_attestation_event_ids.append(event.id)
    # Rows predating revision 31 may contain caller-authored payload text but
    # cannot carry the proof bit, so that text remains untrusted after migration.
    if runtime_owned_input_contract_event_ids:
        connection.executemany(
            """
            UPDATE cayu_events
            SET input_contract_runtime_owned = 1
            WHERE session_id = ?
              AND event_id = ?
              AND event_type IN (
                  'session.started',
                  'session.resumed',
                  'session.message.queued',
                  'session.message.delivered'
              )
              AND json_type(payload_json, '$.input_contract') = 'text'
            """,
            [(session_id, event_id) for event_id in runtime_owned_input_contract_event_ids],
        )
    if runtime_owned_file_attestation_event_ids:
        connection.executemany(
            """
            UPDATE cayu_events
            SET file_attachment_attestations_runtime_owned = 1
            WHERE session_id = ?
              AND event_id = ?
              AND event_type = 'model.started'
              AND json_type(payload_json, '$.file_attachment_attestations') = 'text'
            """,
            [(session_id, event_id) for event_id in runtime_owned_file_attestation_event_ids],
        )
    connection.executemany(
        """
        INSERT INTO cayu_persisted_event_side_effects (
            session_id, event_id, event_sequence, status, attempts, updated_at
        )
        SELECT session_id, event_id, sequence, 'pending', 0, timestamp
        FROM cayu_events
        WHERE session_id = ?
          AND event_id = ?
          AND event_type <> 'runtime.sink.failed'
        """,
        [(session_id, event_id) for event_id in event_ids],
    )


def _append_events_in_transaction(
    connection: sqlite3.Connection,
    session_id: str,
    events: Sequence[Event],
    *,
    activity_at: datetime,
) -> None:
    """Append events and their delivery outbox rows in the caller's transaction."""

    if not events:
        return
    from cayu.runtime.pending_actions import pending_action_event_storage_values

    _touch_session_activity(connection, session_id, activity_at)
    _publish_budget_reservation_identities(connection, list(events))
    rows = []
    for event in events:
        lookup_key, projection, projection_bytes = pending_action_event_storage_values(event)
        rows.append(
            (
                session_id,
                event.id,
                event.interaction_id,
                str(event.type),
                sqlite_support.format_datetime(event.timestamp),
                event.agent_name,
                event.environment_name,
                event.workflow_name,
                event.tool_name,
                sqlite_support.json_dumps(event.payload),
                lookup_key,
                projection,
                projection_bytes,
            )
        )
    connection.executemany(
        """
        INSERT INTO cayu_events (
            session_id,
            event_id,
            interaction_id,
            event_type,
            timestamp,
            agent_name,
            environment_name,
            workflow_name,
            tool_name,
            payload_json,
            pending_action_lookup_key,
            pending_action_projection_json,
            pending_action_projection_bytes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    terminal_events = tuple(
        event
        for event in events
        if event.type
        in {
            EventType.SESSION_COMPLETED,
            EventType.SESSION_FAILED,
            EventType.SESSION_INTERRUPTED,
        }
    )
    if terminal_events:
        session = _load_session(connection, session_id)
        if session is None:  # pragma: no cover - activity update already authenticated it
            raise KeyError(f"Session not found: {session_id}")
        checkpoint = _load_checkpoint_state(connection, session_id)
        terminal_receipts = tuple(
            receipt
            for event in terminal_events
            if (
                receipt := _invocation_terminal_event_receipt_record(
                    session=session,
                    checkpoint=checkpoint,
                    event=event,
                )
            )
            is not None
        )
        connection.executemany(
            "INSERT INTO cayu_session_operations "
            "(session_id, idempotency_key, record_json, updated_at) "
            "VALUES (?, ?, ?, ?)",
            [
                (
                    session_id,
                    receipt_key,
                    sqlite_support.json_dumps(receipt_record),
                    sqlite_support.format_datetime(activity_at),
                )
                for receipt_key, receipt_record in terminal_receipts
            ],
        )
    _enqueue_persisted_event_side_effects(connection, session_id, events)


def _append_event_once_in_transaction(
    connection: sqlite3.Connection,
    event: Event,
    *,
    activity_at: datetime,
) -> Event:
    """Return existing exact evidence or append it in the caller's transaction."""

    row = connection.execute(
        "SELECT * FROM cayu_events WHERE session_id = ? AND event_id = ?",
        (event.session_id, event.id),
    ).fetchone()
    if row is not None:
        return _event_from_row(row)
    _append_events_in_transaction(
        connection,
        event.session_id,
        [event],
        activity_at=activity_at,
    )
    return event


def _queued_session_message_from_row(row: sqlite3.Row) -> SessionQueuedMessage:
    requested_by = row["requested_by_json"]
    message_json = row["message_json"]
    return SessionQueuedMessage(
        queue_id=row["queue_id"],
        session_id=row["session_id"],
        idempotency_key=row["idempotency_key"],
        content=row["content"],
        message=(
            None if message_json is None else Message.model_validate(json.loads(message_json))
        ),
        delivery_mode=row["delivery_mode"],
        status=row["status"],
        ordering_key=row["ordering_key"],
        accepted_run_epoch=row["accepted_run_epoch"],
        accepted_transcript_cursor=row["accepted_transcript_cursor"],
        accepted_event_id=row["accepted_event_id"],
        accepted_at=sqlite_support.parse_datetime(row["accepted_at"]),
        requested_by=(
            None
            if requested_by is None
            else ResolutionActor.model_validate(json.loads(requested_by))
        ),
        delivered_run_epoch=row["delivered_run_epoch"],
        delivered_transcript_cursor=row["delivered_transcript_cursor"],
        delivered_event_id=row["delivered_event_id"],
        delivered_at=(
            None
            if row["delivered_at"] is None
            else sqlite_support.parse_datetime(row["delivered_at"])
        ),
    )


class SQLiteSessionStore(SessionStore):
    """SQLite-backed session store for durable local runtime state."""

    supports_usage_aggregates: ClassVar[bool] = True
    supports_mcp_manifest_history: ClassVar[bool] = True
    supports_public_authority_aliases: ClassVar[bool] = True
    supports_targeted_tool_grants: ClassVar[bool] = True
    supports_session_topology: ClassVar[bool] = True
    supports_session_lineage: ClassVar[bool] = True
    child_session_notification_version: ClassVar[int | None] = 1
    supports_terminal_session_evidence: ClassVar[bool] = True
    supports_runner_owned_interrupted_evidence: ClassVar[bool] = True
    supports_execution_profile_admission: ClassVar[bool] = True
    supports_active_invocation_execution_profiles: ClassVar[bool] = True
    invocation_lifecycle_command_version: ClassVar[int | None] = 1
    supports_pending_session_initial_checkpoint: ClassVar[bool] = True
    supports_profiled_forks: ClassVar[bool] = True
    supports_atomic_session_operation_initialization: ClassVar[bool] = True
    supports_atomic_model_completion_stage_release: ClassVar[bool] = True
    supports_completion_result_event_publication_reservations: ClassVar[bool] = True
    supports_transcript_search: ClassVar[bool] = True
    supports_recall_evidence: ClassVar[bool] = True
    supports_owned_off_thread_session_commit_guards: ClassVar[bool] = True

    def __init__(
        self,
        path: str | Path,
        *,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
        read_only: bool = False,
        public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
        ownership_clock: Callable[[], datetime] | None = None,
    ) -> None:
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str:
            db_path = Path(require_nonblank(path, "path"))
        else:
            raise TypeError("SQLiteSessionStore path must be a string or Path.")
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")
        if type(read_only) is not bool:
            raise TypeError("read_only must be a bool.")
        configured_read_only = read_only
        diagnostic_source_missing = sqlite_support.diagnostic_sqlite_source_missing(db_path)
        if (
            sqlite_support.current_diagnostic_store_inspection() is not None
            and str(db_path) != ":memory:"
        ):
            if diagnostic_source_missing:
                schema_mode = schema.SchemaMode.CREATE
                read_only = False
            else:
                schema_mode = schema.SchemaMode.VALIDATE
                read_only = True
        self.service_durability = (
            RuntimeStoreDurability.READ_ONLY
            if configured_read_only
            else (
                RuntimeStoreDurability.DEVELOPMENT
                if str(db_path) == ":memory:"
                else RuntimeStoreDurability.DURABLE
            )
        )
        if public_authority_alias_codec is not None and not isinstance(
            public_authority_alias_codec,
            PublicAuthorityAliasCodec,
        ):
            raise TypeError("public_authority_alias_codec must be a PublicAuthorityAliasCodec.")
        if read_only and schema_mode is not schema.SchemaMode.VALIDATE:
            raise ValueError("read_only SQLite stores require schema_mode=validate.")

        self.path = db_path
        self._diagnostic_source_missing = diagnostic_source_missing
        self._schema_mode = schema_mode
        self._read_only = read_only
        self._public_authority_alias_codec = public_authority_alias_codec
        self._ownership_clock = utc_clock(ownership_clock)
        self._lock = asyncio.Lock()
        self._detached_read_tasks: set[asyncio.Task[object]] = set()
        effective_db_path = Path(":memory:") if diagnostic_source_missing else db_path
        self._connection = (
            self._connect_read_only(effective_db_path)
            if read_only
            else self._connect(effective_db_path)
        )
        try:
            self._register_public_authority_alias_sql_function(self._connection)
            self._initialize_schema()
            self._initialize_public_authority_alias_registry()
            if diagnostic_source_missing:
                self._connection.execute("PRAGMA query_only = ON")
                self._read_only = True
        except BaseException:
            self._connection.close()
            raise
        # Hot-path queries run on a dedicated read-only connection in worker
        # threads so the event loop never blocks on SQLite I/O and reads never
        # queue behind the writer connection's transactions. In-memory
        # databases are private to their connection, so they fall back to the
        # writer connection (and its lock).
        if self._read_only or str(effective_db_path) == ":memory:":
            self._read_connection = self._connection
            self._read_lock = self._lock
        else:
            self._read_connection = self._connect_read_only(effective_db_path)
            self._read_lock = asyncio.Lock()

    @property
    def public_authority_alias_codec(self) -> PublicAuthorityAliasCodec | None:
        """Return the immutable codec bound to this store's durable alias registry."""

        return self._public_authority_alias_codec

    def _register_public_authority_alias_sql_function(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        codec = self._public_authority_alias_codec

        def public_authority_alias(
            private_value: object,
            field_name: object,
            scope_session_id: object,
        ) -> str | None:
            if codec is None:
                return None
            if type(private_value) is not str or type(field_name) is not str:
                raise ValueError("Public authority alias source must be text.")
            if scope_session_id is not None and type(scope_session_id) is not str:
                raise ValueError("Public authority alias scope must be text or null.")
            return codec.encode(
                private_value,
                field_name=field_name,
                session_id=scope_session_id,
            )

        connection.create_function(
            "cayu_public_authority_alias",
            3,
            public_authority_alias,
            deterministic=True,
        )

        def public_authority_aliases(
            private_value: object,
            field_name: object,
            scope_session_id: object,
        ) -> str:
            if codec is None:
                return "[]"
            if type(private_value) is not str or type(field_name) is not str:
                raise ValueError("Public authority alias source must be text.")
            if scope_session_id is not None and type(scope_session_id) is not str:
                raise ValueError("Public authority alias scope must be text or null.")
            return json.dumps(
                codec.aliases(
                    private_value,
                    field_name=field_name,
                    session_id=scope_session_id,
                ),
                separators=(",", ":"),
            )

        connection.create_function(
            "cayu_public_authority_aliases",
            3,
            public_authority_aliases,
            deterministic=True,
        )
        connection.create_function(
            "cayu_public_authority_active_key_id",
            0,
            lambda: None if codec is None else codec.keyring.active_key_id,
            deterministic=True,
        )
        connection.create_function(
            "cayu_public_authority_keyring_fingerprint",
            0,
            lambda: None if codec is None else codec.keyring_fingerprint(),
            deterministic=True,
        )

    def _initialize_public_authority_alias_registry(self) -> None:
        """Validate key continuity and backfill each newly configured signing key."""

        codec = self._public_authority_alias_codec
        if codec is None:
            initialized = self._connection.execute(
                "SELECT EXISTS(SELECT 1 FROM cayu_public_authority_alias_config)"
            ).fetchone()[0]
            if initialized:
                raise ValueError(
                    "A public authority alias codec is required for this initialized store."
                )
            return

        configured = tuple(
            (key_id, codec.key_fingerprint(key_id)) for key_id in codec.keyring.key_ids
        )
        if self._read_only:
            rows: dict[str, tuple[object, object]] = {}
            for row in self._connection.execute(
                "SELECT key_id, fingerprint, backfill_completed "
                "FROM cayu_public_authority_alias_keys"
            ):
                if type(row["key_id"]) is str:
                    rows[row["key_id"]] = (row["fingerprint"], row["backfill_completed"])
            for key_id, fingerprint in configured:
                existing = rows.get(key_id)
                if existing is None or existing[1] != 1:
                    raise ValueError(
                        "Read-only store has not completed the configured alias-key backfill."
                    )
                if not _alias_key_fingerprint_matches(existing[0], fingerprint):
                    raise ValueError(
                        "Public authority alias key material conflicts with durable state."
                    )
            config = self._connection.execute(
                "SELECT active_key_id, keyring_fingerprint "
                "FROM cayu_public_authority_alias_config "
                "WHERE singleton = 1"
            ).fetchone()
            if (
                config is None
                or config["active_key_id"] != codec.keyring.active_key_id
                or config["keyring_fingerprint"] != codec.keyring_fingerprint()
            ):
                raise ValueError("Read-only store public authority alias active key is stale.")
            return

        try:
            with self._connection:
                for key_id, fingerprint in configured:
                    self._connection.execute(
                        """
                        INSERT INTO cayu_public_authority_alias_keys (
                            key_id, fingerprint, backfill_completed
                        ) VALUES (?, ?, 0)
                        ON CONFLICT(key_id) DO NOTHING
                        """,
                        (key_id, fingerprint),
                    )
                    row = self._connection.execute(
                        """
                        SELECT fingerprint, backfill_completed
                        FROM cayu_public_authority_alias_keys
                        WHERE key_id = ?
                        """,
                        (key_id,),
                    ).fetchone()
                    if row is None:  # pragma: no cover - guarded by the insert above
                        raise RuntimeError("Public authority alias key state was not persisted.")
                    if not _alias_key_fingerprint_matches(row["fingerprint"], fingerprint):
                        raise ValueError(
                            "Public authority alias key material conflicts with durable state."
                        )

                pending = self._connection.execute(
                    """
                    SELECT EXISTS(
                        SELECT 1
                        FROM cayu_public_authority_alias_keys
                        WHERE key_id IN ({}) AND backfill_completed = 0
                    )
                    """.format(", ".join("?" for _ in configured)),
                    tuple(key_id for key_id, _fingerprint in configured),
                ).fetchone()[0]
                if pending:
                    self._backfill_public_authority_aliases()
                    self._connection.executemany(
                        """
                        UPDATE cayu_public_authority_alias_keys
                        SET backfill_completed = 1
                        WHERE key_id = ?
                        """,
                        ((key_id,) for key_id, _fingerprint in configured),
                    )
                config = self._connection.execute(
                    "SELECT active_key_id, keyring_fingerprint, generation, "
                    "retired_key_ids_json "
                    "FROM cayu_public_authority_alias_config WHERE singleton = 1"
                ).fetchone()
                desired_active = codec.keyring.active_key_id
                desired_keyring_fingerprint = codec.keyring_fingerprint()
                if config is None:
                    self._connection.execute(
                        "INSERT INTO cayu_public_authority_alias_config "
                        "(singleton, active_key_id, keyring_fingerprint, generation, "
                        "retired_key_ids_json) VALUES (1, ?, ?, 1, '[]')",
                        (desired_active, desired_keyring_fingerprint),
                    )
                elif (
                    config["active_key_id"] != desired_active
                    or config["keyring_fingerprint"] != desired_keyring_fingerprint
                ):
                    retired = json.loads(config["retired_key_ids_json"])
                    if type(retired) is not list or not all(
                        type(value) is str for value in retired
                    ):
                        raise ValueError("Public authority alias rotation state is malformed.")
                    if config["active_key_id"] != desired_active and desired_active in retired:
                        raise ValueError(
                            "A retired public authority alias key cannot become active again."
                        )
                    if config["active_key_id"] != desired_active:
                        retired.append(str(config["active_key_id"]))
                    self._connection.execute(
                        "UPDATE cayu_public_authority_alias_config "
                        "SET active_key_id = ?, keyring_fingerprint = ?, generation = ?, "
                        "retired_key_ids_json = ? "
                        "WHERE singleton = 1",
                        (
                            desired_active,
                            desired_keyring_fingerprint,
                            int(config["generation"]) + 1,
                            json.dumps(list(dict.fromkeys(retired)), separators=(",", ":")),
                        ),
                    )
        except sqlite3.IntegrityError:
            raise ValueError(
                "Public authority alias registry conflicts with durable authority."
            ) from None

    def _backfill_public_authority_aliases(self) -> None:
        self._connection.execute(
            """
            INSERT OR IGNORE INTO cayu_public_authority_aliases (
                field_name, scope_session_id, public_alias, private_value
            )
            SELECT
                'session_id',
                '',
                alias.value,
                session.id
            FROM cayu_sessions AS session,
                 json_each(
                     cayu_public_authority_aliases(session.id, 'session_id', NULL)
                 ) AS alias
            """
        )
        self._connection.execute(
            """
            INSERT OR IGNORE INTO cayu_public_authority_aliases (
                field_name, scope_session_id, public_alias, private_value
            )
            SELECT
                'tool_ref',
                grant_record.session_id,
                alias.value,
                grant_record.grant_id
            FROM cayu_targeted_tool_grants AS grant_record,
                 json_each(cayu_public_authority_aliases(
                     grant_record.grant_id, 'tool_ref', grant_record.session_id
                 )) AS alias
            """
        )
        self._connection.execute(
            """
            INSERT OR IGNORE INTO cayu_public_authority_aliases (
                field_name, scope_session_id, public_alias, private_value
            )
            SELECT DISTINCT
                'interaction_id',
                event.session_id,
                alias.value,
                event.interaction_id
            FROM cayu_events AS event,
                 json_each(cayu_public_authority_aliases(
                     event.interaction_id, 'interaction_id', event.session_id
                 )) AS alias
            WHERE event.interaction_id IS NOT NULL
            """
        )
        self._connection.execute(
            """
            INSERT OR IGNORE INTO cayu_public_authority_aliases (
                field_name, scope_session_id, public_alias, private_value
            )
            SELECT DISTINCT
                'interaction_id',
                transcript.session_id,
                alias.value,
                transcript.interaction_id
            FROM cayu_transcript_messages AS transcript,
                 json_each(cayu_public_authority_aliases(
                     transcript.interaction_id, 'interaction_id', transcript.session_id
                 )) AS alias
            WHERE transcript.interaction_id IS NOT NULL
            """
        )
        self._connection.execute(
            """
            INSERT OR IGNORE INTO cayu_public_authority_aliases (
                field_name, scope_session_id, public_alias, private_value
            )
            SELECT DISTINCT
                'interaction_id',
                event.session_id,
                alias.value,
                interaction.value
            FROM cayu_events AS event,
                 json_each(event.payload_json, '$.interaction_ids') AS interaction,
                 json_each(cayu_public_authority_aliases(
                     interaction.value, 'interaction_id', event.session_id
                 )) AS alias
            WHERE event.event_type = 'turn.completed'
              AND json_valid(event.payload_json)
              AND json_type(event.payload_json, '$.interaction_ids') = 'array'
              AND interaction.type = 'text'
              AND trim(interaction.value) <> ''
            """
        )

    async def _run_read(self, query: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run a cancellable read while retaining physical connection ownership."""

        def guarded(connection: sqlite3.Connection) -> _T:
            self._require_current_public_authority_configuration(connection)
            return query(connection)

        owner = asyncio.create_task(
            _run_off_thread_with_connection_ownership(
                self._read_lock,
                self._read_connection,
                guarded,
                interrupt_on_cancellation=True,
            ),
            name="cayu-sqlite-read-owner",
        )
        try:
            return await asyncio.shield(owner)
        except asyncio.CancelledError:
            owner.cancel()
            self._retain_detached_read_task(owner)
            raise

    def _retain_detached_read_task(self, task: asyncio.Task[object]) -> None:
        """Observe a cancelled caller's physical read until the worker settles."""

        self._detached_read_tasks.add(task)

        def settled(completed: asyncio.Task[object]) -> None:
            self._detached_read_tasks.discard(completed)
            try:
                failure = completed.exception()
            except asyncio.CancelledError:
                return
            if failure is not None:
                completed.get_loop().call_exception_handler(
                    {
                        "message": "Detached SQLite read failed after caller cancellation",
                        "exception": failure,
                        "task": completed,
                    }
                )

        task.add_done_callback(settled)

    async def _run_write(self, statement: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run a write statement off the event loop on the writer connection."""

        def guarded(connection: sqlite3.Connection) -> _T:
            # This is the fail-fast check. The session/event/transcript BEFORE
            # INSERT triggers repeat it after SQLite has acquired the writer
            # transaction, closing the cross-process key-rotation race between
            # this check and an identity-producing statement.
            self._require_current_public_authority_configuration(connection)
            return statement(connection)

        return await _run_off_thread_with_connection_ownership(
            self._lock,
            self._connection,
            guarded,
        )

    def _require_current_public_authority_configuration(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        codec = self.public_authority_alias_codec
        row = connection.execute(
            "SELECT active_key_id, keyring_fingerprint "
            "FROM cayu_public_authority_alias_config WHERE singleton = 1"
        ).fetchone()
        if codec is None:
            if row is not None:
                raise RuntimeError(
                    "SQLite public authority aliases require the deployment keyring."
                )
            return
        if (
            row is None
            or row["active_key_id"] != codec.keyring.active_key_id
            or row["keyring_fingerprint"] != codec.keyring_fingerprint()
        ):
            raise RuntimeError(
                "SQLite public authority alias key configuration is stale; reopen the store."
            )

    async def register_public_authority_alias(
        self,
        public_alias: str,
        *,
        field_name: str,
        private_value: str,
        scope_session_id: str | None = None,
    ) -> None:
        """Atomically register one codec-authenticated public authority alias."""

        field_name, scope_key, public_alias = _public_authority_alias_store_key(
            public_alias,
            field_name=field_name,
            private_value=private_value,
            scope_session_id=scope_session_id,
        )
        codec = self.public_authority_alias_codec
        if codec is None or not codec.matches(
            public_alias,
            private_value,
            field_name=field_name,
            session_id=scope_session_id,
        ):
            raise ValueError("Public authority alias lacks valid store-configured provenance.")

        def statement(connection: sqlite3.Connection) -> None:
            with connection:
                try:
                    connection.execute(
                        """
                        INSERT INTO cayu_public_authority_aliases (
                            field_name,
                            scope_session_id,
                            public_alias,
                            private_value
                        ) VALUES (?, ?, ?, ?)
                        ON CONFLICT(field_name, scope_session_id, public_alias) DO NOTHING
                        """,
                        (field_name, scope_key, public_alias, private_value),
                    )
                except sqlite3.IntegrityError:
                    raise ValueError(
                        "Public authority alias conflicts with existing private authority."
                    ) from None
                row = connection.execute(
                    """
                    SELECT private_value
                    FROM cayu_public_authority_aliases
                    WHERE field_name = ?
                      AND scope_session_id = ?
                      AND public_alias = ?
                    """,
                    (field_name, scope_key, public_alias),
                ).fetchone()
                if row is None:  # pragma: no cover - guarded by the insert above
                    raise RuntimeError("Public authority alias registration was not persisted.")
                stored = str(row["private_value"])
                if not hmac.compare_digest(
                    stored.encode("utf-8"),
                    private_value.encode("utf-8"),
                ):
                    raise ValueError(
                        "Public authority alias conflicts with existing private authority."
                    )

        await self._run_write(statement)

    async def resolve_public_authority_alias(
        self,
        public_alias: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> str | None:
        """Resolve one exact alias through its indexed authority scope."""

        field_name, scope_key, public_alias = _public_authority_alias_store_key(
            public_alias,
            field_name=field_name,
            private_value=None,
            scope_session_id=scope_session_id,
        )

        def query(connection: sqlite3.Connection) -> str | None:
            row = connection.execute(
                """
                SELECT private_value
                FROM cayu_public_authority_aliases
                WHERE field_name = ?
                  AND scope_session_id = ?
                  AND public_alias = ?
                """,
                (field_name, scope_key, public_alias),
            ).fetchone()
            if row is None:
                return None
            return _authenticated_public_authority_alias_private_value(
                self.public_authority_alias_codec,
                public_alias,
                row["private_value"],
                field_name=field_name,
                scope_session_id=scope_session_id,
            )

        return await self._run_read(query)

    async def issue_targeted_tool_grants(
        self,
        session_id: str,
        *,
        expected_run_epoch: int,
        records: tuple[TargetedToolGrantRecord, ...],
        events: tuple[Event, ...],
    ) -> TargetedToolGrantIssueResult:
        session_id = require_clean_nonblank(session_id, "session_id")
        if type(expected_run_epoch) is not int or expected_run_epoch < 0:
            raise ValueError("expected_run_epoch must be a non-negative integer.")
        if type(records) is not tuple or type(events) is not tuple:
            raise TypeError("records and events must be tuples.")
        if len(records) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
            raise ValueError("Targeted grant issuance exceeds the bounded request count.")
        copied_records = tuple(copy_targeted_tool_grant_record(record) for record in records)
        copied_events = tuple(
            Event.model_validate(event.model_dump(mode="python")) for event in events
        )
        if len(copied_records) != len(copied_events):
            raise ValueError("Each targeted grant record requires one issuance event.")
        if len({record.request_id for record in copied_records}) != len(copied_records):
            raise ValueError("Targeted grant records must have unique request identities.")
        if len({record.tool_id for record in copied_records}) != len(copied_records):
            raise ValueError("Targeted grant records must have unique tool identities.")
        interaction_ids = {record.interaction_id for record in copied_records}
        if len(interaction_ids) > 1:
            raise ValueError("Targeted grant records must share one interaction scope.")
        codec = self.public_authority_alias_codec
        if codec is None:
            raise RuntimeError("Targeted grants require a public authority alias codec.")
        for record, event in zip(copied_records, copied_events, strict=True):
            if record.session_id != session_id:
                raise ValueError("Targeted grant scope is inconsistent.")
            validate_targeted_tool_grant_reference(record, codec)
            validate_targeted_tool_grant_issuance_evidence(record, event)

        def statement(
            connection: sqlite3.Connection,
        ) -> tuple[
            tuple[TargetedToolGrantRecord, ...],
            tuple[TargetedToolGrantIssueOutcome, ...],
            tuple[Event, ...],
        ]:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                session_row = connection.execute(
                    "SELECT agent_name, environment_name, status, run_epoch, invocation_json "
                    "FROM cayu_sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                if session_row is None:
                    raise KeyError(f"Session not found: {session_id}")
                if int(session_row["run_epoch"]) != expected_run_epoch:
                    raise SessionRunFenced(
                        f"Session source run epoch is stale: expected {expected_run_epoch}, "
                        f"current {session_row['run_epoch']}."
                    )
                if str(session_row["status"]) != str(SessionStatus.RUNNING):
                    raise SessionStatusConflict("Targeted grants require a running session.")
                if interaction_ids:
                    lifecycle_placeholders = ", ".join(
                        "?" for _ in INTERACTION_LIFECYCLE_EVENT_TYPES
                    )
                    latest_interaction = connection.execute(
                        "SELECT interaction_id, event_type FROM cayu_events "
                        "WHERE session_id = ? "
                        f"AND event_type IN ({lifecycle_placeholders}) "
                        "ORDER BY sequence DESC LIMIT 1",
                        (
                            session_id,
                            *(str(value) for value in INTERACTION_LIFECYCLE_EVENT_TYPES),
                        ),
                    ).fetchone()
                    if (
                        latest_interaction is None
                        or latest_interaction["interaction_id"] != next(iter(interaction_ids))
                        or EventType(str(latest_interaction["event_type"]))
                        in INTERACTION_TERMINAL_EVENT_TYPES
                    ):
                        raise ValueError("Targeted grants require the current open interaction.")
                    interaction_started_row = connection.execute(
                        "SELECT * FROM cayu_events WHERE session_id = ? "
                        "AND interaction_id = ? AND event_type = ? "
                        "ORDER BY sequence ASC LIMIT 1",
                        (
                            session_id,
                            next(iter(interaction_ids)),
                            str(EventType.INTERACTION_STARTED),
                        ),
                    ).fetchone()
                    if interaction_started_row is None:
                        raise RuntimeError("Targeted grant issuance lost interaction admission.")
                    validate_targeted_tool_grant_batch_evidence(
                        copied_records,
                        _event_from_row(interaction_started_row),
                    )
                invocation = SessionInvocation.model_validate_json(session_row["invocation_json"])
                resolved: list[TargetedToolGrantRecord] = []
                outcomes: list[TargetedToolGrantIssueOutcome] = []
                resolved_events: list[Event] = []
                new_events: list[Event] = []
                for record, event in zip(copied_records, copied_events, strict=True):
                    if (
                        record.session_id != session_id
                        or record.agent_name != session_row["agent_name"]
                        or record.environment_name != session_row["environment_name"]
                        or record.principal != invocation.origin.subject
                        or record.tenant != invocation.origin.tenant
                    ):
                        raise ValueError("Targeted grant scope is inconsistent.")
                    existing_row = connection.execute(
                        "SELECT * FROM cayu_targeted_tool_grants "
                        "WHERE session_id = ? AND interaction_id = ? "
                        "AND (request_id = ? OR tool_id = ?) LIMIT 2",
                        (
                            session_id,
                            record.interaction_id,
                            record.request_id,
                            record.tool_id,
                        ),
                    ).fetchone()
                    if existing_row is not None:
                        existing = _targeted_tool_grant_from_row(existing_row)
                        _validate_targeted_tool_use_counts(connection, (existing,))
                        if existing.request_id != record.request_id:
                            raise ValueError(
                                "Targeted grant tool identity conflicts with durable authority."
                            )
                        if existing.grant_id != record.grant_id:
                            raise ValueError(
                                "Targeted grant request identity conflicts with durable authority."
                            )
                        resolved.append(targeted_tool_grant_with_active_reference(existing, codec))
                        outcomes.append(TargetedToolGrantIssueOutcome.REUSED)
                        issued_row = connection.execute(
                            "SELECT * FROM cayu_events WHERE session_id = ? AND event_id = ?",
                            (session_id, event.id),
                        ).fetchone()
                        if issued_row is None:
                            raise RuntimeError("Targeted grant lost its durable issuance evidence.")
                        validate_targeted_tool_grant_issuance_evidence(
                            existing,
                            _event_from_row(issued_row),
                        )
                        reused_event = targeted_tool_grant_event(
                            existing,
                            event_type=EventType.TARGETED_TOOL_GRANT_REUSED,
                            timestamp=event.timestamp,
                            outcome=TargetedToolGrantIssueOutcome.REUSED.value,
                            event_id_suffix="reused",
                        )
                        resolved_events.append(
                            _append_event_once_in_transaction(
                                connection,
                                reused_event,
                                activity_at=reused_event.timestamp,
                            )
                        )
                        continue
                    collision = connection.execute(
                        "SELECT 1 FROM cayu_targeted_tool_grants WHERE grant_id = ?",
                        (record.grant_id,),
                    ).fetchone()
                    if collision is not None:
                        raise ValueError("Targeted grant identity collides with authority.")
                    for public_alias in codec.aliases(
                        record.grant_id,
                        field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                        session_id=session_id,
                    ):
                        field_name, scope_key, public_alias = _public_authority_alias_store_key(
                            public_alias,
                            field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                            private_value=record.grant_id,
                            scope_session_id=session_id,
                        )
                        connection.execute(
                            "INSERT INTO cayu_public_authority_aliases "
                            "(field_name, scope_session_id, public_alias, private_value) "
                            "VALUES (?, ?, ?, ?) "
                            "ON CONFLICT(field_name, scope_session_id, public_alias) DO NOTHING",
                            (field_name, scope_key, public_alias, record.grant_id),
                        )
                        stored_alias = connection.execute(
                            "SELECT private_value FROM cayu_public_authority_aliases "
                            "WHERE field_name = ? AND scope_session_id = ? AND public_alias = ?",
                            (field_name, scope_key, public_alias),
                        ).fetchone()
                        if stored_alias is None or stored_alias["private_value"] != record.grant_id:
                            raise ValueError("Targeted tool reference collides with authority.")
                    connection.execute(
                        """
                        INSERT INTO cayu_targeted_tool_grants (
                            grant_id, session_id, interaction_id, request_id, tool_ref,
                            generation_id, tool_id, tool_name, catalogue_revision,
                            descriptor_version, issued_at, expires_at, max_calls,
                            used_calls, revoked_at, record_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.grant_id,
                            record.session_id,
                            record.interaction_id,
                            record.request_id,
                            record.tool_ref,
                            record.generation_id,
                            record.tool_id,
                            record.tool_name,
                            record.catalogue_revision,
                            record.descriptor_version,
                            sqlite_support.format_datetime(record.issued_at),
                            sqlite_support.format_datetime(record.expires_at),
                            record.max_calls,
                            record.used_calls,
                            None,
                            sqlite_support.json_dumps(record.model_dump(mode="json")),
                        ),
                    )
                    resolved.append(record)
                    outcomes.append(TargetedToolGrantIssueOutcome.ISSUED)
                    resolved_events.append(event)
                    new_events.append(event)
                if interaction_ids:
                    interaction_count = connection.execute(
                        "SELECT COUNT(*) FROM cayu_targeted_tool_grants "
                        "WHERE session_id = ? AND interaction_id = ?",
                        (session_id, next(iter(interaction_ids))),
                    ).fetchone()[0]
                    if interaction_count > TARGETED_TOOL_GRANT_MAX_REQUESTS:
                        raise ValueError("Targeted grant interaction exceeds its bounded count.")
                _append_events_in_transaction(
                    connection,
                    session_id,
                    new_events,
                    activity_at=self._ownership_clock(),
                )
                return tuple(resolved), tuple(outcomes), tuple(resolved_events)

        resolved_records, outcomes, resolved_events = await self._run_write(statement)
        return TargetedToolGrantIssueResult(
            records=resolved_records,
            outcomes=outcomes,
            events=resolved_events,
        )

    async def list_targeted_tool_grants(
        self,
        session_id: str,
        *,
        interaction_id: str | None = None,
        limit: int = TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS,
    ) -> tuple[TargetedToolGrantRecord, ...]:
        session_id = require_clean_nonblank(session_id, "session_id")
        if interaction_id is not None:
            interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
        if type(limit) is not int or not 1 <= limit <= TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS:
            raise ValueError(
                f"limit must be between 1 and {TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS}."
            )

        def query(connection: sqlite3.Connection) -> tuple[TargetedToolGrantRecord, ...]:
            with connection:
                connection.execute("BEGIN")
                if not _session_exists(connection, session_id):
                    raise KeyError(f"Session not found: {session_id}")
                if interaction_id is None:
                    rows = connection.execute(
                        "SELECT * FROM cayu_targeted_tool_grants "
                        "WHERE session_id = ? ORDER BY issued_at, grant_id LIMIT ?",
                        (session_id, limit + 1),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        "SELECT * FROM cayu_targeted_tool_grants "
                        "WHERE session_id = ? AND interaction_id = ? "
                        "ORDER BY issued_at, grant_id LIMIT ?",
                        (session_id, interaction_id, limit + 1),
                    ).fetchall()
                if len(rows) > limit:
                    raise ValueError("Targeted grant inspection exceeds its bounded result limit.")
                if not rows:
                    return ()
                records = tuple(_targeted_tool_grant_from_row(row) for row in rows)
                _validate_targeted_tool_use_counts(connection, records)
                codec = self.public_authority_alias_codec
                if codec is None:
                    raise RuntimeError("Targeted grants require a public authority alias codec.")
                return tuple(
                    targeted_tool_grant_with_active_reference(
                        record,
                        codec,
                    )
                    for record in records
                )

        return await self._run_read(query)

    async def load_targeted_tool_grant_state(
        self,
        session_id: str,
    ) -> TargetedToolGrantStateSnapshot:
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection: sqlite3.Connection) -> TargetedToolGrantStateSnapshot:
            with connection:
                connection.execute("BEGIN")
                if not _session_exists(connection, session_id):
                    raise KeyError(f"Session not found: {session_id}")
                grant_rows = connection.execute(
                    "SELECT * FROM cayu_targeted_tool_grants "
                    "WHERE session_id = ? ORDER BY issued_at, grant_id",
                    (session_id,),
                ).fetchall()
                use_rows = connection.execute(
                    "SELECT * FROM cayu_targeted_tool_grant_uses "
                    "WHERE session_id = ? ORDER BY bound_at, use_id",
                    (session_id,),
                ).fetchall()
                if not grant_rows:
                    if use_rows:
                        raise ValueError("Targeted grant uses exist without grant records.")
                    return TargetedToolGrantStateSnapshot()
                codec = self.public_authority_alias_codec
                if codec is None:
                    raise RuntimeError("Targeted grants require a public authority alias codec.")
                records: list[TargetedToolGrantRecord] = []
                for row in grant_rows:
                    record = _targeted_tool_grant_from_row(row)
                    records.append(targeted_tool_grant_with_active_reference(record, codec))
                return TargetedToolGrantStateSnapshot(
                    records=tuple(records),
                    uses=tuple(_targeted_tool_use_from_row(row) for row in use_rows),
                )

        return await self._run_read(query)

    async def bind_targeted_tool_grant_use(
        self,
        request: TargetedToolUseRequest,
        *,
        observed_at: datetime,
    ) -> TargetedToolUseResult:
        if type(request) is not TargetedToolUseRequest:
            raise TypeError("request must be a TargetedToolUseRequest.")
        request = TargetedToolUseRequest.model_validate(request.model_dump(mode="python"))
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware.")
        observed_at = observed_at.astimezone(UTC)
        codec = self.public_authority_alias_codec
        if codec is None:
            raise RuntimeError("Targeted grants require a public authority alias codec.")

        def statement(
            connection: sqlite3.Connection,
        ) -> tuple[TargetedToolUseResult, Event | None]:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                session_row = connection.execute(
                    "SELECT agent_name, environment_name, status, run_epoch "
                    "FROM cayu_sessions WHERE id = ?",
                    (request.session_id,),
                ).fetchone()
                if session_row is None:
                    raise KeyError(f"Session not found: {request.session_id}")
                if int(session_row["run_epoch"]) != request.expected_run_epoch:
                    raise SessionRunFenced(
                        "Session source run epoch is stale: expected "
                        f"{request.expected_run_epoch}, current {session_row['run_epoch']}."
                    )
                if str(session_row["status"]) != str(SessionStatus.RUNNING):
                    raise SessionStatusConflict("Targeted tool use requires a running session.")

                def unresolved(
                    reason: TargetedToolUseRejectionReason,
                ) -> tuple[TargetedToolUseResult, Event]:
                    session_agent_name = str(session_row["agent_name"])
                    session_environment_name = (
                        None
                        if session_row["environment_name"] is None
                        else str(session_row["environment_name"])
                    )
                    event = targeted_tool_unresolved_rejection_event(
                        request,
                        reason=reason,
                        timestamp=observed_at,
                        agent_name=session_agent_name,
                        environment_name=session_environment_name,
                    )
                    persisted = _append_event_once_in_transaction(
                        connection,
                        event,
                        activity_at=observed_at,
                    )
                    validate_targeted_tool_unresolved_rejection_evidence(
                        request,
                        reason=reason,
                        event=persisted,
                        agent_name=session_agent_name,
                        environment_name=session_environment_name,
                    )
                    return (
                        TargetedToolUseResult(
                            disposition=TargetedToolUseDisposition.REJECTED,
                            reason=reason,
                            event=persisted,
                        ),
                        persisted,
                    )

                try:
                    parsed = parse_public_authority_alias(request.tool_ref)
                    well_formed = (
                        parsed is not None
                        and parsed.field_name == TARGETED_TOOL_REFERENCE_FIELD_NAME
                    )
                except (TypeError, ValueError):
                    well_formed = False
                if not well_formed:
                    return unresolved(TargetedToolUseRejectionReason.MALFORMED)
                aliases = connection.execute(
                    "SELECT scope_session_id, private_value "
                    "FROM cayu_public_authority_aliases "
                    "WHERE field_name = ? AND public_alias = ? LIMIT 2",
                    (TARGETED_TOOL_REFERENCE_FIELD_NAME, request.tool_ref),
                ).fetchall()
                if not aliases:
                    return unresolved(TargetedToolUseRejectionReason.UNKNOWN)
                if len(aliases) != 1:
                    raise RuntimeError("Targeted tool reference registry is ambiguous.")
                scope_session_id = str(aliases[0]["scope_session_id"])
                grant_id = str(aliases[0]["private_value"])
                if not codec.matches(
                    request.tool_ref,
                    grant_id,
                    field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                    session_id=scope_session_id,
                ):
                    return unresolved(TargetedToolUseRejectionReason.UNKNOWN)
                if scope_session_id != request.session_id:
                    return unresolved(TargetedToolUseRejectionReason.CROSS_SESSION)
                grant_row = connection.execute(
                    "SELECT * FROM cayu_targeted_tool_grants WHERE grant_id = ?",
                    (grant_id,),
                ).fetchone()
                if grant_row is None:
                    raise RuntimeError("Targeted tool reference lost its grant record.")
                record = _targeted_tool_grant_from_row(grant_row)
                _validate_targeted_tool_use_counts(connection, (record,))

                def rejected(
                    reason: TargetedToolUseRejectionReason,
                ) -> tuple[TargetedToolUseResult, Event]:
                    if reason is TargetedToolUseRejectionReason.EXPIRED:
                        expiry_event = targeted_tool_grant_event(
                            record,
                            event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                            timestamp=observed_at,
                            outcome="expired",
                            event_id_suffix="expired",
                            rejection_reason=reason,
                        )
                        persisted_expiry = _append_event_once_in_transaction(
                            connection,
                            expiry_event,
                            activity_at=observed_at,
                        )
                        validate_targeted_tool_grant_lifecycle_event(
                            record,
                            persisted_expiry,
                            event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                            outcome="expired",
                            event_id_suffix="expired",
                            rejection_reason=reason,
                            require_current_call_count=False,
                        )
                    rejection_event = targeted_tool_use_rejection_event(
                        record,
                        request,
                        reason=reason,
                        timestamp=observed_at,
                    )
                    persisted = _append_event_once_in_transaction(
                        connection,
                        rejection_event,
                        activity_at=observed_at,
                    )
                    validate_targeted_tool_use_rejection_evidence(
                        record,
                        request,
                        reason=reason,
                        event=persisted,
                    )
                    return (
                        TargetedToolUseResult(
                            disposition=TargetedToolUseDisposition.REJECTED,
                            reason=reason,
                            grant=record,
                            event=persisted,
                        ),
                        persisted,
                    )

                terminal_placeholders = ", ".join("?" for _ in INTERACTION_TERMINAL_EVENT_TYPES)
                interaction_ended = connection.execute(
                    "SELECT 1 FROM cayu_events WHERE session_id = ? AND interaction_id = ? "
                    f"AND event_type IN ({terminal_placeholders}) LIMIT 1",
                    (
                        request.session_id,
                        record.interaction_id,
                        *(str(event_type) for event_type in INTERACTION_TERMINAL_EVENT_TYPES),
                    ),
                ).fetchone()
                if interaction_ended is not None:
                    return rejected(TargetedToolUseRejectionReason.EXPIRED)
                use_rows = connection.execute(
                    "SELECT * FROM cayu_targeted_tool_grant_uses "
                    "WHERE session_id = ? AND interaction_id = ? "
                    "AND (invocation_id = ? OR outer_tool_call_id = ?) LIMIT 2",
                    (
                        request.session_id,
                        request.interaction_id,
                        request.invocation_id,
                        request.outer_tool_call_id,
                    ),
                ).fetchall()
                if use_rows:
                    scope_rejection = targeted_tool_use_scope_rejection_reason(record, request)
                    if scope_rejection is not None:
                        return rejected(scope_rejection)
                    if len(use_rows) != 1:
                        return rejected(TargetedToolUseRejectionReason.ALTERED_REPLAY)
                    binding = _targeted_tool_use_from_row(use_rows[0])
                    candidate = targeted_tool_use_binding(
                        grant_id,
                        request,
                        bound_at=binding.bound_at,
                    )
                    if binding != candidate:
                        return rejected(TargetedToolUseRejectionReason.ALTERED_REPLAY)
                    expected_event = targeted_tool_grant_event(
                        record,
                        event_type=EventType.TARGETED_TOOL_REFERENCE_CONSUMED,
                        timestamp=binding.bound_at,
                        outcome=TargetedToolUseDisposition.BOUND.value,
                        event_id_suffix=f"use:{binding.use_id}",
                        binding=binding,
                    )
                    event_row = connection.execute(
                        "SELECT * FROM cayu_events WHERE session_id = ? AND event_id = ?",
                        (request.session_id, expected_event.id),
                    ).fetchone()
                    if event_row is None:
                        raise RuntimeError("Targeted tool use lost its durable event evidence.")
                    validate_targeted_tool_grant_lifecycle_event(
                        record,
                        _event_from_row(event_row),
                        event_type=EventType.TARGETED_TOOL_REFERENCE_CONSUMED,
                        outcome=TargetedToolUseDisposition.BOUND.value,
                        event_id_suffix=f"use:{binding.use_id}",
                        binding=binding,
                        require_current_call_count=False,
                    )
                    rejoined_event = targeted_tool_grant_event(
                        record,
                        event_type=EventType.TARGETED_TOOL_REFERENCE_REJOINED,
                        timestamp=observed_at,
                        outcome=TargetedToolUseDisposition.REJOINED.value,
                        event_id_suffix=f"rejoined:{binding.use_id}",
                        binding=binding,
                    )
                    persisted_rejoin = _append_event_once_in_transaction(
                        connection,
                        rejoined_event,
                        activity_at=observed_at,
                    )
                    return (
                        TargetedToolUseResult(
                            disposition=TargetedToolUseDisposition.REJOINED,
                            grant=record,
                            binding=binding,
                            event=persisted_rejoin,
                        ),
                        persisted_rejoin,
                    )
                rejection = targeted_tool_use_rejection_reason(
                    record,
                    request,
                    observed_at=observed_at,
                )
                if rejection is not None:
                    return rejected(rejection)
                binding = targeted_tool_use_binding(
                    grant_id,
                    request,
                    bound_at=observed_at,
                )
                updated = TargetedToolGrantRecord.model_validate(
                    record.model_copy(update={"used_calls": record.used_calls + 1}).model_dump(
                        mode="python"
                    )
                )
                connection.execute(
                    """
                    INSERT INTO cayu_targeted_tool_grant_uses (
                        use_id, grant_id, session_id, interaction_id, model_step_id,
                        outer_tool_call_id, arguments_sha256, invocation_id,
                        bound_at, record_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        binding.use_id,
                        binding.grant_id,
                        binding.session_id,
                        binding.interaction_id,
                        binding.model_step_id,
                        binding.outer_tool_call_id,
                        binding.arguments_sha256,
                        binding.invocation_id,
                        sqlite_support.format_datetime(binding.bound_at),
                        sqlite_support.json_dumps(binding.model_dump(mode="json")),
                    ),
                )
                connection.execute(
                    "UPDATE cayu_targeted_tool_grants SET used_calls = ?, record_json = ? "
                    "WHERE grant_id = ? AND used_calls = ?",
                    (
                        updated.used_calls,
                        sqlite_support.json_dumps(updated.model_dump(mode="json")),
                        grant_id,
                        record.used_calls,
                    ),
                )
                event = targeted_tool_grant_event(
                    updated,
                    event_type=EventType.TARGETED_TOOL_REFERENCE_CONSUMED,
                    timestamp=observed_at,
                    outcome=TargetedToolUseDisposition.BOUND.value,
                    event_id_suffix=f"use:{binding.use_id}",
                    binding=binding,
                )
                _append_events_in_transaction(
                    connection,
                    request.session_id,
                    [event],
                    activity_at=observed_at,
                )
                return (
                    TargetedToolUseResult(
                        disposition=TargetedToolUseDisposition.BOUND,
                        grant=updated,
                        binding=binding,
                        event=event,
                    ),
                    event,
                )

        result, new_event = await self._run_write(statement)
        if result.disposition is TargetedToolUseDisposition.REJECTED:
            return result
        binding = result.binding
        if binding is None or result.grant is None:  # pragma: no cover - model invariant
            raise AssertionError("Accepted targeted tool use lost its binding.")
        if new_event is None:  # pragma: no cover - transaction invariant
            raise RuntimeError("Accepted targeted tool use lost its durable event evidence.")
        if result.event is None:  # pragma: no cover - model invariant
            raise RuntimeError("Accepted targeted tool use lost its result event evidence.")
        return result

    async def revoke_targeted_tool_grant(
        self,
        tool_ref: str,
        *,
        session_id: str,
        expected_run_epoch: int,
        reason: str,
        revoked_at: datetime,
    ) -> TargetedToolGrantRecord | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if type(expected_run_epoch) is not int or expected_run_epoch < 0:
            raise ValueError("expected_run_epoch must be a non-negative integer.")
        reason = validate_targeted_tool_grant_revocation_reason(reason)
        if revoked_at.tzinfo is None or revoked_at.utcoffset() is None:
            raise ValueError("revoked_at must be timezone-aware.")
        revoked_at = revoked_at.astimezone(UTC)
        codec = self.public_authority_alias_codec
        if codec is None:
            raise RuntimeError("Targeted grants require a public authority alias codec.")

        def statement(
            connection: sqlite3.Connection,
        ) -> tuple[TargetedToolGrantRecord | None, Event | None]:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                session_row = connection.execute(
                    "SELECT run_epoch FROM cayu_sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                if session_row is None:
                    raise KeyError(f"Session not found: {session_id}")
                if int(session_row["run_epoch"]) != expected_run_epoch:
                    raise SessionRunFenced(
                        f"Session source run epoch is stale: expected {expected_run_epoch}, "
                        f"current {session_row['run_epoch']}."
                    )
                try:
                    parsed = parse_public_authority_alias(tool_ref)
                except (TypeError, ValueError):
                    return None, None
                if parsed is None or parsed.field_name != TARGETED_TOOL_REFERENCE_FIELD_NAME:
                    return None, None
                alias_row = connection.execute(
                    "SELECT scope_session_id, private_value "
                    "FROM cayu_public_authority_aliases "
                    "WHERE field_name = ? AND public_alias = ? LIMIT 2",
                    (TARGETED_TOOL_REFERENCE_FIELD_NAME, tool_ref),
                ).fetchall()
                if not alias_row:
                    return None, None
                if len(alias_row) != 1:
                    raise RuntimeError("Targeted tool reference registry is ambiguous.")
                scope = str(alias_row[0]["scope_session_id"])
                grant_id = str(alias_row[0]["private_value"])
                if scope != session_id or not codec.matches(
                    tool_ref,
                    grant_id,
                    field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                    session_id=scope,
                ):
                    return None, None
                row = connection.execute(
                    "SELECT * FROM cayu_targeted_tool_grants WHERE grant_id = ?",
                    (grant_id,),
                ).fetchone()
                if row is None:
                    raise RuntimeError("Targeted tool reference lost its grant record.")
                record = _targeted_tool_grant_from_row(row)
                _validate_targeted_tool_use_counts(connection, (record,))
                if record.revoked_at is not None:
                    if record.revocation_reason != reason:
                        raise ValueError("Targeted grant was revoked with a different reason.")
                    expected_event = targeted_tool_grant_event(
                        record,
                        event_type=EventType.TARGETED_TOOL_GRANT_REVOKED,
                        timestamp=record.revoked_at,
                        outcome="revoked",
                        event_id_suffix="revoked",
                    )
                    event_row = connection.execute(
                        "SELECT * FROM cayu_events WHERE session_id = ? AND event_id = ?",
                        (session_id, expected_event.id),
                    ).fetchone()
                    if event_row is None:
                        raise RuntimeError(
                            "Targeted grant revocation lost its durable event evidence."
                        )
                    persisted_event = _event_from_row(event_row)
                    validate_targeted_tool_grant_revocation_evidence(
                        record,
                        persisted_event,
                    )
                    return record, persisted_event
                latest_use_row = connection.execute(
                    "SELECT MAX(bound_at) AS latest_bound_at "
                    "FROM cayu_targeted_tool_grant_uses WHERE grant_id = ?",
                    (grant_id,),
                ).fetchone()
                latest_bound_at = latest_use_row["latest_bound_at"]
                if latest_bound_at is not None and (
                    sqlite_support.parse_datetime(str(latest_bound_at)) > revoked_at
                ):
                    raise ValueError("revoked_at cannot precede a bound targeted tool use.")
                updated = TargetedToolGrantRecord.model_validate(
                    record.model_copy(
                        update={"revoked_at": revoked_at, "revocation_reason": reason}
                    ).model_dump(mode="python")
                )
                connection.execute(
                    "UPDATE cayu_targeted_tool_grants SET revoked_at = ?, record_json = ? "
                    "WHERE grant_id = ? AND revoked_at IS NULL",
                    (
                        sqlite_support.format_datetime(revoked_at),
                        sqlite_support.json_dumps(updated.model_dump(mode="json")),
                        grant_id,
                    ),
                )
                event = targeted_tool_grant_event(
                    updated,
                    event_type=EventType.TARGETED_TOOL_GRANT_REVOKED,
                    timestamp=revoked_at,
                    outcome="revoked",
                    event_id_suffix="revoked",
                )
                _append_events_in_transaction(
                    connection,
                    session_id,
                    [event],
                    activity_at=revoked_at,
                )
                return updated, event

        record, event = await self._run_write(statement)
        if record is None:
            return None
        if event is None:  # pragma: no cover - transaction invariant
            raise RuntimeError("Targeted grant revocation lost its durable event evidence.")
        return record

    async def reconstruct_targeted_tool_grants(
        self,
        session_id: str,
        *,
        expected_run_epoch: int,
        interaction_id: str,
        generation_id: str,
        agent_name: str,
        task_id: str | None,
        environment_name: str | None,
        principal: str | None,
        tenant: str | None,
        catalogue_revision: str,
        descriptors_by_id: Mapping[str, tuple[str, str, str]],
        capability_ceiling_names: frozenset[str],
        observed_at: datetime,
    ) -> TargetedToolGrantReconstructionResult:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
        if type(expected_run_epoch) is not int or expected_run_epoch < 0:
            raise ValueError("expected_run_epoch must be a non-negative integer.")
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware.")
        observed_at = observed_at.astimezone(UTC)

        def statement(connection: sqlite3.Connection) -> TargetedToolGrantReconstructionResult:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                session_row = connection.execute(
                    "SELECT status, run_epoch FROM cayu_sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                if session_row is None:
                    raise KeyError(f"Session not found: {session_id}")
                if int(session_row["run_epoch"]) != expected_run_epoch:
                    raise SessionRunFenced(
                        f"Session source run epoch is stale: expected {expected_run_epoch}, "
                        f"current {session_row['run_epoch']}."
                    )
                if str(session_row["status"]) != str(SessionStatus.RUNNING):
                    raise SessionStatusConflict("Grant reconstruction requires a running session.")
                rows = connection.execute(
                    "SELECT * FROM cayu_targeted_tool_grants "
                    "WHERE session_id = ? AND interaction_id = ? "
                    "ORDER BY issued_at, grant_id LIMIT ?",
                    (session_id, interaction_id, TARGETED_TOOL_GRANT_MAX_REQUESTS + 1),
                ).fetchall()
                if len(rows) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
                    raise ValueError("Targeted grant interaction exceeds its bounded count.")
                records = tuple(_targeted_tool_grant_from_row(row) for row in rows)
                _validate_targeted_tool_use_counts(connection, records)
                interaction_started_row = connection.execute(
                    "SELECT * FROM cayu_events WHERE session_id = ? "
                    "AND interaction_id = ? AND event_type = ? "
                    "ORDER BY sequence ASC LIMIT 1",
                    (session_id, interaction_id, str(EventType.INTERACTION_STARTED)),
                ).fetchone()
                if interaction_started_row is None:
                    raise RuntimeError("Targeted grant reconstruction lost interaction admission.")
                validate_targeted_tool_grant_batch_evidence(
                    records,
                    _event_from_row(interaction_started_row),
                )
                placeholders = ", ".join("?" for _ in INTERACTION_TERMINAL_EVENT_TYPES)
                interaction_ended = (
                    connection.execute(
                        "SELECT 1 FROM cayu_events WHERE session_id = ? AND interaction_id = ? "
                        f"AND event_type IN ({placeholders}) LIMIT 1",
                        (
                            session_id,
                            interaction_id,
                            *(str(value) for value in INTERACTION_TERMINAL_EVENT_TYPES),
                        ),
                    ).fetchone()
                    is not None
                )
                valid: list[TargetedToolGrantRecord] = []
                rejected: list[tuple[str, TargetedToolUseRejectionReason]] = []
                events: list[Event] = []
                for record in records:
                    reason = targeted_tool_grant_reconstruction_rejection_reason(
                        record,
                        generation_id=generation_id,
                        agent_name=agent_name,
                        task_id=task_id,
                        environment_name=environment_name,
                        principal=principal,
                        tenant=tenant,
                        catalogue_revision=catalogue_revision,
                        descriptors_by_id=descriptors_by_id,
                        capability_ceiling_names=capability_ceiling_names,
                        observed_at=observed_at,
                        interaction_ended=interaction_ended,
                    )
                    if reason is None:
                        valid.append(record)
                        event = targeted_tool_grant_event(
                            record,
                            event_type=EventType.TARGETED_TOOL_GRANT_RECONSTRUCTED,
                            timestamp=observed_at,
                            outcome="reconstructed",
                            event_id_suffix="reconstructed",
                        )
                    else:
                        rejected.append((record.grant_id, reason))
                        if reason is TargetedToolUseRejectionReason.EXPIRED:
                            persisted_expiry = _append_event_once_in_transaction(
                                connection,
                                targeted_tool_grant_event(
                                    record,
                                    event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                                    timestamp=observed_at,
                                    outcome="expired",
                                    event_id_suffix="expired",
                                    rejection_reason=reason,
                                ),
                                activity_at=observed_at,
                            )
                            validate_targeted_tool_grant_lifecycle_event(
                                record,
                                persisted_expiry,
                                event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                                outcome="expired",
                                event_id_suffix="expired",
                                rejection_reason=reason,
                                require_current_call_count=False,
                            )
                        event = targeted_tool_grant_event(
                            record,
                            event_type=EventType.TARGETED_TOOL_GRANT_RECONSTRUCTED,
                            timestamp=observed_at,
                            outcome="rejected",
                            event_id_suffix=f"reconstruction-rejected:{reason.value}",
                            rejection_reason=reason,
                        )
                    persisted = _append_event_once_in_transaction(
                        connection,
                        event,
                        activity_at=observed_at,
                    )
                    validate_targeted_tool_grant_lifecycle_event(
                        record,
                        persisted,
                        event_type=EventType.TARGETED_TOOL_GRANT_RECONSTRUCTED,
                        outcome="reconstructed" if reason is None else "rejected",
                        event_id_suffix=(
                            "reconstructed"
                            if reason is None
                            else f"reconstruction-rejected:{reason.value}"
                        ),
                        rejection_reason=reason,
                        require_current_call_count=False,
                    )
                    events.append(persisted)
                return TargetedToolGrantReconstructionResult(
                    valid=tuple(valid),
                    rejected=tuple(rejected),
                    events=tuple(events),
                )

        return await self._run_write(statement)

    async def public_authority_private_value_exists(
        self,
        private_value: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> bool:
        codec = self.public_authority_alias_codec
        if codec is None:
            raise RuntimeError("Public authority alias codec is unavailable.")
        probe = codec.encode(
            private_value,
            field_name=field_name,
            session_id=scope_session_id,
        )
        field_name, scope_key, _probe = _public_authority_alias_store_key(
            probe,
            field_name=field_name,
            private_value=private_value,
            scope_session_id=scope_session_id,
        )

        def query(connection: sqlite3.Connection) -> bool:
            return (
                connection.execute(
                    """
                    SELECT EXISTS(
                        SELECT 1
                        FROM cayu_public_authority_aliases
                        WHERE field_name = ?
                          AND scope_session_id = ?
                          AND private_value = ?
                    )
                    """,
                    (field_name, scope_key, private_value),
                ).fetchone()[0]
                == 1
            )

        return await self._run_read(query)

    async def create(
        self,
        request: RunRequest,
        *,
        identity: SessionIdentity,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        checkpoint_transform: CheckpointTransform | None = None,
        result_checkpoint_transform: CheckpointTransform | None = None,
        operation_initializer: SessionOperationInitializer | None = None,
    ) -> Session:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        request = copy_run_request(request)
        identity = copy_session_identity(identity)
        if result_checkpoint_transform is not None and not callable(result_checkpoint_transform):
            raise TypeError("result_checkpoint_transform must be callable.")
        async with self._lock:
            self._require_current_public_authority_configuration(self._connection)
            if request.session_id is not None and request.parent_session_id == request.session_id:
                raise ValueError("Session cannot be its own parent.")
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                with self._connection:
                    created_at = self._ownership_clock()
                    parent_session = (
                        None
                        if request.parent_session_id is None
                        else _load_session(self._connection, request.parent_session_id)
                    )
                    if request.parent_session_id is not None and parent_session is None:
                        raise ValueError(f"Parent session not found: {request.parent_session_id}")
                    session = sqlite_support.session_from_request(
                        request,
                        identity=identity,
                        parent_session=parent_session,
                        created_at=created_at,
                    )
                    admission = _copy_optional_interaction_admission(
                        session.id,
                        interaction_started_event,
                        interaction_source_messages,
                        defer_transcript=True,
                    )
                    if admission is not None:
                        session = session.model_copy(
                            update={"status": SessionStatus.RUNNING, "run_epoch": 1}
                        )
                    initial_operation_records = _prepare_initial_session_operation_records(
                        session,
                        operation_initializer,
                    )
                    if session.parent_session_id == session.id:
                        raise ValueError("Session cannot be its own parent.")
                    self._connection.execute(
                        """
                        INSERT INTO cayu_sessions (
                            id,
                            instance_id,
                            agent_name,
                            provider_name,
                            model,
                            parent_session_id,
                            causal_budget_id,
                            runtime_name,
                            runtime_version,
                            environment_name,
                            status,
                            created_at,
                            updated_at,
                            last_activity_at,
                            run_epoch,
                            invocation_json,
                            metadata_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session.id,
                            session.instance_id,
                            session.agent_name,
                            session.provider_name,
                            session.model,
                            session.parent_session_id,
                            session.causal_budget_id,
                            session.runtime_name,
                            session.runtime_version,
                            session.environment_name,
                            str(session.status),
                            sqlite_support.format_datetime(session.created_at),
                            sqlite_support.format_datetime(session.updated_at),
                            sqlite_support.format_datetime(session.last_activity_at),
                            session.run_epoch,
                            sqlite_support.json_dumps(session.invocation.model_dump(mode="json")),
                            sqlite_support.json_dumps(session.metadata),
                        ),
                    )
                    if initial_operation_records:
                        self._connection.executemany(
                            "INSERT INTO cayu_session_operations "
                            "(session_id, idempotency_key, record_json, updated_at) "
                            "VALUES (?, ?, ?, ?)",
                            [
                                (
                                    session.id,
                                    key,
                                    sqlite_support.json_dumps(record),
                                    sqlite_support.format_datetime(session.updated_at),
                                )
                                for key, record in initial_operation_records.items()
                            ],
                        )
                    if session.labels:
                        self._connection.executemany(
                            """
                            INSERT INTO cayu_session_labels (session_id, key, value)
                            VALUES (?, ?, ?)
                            """,
                            sqlite_support.session_label_row_values(session),
                        )
                    if admission is not None:
                        started_event, source_messages = admission
                        interaction_id = started_event.interaction_id
                        if interaction_id is None:
                            raise AssertionError("Interaction admission lost its identity.")
                        deferred_input = deferred_interaction_input_for_run_request(
                            request,
                            session_id=session.id,
                            interaction_id=interaction_id,
                            source_messages=source_messages,
                        )
                        lookup_key, projection, projection_bytes = (
                            pending_action_event_storage_values(started_event)
                        )
                        self._connection.execute(
                            """
                            INSERT INTO cayu_events (
                                session_id, event_id, interaction_id, event_type,
                                timestamp, agent_name, environment_name, workflow_name,
                                tool_name, payload_json, pending_action_lookup_key,
                                pending_action_projection_json,
                                pending_action_projection_bytes
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                session.id,
                                started_event.id,
                                interaction_id,
                                str(started_event.type),
                                sqlite_support.format_datetime(started_event.timestamp),
                                started_event.agent_name,
                                started_event.environment_name,
                                started_event.workflow_name,
                                started_event.tool_name,
                                sqlite_support.json_dumps(started_event.payload),
                                lookup_key,
                                projection,
                                projection_bytes,
                            ),
                        )
                        _enqueue_persisted_event_side_effects(
                            self._connection,
                            session.id,
                            [started_event],
                        )
                        self._connection.execute(
                            "INSERT INTO cayu_deferred_interaction_inputs "
                            "(session_id, interaction_id, source_messages_json) "
                            "VALUES (?, ?, ?)",
                            (
                                session.id,
                                interaction_id,
                                sqlite_support.json_dumps(
                                    deferred_interaction_input_storage_payload(deferred_input)
                                ),
                            ),
                        )
                        self._connection.execute(
                            """
                            INSERT INTO cayu_checkpoints (
                                session_id, state_json, updated_at,
                                pending_action_source_bytes,
                                pending_action_tool_call_count,
                                pending_action_flags,
                                pending_action_metrics_ready
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            sqlite_support.checkpoint_row_values(
                                session.id,
                                _initial_transcript_pending_checkpoint(
                                    session,
                                    interaction_id,
                                    checkpoint_transform=checkpoint_transform,
                                ),
                                session.updated_at,
                            ),
                        )
                    elif checkpoint_transform is not None:
                        transformed = checkpoint_transform(session.model_copy(deep=True), None)
                        if transformed is not None:
                            transformed = (
                                _replace_checkpoint_preserving_completion_result_event_publications(
                                    None,
                                    copy_durable_json_object(transformed, "checkpoint"),
                                    session_id=session.id,
                                )
                            )
                            self._connection.execute(
                                """
                                INSERT INTO cayu_checkpoints (
                                    session_id, state_json, updated_at,
                                    pending_action_source_bytes,
                                    pending_action_tool_call_count,
                                    pending_action_flags,
                                    pending_action_metrics_ready
                                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                sqlite_support.checkpoint_row_values(
                                    session.id,
                                    transformed,
                                    session.updated_at,
                                ),
                            )
                    if result_checkpoint_transform is not None:
                        current_checkpoint = self._load_checkpoint_unlocked(session.id)
                        transformed = result_checkpoint_transform(
                            session.model_copy(deep=True),
                            _copy_checkpoint_for_transform(
                                current_checkpoint,
                                session_id=session.id,
                            ),
                        )
                        if transformed is None:
                            raise ValueError(
                                "Result checkpoint transform must return a checkpoint."
                            )
                        self._connection.execute(
                            """
                            INSERT INTO cayu_checkpoints (
                                session_id, state_json, updated_at,
                                pending_action_source_bytes,
                                pending_action_tool_call_count,
                                pending_action_flags,
                                pending_action_metrics_ready
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(session_id) DO UPDATE SET
                                state_json = excluded.state_json,
                                updated_at = excluded.updated_at,
                                pending_action_source_bytes = excluded.pending_action_source_bytes,
                                pending_action_tool_call_count = excluded.pending_action_tool_call_count,
                                pending_action_flags = excluded.pending_action_flags,
                                pending_action_metrics_ready = excluded.pending_action_metrics_ready
                            """,
                            sqlite_support.checkpoint_row_values(
                                session.id,
                                _checkpoint_transform_result_preserving_completion_result_event_publications(
                                    current_checkpoint,
                                    transformed,
                                    session_id=session.id,
                                ),
                                session.updated_at,
                            ),
                        )
            except sqlite3.IntegrityError as exc:
                if self._session_exists_unlocked(session.id):
                    raise ValueError(f"Session already exists: {session.id}") from exc
                if session.parent_session_id is not None and not self._session_exists_unlocked(
                    session.parent_session_id
                ):
                    raise ValueError(
                        f"Parent session not found: {session.parent_session_id}"
                    ) from exc
                raise
            if admission is not None:
                _activate_session_run_fence(session)
            return session.model_copy(deep=True)

    async def create_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        system_prompt_replacement: ForkSystemPromptReplacement | None = None,
        expected_source_run_epoch: int,
        operation_initializer: SessionOperationInitializer | None = None,
    ) -> Session:
        return await self._create_fork(
            source_session_id=source_session_id,
            fork=fork,
            source_statuses=source_statuses,
            transcript_cursor=transcript_cursor,
            checkpoint_transform=checkpoint_transform,
            system_prompt_replacement=system_prompt_replacement,
            expected_source_run_epoch=expected_source_run_epoch,
            transcript_validator=None,
            operation_initializer=operation_initializer,
        )

    async def create_fork_with_transcript_validation(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        system_prompt_replacement: ForkSystemPromptReplacement | None = None,
        expected_source_run_epoch: int,
        transcript_validator: ForkTranscriptValidator,
        operation_initializer: SessionOperationInitializer | None = None,
    ) -> Session:
        return await self._create_fork(
            source_session_id=source_session_id,
            fork=fork,
            source_statuses=source_statuses,
            transcript_cursor=transcript_cursor,
            checkpoint_transform=checkpoint_transform,
            system_prompt_replacement=system_prompt_replacement,
            expected_source_run_epoch=expected_source_run_epoch,
            transcript_validator=transcript_validator,
            operation_initializer=operation_initializer,
        )

    async def create_profiled_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        system_prompt_replacement: ForkSystemPromptReplacement | None = None,
        expected_source_run_epoch: int,
        relationship: SessionForkProfileRelationship,
        events: list[Event],
        transcript_validator: ForkTranscriptValidator | None = None,
        checkpoint_authority_decoder: ForkCheckpointAuthorityDecoder | None = None,
        operation_initializer: SessionOperationInitializer | None = None,
    ) -> ProfiledSessionForkResult:
        relationship, copied_events = _copy_profiled_fork_authority(
            fork=fork,
            relationship=relationship,
            events=events,
        )
        created = await self._create_fork(
            source_session_id=source_session_id,
            fork=fork,
            source_statuses=source_statuses,
            transcript_cursor=transcript_cursor,
            checkpoint_transform=checkpoint_transform,
            system_prompt_replacement=system_prompt_replacement,
            expected_source_run_epoch=expected_source_run_epoch,
            transcript_validator=transcript_validator,
            profile_relationship=relationship,
            events=copied_events,
            checkpoint_authority_decoder=checkpoint_authority_decoder,
            operation_initializer=operation_initializer,
        )
        return ProfiledSessionForkResult(session=created, events=tuple(copied_events))

    async def _create_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        system_prompt_replacement: ForkSystemPromptReplacement | None,
        expected_source_run_epoch: int,
        transcript_validator: ForkTranscriptValidator | None,
        profile_relationship: SessionForkProfileRelationship | None = None,
        events: list[Event] | None = None,
        checkpoint_authority_decoder: ForkCheckpointAuthorityDecoder | None = None,
        operation_initializer: SessionOperationInitializer | None = None,
    ) -> Session:
        source_session_id, fork, allowed_statuses, transcript_cursor = (
            _prepare_session_fork_request(
                source_session_id=source_session_id,
                fork=fork,
                source_statuses=source_statuses,
                transcript_cursor=transcript_cursor,
            )
        )
        fork = fork.model_copy(update={"instance_id": str(uuid4())})

        async with self._lock:
            self._require_current_public_authority_configuration(self._connection)
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                source_session = _validate_session_fork_source(
                    source_session=self._load_unlocked(source_session_id),
                    source_session_id=source_session_id,
                    fork=fork,
                    allowed_statuses=allowed_statuses,
                    expected_source_run_epoch=expected_source_run_epoch,
                    profile_relationship=profile_relationship,
                )
                active_stage = self._connection.execute(
                    "SELECT 1 FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (
                        source_session_id,
                        MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                    ),
                ).fetchone()
                if active_stage is not None:
                    raise SessionForkActiveModelStageConflict(
                        "Cannot fork a session while a model-completion stage is active."
                    )
                source_checkpoint = self._load_checkpoint_unlocked(source_session_id)
                source_checkpoint_present = source_checkpoint is not None
                if profile_relationship is not None:
                    profile_checkpoint = None
                    profile_failure: ValueError | None = None
                    try:
                        profile_checkpoint = (
                            source_checkpoint
                            if checkpoint_authority_decoder is None
                            else checkpoint_authority_decoder(
                                None
                                if source_checkpoint is None
                                else copy_durable_json_object(
                                    source_checkpoint,
                                    "source checkpoint",
                                )
                            )
                        )
                        _validate_profiled_fork_authority(
                            source_session=source_session,
                            source_checkpoint=profile_checkpoint,
                            fork=fork,
                            relationship=profile_relationship,
                            events=() if events is None else events,
                            transcript_cursor=transcript_cursor,
                            checkpoint_transform=checkpoint_transform,
                            system_prompt_replacement=system_prompt_replacement,
                            transcript_validator=transcript_validator,
                        )
                    except Exception as exc:
                        profile_failure = _profiled_fork_authority_validation_error(exc)
                        source_checkpoint = None
                    finally:
                        profile_checkpoint = None
                    if profile_failure is not None:
                        raise profile_failure from None
                source_transcript_cursor = _transcript_cursor(
                    self._connection,
                    source_session_id,
                )
                if transcript_cursor is not None and transcript_cursor > source_transcript_cursor:
                    raise ValueError("transcript_cursor is greater than source transcript length.")
                selected_transcript_rows = self._connection.execute(
                    """
                    SELECT session_order, message_json, interaction_id
                    FROM cayu_transcript_messages
                    WHERE session_id = ?
                      AND session_order <= ?
                    ORDER BY session_order ASC
                    """,
                    (
                        source_session_id,
                        (
                            source_transcript_cursor
                            if transcript_cursor is None
                            else transcript_cursor
                        ),
                    ),
                ).fetchall()
                copied_messages = [
                    Message(**json.loads(row["message_json"])) for row in selected_transcript_rows
                ]
                copied_interaction_ids = [row["interaction_id"] for row in selected_transcript_rows]
                source_transcript_snapshot = (
                    None
                    if transcript_validator is None
                    else TranscriptSnapshot(
                        records=[
                            TranscriptRecord(
                                index=int(row["session_order"]) - 1,
                                interaction_id=row["interaction_id"],
                                message=copied_messages[position],
                            )
                            for position, row in enumerate(selected_transcript_rows)
                        ],
                        cursor=source_transcript_cursor,
                    )
                )
                selected_transcript_rows.clear()
                copied_messages, copied_interaction_ids = apply_fork_system_prompt_replacement(
                    copied_messages,
                    copied_interaction_ids,
                    system_prompt_replacement,
                )
                if not fork_transcript_is_accepted(
                    copied_messages,
                    source_transcript_snapshot,
                    transcript_validator,
                ):
                    copied_messages.clear()
                    copied_messages = []
                    source_transcript_snapshot = None
                    raise ValueError(FORK_TRANSCRIPT_VALIDATION_ERROR) from None
                source_transcript_snapshot = None
                copied_checkpoint = None
                if checkpoint_transform is not None:
                    checkpoint_input = source_checkpoint
                    copied_checkpoint = transform_fork_checkpoint(
                        source_session,
                        checkpoint_input,
                        checkpoint_transform,
                    )
                    checkpoint_input = None
                    if copied_checkpoint is not None:
                        copied_checkpoint = copy_durable_json_object(
                            copied_checkpoint,
                            "checkpoint",
                        )
                if profile_relationship is not None:
                    _validate_profiled_fork_checkpoint_result(
                        relationship=profile_relationship,
                        source_checkpoint_present=source_checkpoint_present,
                        copied_checkpoint=copied_checkpoint,
                    )
                initial_operation_records = _prepare_initial_session_operation_records(
                    fork,
                    operation_initializer,
                )

                self._connection.execute(
                    """
                    INSERT INTO cayu_sessions (
                        id,
                        instance_id,
                        agent_name,
                        provider_name,
                        model,
                        parent_session_id,
                        causal_budget_id,
                        runtime_name,
                        runtime_version,
                        environment_name,
                        status,
                        created_at,
                        updated_at,
                        last_activity_at,
                        run_epoch,
                        invocation_json,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    sqlite_support.session_to_row_values(fork),
                )
                if initial_operation_records:
                    self._connection.executemany(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record_json, updated_at) "
                        "VALUES (?, ?, ?, ?)",
                        [
                            (
                                fork.id,
                                key,
                                sqlite_support.json_dumps(record),
                                sqlite_support.format_datetime(fork.updated_at),
                            )
                            for key, record in initial_operation_records.items()
                        ],
                    )
                if fork.labels:
                    self._connection.executemany(
                        """
                        INSERT INTO cayu_session_labels (session_id, key, value)
                        VALUES (?, ?, ?)
                        """,
                        sqlite_support.session_label_row_values(fork),
                    )
                if copied_messages:
                    self._connection.executemany(
                        """
                        INSERT INTO cayu_transcript_messages (
                            session_id,
                            role,
                            interaction_id,
                            message_json,
                            transcript_search_document
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                fork.id,
                                str(message.role),
                                copied_interaction_ids[index],
                                sqlite_support.json_dumps(message.model_dump(mode="json")),
                                transcript_search_document(message),
                            )
                            for index, message in enumerate(copied_messages)
                        ],
                    )
                if copied_checkpoint is not None:
                    self._connection.execute(
                        """
                        INSERT INTO cayu_checkpoints (
                            session_id, state_json, updated_at,
                            pending_action_source_bytes,
                            pending_action_tool_call_count,
                            pending_action_flags,
                            pending_action_metrics_ready
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        sqlite_support.checkpoint_row_values(
                            fork.id, copied_checkpoint, fork.updated_at
                        ),
                    )
                if events:
                    from cayu.runtime.pending_actions import pending_action_event_storage_values

                    _touch_session_activity(self._connection, fork.id, self._ownership_clock())
                    rows = []
                    for event in events:
                        lookup_key, projection, projection_bytes = (
                            pending_action_event_storage_values(event)
                        )
                        rows.append(
                            (
                                fork.id,
                                event.id,
                                event.interaction_id,
                                str(event.type),
                                sqlite_support.format_datetime(event.timestamp),
                                event.agent_name,
                                event.environment_name,
                                event.workflow_name,
                                event.tool_name,
                                sqlite_support.json_dumps(event.payload),
                                lookup_key,
                                projection,
                                projection_bytes,
                            )
                        )
                    self._connection.executemany(
                        """
                        INSERT INTO cayu_events (
                            session_id, event_id, interaction_id, event_type, timestamp,
                            agent_name, environment_name, workflow_name, tool_name,
                            payload_json, pending_action_lookup_key,
                            pending_action_projection_json, pending_action_projection_bytes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        rows,
                    )
                    _enqueue_persisted_event_side_effects(
                        self._connection,
                        fork.id,
                        events,
                    )
                self._connection.commit()
            except sqlite3.IntegrityError as exc:
                self._connection.rollback()
                if self._session_exists_unlocked(fork.id):
                    raise ValueError(f"Session already exists: {fork.id}") from exc
                raise
            except Exception:
                self._connection.rollback()
                raise

            loaded = self._load_unlocked(fork.id)
            if loaded is None:
                raise KeyError(f"Session not found: {fork.id}")
            return loaded

    async def load(self, session_id: str) -> Session | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        return await self._run_read(lambda connection: _load_session(connection, session_id))

    async def load_state(self, session_id: str) -> SessionStateSnapshot | None:
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection: sqlite3.Connection) -> SessionStateSnapshot | None:
            row = connection.execute(
                """
                SELECT id, status, updated_at, last_activity_at
                FROM cayu_sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return SessionStateSnapshot(
                id=row["id"],
                status=SessionStatus(row["status"]),
                updated_at=sqlite_support.parse_datetime(row["updated_at"]),
                last_activity_at=sqlite_support.parse_datetime(row["last_activity_at"]),
            )

        return await self._run_read(query)

    async def load_invocation_snapshot(
        self,
        session_id: str,
    ) -> SessionInvocationSnapshot | None:
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection: sqlite3.Connection) -> SessionInvocationSnapshot | None:
            row = connection.execute(
                "SELECT id, instance_id, status, invocation_json FROM cayu_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return SessionInvocationSnapshot(
                id=row["id"],
                session_instance_id=row["instance_id"],
                status=SessionStatus(row["status"]),
                invocation=SessionInvocation.model_validate(json.loads(row["invocation_json"])),
            )

        return await self._run_read(query)

    async def create_recall_receipt(self, receipt: RecallReceipt) -> RecallReceipt:
        copied = copy_recall_receipt(receipt)
        document = memory_evidence_document_bytes(copied, "recall receipt")

        def statement(connection: sqlite3.Connection) -> RecallReceipt:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM cayu_recall_receipts WHERE receipt_id = ?",
                    (copied.receipt_id,),
                ).fetchone()
                if row is not None:
                    current = _sqlite_recall_receipt(row)
                    if memory_evidence_document_bytes(current, "stored recall receipt") != document:
                        raise RecallEvidenceConflict("Recall receipt", copied.receipt_id)
                    connection.commit()
                    return current
                if not connection.execute(
                    "SELECT 1 FROM cayu_sessions WHERE id = ?",
                    (copied.session_id,),
                ).fetchone():
                    raise KeyError(f"Session not found: {copied.session_id}")
                connection.execute(
                    """
                    INSERT INTO cayu_recall_receipts (
                        receipt_id, session_id, interaction_id, model_step_id,
                        created_at, receipt_json, document_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        copied.receipt_id,
                        copied.session_id,
                        copied.interaction_id,
                        copied.model_step_id,
                        sqlite_support.format_datetime(copied.created_at),
                        document.decode("utf-8"),
                        len(document),
                    ),
                )
                connection.commit()
                return copied
            except BaseException:
                connection.rollback()
                raise

        return copy_recall_receipt(await self._run_write(statement))

    async def load_recall_receipt(
        self,
        session_id: str,
        receipt_id: str,
    ) -> RecallReceipt | None:
        session_id = require_memory_evidence_session_id(session_id)
        receipt_id = require_memory_evidence_id(receipt_id, "receipt_id")

        def query(connection: sqlite3.Connection) -> RecallReceipt | None:
            row = connection.execute(
                """
                SELECT * FROM cayu_recall_receipts
                WHERE session_id = ? AND receipt_id = ?
                """,
                (session_id, receipt_id),
            ).fetchone()
            return None if row is None else _sqlite_recall_receipt(row)

        loaded = await self._run_read(query)
        return None if loaded is None else copy_recall_receipt(loaded)

    async def list_recall_receipts(
        self,
        query: RecallEvidenceQuery,
    ) -> RecallReceiptPage:
        copied_query = RecallEvidenceQuery.model_validate(query.model_dump(mode="python"))
        query_fingerprint = copied_query.fingerprint("receipt")
        after = (
            None
            if copied_query.cursor is None
            else decode_recall_evidence_cursor(
                copied_query.cursor,
                record_kind="receipt",
                query_fingerprint=query_fingerprint,
            )
        )

        def read(connection: sqlite3.Connection) -> RecallReceiptPage:
            clauses = ["session_id = ?"]
            parameters: list[object] = [copied_query.session_id]
            if copied_query.interaction_id is not None:
                clauses.append("interaction_id = ?")
                parameters.append(copied_query.interaction_id)
            if copied_query.model_step_id is not None:
                clauses.append("model_step_id = ?")
                parameters.append(copied_query.model_step_id)
            if after is not None:
                clauses.append("(created_at, receipt_id) > (?, ?)")
                created_at = sqlite_support.format_datetime(after[0])
                parameters.extend((created_at, after[1]))
            where = " AND ".join(clauses)
            rows = connection.execute(
                f"""
                SELECT * FROM cayu_recall_receipts
                WHERE {where}
                ORDER BY created_at, receipt_id COLLATE BINARY
                LIMIT ?
                """,
                (*parameters, copied_query.limit + 1),
            ).fetchall()
            retained: list[RecallReceipt] = []
            retained_bytes = 2
            for row in rows:
                if len(retained) >= copied_query.limit:
                    break
                receipt = _sqlite_recall_receipt(row)
                document_bytes = len(
                    memory_evidence_document_bytes(receipt, "recall receipt page item")
                )
                separator_bytes = 1 if retained else 0
                if retained_bytes + separator_bytes + document_bytes > copied_query.max_bytes:
                    break
                retained.append(receipt)
                retained_bytes += separator_bytes + document_bytes
            truncated = len(retained) < len(rows)
            return RecallReceiptPage(
                items=tuple(retained),
                next_cursor=(
                    encode_recall_evidence_cursor(
                        record_kind="receipt",
                        query_fingerprint=query_fingerprint,
                        created_at=retained[-1].created_at,
                        record_id=retained[-1].receipt_id,
                    )
                    if truncated and retained
                    else None
                ),
                truncated=truncated,
            )

        return await self._run_read(read)

    async def create_context_exposure(
        self,
        exposure: ContextExposure,
        item_exposures: tuple[RecallItemExposure, ...] = (),
    ) -> ContextExposure:
        copied = copy_context_exposure(exposure)
        copied_items = tuple(copy_recall_item_exposure(item) for item in item_exposures)
        validate_new_context_exposure(copied, copied_items)
        document = memory_evidence_document_bytes(copied, "context exposure")
        item_documents = tuple(
            memory_evidence_document_bytes(item, "recall item exposure") for item in copied_items
        )

        def statement(connection: sqlite3.Connection) -> ContextExposure:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM cayu_context_exposures WHERE exposure_id = ?",
                    (copied.exposure_id,),
                ).fetchone()
                if row is not None:
                    current = _sqlite_context_exposure(row)
                    current_item_rows = connection.execute(
                        """
                        SELECT item.*
                        FROM cayu_recall_item_exposures AS item
                        WHERE item.exposure_id = ?
                        ORDER BY item.ordinal
                        LIMIT ?
                        """,
                        (copied.exposure_id, MAX_RECALL_RECEIPT_ITEMS + 1),
                    ).fetchall()
                    if len(current_item_rows) > MAX_RECALL_RECEIPT_ITEMS:
                        raise ValueError("Stored recall item exposures exceed their count bound.")
                    current_items = _sqlite_recall_item_exposures(current_item_rows)
                    if (
                        not context_exposure_creation_matches(current, copied)
                        or tuple(
                            memory_evidence_document_bytes(item, "stored recall item exposure")
                            for item in current_items
                        )
                        != item_documents
                    ):
                        raise RecallEvidenceConflict("Context exposure", copied.exposure_id)
                    connection.commit()
                    return current
                if not connection.execute(
                    "SELECT 1 FROM cayu_sessions WHERE id = ?",
                    (copied.session_id,),
                ).fetchone():
                    raise KeyError(f"Session not found: {copied.session_id}")

                model_attempt_collision = connection.execute(
                    """
                    SELECT exposure_id
                    FROM cayu_context_exposures
                    WHERE session_id = ? AND model_attempt_id = ?
                    """,
                    (copied.session_id, copied.model_attempt_id),
                ).fetchone()
                if model_attempt_collision is not None:
                    raise RecallEvidenceConflict(
                        "Model-attempt exposure",
                        model_attempt_collision[0],
                    )
                provider_attempt_collision = connection.execute(
                    """
                    SELECT exposure_id
                    FROM cayu_context_exposures
                    WHERE session_id = ? AND provider_attempt_id = ?
                    """,
                    (copied.session_id, copied.provider_attempt_id),
                ).fetchone()
                if provider_attempt_collision is not None:
                    raise RecallEvidenceConflict(
                        "Provider-attempt exposure",
                        provider_attempt_collision[0],
                    )

                receipts: dict[str, RecallReceipt] = {}
                for receipt_id in copied.receipt_ids:
                    receipt_row = connection.execute(
                        "SELECT * FROM cayu_recall_receipts WHERE receipt_id = ?",
                        (receipt_id,),
                    ).fetchone()
                    if receipt_row is None:
                        raise KeyError(f"Recall receipt not found: {receipt_id}")
                    receipt = _sqlite_recall_receipt(receipt_row)
                    validate_context_exposure_receipt_scope(copied, receipt)
                    receipts[receipt_id] = receipt
                for item in copied_items:
                    if not recall_item_exposure_matches_receipt_item(
                        item,
                        receipts[item.receipt_id],
                    ):
                        raise ValueError(
                            "Recall item exposure differs from its immutable receipt item."
                        )
                connection.execute(
                    """
                    INSERT INTO cayu_context_exposures (
                        exposure_id, session_id, interaction_id, model_step_id,
                        model_attempt_id, provider_attempt_id, state, state_revision,
                        created_at, updated_at, exposure_json, document_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        copied.exposure_id,
                        copied.session_id,
                        copied.interaction_id,
                        copied.model_step_id,
                        copied.model_attempt_id,
                        copied.provider_attempt_id,
                        str(copied.state),
                        copied.state_revision,
                        sqlite_support.format_datetime(copied.created_at),
                        sqlite_support.format_datetime(copied.updated_at),
                        document.decode("utf-8"),
                        len(document),
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO cayu_recall_item_exposures (
                        exposure_id, ordinal, receipt_id, receipt_item_ordinal,
                        item_json, document_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            item.exposure_id,
                            item.ordinal,
                            item.receipt_id,
                            item.receipt_item_ordinal,
                            item_document.decode("utf-8"),
                            len(item_document),
                        )
                        for item, item_document in zip(
                            copied_items,
                            item_documents,
                            strict=True,
                        )
                    ),
                )
                connection.commit()
                return copied
            except BaseException:
                connection.rollback()
                raise

        return copy_context_exposure(await self._run_write(statement))

    async def load_context_exposure(
        self,
        session_id: str,
        exposure_id: str,
    ) -> ContextExposure | None:
        session_id = require_memory_evidence_session_id(session_id)
        exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")

        def query(connection: sqlite3.Connection) -> ContextExposure | None:
            row = connection.execute(
                """
                SELECT * FROM cayu_context_exposures
                WHERE session_id = ? AND exposure_id = ?
                """,
                (session_id, exposure_id),
            ).fetchone()
            return None if row is None else _sqlite_context_exposure(row)

        loaded = await self._run_read(query)
        return None if loaded is None else copy_context_exposure(loaded)

    async def load_recall_item_exposures(
        self,
        session_id: str,
        exposure_id: str,
    ) -> tuple[RecallItemExposure, ...]:
        session_id = require_memory_evidence_session_id(session_id)
        exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")

        def query(connection: sqlite3.Connection) -> tuple[RecallItemExposure, ...]:
            rows = connection.execute(
                """
                SELECT item.*
                FROM cayu_recall_item_exposures AS item
                JOIN cayu_context_exposures AS exposure
                  ON exposure.exposure_id = item.exposure_id
                WHERE exposure.session_id = ? AND exposure.exposure_id = ?
                ORDER BY item.ordinal
                LIMIT ?
                """,
                (session_id, exposure_id, MAX_RECALL_RECEIPT_ITEMS + 1),
            ).fetchall()
            if len(rows) > MAX_RECALL_RECEIPT_ITEMS:
                raise ValueError("Stored recall item exposures exceed their count bound.")
            return _sqlite_recall_item_exposures(rows)

        return tuple(copy_recall_item_exposure(item) for item in await self._run_read(query))

    async def list_context_exposures(
        self,
        query: RecallEvidenceQuery,
    ) -> ContextExposurePage:
        copied_query = RecallEvidenceQuery.model_validate(query.model_dump(mode="python"))
        query_fingerprint = copied_query.fingerprint("exposure")
        after = (
            None
            if copied_query.cursor is None
            else decode_recall_evidence_cursor(
                copied_query.cursor,
                record_kind="exposure",
                query_fingerprint=query_fingerprint,
            )
        )

        def read(connection: sqlite3.Connection) -> ContextExposurePage:
            clauses = ["session_id = ?"]
            parameters: list[object] = [copied_query.session_id]
            if copied_query.interaction_id is not None:
                clauses.append("interaction_id = ?")
                parameters.append(copied_query.interaction_id)
            if copied_query.model_step_id is not None:
                clauses.append("model_step_id = ?")
                parameters.append(copied_query.model_step_id)
            if after is not None:
                clauses.append("(created_at, exposure_id) > (?, ?)")
                created_at = sqlite_support.format_datetime(after[0])
                parameters.extend((created_at, after[1]))
            where = " AND ".join(clauses)
            rows = connection.execute(
                f"""
                SELECT * FROM cayu_context_exposures
                WHERE {where}
                ORDER BY created_at, exposure_id COLLATE BINARY
                LIMIT ?
                """,
                (*parameters, copied_query.limit + 1),
            ).fetchall()
            retained: list[ContextExposure] = []
            retained_bytes = 2
            for row in rows:
                if len(retained) >= copied_query.limit:
                    break
                exposure = _sqlite_context_exposure(row)
                document_bytes = len(
                    memory_evidence_document_bytes(exposure, "context exposure page item")
                )
                separator_bytes = 1 if retained else 0
                if retained_bytes + separator_bytes + document_bytes > copied_query.max_bytes:
                    break
                retained.append(exposure)
                retained_bytes += separator_bytes + document_bytes
            truncated = len(retained) < len(rows)
            return ContextExposurePage(
                items=tuple(retained),
                next_cursor=(
                    encode_recall_evidence_cursor(
                        record_kind="exposure",
                        query_fingerprint=query_fingerprint,
                        created_at=retained[-1].created_at,
                        record_id=retained[-1].exposure_id,
                    )
                    if truncated and retained
                    else None
                ),
                truncated=truncated,
            )

        return await self._run_read(read)

    async def transition_context_exposure(
        self,
        session_id: str,
        exposure_id: str,
        request: ContextExposureTransitionRequest,
    ) -> ContextExposure:
        session_id = require_memory_evidence_session_id(session_id)
        exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")
        copied_request = ContextExposureTransitionRequest.model_validate(
            request.model_dump(mode="python")
        )

        def statement(connection: sqlite3.Connection) -> ContextExposure:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT * FROM cayu_context_exposures
                    WHERE session_id = ? AND exposure_id = ?
                    """,
                    (session_id, exposure_id),
                ).fetchone()
                if row is None:
                    raise KeyError(f"Context exposure not found: {exposure_id}")
                current = _sqlite_context_exposure(row)
                replay = next(
                    (
                        transition
                        for transition in current.transitions
                        if transition.transition_id == copied_request.transition_id
                    ),
                    None,
                )
                if replay is not None:
                    if not context_exposure_transition_replays(current, copied_request):
                        raise RecallEvidenceConflict(
                            "Context exposure transition",
                            copied_request.transition_id,
                        )
                    connection.commit()
                    return current
                updated = append_context_exposure_transition(current, copied_request)
                updated_document = memory_evidence_document_bytes(
                    updated,
                    "context exposure",
                )
                cursor = connection.execute(
                    """
                    UPDATE cayu_context_exposures
                    SET state = ?, state_revision = ?, updated_at = ?,
                        exposure_json = ?, document_bytes = ?
                    WHERE session_id = ? AND exposure_id = ?
                      AND state = ? AND state_revision = ?
                    """,
                    (
                        str(updated.state),
                        updated.state_revision,
                        sqlite_support.format_datetime(updated.updated_at),
                        updated_document.decode("utf-8"),
                        len(updated_document),
                        session_id,
                        exposure_id,
                        str(copied_request.expected_state),
                        copied_request.expected_revision,
                    ),
                )
                if cursor.rowcount != 1:
                    latest_row = connection.execute(
                        "SELECT * FROM cayu_context_exposures WHERE exposure_id = ?",
                        (exposure_id,),
                    ).fetchone()
                    if latest_row is None:
                        raise KeyError(f"Context exposure not found: {exposure_id}")
                    latest = _sqlite_context_exposure(latest_row)
                    raise ContextExposureTransitionConflict(
                        exposure_id,
                        expected_state=copied_request.expected_state,
                        expected_revision=copied_request.expected_revision,
                        actual_state=latest.state,
                        actual_revision=latest.state_revision,
                    )
                connection.commit()
                return updated
            except BaseException:
                connection.rollback()
                raise

        return copy_context_exposure(await self._run_write(statement))

    async def inspect_identity(self, session_id: str) -> SessionInspectionIdentity:
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection: sqlite3.Connection) -> SessionInspectionIdentity:
            row = connection.execute(
                """
                SELECT id, agent_name, provider_name, model, parent_session_id,
                       causal_budget_id, runtime_name, runtime_version, environment_name,
                       status, created_at, updated_at, last_activity_at, run_epoch,
                       json_extract(
                           metadata_json,
                           '$."cayu:runtime_build_provenance"'
                       ) AS runtime_build_provenance_json
                FROM cayu_sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(session_id)
            label_rows = connection.execute(
                """
                SELECT key, value,
                       (SELECT COUNT(*)
                        FROM cayu_session_labels
                        WHERE session_id = ?) AS label_count
                FROM cayu_session_labels
                WHERE session_id = ?
                ORDER BY key ASC
                LIMIT ?
                """,
                (session_id, session_id, SESSION_INSPECTION_LABEL_LIMIT),
            ).fetchall()
            label_count = 0 if not label_rows else label_rows[0]["label_count"]
            return SessionInspectionIdentity(
                id=row["id"],
                agent_name=row["agent_name"],
                provider_name=row["provider_name"],
                model=row["model"],
                parent_session_id=row["parent_session_id"],
                causal_budget_id=row["causal_budget_id"],
                runtime_name=row["runtime_name"],
                runtime_version=row["runtime_version"],
                runtime_build_provenance=runtime_build_provenance_from_session_metadata(
                    {}
                    if row["runtime_build_provenance_json"] is None
                    else {
                        RUNTIME_BUILD_PROVENANCE_METADATA_KEY: json.loads(
                            row["runtime_build_provenance_json"]
                        )
                    }
                ),
                environment_name=row["environment_name"],
                status=SessionStatus(row["status"]),
                created_at=sqlite_support.parse_datetime(row["created_at"]),
                updated_at=sqlite_support.parse_datetime(row["updated_at"]),
                last_activity_at=sqlite_support.parse_datetime(row["last_activity_at"]),
                run_epoch=row["run_epoch"],
                labels={label_row["key"]: label_row["value"] for label_row in label_rows},
                label_count=label_count,
                labels_truncated=label_count > len(label_rows),
            )

        return await self._run_read(query)

    async def update_status(self, session_id: str, status: SessionStatus) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(status, SessionStatus):
            raise ValueError("Session status must be a SessionStatus.")
        # Route the unconditional setter through the guarded transition machine so
        # both write paths share one atomic UPDATE-and-check. Allowing every source
        # status preserves update_status semantics (any -> status) while inheriting
        # the row-level not-found guard.
        return await self.transition_status(
            session_id,
            from_statuses=set(SessionStatus),
            to_status=status,
        )

    async def delete_session(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                session = self._load_unlocked(session_id)
                if session is None:
                    self._connection.rollback()
                    return
                if session.status in DELETE_BLOCKED_SESSION_STATUSES:
                    raise ValueError(
                        f"Cannot delete a session while it is {session.status}; "
                        f"interrupt it first: {session_id}"
                    )
                durable_child = self._connection.execute(
                    "SELECT id FROM cayu_sessions "
                    "WHERE parent_session_id = ? "
                    "AND json_extract(metadata_json, '$.subagent.mode') = ? "
                    "ORDER BY id LIMIT 1",
                    (session_id, "durable"),
                ).fetchone()
                if durable_child is not None:
                    raise ValueError(
                        _durable_subagent_parent_delete_block_reason(durable_child["id"])
                    )
                checkpoint = self._load_checkpoint_unlocked(session_id)
                deletion_now = self._ownership_clock()
                active_recovery_claim_id = _active_unexpired_incomplete_recovery_claim_id(
                    checkpoint,
                    now=deletion_now,
                )
                if active_recovery_claim_id is not None:
                    raise ValueError(
                        "Cannot delete a session while incomplete-session recovery claim "
                        f"{active_recovery_claim_id} is active: {session_id}"
                    )
                run_operation = _session_run_operation_from_checkpoint(checkpoint)
                if run_operation is not None:
                    raise ValueError(
                        "Cannot delete a session while terminal publication "
                        f"{run_operation.operation_id} is incomplete: {session_id}"
                    )
                if _queued_dispatch_terminal_receipts_from_checkpoint(checkpoint):
                    raise ValueError(
                        "Cannot delete a session while queued dispatch terminal "
                        f"acknowledgement is incomplete: {session_id}"
                    )
                terminal_evidence_rows = self._connection.execute(
                    f"SELECT {', '.join(_EVENT_COLUMN_NAMES)} FROM cayu_events "
                    "WHERE session_id = ? "
                    f"AND event_type IN ({', '.join('?' for _ in _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES)}) "
                    "ORDER BY sequence DESC LIMIT ?",
                    (
                        session_id,
                        *(
                            str(event_type)
                            for event_type in _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES
                        ),
                        _TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT,
                    ),
                ).fetchall()
                terminal_publication_block = _terminal_publication_delete_block_reason(
                    session=session,
                    checkpoint=checkpoint,
                    evidence_events=[_event_from_row(row) for row in terminal_evidence_rows],
                )
                if terminal_publication_block is not None:
                    raise ValueError(
                        f"Cannot delete a session while {terminal_publication_block}: {session_id}"
                    )
                active_operation_id = _active_unexpired_session_operation_id(
                    checkpoint,
                    now=deletion_now,
                )
                if active_operation_id is not None:
                    raise ValueError(
                        "Cannot delete a session while durable operation "
                        f"{active_operation_id} is active: {session_id}"
                    )
                completion_result_publication_block = (
                    _completion_result_event_publication_delete_block_reason(
                        checkpoint,
                        now=deletion_now,
                    )
                )
                if completion_result_publication_block is not None:
                    raise ValueError(
                        "Cannot delete a session while "
                        f"{completion_result_publication_block}: {session_id}"
                    )
                active_stage = self._connection.execute(
                    "SELECT 1 FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (
                        session_id,
                        MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                    ),
                ).fetchone()
                if active_stage is not None:
                    raise ValueError(
                        "Cannot delete a session while a model-completion stage is active: "
                        f"{session_id}"
                    )
                pending_budget_settlement = self._connection.execute(
                    """
                    SELECT identity.reservation_id
                    FROM cayu_budget_reservation_identities AS identity
                    LEFT JOIN cayu_events AS event
                      ON event.session_id = identity.publication_session_id
                     AND event.event_type IN (
                         'budget.reconciled',
                         'budget.reservation_released'
                     )
                     AND json_extract(event.payload_json, '$.reservation_id')
                         = identity.reservation_id
                    LEFT JOIN cayu_persisted_event_side_effects AS delivery
                      ON delivery.session_id = event.session_id
                     AND delivery.event_id = event.event_id
                    WHERE identity.publication_session_id = ?
                    GROUP BY identity.reservation_id
                    HAVING COUNT(event.event_id) <> 1
                        OR COUNT(
                            CASE WHEN delivery.status = 'delivered' THEN 1 END
                        ) <> 1
                    LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
                if pending_budget_settlement is not None:
                    raise ValueError(
                        "Cannot delete a session while a budget settlement audit "
                        f"event is pending: {session_id}"
                    )
                # ON DELETE CASCADE removes events/labels/checkpoint/transcript;
                # the self-FK is ON DELETE SET NULL so children keep loading.
                self._connection.execute(
                    "DELETE FROM cayu_sessions WHERE id = ?",
                    (session_id,),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    async def update_labels(self, session_id: str, labels: dict[str, str]) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        new_labels = copy_label_map(labels, "labels", allow_reserved=False)
        updated_at = self._ownership_clock()
        expected_run_epoch = _current_session_run_epoch(session_id)
        async with self._lock:
            with self._connection:
                epoch_clause = "" if expected_run_epoch is None else " AND run_epoch = ?"
                params: list[object] = [
                    sqlite_support.format_datetime(updated_at),
                    session_id,
                ]
                if expected_run_epoch is not None:
                    params.append(expected_run_epoch)
                cursor = self._connection.execute(
                    f"UPDATE cayu_sessions SET updated_at = ? WHERE id = ?{epoch_clause}",
                    params,
                )
                if cursor.rowcount != 1:
                    if expected_run_epoch is not None:
                        _raise_session_write_conflict(
                            self._connection, session_id, expected_run_epoch
                        )
                    raise KeyError(f"Session not found: {session_id}")
                self._connection.execute(
                    "DELETE FROM cayu_session_labels WHERE session_id = ?",
                    (session_id,),
                )
                if new_labels:
                    self._connection.executemany(
                        """
                        INSERT INTO cayu_session_labels (session_id, key, value)
                        VALUES (?, ?, ?)
                        """,
                        [(session_id, key, value) for key, value in new_labels.items()],
                    )
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                from cayu.runtime._invocation_lifecycle import (
                    require_invocation_lifecycle_release_capacity,
                )

                require_invocation_lifecycle_release_capacity(
                    self._load_checkpoint_unlocked(session_id),
                    loaded,
                )
            return loaded

    async def update_metadata(self, session_id: str, metadata: dict[str, Any]) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        user_metadata = copy_session_user_metadata(metadata)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                updated_at = self._ownership_clock()
                row = self._connection.execute(
                    "SELECT run_epoch, metadata_json FROM cayu_sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch_value(session_id, row["run_epoch"])
                new_metadata = replace_session_user_metadata(
                    json.loads(row["metadata_json"]),
                    user_metadata,
                )
                self._connection.execute(
                    "UPDATE cayu_sessions SET metadata_json = ?, updated_at = ? WHERE id = ?",
                    (
                        sqlite_support.json_dumps(new_metadata),
                        sqlite_support.format_datetime(updated_at),
                        session_id,
                    ),
                )
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                from cayu.runtime._invocation_lifecycle import (
                    require_invocation_lifecycle_release_capacity,
                )

                require_invocation_lifecycle_release_capacity(
                    self._load_checkpoint_unlocked(session_id),
                    loaded,
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
            return loaded

    async def transition_status(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")

        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                updated_at = self._ownership_clock()
                expected_run_epoch = _current_session_run_epoch(session_id)
                placeholders = ", ".join("?" for _ in allowed_statuses)
                params: list[object] = [
                    str(to_status),
                    sqlite_support.format_datetime(updated_at),
                    sqlite_support.format_datetime(updated_at),
                    1 if to_status == SessionStatus.RUNNING else 0,
                    session_id,
                    *[str(status) for status in allowed_statuses],
                ]
                epoch_clause = ""
                if expected_run_epoch is not None:
                    epoch_clause = " AND run_epoch = ?"
                    params.append(expected_run_epoch)
                cursor = self._connection.execute(
                    f"""
                    UPDATE cayu_sessions
                    SET status = ?, updated_at = ?, last_activity_at = ?,
                        run_epoch = run_epoch + ?
                    WHERE id = ? AND status IN ({placeholders}){epoch_clause}
                    """,
                    params,
                )
                if cursor.rowcount != 1:
                    loaded = self._load_unlocked(session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    if expected_run_epoch is not None and loaded.run_epoch != expected_run_epoch:
                        raise SessionRunFenced(
                            f"Session run epoch no longer owns {session_id}: expected "
                            f"{expected_run_epoch}, current {loaded.run_epoch}."
                        )
                    raise SessionStatusConflict(
                        f"Session status transition not allowed: {loaded.status} -> {to_status}"
                    )
                if to_status is SessionStatus.RUNNING:
                    _require_live_incomplete_recovery_claim_for_run_epoch_transfer(
                        self._load_checkpoint_unlocked(session_id),
                        now=updated_at,
                    )

                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
            if to_status == SessionStatus.RUNNING:
                _activate_session_run_fence(loaded)
            return loaded

    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_transform: CheckpointTransform | None = None,
        store_time_checkpoint_transform: StoreTimeCheckpointTransform | None = None,
        result_checkpoint_transform: CheckpointTransform | None = None,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        continued_interaction_id: str | None = None,
        defer_interaction_source: bool = False,
        model_transition: SessionModelTransition | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
        execution_profile_decision: ExecutionProfileDecision | None = None,
        adopted_runtime_identity: SessionRuntimeIdentity | None = None,
        tool_capability_ceiling: ToolCapabilityCeiling | None = None,
    ) -> Session:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        if (checkpoint_transform is None) == (store_time_checkpoint_transform is None):
            raise TypeError("Exactly one checkpoint transform is required.")
        if result_checkpoint_transform is not None and not callable(result_checkpoint_transform):
            raise TypeError("result_checkpoint_transform must be callable.")
        admission = _copy_transition_interaction_admission(
            session_id,
            interaction_started_event,
            interaction_source_messages,
            continued_interaction_id=continued_interaction_id,
            defer_interaction_source=defer_interaction_source,
        )
        prepared_model_transition = _copy_session_model_transition(
            session_id,
            model_transition,
            interaction_id=(None if admission is None else admission[1]),
            interaction_is_new=(admission is not None and admission[0] is not None),
        )
        prepared_execution_profile = _copy_optional_execution_profile(execution_profile)
        prepared_execution_profile_decision = _copy_optional_execution_profile_decision(
            execution_profile_decision
        )
        prepared_adopted_runtime_identity = (
            None
            if adopted_runtime_identity is None
            else copy_session_runtime_identity(adopted_runtime_identity)
        )
        prepared_tool_capability_ceiling = _copy_optional_tool_capability_ceiling(
            tool_capability_ceiling
        )
        if prepared_execution_profile_decision is not None and admission is None:
            raise ValueError("An execution-profile decision requires atomic interaction admission.")
        if admission is not None and to_status is not SessionStatus.RUNNING:
            raise ValueError("Interaction admission requires a transition to running.")

        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                updated_at = self._ownership_clock()
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, loaded)
                if loaded.status not in allowed_statuses:
                    raise SessionStatusConflict(
                        f"Session status transition not allowed: {loaded.status} -> {to_status}"
                    )
                transition_profile_metadata = _validate_execution_profile_admission(
                    loaded,
                    candidate_profile=prepared_execution_profile,
                    model_transition=prepared_model_transition,
                    decision=prepared_execution_profile_decision,
                )
                transition_metadata = transition_profile_metadata
                if prepared_model_transition is not None:
                    transcript_rows = self._connection.execute(
                        "SELECT message_json FROM cayu_transcript_messages "
                        "WHERE session_id = ? ORDER BY session_order ASC",
                        (session_id,),
                    ).fetchall()
                    _validate_session_model_transition(
                        loaded,
                        [
                            Message.model_validate(json.loads(row["message_json"]))
                            for row in transcript_rows
                        ],
                        _transcript_cursor(self._connection, session_id),
                        prepared_model_transition,
                    )
                    transition_metadata = _session_metadata_after_model_transition(
                        loaded,
                        prepared_model_transition,
                        execution_profile_metadata=transition_profile_metadata,
                    )
                transition_metadata = _session_metadata_after_runtime_identity_adoption(
                    loaded,
                    prepared_adopted_runtime_identity,
                    model_transition=prepared_model_transition,
                    execution_profile_metadata=transition_metadata,
                )
                transition_metadata = _session_metadata_after_tool_capability_ceiling_admission(
                    loaded,
                    prepared_tool_capability_ceiling,
                    transition_metadata=transition_metadata,
                    require_existing_ceiling=prepared_execution_profile is not None,
                )

                current_checkpoint = self._load_checkpoint_unlocked(session_id)
                if to_status is SessionStatus.RUNNING:
                    _require_live_incomplete_recovery_claim_for_run_epoch_transfer(
                        current_checkpoint,
                        now=updated_at,
                    )
                checkpoint_copy = _copy_checkpoint_for_transform(
                    current_checkpoint,
                    session_id=session_id,
                )
                if store_time_checkpoint_transform is not None:
                    transformed_checkpoint = store_time_checkpoint_transform(
                        loaded,
                        checkpoint_copy,
                        updated_at,
                    )
                else:
                    assert checkpoint_transform is not None
                    transformed_checkpoint = checkpoint_transform(
                        loaded,
                        checkpoint_copy,
                    )
                if transformed_checkpoint is not None:
                    transformed_checkpoint = _checkpoint_transform_result_preserving_completion_result_event_publications(
                        current_checkpoint,
                        transformed_checkpoint,
                        session_id=session_id,
                    )

                placeholders = ", ".join("?" for _ in allowed_statuses)
                transition_values = (
                    str(to_status),
                    sqlite_support.format_datetime(updated_at),
                    sqlite_support.format_datetime(updated_at),
                    1 if to_status == SessionStatus.RUNNING else 0,
                )
                if prepared_model_transition is None and transition_metadata is None:
                    cursor = self._connection.execute(
                        f"""
                        UPDATE cayu_sessions
                        SET status = ?, updated_at = ?, last_activity_at = ?,
                            run_epoch = run_epoch + ?
                        WHERE id = ? AND status IN ({placeholders})
                        """,
                        (
                            *transition_values,
                            session_id,
                            *(str(status) for status in allowed_statuses),
                        ),
                    )
                elif prepared_adopted_runtime_identity is not None:
                    target_provider_name = (
                        loaded.provider_name
                        if prepared_model_transition is None
                        else prepared_model_transition.target.provider_name
                    )
                    target_model = (
                        loaded.model
                        if prepared_model_transition is None
                        else prepared_model_transition.target.model
                    )
                    cursor = self._connection.execute(
                        f"""
                        UPDATE cayu_sessions
                        SET status = ?, updated_at = ?, last_activity_at = ?,
                            run_epoch = run_epoch + ?, provider_name = ?, model = ?,
                            runtime_name = ?, runtime_version = ?, metadata_json = ?
                        WHERE id = ? AND status IN ({placeholders})
                        """,
                        (
                            *transition_values,
                            target_provider_name,
                            target_model,
                            prepared_adopted_runtime_identity.runtime_name,
                            prepared_adopted_runtime_identity.runtime_version,
                            sqlite_support.json_dumps(transition_metadata),
                            session_id,
                            *(str(status) for status in allowed_statuses),
                        ),
                    )
                elif prepared_model_transition is not None:
                    cursor = self._connection.execute(
                        f"""
                        UPDATE cayu_sessions
                        SET status = ?, updated_at = ?, last_activity_at = ?,
                            run_epoch = run_epoch + ?, provider_name = ?, model = ?,
                            metadata_json = ?
                        WHERE id = ? AND status IN ({placeholders})
                        """,
                        (
                            *transition_values,
                            prepared_model_transition.target.provider_name,
                            prepared_model_transition.target.model,
                            sqlite_support.json_dumps(transition_metadata),
                            session_id,
                            *(str(status) for status in allowed_statuses),
                        ),
                    )
                else:
                    cursor = self._connection.execute(
                        f"""
                        UPDATE cayu_sessions
                        SET status = ?, updated_at = ?, last_activity_at = ?,
                            run_epoch = run_epoch + ?, metadata_json = ?
                        WHERE id = ? AND status IN ({placeholders})
                        """,
                        (
                            *transition_values,
                            sqlite_support.json_dumps(transition_metadata),
                            session_id,
                            *(str(status) for status in allowed_statuses),
                        ),
                    )
                if cursor.rowcount != 1:
                    current = self._load_unlocked(session_id)
                    if current is None:
                        raise KeyError(f"Session not found: {session_id}")
                    raise SessionStatusConflict(
                        f"Session status transition not allowed: {current.status} -> {to_status}"
                    )
                transition_updates: dict[str, Any] = {
                    "status": to_status,
                    "updated_at": updated_at,
                    "last_activity_at": updated_at,
                    "run_epoch": loaded.run_epoch + (to_status == SessionStatus.RUNNING),
                }
                if prepared_model_transition is not None:
                    transition_updates.update(
                        provider_name=prepared_model_transition.target.provider_name,
                        model=prepared_model_transition.target.model,
                        metadata=transition_metadata,
                    )
                elif transition_metadata is not None:
                    transition_updates["metadata"] = transition_metadata
                if prepared_adopted_runtime_identity is not None:
                    transition_updates.update(
                        runtime_name=prepared_adopted_runtime_identity.runtime_name,
                        runtime_version=prepared_adopted_runtime_identity.runtime_version,
                        metadata=transition_metadata,
                    )
                transitioned = loaded.model_copy(update=transition_updates)
                if result_checkpoint_transform is not None:
                    result_checkpoint = result_checkpoint_transform(
                        transitioned,
                        _copy_checkpoint_for_transform(
                            transformed_checkpoint,
                            session_id=session_id,
                        ),
                    )
                    if result_checkpoint is None:
                        raise ValueError("Result checkpoint transform must return a checkpoint.")
                    transformed_checkpoint = _checkpoint_transform_result_preserving_completion_result_event_publications(
                        transformed_checkpoint,
                        result_checkpoint,
                        session_id=session_id,
                    )
                if transformed_checkpoint is not None:
                    self._connection.execute(
                        """
                        INSERT INTO cayu_checkpoints (
                            session_id, state_json, updated_at,
                            pending_action_source_bytes,
                            pending_action_tool_call_count,
                            pending_action_flags,
                            pending_action_metrics_ready
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(session_id) DO UPDATE SET
                            state_json = excluded.state_json,
                            updated_at = excluded.updated_at,
                            pending_action_source_bytes = excluded.pending_action_source_bytes,
                            pending_action_tool_call_count = excluded.pending_action_tool_call_count,
                            pending_action_flags = excluded.pending_action_flags,
                            pending_action_metrics_ready = excluded.pending_action_metrics_ready
                        """,
                        sqlite_support.checkpoint_row_values(
                            session_id, transformed_checkpoint, updated_at
                        ),
                    )
                if admission is not None:
                    started_event, interaction_id, source_messages, defer_source = admission
                    existing_deferred = self._connection.execute(
                        "SELECT interaction_id FROM cayu_deferred_interaction_inputs "
                        "WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()
                    if existing_deferred is not None and (
                        not defer_source or existing_deferred["interaction_id"] != interaction_id
                    ):
                        raise RuntimeError("Session already has deferred interaction input.")
                    admission_events = []
                    if prepared_execution_profile_decision is not None:
                        admission_events.append(prepared_execution_profile_decision.event)
                    if prepared_model_transition is not None:
                        admission_events.append(prepared_model_transition.event)
                    if started_event is not None:
                        admission_events.append(started_event)
                    for admission_event in admission_events:
                        lookup_key, projection, projection_bytes = (
                            pending_action_event_storage_values(admission_event)
                        )
                        self._connection.execute(
                            """
                            INSERT INTO cayu_events (
                                session_id, event_id, interaction_id, event_type,
                                timestamp, agent_name, environment_name, workflow_name,
                                tool_name, payload_json, pending_action_lookup_key,
                                pending_action_projection_json,
                                pending_action_projection_bytes
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                session_id,
                                admission_event.id,
                                admission_event.interaction_id,
                                str(admission_event.type),
                                sqlite_support.format_datetime(admission_event.timestamp),
                                admission_event.agent_name,
                                admission_event.environment_name,
                                admission_event.workflow_name,
                                admission_event.tool_name,
                                sqlite_support.json_dumps(admission_event.payload),
                                lookup_key,
                                projection,
                                projection_bytes,
                            ),
                        )
                    if admission_events:
                        _enqueue_persisted_event_side_effects(
                            self._connection, session_id, admission_events
                        )
                    if defer_source:
                        deferred_input = DeferredInteractionInput(
                            interaction_id=interaction_id,
                            source_messages=source_messages,
                        )
                        self._connection.execute(
                            "INSERT INTO cayu_deferred_interaction_inputs "
                            "(session_id, interaction_id, source_messages_json) "
                            "VALUES (?, ?, ?) "
                            "ON CONFLICT(session_id) DO UPDATE SET "
                            "interaction_id = excluded.interaction_id, "
                            "source_messages_json = excluded.source_messages_json",
                            (
                                session_id,
                                interaction_id,
                                sqlite_support.json_dumps(
                                    deferred_interaction_input_storage_payload(deferred_input)
                                ),
                            ),
                        )
                    else:
                        self._connection.executemany(
                            "INSERT INTO cayu_transcript_messages "
                            "(session_id, role, interaction_id, message_json, "
                            "transcript_search_document) VALUES (?, ?, ?, ?, ?)",
                            [
                                (
                                    session_id,
                                    str(message.role),
                                    interaction_id,
                                    sqlite_support.json_dumps(message.model_dump(mode="json")),
                                    transcript_search_document(message),
                                )
                                for message in source_messages
                            ],
                        )
                self._connection.commit()
            except sqlite3.IntegrityError as exc:
                transaction_failure = sqlite_support._settle_failed_transaction(
                    self._connection,
                    exc,
                )
                if transaction_failure is not exc:
                    raise transaction_failure from None
                existing_event_id = (
                    None
                    if admission is None
                    else _first_existing_event_id(
                        self._connection,
                        session_id,
                        [
                            *(
                                [prepared_execution_profile_decision.event.id]
                                if prepared_execution_profile_decision is not None
                                else []
                            ),
                            *(
                                [prepared_model_transition.event.id]
                                if prepared_model_transition is not None
                                else []
                            ),
                            *([admission[0].id] if admission[0] is not None else []),
                        ],
                    )
                )
                if existing_event_id is not None:
                    raise ValueError(
                        f"Event already exists for session {session_id}: {existing_event_id}"
                    ) from exc
                raise
            except BaseException as primary:
                transaction_failure = sqlite_support._settle_failed_transaction(
                    self._connection,
                    primary,
                )
                if transaction_failure is not primary:
                    raise transaction_failure from None
                raise

            if to_status == SessionStatus.RUNNING:
                _activate_session_run_fence(transitioned)
            return transitioned

    async def reject_execution_profile_resume(
        self,
        session_id: str,
        *,
        expected_session_instance_id: str | None = None,
        expected_statuses: set[SessionStatus],
        expected_run_epoch: int,
        expected_profile: ExecutionProfileIdentity,
        candidate_profile: ExecutionProfileIdentity,
        event: Event,
        decision: ExecutionProfileDecision | None = None,
        expected_active_invocation_profile_authority: CheckpointValueAuthority | None = None,
    ) -> ExecutionProfileRejectionResult:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        (
            session_id,
            statuses,
            expected_run_epoch,
            expected_profile,
            _candidate_profile,
            copied_event,
        ) = _prepare_execution_profile_rejection(
            session_id,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_profile=expected_profile,
            candidate_profile=candidate_profile,
            event=event,
            decision=decision,
        )
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                session = self._load_unlocked(session_id)
                if session is None:
                    raise KeyError(f"Session not found: {session_id}")
                _validate_execution_profile_rejection_session(
                    session,
                    checkpoint=self._load_checkpoint_unlocked(session_id),
                    expected_session_instance_id=expected_session_instance_id,
                    expected_statuses=statuses,
                    expected_run_epoch=expected_run_epoch,
                    expected_profile=expected_profile,
                    event=copied_event,
                    expected_active_invocation_profile_authority=(
                        expected_active_invocation_profile_authority
                    ),
                )
                existing_row = self._connection.execute(
                    "SELECT * FROM cayu_events WHERE session_id = ? AND event_id = ?",
                    (session_id, copied_event.id),
                ).fetchone()
                if existing_row is not None:
                    existing = _event_from_row(existing_row)
                    if not _execution_profile_rejection_events_equivalent(
                        existing,
                        copied_event,
                    ):
                        raise ValueError(
                            f"Execution-profile rejection id was reused: {copied_event.id}"
                        )
                    self._connection.commit()
                    return ExecutionProfileRejectionResult(event=existing, replayed=True)

                lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                    copied_event
                )
                _touch_session_activity(self._connection, session_id, self._ownership_clock())
                self._connection.execute(
                    """
                    INSERT INTO cayu_events (
                        session_id, event_id, interaction_id, event_type,
                        timestamp, agent_name, environment_name, workflow_name,
                        tool_name, payload_json, pending_action_lookup_key,
                        pending_action_projection_json, pending_action_projection_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        copied_event.id,
                        copied_event.interaction_id,
                        str(copied_event.type),
                        sqlite_support.format_datetime(copied_event.timestamp),
                        copied_event.agent_name,
                        copied_event.environment_name,
                        copied_event.workflow_name,
                        copied_event.tool_name,
                        sqlite_support.json_dumps(copied_event.payload),
                        lookup_key,
                        projection,
                        projection_bytes,
                    ),
                )
                _enqueue_persisted_event_side_effects(
                    self._connection,
                    session_id,
                    [copied_event],
                )
                self._connection.commit()
                return ExecutionProfileRejectionResult(event=copied_event, replayed=False)
            except Exception:
                self._connection.rollback()
                raise

    async def fence_stalled_run(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        inactive_for_seconds: int,
    ) -> Session | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(statuses, "statuses")
        validated_inactive_for_seconds = _validate_inactive_for_seconds(inactive_for_seconds)
        assert validated_inactive_for_seconds is not None
        placeholders = ", ".join("?" for _ in allowed_statuses)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                now = self._ownership_clock()
                inactive_before = utc_duration_cutoff(
                    now,
                    validated_inactive_for_seconds,
                )
                if not self._session_exists_unlocked(session_id):
                    raise KeyError(f"Session not found: {session_id}")
                current_checkpoint = self._load_checkpoint_unlocked(session_id)
                if (
                    active_provider_operation_cancellation_claim_from_checkpoint(
                        current_checkpoint,
                        now=now,
                    )
                    is not None
                    or _incomplete_recovery_claim_from_checkpoint(current_checkpoint) is not None
                ):
                    self._connection.rollback()
                    return None
                if inactive_before is None:
                    self._connection.commit()
                    return None
                cursor = self._connection.execute(
                    f"""
                    UPDATE cayu_sessions
                    SET run_epoch = run_epoch + 1, last_activity_at = ?
                    WHERE id = ? AND status IN ({placeholders}) AND last_activity_at <= ?
                    """,
                    (
                        sqlite_support.format_datetime(now),
                        session_id,
                        *(str(status) for status in allowed_statuses),
                        sqlite_support.format_datetime(inactive_before),
                    ),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
            if cursor.rowcount != 1:
                return None
            loaded = self._load_unlocked(session_id)
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            _activate_session_run_fence(loaded)
            return loaded

    async def reserve_stalled_run_recovery(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        inactive_for_seconds: int | None,
        checkpoint_transform: StoreTimeCheckpointTransform,
    ) -> Session | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(statuses, "statuses")
        inactive_for_seconds = _validate_inactive_for_seconds(inactive_for_seconds)
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                now = self._ownership_clock()
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                current = self._load_checkpoint_unlocked(session_id)
                inactive_before = (
                    None
                    if inactive_for_seconds is None
                    else utc_duration_cutoff(now, inactive_for_seconds)
                )
                if (
                    loaded.status not in allowed_statuses
                    or (
                        inactive_for_seconds is not None
                        and (inactive_before is None or loaded.last_activity_at > inactive_before)
                    )
                    or active_provider_operation_cancellation_claim_from_checkpoint(
                        current,
                        now=now,
                    )
                    is not None
                ):
                    self._connection.commit()
                    return None
                transformed = checkpoint_transform(
                    loaded,
                    _copy_checkpoint_for_transform(current, session_id=session_id),
                    now,
                )
                if transformed is None:
                    self._connection.commit()
                    return None
                transformed = (
                    _checkpoint_transform_result_preserving_completion_result_event_publications(
                        current,
                        transformed,
                        session_id=session_id,
                    )
                )
                self._connection.execute(
                    """
                    INSERT INTO cayu_checkpoints (
                        session_id, state_json, updated_at,
                        pending_action_source_bytes,
                        pending_action_tool_call_count,
                        pending_action_flags,
                        pending_action_metrics_ready
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        state_json = excluded.state_json,
                        updated_at = excluded.updated_at,
                        pending_action_source_bytes = excluded.pending_action_source_bytes,
                        pending_action_tool_call_count = excluded.pending_action_tool_call_count,
                        pending_action_flags = excluded.pending_action_flags,
                        pending_action_metrics_ready = excluded.pending_action_metrics_ready
                    """,
                    sqlite_support.checkpoint_row_values(session_id, transformed, now),
                )
                self._connection.commit()
                return loaded
            except BaseException:
                self._connection.rollback()
                raise

    async def fence_run_and_transform_checkpoint(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        checkpoint_transform: CheckpointTransform,
        result_checkpoint_transform: CheckpointTransform | None = None,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(statuses, "statuses")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        if result_checkpoint_transform is not None and not callable(result_checkpoint_transform):
            raise TypeError("result_checkpoint_transform must be callable.")
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                updated_at = self._ownership_clock()
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                if loaded.status not in allowed_statuses:
                    raise SessionStatusConflict(f"Session status cannot be fenced: {loaded.status}")
                current_checkpoint = self._load_checkpoint_unlocked(session_id)
                _require_live_incomplete_recovery_claim_for_run_epoch_transfer(
                    current_checkpoint,
                    now=updated_at,
                )
                if (
                    active_provider_operation_cancellation_claim_from_checkpoint(
                        current_checkpoint,
                        now=updated_at,
                    )
                    is not None
                ):
                    raise SessionStatusConflict(
                        "Provider-operation cancellation still owns the session run epoch."
                    )
                transformed = checkpoint_transform(
                    loaded,
                    _copy_checkpoint_for_transform(
                        current_checkpoint,
                        session_id=session_id,
                    ),
                )
                if transformed is None:
                    raise ValueError("Fenced checkpoint transform must return a checkpoint.")
                transformed = (
                    _checkpoint_transform_result_preserving_completion_result_event_publications(
                        current_checkpoint,
                        transformed,
                        session_id=session_id,
                    )
                )
                self._connection.execute(
                    "UPDATE cayu_sessions SET run_epoch = run_epoch + 1, "
                    "last_activity_at = ? WHERE id = ?",
                    (sqlite_support.format_datetime(updated_at), session_id),
                )
                fenced = loaded.model_copy(
                    update={
                        "run_epoch": loaded.run_epoch + 1,
                        "last_activity_at": updated_at,
                    }
                )
                if result_checkpoint_transform is not None:
                    result_checkpoint = result_checkpoint_transform(
                        fenced,
                        _copy_checkpoint_for_transform(
                            transformed,
                            session_id=session_id,
                        ),
                    )
                    if result_checkpoint is None:
                        raise ValueError("Result checkpoint transform must return a checkpoint.")
                    transformed = _checkpoint_transform_result_preserving_completion_result_event_publications(
                        transformed,
                        result_checkpoint,
                        session_id=session_id,
                    )
                self._connection.execute(
                    """
                    INSERT INTO cayu_checkpoints (
                        session_id, state_json, updated_at,
                        pending_action_source_bytes,
                        pending_action_tool_call_count,
                        pending_action_flags,
                        pending_action_metrics_ready
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        state_json = excluded.state_json,
                        updated_at = excluded.updated_at,
                        pending_action_source_bytes = excluded.pending_action_source_bytes,
                        pending_action_tool_call_count = excluded.pending_action_tool_call_count,
                        pending_action_flags = excluded.pending_action_flags,
                        pending_action_metrics_ready = excluded.pending_action_metrics_ready
                    """,
                    sqlite_support.checkpoint_row_values(
                        session_id,
                        transformed,
                        updated_at,
                    ),
                )
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
            _activate_session_run_fence(fenced)
            return fenced

    async def transition_status_if_no_queued_messages(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                updated_at = self._ownership_clock()
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, loaded)
                if loaded.status not in allowed_statuses:
                    raise SessionStatusConflict(
                        f"Session status transition not allowed: {loaded.status} -> {to_status}"
                    )
                pending = self._connection.execute(
                    "SELECT 1 FROM cayu_session_message_queue "
                    "WHERE session_id = ? AND status = 'queued' LIMIT 1",
                    (session_id,),
                ).fetchone()
                if pending is not None:
                    raise SessionQueuedMessagesPending(
                        f"Session has durable queued messages: {session_id}"
                    )
                cursor = self._connection.execute(
                    "UPDATE cayu_sessions SET status = ?, updated_at = ?, "
                    "last_activity_at = ?, run_epoch = run_epoch + ? WHERE id = ?",
                    (
                        str(to_status),
                        sqlite_support.format_datetime(updated_at),
                        sqlite_support.format_datetime(updated_at),
                        1 if to_status == SessionStatus.RUNNING else 0,
                        session_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"Session not found: {session_id}")
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
            transitioned = loaded.model_copy(
                update={
                    "status": to_status,
                    "updated_at": updated_at,
                    "last_activity_at": updated_at,
                    "run_epoch": loaded.run_epoch + (to_status == SessionStatus.RUNNING),
                }
            )
            if to_status == SessionStatus.RUNNING:
                _activate_session_run_fence(transitioned)
            return transitioned

    async def publish_interaction_transition(
        self,
        session_id: str,
        *,
        event: Event,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        only_if_no_queued_messages: bool = False,
        model_completion_stage_settlement: ModelCompletionStageSettlementRequest | None = None,
        checkpoint_mutation: dict[str, Any] | None = None,
        expected_session_instance_id: str | None = None,
        expected_active_invocation_profile: ActiveInvocationExecutionProfile | None = None,
        expected_invocation_authority_state: Literal["active", "released"] = "active",
        expected_recovery_claim_id: str | None = None,
    ) -> InteractionTransitionResult:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        expected_invocation_authority_state = (
            _validate_interaction_transition_invocation_authority_parameters(
                expected_session_instance_id=expected_session_instance_id,
                expected_active_invocation_profile=expected_active_invocation_profile,
                expected_invocation_authority_state=expected_invocation_authority_state,
            )
        )
        expected_recovery_claim_id = _validate_interaction_transition_recovery_claim_id(
            expected_recovery_claim_id
        )

        session_id, transition = _prepare_interaction_transition(
            session_id,
            event=event,
            from_statuses=from_statuses,
            to_status=to_status,
            only_if_no_queued_messages=only_if_no_queued_messages,
            model_completion_stage_settlement=model_completion_stage_settlement,
            checkpoint_mutation=checkpoint_mutation,
        )
        copied_event = transition.event
        allowed_statuses = set(transition.from_statuses)
        target_status = transition.to_status
        conditional = transition.only_if_no_queued_messages
        settlement_request = transition.model_completion_stage_settlement
        checkpoint_mutation_request = transition.checkpoint_mutation
        receipt_storage_key = _interaction_transition_storage_key(copied_event.id)

        def statement(connection: sqlite3.Connection) -> InteractionTransitionResult:
            try:
                connection.execute("BEGIN IMMEDIATE")
                loaded = _load_session(connection, session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                if expected_active_invocation_profile is None:
                    _assert_session_run_epoch(session_id, loaded)
                receipt_row = connection.execute(
                    "SELECT record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, receipt_storage_key),
                ).fetchone()
                existing_row = connection.execute(
                    f"SELECT {', '.join(_EVENT_COLUMN_NAMES)} FROM cayu_events "
                    "WHERE session_id = ? AND event_id = ?",
                    (session_id, copied_event.id),
                ).fetchone()
                if receipt_row is not None:
                    receipt = _reconstruct_interaction_transition_receipt(
                        copy_durable_json_object(
                            json.loads(receipt_row["record_json"]),
                            "interaction transition receipt",
                        ),
                        transition=transition,
                    )
                    _validate_interaction_transition_receipt_authority(
                        receipt,
                        current_session=loaded,
                        current_checkpoint=_load_checkpoint_state(connection, session_id),
                        expected_session_instance_id=expected_session_instance_id,
                        expected_active_invocation_profile=expected_active_invocation_profile,
                        expected_invocation_authority_state=(expected_invocation_authority_state),
                        expected_recovery_claim_id=expected_recovery_claim_id,
                    )
                    if existing_row is not None and _event_from_row(existing_row) != receipt.event:
                        raise RuntimeError(
                            "Interaction transition receipt conflicts with retained event history."
                        )
                    connection.commit()
                    return InteractionTransitionResult(
                        session=receipt.session,
                        event=receipt.event,
                        status_changed=receipt.status_changed,
                        replayed=True,
                    )
                if existing_row is not None:
                    raise RuntimeError(
                        "Interaction transition event exists without its immutable receipt."
                    )
                active_recovery_claim_id = _active_unexpired_incomplete_recovery_claim_id(
                    _load_checkpoint_state(connection, session_id),
                    now=self._ownership_clock(),
                )
                if expected_recovery_claim_id is None:
                    if active_recovery_claim_id is not None:
                        raise SessionRunFenced(
                            "Interaction transition is owned by another terminal recovery claim."
                        )
                elif active_recovery_claim_id != expected_recovery_claim_id:
                    raise SessionRunFenced(
                        "Interaction transition lost its exact terminal recovery claim."
                    )
                if expected_active_invocation_profile is not None:
                    from cayu.runtime._invocation_lifecycle import (
                        require_invocation_command_authority,
                        require_released_invocation_command_authority,
                    )

                    assert expected_session_instance_id is not None
                    checkpoint = _load_checkpoint_state(connection, session_id)
                    if expected_invocation_authority_state == "released":
                        require_released_invocation_command_authority(
                            loaded,
                            checkpoint,
                            session_id=session_id,
                            session_instance_id=expected_session_instance_id,
                            active_profile=expected_active_invocation_profile,
                            events=(copied_event,),
                        )
                    else:
                        require_invocation_command_authority(
                            loaded,
                            checkpoint,
                            session_id=session_id,
                            session_instance_id=expected_session_instance_id,
                            run_epochs=frozenset({expected_active_invocation_profile.run_epoch}),
                            active_profile=expected_active_invocation_profile,
                            events=(copied_event,),
                        )
                if loaded.status not in allowed_statuses:
                    raise SessionStatusConflict(
                        f"Session status transition not allowed: {loaded.status} -> {target_status}"
                    )
                queued = False
                if conditional:
                    queued = (
                        connection.execute(
                            "SELECT 1 FROM cayu_session_message_queue "
                            "WHERE session_id = ? AND status = 'queued' LIMIT 1",
                            (session_id,),
                        ).fetchone()
                        is not None
                    )
                updated_at = self._ownership_clock()
                formatted_updated_at = sqlite_support.format_datetime(updated_at)
                settlement_record = None
                settlement_storage_key = None
                if settlement_request is not None:
                    active_row = connection.execute(
                        "SELECT record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key = ?",
                        (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                    ).fetchone()
                    if active_row is None:
                        raise SessionModelCompletionStageConflict(
                            "The interaction transition has no active model-completion "
                            "stage to settle."
                        )
                    active_record = _decode_model_completion_stage_record(active_row["record_json"])
                    marker = _reconstruct_active_model_completion_stage_record(
                        active_record,
                        session_id=session_id,
                    )
                    _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
                        session_id, marker.stage_id
                    )
                    stage_rows = connection.execute(
                        "SELECT idempotency_key, record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key IN (?, ?)",
                        (session_id, preparation_key, terminal_key),
                    ).fetchall()
                    stage_records = {
                        row["idempotency_key"]: _decode_model_completion_stage_record(
                            row["record_json"]
                        )
                        for row in stage_rows
                    }
                    active = _reconstruct_active_model_completion_stage(
                        active_record,
                        stage_records.get(preparation_key),
                        stage_records.get(terminal_key),
                        session_id=session_id,
                    )
                    if active is None:
                        raise SessionModelCompletionStageConflict(
                            "The active model-completion stage disappeared during settlement."
                        )
                    stage = active.stage
                    settlement_storage_key = _model_completion_stage_settlement_storage_key(
                        stage.stage_id
                    )
                    related_keys = (
                        settlement_storage_key,
                        _model_completion_stage_winner_storage_key(stage.logical_step_id),
                        _runtime_publication_storage_key(stage.logical_step_id),
                    )
                    related_rows = connection.execute(
                        "SELECT idempotency_key, record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key IN (?, ?, ?)",
                        (session_id, *related_keys),
                    ).fetchall()
                    related_records = {
                        row["idempotency_key"]: _decode_model_completion_stage_record(
                            row["record_json"]
                        )
                        for row in related_rows
                    }
                    _validate_model_completion_stage_for_settlement(
                        session=loaded,
                        stage=stage,
                        active=active,
                        request=settlement_request,
                        settlement_record=related_records.get(settlement_storage_key),
                        winner_exists=related_keys[1] in related_records,
                        receipt_exists=related_keys[2] in related_records,
                    )
                    settlement_record = _model_completion_stage_settlement_record(
                        stage,
                        request=settlement_request,
                        settled_at=updated_at,
                    )
                if queued:
                    _touch_session_activity(connection, session_id, updated_at)
                else:
                    connection.execute(
                        "UPDATE cayu_sessions SET status = ?, updated_at = ?, "
                        "last_activity_at = ? WHERE id = ?",
                        (
                            str(target_status),
                            formatted_updated_at,
                            formatted_updated_at,
                            session_id,
                        ),
                    )
                    if checkpoint_mutation_request is not None:
                        transformed_checkpoint = _apply_runtime_publication_checkpoint_mutation(
                            RuntimePublicationMutation.model_validate(checkpoint_mutation_request),
                            _load_checkpoint_state(connection, session_id),
                        )
                        if transformed_checkpoint is None:
                            raise AssertionError(
                                "Interaction checkpoint mutation deleted its checkpoint."
                            )
                        connection.execute(
                            """
                            INSERT INTO cayu_checkpoints (
                                session_id, state_json, updated_at,
                                pending_action_source_bytes,
                                pending_action_tool_call_count,
                                pending_action_flags,
                                pending_action_metrics_ready
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(session_id) DO UPDATE SET
                                state_json = excluded.state_json,
                                updated_at = excluded.updated_at,
                                pending_action_source_bytes = excluded.pending_action_source_bytes,
                                pending_action_tool_call_count = excluded.pending_action_tool_call_count,
                                pending_action_flags = excluded.pending_action_flags,
                                pending_action_metrics_ready = excluded.pending_action_metrics_ready
                            """,
                            sqlite_support.checkpoint_row_values(
                                session_id,
                                transformed_checkpoint,
                                updated_at,
                            ),
                        )
                if settlement_record is not None and settlement_storage_key is not None:
                    connection.execute(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record_json, updated_at) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            session_id,
                            settlement_storage_key,
                            sqlite_support.json_dumps(settlement_record),
                            formatted_updated_at,
                        ),
                    )
                    deleted = connection.execute(
                        "DELETE FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key = ?",
                        (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                    )
                    if deleted.rowcount != 1:
                        raise SessionModelCompletionStageConflict(
                            "The active model-completion stage changed during settlement."
                        )
                lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                    copied_event
                )
                connection.execute(
                    """
                    INSERT INTO cayu_events (
                        session_id, event_id, interaction_id, event_type, timestamp,
                        agent_name, environment_name, workflow_name, tool_name,
                        payload_json, pending_action_lookup_key,
                        pending_action_projection_json, pending_action_projection_bytes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        copied_event.id,
                        copied_event.interaction_id,
                        str(copied_event.type),
                        sqlite_support.format_datetime(copied_event.timestamp),
                        copied_event.agent_name,
                        copied_event.environment_name,
                        copied_event.workflow_name,
                        copied_event.tool_name,
                        sqlite_support.json_dumps(copied_event.payload),
                        lookup_key,
                        projection,
                        projection_bytes,
                    ),
                )
                _enqueue_persisted_event_side_effects(
                    connection,
                    session_id,
                    [copied_event],
                )
                transitioned = _load_session(connection, session_id)
                if transitioned is None:
                    raise KeyError(f"Session not found: {session_id}")
                receipt_record = _interaction_transition_receipt_record(
                    session=transitioned,
                    event=copied_event,
                    from_statuses=allowed_statuses,
                    to_status=target_status,
                    only_if_no_queued_messages=conditional,
                    model_completion_stage_settlement=settlement_request,
                    checkpoint_mutation=checkpoint_mutation_request,
                    status_changed=not queued,
                    invocation_session_instance_id=expected_session_instance_id,
                    invocation_active_profile=expected_active_invocation_profile,
                    invocation_authority_state=expected_invocation_authority_state,
                    recovery_claim_id=expected_recovery_claim_id,
                )
                connection.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record_json, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        session_id,
                        receipt_storage_key,
                        sqlite_support.json_dumps(receipt_record),
                        formatted_updated_at,
                    ),
                )
                connection.commit()
                return InteractionTransitionResult(
                    session=transitioned,
                    event=copied_event,
                    status_changed=not queued,
                )
            except BaseException as primary:
                transaction_failure = sqlite_support._settle_failed_transaction(
                    connection,
                    primary,
                )
                if transaction_failure is not primary:
                    raise transaction_failure from None
                raise

        return await self._run_write(statement)

    async def settle_session_invocation(self, command: Any) -> InteractionTransitionResult:
        from cayu.runtime._invocation_lifecycle import (
            SettleInvocationCommand,
            copy_invocation_lifecycle_command,
        )

        copied = copy_invocation_lifecycle_command(command)
        if type(copied) is not SettleInvocationCommand:
            raise TypeError("command must be a SettleInvocationCommand.")
        transition = copied.transition
        kwargs: dict[str, Any] = {
            "event": transition.event,
            "from_statuses": set(transition.from_statuses),
            "to_status": transition.to_status,
            "only_if_no_queued_messages": transition.only_if_no_queued_messages,
            "model_completion_stage_settlement": transition.model_completion_stage_settlement,
            "expected_session_instance_id": copied.expected_session_instance_id,
            "expected_active_invocation_profile": copied.expected_active_profile,
            "expected_invocation_authority_state": copied.expected_authority_state,
        }
        if transition.checkpoint_mutation is not None:
            kwargs["checkpoint_mutation"] = transition.checkpoint_mutation
        return await self.publish_interaction_transition(copied.session_id, **kwargs)

    async def load_interaction_transition_receipt(
        self,
        session_id: str,
        *,
        transition: InteractionTransitionSpec,
        expected_recovery_claim_id: str | None = None,
    ) -> InteractionTransitionReceiptResult | None:
        session_id, copied_transition = _prepare_interaction_transition_receipt_lookup(
            session_id,
            transition=transition,
        )
        copied_event = copied_transition.event
        receipt_storage_key = _interaction_transition_storage_key(copied_event.id)

        def statement(
            connection: sqlite3.Connection,
        ) -> InteractionTransitionReceiptResult | None:
            selected_event_columns = ", ".join(
                f"retained.{column} AS {column}" for column in _EVENT_COLUMN_NAMES
            )
            row = connection.execute(
                f"SELECT operation.record_json AS receipt_record_json, "
                f"{selected_event_columns} "
                "FROM cayu_sessions AS session "
                "LEFT JOIN cayu_session_operations AS operation "
                "ON operation.session_id = session.id AND operation.idempotency_key = ? "
                "LEFT JOIN cayu_events AS retained "
                "ON retained.session_id = session.id AND retained.event_id = ? "
                "WHERE session.id = ?",
                (receipt_storage_key, copied_event.id, session_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"Session not found: {session_id}")
            receipt_record_json = row["receipt_record_json"]
            retained_event_exists = row["event_id"] is not None
            if receipt_record_json is None:
                if retained_event_exists:
                    raise RuntimeError(
                        "Interaction transition event exists without its immutable receipt."
                    )
                return None
            receipt = _reconstruct_interaction_transition_receipt(
                copy_durable_json_object(
                    json.loads(receipt_record_json),
                    "interaction transition receipt",
                ),
                transition=copied_transition,
            )
            _validate_interaction_transition_receipt_recovery_authority(
                receipt,
                current_checkpoint=_load_checkpoint_state(connection, session_id),
                expected_recovery_claim_id=expected_recovery_claim_id,
            )
            if retained_event_exists and _event_from_row(row) != receipt.event:
                raise RuntimeError(
                    "Interaction transition receipt conflicts with retained event history."
                )
            return InteractionTransitionReceiptResult(
                session=receipt.session,
                transition=_interaction_transition_spec_from_receipt(receipt),
                status_changed=receipt.status_changed,
            )

        return await self._run_read(statement)

    async def _load_interaction_transition_receipt_by_event_id(
        self,
        session_id: str,
        *,
        event_id: str,
        expected_session_instance_id: str,
        expected_active_invocation_profile: ActiveInvocationExecutionProfile,
    ) -> InteractionTransitionReceiptResult | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        event_id = require_clean_nonblank(event_id, "event_id")
        receipt_storage_key = _interaction_transition_storage_key(event_id)

        def statement(
            connection: sqlite3.Connection,
        ) -> InteractionTransitionReceiptResult | None:
            selected_event_columns = ", ".join(
                f"retained.{column} AS {column}" for column in _EVENT_COLUMN_NAMES
            )
            row = connection.execute(
                f"SELECT operation.record_json AS receipt_record_json, "
                f"{selected_event_columns} "
                "FROM cayu_sessions AS session "
                "LEFT JOIN cayu_session_operations AS operation "
                "ON operation.session_id = session.id AND operation.idempotency_key = ? "
                "LEFT JOIN cayu_events AS retained "
                "ON retained.session_id = session.id AND retained.event_id = ? "
                "WHERE session.id = ?",
                (receipt_storage_key, event_id, session_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"Session not found: {session_id}")
            receipt_record_json = row["receipt_record_json"]
            retained_event_exists = row["event_id"] is not None
            if receipt_record_json is None:
                if retained_event_exists:
                    raise RuntimeError(
                        "Interaction transition event exists without its immutable receipt."
                    )
                return None
            receipt = _load_interaction_transition_receipt(
                copy_durable_json_object(
                    json.loads(receipt_record_json),
                    "interaction transition receipt",
                )
            )
            current_session = _load_session(connection, session_id)
            if current_session is None:  # pragma: no cover - selected above
                raise KeyError(f"Session not found: {session_id}")
            _validate_invocation_release_settlement_receipt_authority(
                receipt,
                current_session=current_session,
                expected_session_instance_id=expected_session_instance_id,
                expected_active_invocation_profile=expected_active_invocation_profile,
            )
            if receipt.event.id != event_id:
                raise RuntimeError("Interaction transition receipt has a conflicting event ID.")
            if retained_event_exists and _event_from_row(row) != receipt.event:
                raise RuntimeError(
                    "Interaction transition receipt conflicts with retained event history."
                )
            return InteractionTransitionReceiptResult(
                session=receipt.session,
                transition=_interaction_transition_spec_from_receipt(receipt),
                status_changed=receipt.status_changed,
            )

        return await self._run_read(statement)

    async def release_run_fence(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        expected_run_epoch = _current_session_run_epoch(session_id)
        if expected_run_epoch is None:
            _deactivate_session_interaction(session_id)
            return
        try:
            async with self._lock:
                with self._connection:
                    self._connection.execute(
                        "UPDATE cayu_sessions SET run_epoch = run_epoch + 1 "
                        "WHERE id = ? AND run_epoch = ?",
                        (session_id, expected_run_epoch),
                    )
        finally:
            _deactivate_session_run_fence(session_id)
            _deactivate_session_interaction(session_id)

    async def release_session_invocation(self, command: Any) -> Any:
        from cayu.runtime._invocation_lifecycle import (
            InvocationReleaseResult,
            ReleaseInvocationCommand,
            _invocation_lifecycle_receipt_ledger_from_checkpoint,
            checkpoint_with_invocation_lifecycle_receipt,
            copy_invocation_lifecycle_command,
            invocation_release_replay_from_state,
            require_invocation_command_authority,
            require_invocation_release_store_authority,
        )

        copied = copy_invocation_lifecycle_command(command)
        if type(copied) is not ReleaseInvocationCommand:
            raise TypeError("command must be a ReleaseInvocationCommand.")
        require_invocation_release_store_authority(copied)

        def statement(connection: sqlite3.Connection) -> Any:
            with sqlite_support._transaction(connection):
                session = _load_session(connection, copied.session_id)
                if session is None:
                    raise KeyError(f"Session not found: {copied.session_id}")
                checkpoint = _load_checkpoint_state(connection, copied.session_id)
                ledger = _invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
                replay = invocation_release_replay_from_state(
                    session,
                    checkpoint,
                    copied,
                    _ledger=ledger,
                )
                if replay is not None:
                    connection.commit()
                    return replay
                if copied.terminal_session_event is not None:
                    terminal_event_row = connection.execute(
                        "SELECT event.*, operation.record_json "
                        "FROM cayu_events AS event "
                        "LEFT JOIN cayu_session_operations AS operation "
                        "ON operation.session_id = event.session_id "
                        "AND operation.idempotency_key = ? "
                        "WHERE event.session_id = ? AND event.event_id = ?",
                        (
                            _invocation_terminal_event_storage_key(
                                copied.terminal_session_event.id
                            ),
                            copied.session_id,
                            copied.terminal_session_event.id,
                        ),
                    ).fetchone()
                    _require_invocation_release_terminal_session_event(
                        (
                            None
                            if terminal_event_row is None
                            or terminal_event_row["record_json"] is None
                            else copy_durable_json_object(
                                json.loads(terminal_event_row["record_json"]),
                                "invocation terminal-event receipt",
                            )
                        ),
                        (
                            None
                            if terminal_event_row is None
                            else _event_from_row(terminal_event_row)
                        ),
                        current_session=session,
                        expected_event=copied.terminal_session_event,
                        expected_session_instance_id=copied.expected_session_instance_id,
                        expected_active_invocation_profile=copied.expected_active_profile,
                    )
                elif copied.settlement_transition is None:
                    assert copied.recovery_claim_id is not None
                    _require_invocation_release_recovery_claim(
                        checkpoint,
                        current_session=session,
                        recovery_claim_id=copied.recovery_claim_id,
                    )
                else:
                    settlement_row = connection.execute(
                        "SELECT operation.record_json "
                        "FROM cayu_session_operations AS operation "
                        "WHERE operation.session_id = ? AND operation.idempotency_key = ?",
                        (
                            copied.session_id,
                            _interaction_transition_storage_key(
                                copied.settlement_transition.event.id
                            ),
                        ),
                    ).fetchone()
                    if settlement_row is None:
                        raise SessionRunFenced(
                            "Invocation release lacks exact durable terminal settlement."
                        )
                    _require_invocation_release_settlement_record(
                        copy_durable_json_object(
                            json.loads(settlement_row["record_json"]),
                            "interaction transition receipt",
                        ),
                        current_session=session,
                        transition=copied.settlement_transition,
                        expected_session_instance_id=copied.expected_session_instance_id,
                        expected_active_invocation_profile=copied.expected_active_profile,
                    )
                require_invocation_command_authority(
                    session,
                    checkpoint,
                    session_id=copied.session_id,
                    session_instance_id=copied.expected_session_instance_id,
                    run_epochs=frozenset({copied.expected_run_epoch}),
                    active_profile=copied.expected_active_profile,
                )
                connection.execute(
                    "UPDATE cayu_sessions SET run_epoch = run_epoch + 1 "
                    "WHERE id = ? AND run_epoch = ?",
                    (copied.session_id, copied.expected_run_epoch),
                )
                session = session.model_copy(update={"run_epoch": copied.expected_run_epoch + 1})
                updated_checkpoint = checkpoint_with_invocation_lifecycle_receipt(
                    checkpoint,
                    copied,
                    active_profile=copied.expected_active_profile,
                    result_session=session,
                    _ledger=ledger,
                )
                connection.execute(
                    "INSERT INTO cayu_checkpoints ("
                    "session_id, state_json, updated_at, pending_action_source_bytes, "
                    "pending_action_tool_call_count, pending_action_flags, "
                    "pending_action_metrics_ready) VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET "
                    "state_json = excluded.state_json, updated_at = excluded.updated_at, "
                    "pending_action_source_bytes = excluded.pending_action_source_bytes, "
                    "pending_action_tool_call_count = excluded.pending_action_tool_call_count, "
                    "pending_action_flags = excluded.pending_action_flags, "
                    "pending_action_metrics_ready = excluded.pending_action_metrics_ready",
                    sqlite_support.checkpoint_row_values(
                        copied.session_id,
                        updated_checkpoint,
                        session.updated_at,
                    ),
                )
                return InvocationReleaseResult(
                    session=session,
                    active_profile=copied.expected_active_profile,
                    replayed=False,
                )

        released = False
        try:
            result = await self._run_write(statement)
            released = True
            return result
        finally:
            if released and (
                _current_session_run_epoch(copied.session_id) == copied.expected_run_epoch
            ):
                _deactivate_session_run_fence(copied.session_id)
                _deactivate_session_interaction(copied.session_id)

    async def append_event(self, session_id: str, event: Event) -> None:
        await self.append_events(session_id, [event])

    async def claim_budget_reservation_identity(
        self,
        *,
        reservation_id: str,
        publication_session_id: str,
        publication_id: str,
    ) -> None:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        publication_session_id = require_clean_nonblank(
            publication_session_id,
            "publication_session_id",
        )
        publication_id = require_clean_nonblank(publication_id, "publication_id")
        expected_run_epoch = _current_session_run_epoch(publication_session_id)

        def statement(connection: sqlite3.Connection) -> None:
            with connection:
                # The conditional no-op acquires SQLite's writer lock while it
                # validates the session epoch, so a takeover cannot interleave
                # between this check and the registry claim below.
                if expected_run_epoch is None:
                    cursor = connection.execute(
                        "UPDATE cayu_sessions SET run_epoch = run_epoch WHERE id = ?",
                        (publication_session_id,),
                    )
                else:
                    cursor = connection.execute(
                        "UPDATE cayu_sessions SET run_epoch = run_epoch "
                        "WHERE id = ? AND run_epoch = ?",
                        (publication_session_id, expected_run_epoch),
                    )
                if cursor.rowcount != 1:
                    if expected_run_epoch is not None:
                        _raise_session_write_conflict(
                            connection,
                            publication_session_id,
                            expected_run_epoch,
                        )
                    raise KeyError(f"Session not found: {publication_session_id}")
                _claim_budget_reservation_identity(
                    connection,
                    reservation_id=reservation_id,
                    publication_session_id=publication_session_id,
                    publication_id=publication_id,
                )

        await self._run_write(statement)

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        session_id, copied_events = _copy_session_event_batch(session_id, events)

        def statement(connection: sqlite3.Connection) -> None:
            if not copied_events:
                if not _session_exists(connection, session_id):
                    raise KeyError(f"Session not found: {session_id}")
                return

            try:
                connection.execute("BEGIN IMMEDIATE")
                if not _session_exists(connection, session_id):
                    raise KeyError(f"Session not found: {session_id}")
                activity_at = self._ownership_clock()
                _append_events_in_transaction(
                    connection,
                    session_id,
                    copied_events,
                    activity_at=activity_at,
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                existing_event_id = _first_existing_event_id(
                    connection,
                    session_id,
                    [event.id for event in copied_events],
                )
                if existing_event_id is not None:
                    raise ValueError(
                        f"Event already exists for session {session_id}: {existing_event_id}"
                    ) from exc
                if "idx_cayu_events_budget_reservation_identity" in str(exc):
                    raise BudgetReservationIdentityConflict(
                        "Budget ledger reused a reservation identity."
                    ) from exc
                raise
            except BaseException:
                connection.rollback()
                raise

        await self._run_write(statement)

    async def append_workflow_step_started(
        self,
        session_id: str,
        event: Event,
        *,
        workflow_name: str,
        attempt_id: str,
    ) -> bool:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        session_id, copied_event, workflow_name, attempt_id = _copy_workflow_step_reservation(
            session_id,
            event,
            workflow_name=workflow_name,
            attempt_id=attempt_id,
        )

        def statement(connection: sqlite3.Connection) -> bool:
            try:
                connection.execute("BEGIN IMMEDIATE")
                if not _session_exists(connection, session_id):
                    raise KeyError(f"Session not found: {session_id}")
                row = connection.execute(
                    """
                    SELECT json_extract(payload_json, '$.attempt_id')
                    FROM cayu_events
                    WHERE session_id = ?
                      AND workflow_name = ?
                      AND event_type = ?
                    ORDER BY sequence DESC
                    LIMIT 1
                    """,
                    (session_id, workflow_name, WORKFLOW_ATTEMPT_EVENT_TYPE),
                ).fetchone()
                if row is None or row[0] != attempt_id:
                    connection.rollback()
                    return False
                if _first_existing_event_id(connection, session_id, [copied_event.id]) is not None:
                    connection.rollback()
                    return False

                _touch_session_activity(connection, session_id, self._ownership_clock())
                lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                    copied_event
                )
                connection.execute(
                    """
                    INSERT INTO cayu_events (
                        session_id,
                        event_id,
                        interaction_id,
                        event_type,
                        timestamp,
                        agent_name,
                        environment_name,
                        workflow_name,
                        tool_name,
                        payload_json,
                        pending_action_lookup_key,
                        pending_action_projection_json,
                        pending_action_projection_bytes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        copied_event.id,
                        copied_event.interaction_id,
                        str(copied_event.type),
                        sqlite_support.format_datetime(copied_event.timestamp),
                        copied_event.agent_name,
                        copied_event.environment_name,
                        copied_event.workflow_name,
                        copied_event.tool_name,
                        sqlite_support.json_dumps(copied_event.payload),
                        lookup_key,
                        projection,
                        projection_bytes,
                    ),
                )
                _enqueue_persisted_event_side_effects(
                    connection,
                    session_id,
                    [copied_event],
                )
                connection.commit()
                return True
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                existing_event_id = _first_existing_event_id(
                    connection,
                    session_id,
                    [copied_event.id],
                )
                if existing_event_id is not None:
                    return False
                raise exc
            except Exception:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def load_mcp_manifest_baselines(
        self,
        history_keys: tuple[str, ...],
    ) -> McpManifestBaselineLoadResult:
        keys = _validate_mcp_manifest_history_keys(history_keys)

        def query(connection: sqlite3.Connection) -> McpManifestBaselineLoadResult:
            result: dict[str, McpManifestBaseline] = {}
            for key in keys:
                row = connection.execute(
                    "SELECT generation, baseline_json FROM cayu_mcp_manifest_baselines "
                    "WHERE history_key = ?",
                    (key,),
                ).fetchone()
                if row is not None:
                    result[key] = _stored_mcp_manifest_baseline_json(
                        key,
                        row["generation"],
                        row["baseline_json"],
                    )
            return McpManifestBaselineLoadResult(baselines=result)

        return await self._run_read(query)

    async def compare_and_publish_mcp_manifest_checks(
        self,
        session_id: str,
        *,
        expected_generations: dict[str, int | None],
        baseline_updates: dict[str, McpManifestBaseline],
        events: list[Event],
    ) -> McpManifestPublicationResult:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        session_id, expected, updates, copied_events = _copy_mcp_manifest_publication(
            session_id,
            expected_generations=expected_generations,
            baseline_updates=baseline_updates,
            events=events,
        )

        def statement(connection: sqlite3.Connection) -> McpManifestPublicationResult:
            try:
                connection.execute("BEGIN IMMEDIATE")
                session = _load_session(connection, session_id)
                if session is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, session)
                current: dict[str, McpManifestBaseline] = {}
                for key in expected:
                    row = connection.execute(
                        "SELECT generation, baseline_json "
                        "FROM cayu_mcp_manifest_baselines "
                        "WHERE history_key = ?",
                        (key,),
                    ).fetchone()
                    if row is not None:
                        current[key] = _stored_mcp_manifest_baseline_json(
                            key,
                            row["generation"],
                            row["baseline_json"],
                        )
                if any(
                    expected_generation
                    != (None if (baseline := current.get(key)) is None else baseline.generation)
                    for key, expected_generation in expected.items()
                ):
                    connection.rollback()
                    return McpManifestPublicationResult(
                        published=False,
                        baselines=current,
                    )

                _validate_mcp_manifest_publication_state(
                    expected_generations=expected,
                    current_baselines=current,
                    baseline_updates=updates,
                    events=copied_events,
                )
                _touch_session_activity(connection, session_id, self._ownership_clock())
                event_rows = []
                for event in copied_events:
                    lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                        event
                    )
                    event_rows.append(
                        (
                            session_id,
                            event.id,
                            event.interaction_id,
                            str(event.type),
                            sqlite_support.format_datetime(event.timestamp),
                            event.agent_name,
                            event.environment_name,
                            event.workflow_name,
                            event.tool_name,
                            sqlite_support.json_dumps(event.payload),
                            lookup_key,
                            projection,
                            projection_bytes,
                        )
                    )
                connection.executemany(
                    """
                    INSERT INTO cayu_events (
                        session_id, event_id, interaction_id, event_type, timestamp, agent_name,
                        environment_name, workflow_name, tool_name, payload_json,
                        pending_action_lookup_key, pending_action_projection_json,
                        pending_action_projection_bytes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    event_rows,
                )
                _enqueue_persisted_event_side_effects(
                    connection,
                    session_id,
                    copied_events,
                )
                updated_at = sqlite_support.format_datetime(self._ownership_clock())
                for key, baseline in updates.items():
                    connection.execute(
                        """
                        INSERT INTO cayu_mcp_manifest_baselines (
                            history_key, generation, baseline_json, updated_at
                        )
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(history_key) DO UPDATE SET
                            generation = excluded.generation,
                            baseline_json = excluded.baseline_json,
                            updated_at = excluded.updated_at
                        """,
                        (
                            key,
                            baseline.generation,
                            sqlite_support.json_dumps(baseline.model_dump(mode="json")),
                            updated_at,
                        ),
                    )
                    current[key] = baseline.model_copy(deep=True)
                connection.commit()
                return McpManifestPublicationResult(
                    published=True,
                    baselines=current,
                )
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                existing_event_id = _first_existing_event_id(
                    connection,
                    session_id,
                    [event.id for event in copied_events],
                )
                if existing_event_id is not None:
                    raise ValueError(
                        f"Event already exists for session {session_id}: {existing_event_id}"
                    ) from exc
                raise
            except Exception:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def claim_persisted_event_side_effect(
        self,
        *,
        session_id: str | None = None,
        event_id: str | None = None,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectClaim | None:
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        if event_id is not None:
            event_id = require_clean_nonblank(event_id, "event_id")
        if (session_id is None) != (event_id is None):
            raise ValueError("session_id and event_id must be supplied together.")
        if type(lease_seconds) not in {int, float} or lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than 0.")

        def statement(connection: sqlite3.Connection) -> PersistedEventSideEffectClaim | None:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = self._ownership_clock()
                lease_expires_at = now + timedelta(seconds=float(lease_seconds))
                formatted_now = sqlite_support.format_datetime(now)
                filters = [
                    "(status = 'pending' "
                    "OR (status = 'failed' AND "
                    "(next_attempt_at IS NULL OR next_attempt_at <= ?)) "
                    "OR (status = 'leased' AND lease_expires_at <= ?))"
                ]
                params: list[object] = [formatted_now, formatted_now]
                if session_id is not None and event_id is not None:
                    filters.extend(["session_id = ?", "event_id = ?"])
                    params.extend([session_id, event_id])
                delivery_row = connection.execute(
                    "SELECT * FROM cayu_persisted_event_side_effects WHERE "
                    + " AND ".join(filters)
                    + " ORDER BY event_sequence ASC LIMIT 1",
                    params,
                ).fetchone()
                if delivery_row is None:
                    connection.commit()
                    return None
                claim_id = str(uuid4())
                attempt = int(delivery_row["attempts"]) + 1
                connection.execute(
                    "UPDATE cayu_persisted_event_side_effects "
                    "SET status = 'leased', attempts = ?, claim_id = ?, "
                    "lease_expires_at = ?, next_attempt_at = NULL, "
                    "last_error = NULL, updated_at = ? "
                    "WHERE session_id = ? AND event_id = ?",
                    (
                        attempt,
                        claim_id,
                        sqlite_support.format_datetime(lease_expires_at),
                        sqlite_support.format_datetime(now),
                        delivery_row["session_id"],
                        delivery_row["event_id"],
                    ),
                )
                event_row = connection.execute(
                    f"SELECT {', '.join(_EVENT_COLUMN_NAMES)} FROM cayu_events "
                    "WHERE session_id = ? AND event_id = ?",
                    (delivery_row["session_id"], delivery_row["event_id"]),
                ).fetchone()
                if event_row is None:
                    raise RuntimeError("Persisted side-effect delivery lost its source event.")
                connection.commit()
                return PersistedEventSideEffectClaim(
                    session_id=delivery_row["session_id"],
                    event_id=delivery_row["event_id"],
                    event_sequence=delivery_row["event_sequence"],
                    event=_event_from_row(event_row),
                    attempt=attempt,
                    claim_id=claim_id,
                    lease_expires_at=lease_expires_at,
                )
            except Exception:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def get_persisted_event_side_effect_delivery(
        self,
        *,
        session_id: str,
        event_id: str,
    ) -> PersistedEventSideEffectDelivery | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        event_id = require_clean_nonblank(event_id, "event_id")

        def query(connection: sqlite3.Connection) -> PersistedEventSideEffectDelivery | None:
            row = connection.execute(
                "SELECT * FROM cayu_persisted_event_side_effects "
                "WHERE session_id = ? AND event_id = ?",
                (session_id, event_id),
            ).fetchone()
            return None if row is None else _persisted_event_side_effect_delivery_from_row(row)

        return await self._run_read(query)

    async def mark_persisted_event_side_effect_delivered(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        claim = PersistedEventSideEffectClaim.model_validate(claim)
        return await self._finish_persisted_event_side_effect_claim(
            claim,
            status=PersistedEventSideEffectStatus.DELIVERED,
            error=None,
            retry_delay_seconds=None,
        )

    async def mark_persisted_event_side_effect_failed(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        error: str,
        max_attempts: int,
        retry_delay_seconds: float,
    ) -> PersistedEventSideEffectDelivery:
        claim = PersistedEventSideEffectClaim.model_validate(claim)
        error = validate_persisted_event_side_effect_error(error)
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be an integer greater than or equal to 1.")
        if (
            type(retry_delay_seconds) not in {int, float}
            or not math.isfinite(retry_delay_seconds)
            or retry_delay_seconds < 0
        ):
            raise ValueError("retry_delay_seconds must be a finite non-negative number.")
        dead_lettered = claim.attempt >= max_attempts
        return await self._finish_persisted_event_side_effect_claim(
            claim,
            status=(
                PersistedEventSideEffectStatus.DEAD_LETTERED
                if dead_lettered
                else PersistedEventSideEffectStatus.FAILED
            ),
            error=error,
            retry_delay_seconds=(None if dead_lettered else float(retry_delay_seconds)),
        )

    async def _finish_persisted_event_side_effect_claim(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        status: PersistedEventSideEffectStatus,
        error: str | None,
        retry_delay_seconds: float | None,
    ) -> PersistedEventSideEffectDelivery:
        def statement(connection: sqlite3.Connection) -> PersistedEventSideEffectDelivery:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = self._ownership_clock()
                next_attempt_at = (
                    None
                    if retry_delay_seconds is None
                    else now + timedelta(seconds=retry_delay_seconds)
                )
                cursor = connection.execute(
                    "UPDATE cayu_persisted_event_side_effects "
                    "SET status = ?, claim_id = NULL, lease_expires_at = NULL, "
                    "next_attempt_at = ?, last_error = ?, updated_at = ? "
                    "WHERE session_id = ? AND event_id = ? AND status = 'leased' "
                    "AND claim_id = ? AND attempts = ?",
                    (
                        str(status),
                        (
                            None
                            if next_attempt_at is None
                            else sqlite_support.format_datetime(next_attempt_at)
                        ),
                        error,
                        sqlite_support.format_datetime(now),
                        claim.session_id,
                        claim.event_id,
                        claim.claim_id,
                        claim.attempt,
                    ),
                )
                if cursor.rowcount != 1:
                    existing = connection.execute(
                        "SELECT 1 FROM cayu_persisted_event_side_effects "
                        "WHERE session_id = ? AND event_id = ?",
                        (claim.session_id, claim.event_id),
                    ).fetchone()
                    if existing is None:
                        raise ValueError("Persisted event side-effect delivery was not found.")
                    raise PersistedEventSideEffectClaimLost(
                        "Persisted event side-effect claim is no longer active."
                    )
                row = connection.execute(
                    "SELECT * FROM cayu_persisted_event_side_effects "
                    "WHERE session_id = ? AND event_id = ?",
                    (claim.session_id, claim.event_id),
                ).fetchone()
                if row is None:
                    raise RuntimeError("Persisted event side-effect delivery disappeared.")
                delivery = _persisted_event_side_effect_delivery_from_row(row)
                connection.commit()
                return delivery
            except Exception:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def list_persisted_event_side_effect_deliveries(
        self,
        *,
        statuses: set[PersistedEventSideEffectStatus] | None = None,
        claimable_only: bool = False,
        after_sequence: int | None = None,
        limit: int = 100,
    ) -> list[PersistedEventSideEffectDelivery]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000.")
        if type(claimable_only) is not bool:
            raise TypeError("claimable_only must be a bool.")
        if after_sequence is not None and (type(after_sequence) is not int or after_sequence < 0):
            raise ValueError("after_sequence must be a non-negative integer.")
        selected_statuses = (
            None
            if statuses is None
            else sorted(str(PersistedEventSideEffectStatus(status)) for status in statuses)
        )

        def query(connection: sqlite3.Connection) -> list[PersistedEventSideEffectDelivery]:
            clauses: list[str] = []
            params: list[object] = []
            if after_sequence is not None:
                clauses.append("event_sequence > ?")
                params.append(after_sequence)
            if selected_statuses is not None:
                if not selected_statuses:
                    return []
                placeholders = ", ".join("?" for _ in selected_statuses)
                clauses.append(f"status IN ({placeholders})")
                params.extend(selected_statuses)
            if claimable_only:
                clauses.append(
                    "(status = 'pending' "
                    "OR (status = 'failed' AND "
                    "(next_attempt_at IS NULL OR next_attempt_at <= ?)) "
                    "OR (status = 'leased' AND lease_expires_at <= ?))"
                )
                formatted_now = sqlite_support.format_datetime(self._ownership_clock())
                params.extend([formatted_now, formatted_now])
            where = "" if not clauses else "WHERE " + " AND ".join(clauses)
            params.append(limit)
            rows = connection.execute(
                "SELECT * FROM cayu_persisted_event_side_effects "
                f"{where} ORDER BY event_sequence ASC LIMIT ?",
                params,
            ).fetchall()
            return [_persisted_event_side_effect_delivery_from_row(row) for row in rows]

        return await self._run_read(query)

    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
    ) -> EnqueueSessionMessageResult:
        request = copy_enqueue_session_message_request(request)

        def statement(connection: sqlite3.Connection) -> EnqueueSessionMessageResult:
            from cayu.runtime.pending_actions import pending_action_event_storage_values

            try:
                connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(request.session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {request.session_id}")
                existing_row = connection.execute(
                    "SELECT * FROM cayu_session_message_queue "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (request.session_id, request.idempotency_key),
                ).fetchone()
                if existing_row is not None:
                    existing = _queued_session_message_from_row(existing_row)
                    _validate_equivalent_queued_session_message(existing, request)
                    event_row = connection.execute(
                        f"SELECT {', '.join(_EVENT_COLUMN_NAMES)} FROM cayu_events "
                        "WHERE session_id = ? AND event_id = ?",
                        (request.session_id, existing.accepted_event_id),
                    ).fetchone()
                    if event_row is None:
                        raise RuntimeError(
                            "Queued session message is missing its durable acceptance event."
                        )
                    connection.commit()
                    return EnqueueSessionMessageResult(
                        message=existing,
                        event=_event_from_row(event_row),
                        replayed=True,
                    )
                if loaded.status not in {SessionStatus.PENDING, SessionStatus.RUNNING}:
                    raise SessionStatusConflict(
                        "Session messages may be enqueued only while a session is pending or running."
                    )
                checkpoint = _load_checkpoint_state(connection, request.session_id)
                if (
                    checkpoint is not None
                    and PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY in checkpoint
                ):
                    raise SessionStatusConflict(
                        "Session messages cannot be enqueued while completion finalization "
                        "is pending."
                    )
                transcript_cursor = _transcript_cursor(connection, request.session_id)
                accepted_at = self._ownership_clock()
                queue_id = str(uuid4())
                accepted_event_id = str(uuid4())
                cursor = connection.execute(
                    """
                    INSERT INTO cayu_session_message_queue (
                        queue_id, session_id, idempotency_key, content, message_json,
                        delivery_mode, status, requested_by_json,
                        accepted_run_epoch, accepted_transcript_cursor,
                        accepted_event_id, accepted_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)
                    """,
                    (
                        queue_id,
                        request.session_id,
                        request.idempotency_key,
                        request.content,
                        (
                            None
                            if request.message is None
                            else sqlite_support.json_dumps(request.message.model_dump(mode="json"))
                        ),
                        str(request.delivery_mode),
                        (
                            None
                            if request.requested_by is None
                            else sqlite_support.json_dumps(
                                resolution_actor_payload(request.requested_by)
                            )
                        ),
                        loaded.run_epoch,
                        transcript_cursor,
                        accepted_event_id,
                        sqlite_support.format_datetime(accepted_at),
                    ),
                )
                ordering_key = cursor.lastrowid
                if type(ordering_key) is not int:
                    raise RuntimeError("SQLite queue insert did not return an ordering key.")
                accepted_message = enqueue_session_message_input(request)
                accepted_event = event_with_runtime_payload_authority(
                    Event(
                        id=accepted_event_id,
                        type=EventType.SESSION_MESSAGE_QUEUED,
                        session_id=request.session_id,
                        agent_name=loaded.agent_name,
                        environment_name=loaded.environment_name,
                        timestamp=accepted_at,
                        payload={
                            **_queued_session_message_event_payload(
                                queue_id=queue_id,
                                delivery_mode=request.delivery_mode,
                                ordering_key=ordering_key,
                                actor=request.requested_by,
                                run_epoch=loaded.run_epoch,
                                transcript_cursor=transcript_cursor,
                            ),
                            SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY: (
                                session_messages_input_contract_evidence(
                                    (accepted_message,),
                                    message_start_index=transcript_cursor,
                                    redactions_applied=request._input_redactions_applied,
                                    structured_output_requested=False,
                                )
                            ),
                        },
                    ),
                    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
                )
                lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                    accepted_event
                )
                connection.execute(
                    """
                    INSERT INTO cayu_events (
                        session_id, event_id, interaction_id, event_type, timestamp, agent_name,
                        environment_name, workflow_name, tool_name, payload_json,
                        pending_action_lookup_key, pending_action_projection_json,
                        pending_action_projection_bytes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request.session_id,
                        accepted_event.id,
                        accepted_event.interaction_id,
                        str(accepted_event.type),
                        sqlite_support.format_datetime(accepted_event.timestamp),
                        accepted_event.agent_name,
                        accepted_event.environment_name,
                        accepted_event.workflow_name,
                        accepted_event.tool_name,
                        sqlite_support.json_dumps(accepted_event.payload),
                        lookup_key,
                        projection,
                        projection_bytes,
                    ),
                )
                _enqueue_persisted_event_side_effects(
                    connection,
                    request.session_id,
                    [accepted_event],
                )
                _touch_session_activity(connection, request.session_id, accepted_at)
                connection.commit()
                stored_row = connection.execute(
                    "SELECT * FROM cayu_session_message_queue WHERE queue_id = ?",
                    (queue_id,),
                ).fetchone()
                if stored_row is None:
                    raise RuntimeError("Queued session message disappeared after acceptance.")
                return EnqueueSessionMessageResult(
                    message=_queued_session_message_from_row(stored_row),
                    event=accepted_event,
                )
            except Exception:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def deliver_queued_session_messages(
        self,
        session_id: str,
        *,
        include_on_idle: bool,
        delivery_id: str | None = None,
        eligible_through: int | None = None,
        limit: int = SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
        interaction_id: str | None = None,
        interaction_started_event: Event | None = None,
    ) -> SessionMessageDeliveryBatch:
        session_id = require_clean_nonblank(session_id, "session_id")
        delivery_id = (
            str(uuid4())
            if delivery_id is None
            else require_clean_nonblank(delivery_id, "delivery_id")
        )
        if interaction_id is not None:
            interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
        interaction_started_event = _copy_queued_interaction_started_event(
            session_id,
            interaction_id,
            interaction_started_event,
        )
        if type(include_on_idle) is not bool:
            raise TypeError("include_on_idle must be a bool.")
        eligible_through = _validate_message_delivery_eligible_through(eligible_through)
        if type(limit) is not int or not 1 <= limit <= SESSION_MESSAGE_DELIVERY_BATCH_LIMIT:
            raise ValueError(f"limit must be between 1 and {SESSION_MESSAGE_DELIVERY_BATCH_LIMIT}.")

        def statement(connection: sqlite3.Connection) -> SessionMessageDeliveryBatch:
            from cayu.runtime.pending_actions import pending_action_event_storage_values

            try:
                connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, loaded)
                delivery_row = connection.execute(
                    "SELECT * FROM cayu_session_message_deliveries WHERE delivery_id = ?",
                    (delivery_id,),
                ).fetchone()
                if delivery_row is not None:
                    stored_started_event = (
                        None
                        if delivery_row["interaction_started_event_json"] is None
                        else Event.model_validate_json(
                            delivery_row["interaction_started_event_json"]
                        )
                    )
                    if (
                        delivery_row["session_id"] != session_id
                        or bool(delivery_row["include_on_idle"]) != include_on_idle
                        or delivery_row["requested_eligible_through"] != eligible_through
                        or delivery_row["batch_limit"] != limit
                        or delivery_row["interaction_id"] != interaction_id
                        or stored_started_event != interaction_started_event
                    ):
                        raise ValueError(
                            "delivery_id was already used for a different queue delivery."
                        )
                    queue_ids = json.loads(delivery_row["queue_ids_json"])
                    replayed_messages: list[SessionQueuedMessage] = []
                    replayed_events = [
                        Event.model_validate(event)
                        for event in json.loads(delivery_row["events_json"])
                    ]
                    for queue_id in queue_ids:
                        queued_row = connection.execute(
                            "SELECT * FROM cayu_session_message_queue WHERE queue_id = ?",
                            (queue_id,),
                        ).fetchone()
                        if queued_row is None:
                            raise RuntimeError("Queue delivery replay lost a delivered message.")
                        replayed_messages.append(_queued_session_message_from_row(queued_row))
                    connection.commit()
                    return SessionMessageDeliveryBatch(
                        messages=tuple(replayed_messages),
                        events=tuple(replayed_events),
                        delivery_id=delivery_id,
                        interaction_id=interaction_id,
                        eligible_through=delivery_row["eligible_through"],
                        has_more=bool(delivery_row["has_more"]),
                        replayed=True,
                    )
                if loaded.status != SessionStatus.RUNNING:
                    raise SessionStatusConflict(
                        "Queued session messages may be delivered only while running."
                    )
                boundary = eligible_through
                if boundary is None:
                    # ``ordering_key`` is a global AUTOINCREMENT primary key.
                    # Reading its global maximum is an end-of-index lookup and
                    # still fences every message this session could currently
                    # contain; BEGIN IMMEDIATE prevents a same-session enqueue
                    # from crossing the boundary during this transaction.
                    boundary_row = connection.execute(
                        "SELECT COALESCE(MAX(ordering_key), 0) AS boundary "
                        "FROM cayu_session_message_queue"
                    ).fetchone()
                    boundary = boundary_row["boundary"]
                rows = connection.execute(
                    "SELECT * FROM cayu_session_message_queue "
                    "WHERE session_id = ? AND status = 'queued' "
                    "AND delivery_mode = 'next_turn' AND ordering_key <= ? "
                    "ORDER BY ordering_key ASC LIMIT ?",
                    (session_id, boundary, limit),
                ).fetchall()
                if not rows and include_on_idle:
                    rows = connection.execute(
                        "SELECT * FROM cayu_session_message_queue "
                        "WHERE session_id = ? AND status = 'queued' "
                        "AND delivery_mode = 'on_idle' AND ordering_key <= ? "
                        "ORDER BY ordering_key ASC LIMIT ?",
                        (session_id, boundary, limit),
                    ).fetchall()
                if not rows:
                    connection.execute(
                        """
                        INSERT INTO cayu_session_message_deliveries (
                            delivery_id, session_id, interaction_id, include_on_idle,
                            requested_eligible_through, eligible_through, batch_limit,
                            has_more, interaction_started_event_json, queue_ids_json,
                            events_json, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, '[]', '[]', ?)
                        """,
                        (
                            delivery_id,
                            session_id,
                            interaction_id,
                            include_on_idle,
                            eligible_through,
                            boundary,
                            limit,
                            (
                                None
                                if interaction_started_event is None
                                else sqlite_support.json_dumps(
                                    interaction_started_event.model_dump(mode="json")
                                )
                            ),
                            sqlite_support.format_datetime(self._ownership_clock()),
                        ),
                    )
                    connection.commit()
                    return SessionMessageDeliveryBatch(
                        delivery_id=delivery_id,
                        interaction_id=interaction_id,
                        eligible_through=boundary,
                        has_more=False,
                    )
                transcript_cursor = _transcript_cursor(connection, session_id)
                delivered_at = self._ownership_clock()
                updated_messages: list[SessionQueuedMessage] = []
                delivery_events: list[Event] = []
                transcript_messages: list[Message] = []
                for offset, row in enumerate(rows, start=1):
                    queued_message = _queued_session_message_from_row(row)
                    delivered_cursor = transcript_cursor + offset
                    delivered_message = queued_session_message_input(queued_message)
                    delivery_event = event_with_runtime_payload_authority(
                        Event(
                            type=EventType.SESSION_MESSAGE_DELIVERED,
                            session_id=session_id,
                            interaction_id=interaction_id,
                            agent_name=loaded.agent_name,
                            environment_name=loaded.environment_name,
                            timestamp=delivered_at,
                            payload={
                                **_queued_session_message_event_payload(
                                    queue_id=queued_message.queue_id,
                                    delivery_mode=queued_message.delivery_mode,
                                    ordering_key=queued_message.ordering_key,
                                    actor=queued_message.requested_by,
                                    run_epoch=loaded.run_epoch,
                                    transcript_cursor=delivered_cursor,
                                ),
                                "accepted_run_epoch": queued_message.accepted_run_epoch,
                                "accepted_transcript_cursor": (
                                    queued_message.accepted_transcript_cursor
                                ),
                                SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY: (
                                    session_messages_input_contract_evidence(
                                        (delivered_message,),
                                        message_start_index=delivered_cursor - 1,
                                        redactions_applied=False,
                                        structured_output_requested=False,
                                    )
                                ),
                            },
                        ),
                        SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
                    )
                    updated = queued_message.model_copy(
                        update={
                            "status": SessionMessageQueueStatus.DELIVERED,
                            "delivered_run_epoch": loaded.run_epoch,
                            "delivered_transcript_cursor": delivered_cursor,
                            "delivered_event_id": delivery_event.id,
                            "delivered_at": delivered_at,
                        },
                        deep=True,
                    )
                    updated_messages.append(updated)
                    delivery_events.append(delivery_event)
                    transcript_messages.append(delivered_message)
                connection.executemany(
                    "INSERT INTO cayu_transcript_messages "
                    "(session_id, role, interaction_id, message_json, "
                    "transcript_search_document) VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            session_id,
                            str(message.role),
                            interaction_id,
                            sqlite_support.json_dumps(message.model_dump(mode="json")),
                            transcript_search_document(message),
                        )
                        for message in transcript_messages
                    ],
                )
                for updated in updated_messages:
                    connection.execute(
                        "UPDATE cayu_session_message_queue SET status = 'delivered', "
                        "delivered_run_epoch = ?, delivered_transcript_cursor = ?, "
                        "delivered_event_id = ?, delivered_at = ? "
                        "WHERE queue_id = ? AND status = 'queued'",
                        (
                            updated.delivered_run_epoch,
                            updated.delivered_transcript_cursor,
                            updated.delivered_event_id,
                            sqlite_support.format_datetime(delivered_at),
                            updated.queue_id,
                        ),
                    )
                persisted_events = [
                    *([interaction_started_event] if interaction_started_event is not None else []),
                    *delivery_events,
                ]
                event_rows = []
                for event in persisted_events:
                    lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                        event
                    )
                    event_rows.append(
                        (
                            session_id,
                            event.id,
                            event.interaction_id,
                            str(event.type),
                            sqlite_support.format_datetime(event.timestamp),
                            event.agent_name,
                            event.environment_name,
                            event.workflow_name,
                            event.tool_name,
                            sqlite_support.json_dumps(event.payload),
                            lookup_key,
                            projection,
                            projection_bytes,
                        )
                    )
                connection.executemany(
                    "INSERT INTO cayu_events (session_id, event_id, interaction_id, "
                    "event_type, timestamp, "
                    "agent_name, environment_name, workflow_name, tool_name, payload_json, "
                    "pending_action_lookup_key, pending_action_projection_json, "
                    "pending_action_projection_bytes) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    event_rows,
                )
                _enqueue_persisted_event_side_effects(
                    connection,
                    session_id,
                    persisted_events,
                )
                _touch_session_activity(connection, session_id, delivered_at)
                remaining_mode_sql = (
                    "delivery_mode IN ('next_turn', 'on_idle')"
                    if include_on_idle
                    else "delivery_mode = 'next_turn'"
                )
                remaining = connection.execute(
                    "SELECT 1 FROM cayu_session_message_queue WHERE session_id = ? "
                    "AND status = 'queued' AND ordering_key <= ? "
                    f"AND {remaining_mode_sql} LIMIT 1",
                    (session_id, boundary),
                ).fetchone()
                connection.execute(
                    """
                    INSERT INTO cayu_session_message_deliveries (
                        delivery_id, session_id, interaction_id, include_on_idle,
                        requested_eligible_through, eligible_through, batch_limit,
                        has_more, interaction_started_event_json, queue_ids_json,
                        events_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        delivery_id,
                        session_id,
                        interaction_id,
                        include_on_idle,
                        eligible_through,
                        boundary,
                        limit,
                        remaining is not None,
                        (
                            None
                            if interaction_started_event is None
                            else sqlite_support.json_dumps(
                                interaction_started_event.model_dump(mode="json")
                            )
                        ),
                        sqlite_support.json_dumps(
                            [message.queue_id for message in updated_messages]
                        ),
                        sqlite_support.json_dumps(
                            [event.model_dump(mode="json") for event in persisted_events]
                        ),
                        sqlite_support.format_datetime(delivered_at),
                    ),
                )
                connection.commit()
                return SessionMessageDeliveryBatch(
                    messages=tuple(updated_messages),
                    events=tuple(persisted_events),
                    delivery_id=delivery_id,
                    interaction_id=interaction_id,
                    eligible_through=boundary,
                    has_more=remaining is not None,
                )
            except Exception:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def publish_checkpoint_and_events(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        return await self._publish_checkpoint_and_events(
            session_id,
            checkpoint_transform=checkpoint_transform,
            operation_idempotency_key=None,
            operation_transform=None,
            store_time_operation_transform=None,
            operation_commit_guard=None,
            operation_commit_time_guard=None,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
            preserve_completion_result_publications=True,
        )

    async def _publish_completion_result_event_publication(
        self,
        session_id: str,
        *,
        checkpoint_transform: StoreTimeCheckpointTransform,
        events: list[Event],
    ) -> Session:
        return await self._publish_checkpoint_and_events(
            session_id,
            checkpoint_transform=None,
            store_time_checkpoint_transform=checkpoint_transform,
            operation_idempotency_key=None,
            operation_transform=None,
            store_time_operation_transform=None,
            operation_commit_guard=None,
            operation_commit_time_guard=None,
            events=events,
            expected_statuses=None,
            expected_run_epoch=None,
            expected_transcript_cursor=None,
            preserve_completion_result_publications=False,
        )

    async def load_session_operation(
        self,
        session_id: str,
        idempotency_key: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        idempotency_key = _reject_reserved_runtime_publication_key(
            idempotency_key,
            "idempotency_key",
        )

        checkpoint_root_key = (
            "__cayu_no_checkpoint_root_guard__"
            if checkpoint_root_guard is None
            else checkpoint_root_guard.key
        )
        checkpoint_root_path = f"$.{checkpoint_root_key}"

        def query(connection: sqlite3.Connection) -> dict[str, Any] | None:
            row = connection.execute(
                f"""
                SELECT
                    cayu_session_operations.record_json,
                    json_type(
                        cayu_checkpoints.state_json,
                        '{checkpoint_root_path}'
                    ) AS checkpoint_root_field_type,
                    CASE
                        WHEN json_type(
                            cayu_checkpoints.state_json,
                            '{checkpoint_root_path}'
                        ) = 'integer'
                        THEN substr(
                            CAST(json_extract(
                                cayu_checkpoints.state_json,
                                '{checkpoint_root_path}'
                            ) AS TEXT),
                            1,
                            {CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS + 1}
                        )
                    END AS checkpoint_root_field_scalar
                FROM cayu_sessions
                LEFT JOIN cayu_session_operations
                    ON cayu_session_operations.session_id = cayu_sessions.id
                    AND cayu_session_operations.idempotency_key = ?
                LEFT JOIN cayu_checkpoints
                    ON cayu_checkpoints.session_id = cayu_sessions.id
                WHERE cayu_sessions.id = ?
                """,
                (idempotency_key, session_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"Session not found: {session_id}")
            scalar_text = row["checkpoint_root_field_scalar"]
            if checkpoint_root_guard is not None:
                checkpoint_root_guard.validate(
                    session_id,
                    checkpoint_root_field_projection_from_storage(
                        json_type=row["checkpoint_root_field_type"],
                        scalar_text=scalar_text,
                    ),
                )
            record_json = row["record_json"]
            return None if record_json is None else json.loads(record_json)

        return await self._run_read(query)

    async def _load_runtime_publication_receipt_record(
        self,
        session_id: str,
        storage_key: str,
        publication_id: str,
    ) -> dict[str, Any] | None:
        def query(connection: sqlite3.Connection) -> dict[str, Any] | None:
            connection.execute("BEGIN")
            try:
                if not _session_exists(connection, session_id):
                    raise KeyError(f"Session not found: {session_id}")
                row = connection.execute(
                    "SELECT record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, storage_key),
                ).fetchone()
                if row is None:
                    return None
                record = _decode_runtime_publication_record(row["record_json"])
                receipt = _reconstruct_runtime_publication_receipt(
                    record,
                    storage_key=storage_key,
                    session_id=session_id,
                    publication_id=publication_id,
                )
                self._validate_runtime_publication_material(connection, receipt)
                return record
            finally:
                connection.rollback()

        return await self._run_read(query)

    def _validate_runtime_publication_material(
        self,
        connection: sqlite3.Connection,
        receipt: RuntimePublicationReceipt,
    ) -> None:
        try:
            transcript_rows = connection.execute(
                "SELECT interaction_id, message_json FROM cayu_transcript_messages "
                "WHERE session_id = ? AND session_order > ? AND session_order <= ? "
                "ORDER BY session_order ASC",
                (
                    receipt.session_id,
                    receipt.transcript_start_cursor,
                    receipt.transcript_end_cursor,
                ),
            ).fetchall()
            transcript = [Message(**json.loads(row["message_json"])) for row in transcript_rows]
            transcript_interaction_ids = [row["interaction_id"] for row in transcript_rows]

            referenced_event_ids = _runtime_publication_referenced_event_ids(
                receipt.referenced_events
            )
            requested_event_ids = tuple(
                dict.fromkeys((*receipt.appended_event_ids, *referenced_event_ids))
            )
            events_by_id: dict[str, Event] = {}
            if requested_event_ids:
                placeholders = ", ".join("?" for _ in requested_event_ids)
                rows = connection.execute(
                    f"SELECT {', '.join(_EVENT_COLUMN_NAMES)} FROM cayu_events "
                    f"WHERE session_id = ? AND event_id IN ({placeholders})",
                    (receipt.session_id, *requested_event_ids),
                ).fetchall()
                events_by_id = {row["event_id"]: _event_from_row(row) for row in rows}
            _validate_runtime_publication_durable_material(
                receipt,
                transcript_messages=transcript,
                transcript_interaction_ids=transcript_interaction_ids,
                appended_events=(
                    events_by_id[event_id]
                    for event_id in receipt.appended_event_ids
                    if event_id in events_by_id
                ),
                durable_referenced_events=(
                    events_by_id[event_id]
                    for event_id in referenced_event_ids
                    if event_id in events_by_id
                ),
            )
        except SessionRuntimePublicationConflict:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SessionRuntimePublicationConflict(
                "The durable runtime publication material is malformed."
            ) from exc

    async def _load_model_completion_stage_records(
        self,
        session_id: str,
        preparation_storage_key: str,
        terminal_storage_key: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        def query(
            connection: sqlite3.Connection,
        ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            rows = connection.execute(
                "SELECT idempotency_key, record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key IN (?, ?)",
                (session_id, preparation_storage_key, terminal_storage_key),
            ).fetchall()
            records = {
                row["idempotency_key"]: _decode_model_completion_stage_record(row["record_json"])
                for row in rows
            }
            return records.get(preparation_storage_key), records.get(terminal_storage_key)

        return await self._run_read(query)

    async def _load_model_completion_stage_settlement_record(
        self,
        session_id: str,
        settlement_storage_key: str,
    ) -> dict[str, Any] | None:
        def query(connection: sqlite3.Connection) -> dict[str, Any] | None:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            row = connection.execute(
                "SELECT record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key = ?",
                (session_id, settlement_storage_key),
            ).fetchone()
            return (
                None if row is None else _decode_model_completion_stage_record(row["record_json"])
            )

        return await self._run_read(query)

    async def _load_model_completion_stage_dispatch_record(
        self,
        session_id: str,
        dispatch_storage_key: str,
    ) -> dict[str, Any] | None:
        def query(connection: sqlite3.Connection) -> dict[str, Any] | None:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            row = connection.execute(
                "SELECT record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key = ?",
                (session_id, dispatch_storage_key),
            ).fetchone()
            return (
                None if row is None else _decode_model_completion_stage_record(row["record_json"])
            )

        return await self._run_read(query)

    async def _load_active_model_completion_stage_records(
        self,
        session_id: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        def query(
            connection: sqlite3.Connection,
        ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
            try:
                connection.execute("BEGIN")
                if not _session_exists(connection, session_id):
                    raise KeyError(f"Session not found: {session_id}")
                row = connection.execute(
                    "SELECT record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                ).fetchone()
                if row is None:
                    connection.rollback()
                    return None, None, None
                active_record = _decode_model_completion_stage_record(row["record_json"])
                marker = _reconstruct_active_model_completion_stage_record(
                    active_record,
                    session_id=session_id,
                )
                _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
                    session_id, marker.stage_id
                )
                rows = connection.execute(
                    "SELECT idempotency_key, record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key IN (?, ?)",
                    (session_id, preparation_key, terminal_key),
                ).fetchall()
                records = {
                    record_row["idempotency_key"]: _decode_model_completion_stage_record(
                        record_row["record_json"]
                    )
                    for record_row in rows
                }
                connection.rollback()
                return (
                    active_record,
                    records.get(preparation_key),
                    records.get(terminal_key),
                )
            except BaseException:
                connection.rollback()
                raise

        return await self._run_read(query)

    async def _mark_model_completion_stage_dispatched_atomic(
        self,
        session_id: str,
        *,
        stage: ModelCompletionStage,
        consume_child_session_notifications: bool,
    ) -> ModelCompletionStageDispatch:
        _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
            session_id,
            stage.stage_id,
        )
        settlement_key = _model_completion_stage_settlement_storage_key(stage.stage_id)
        dispatch_key = _model_completion_stage_dispatch_storage_key(stage.stage_id)

        def statement(connection: sqlite3.Connection) -> ModelCompletionStageDispatch:
            try:
                connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                rows = connection.execute(
                    "SELECT idempotency_key, record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key IN (?, ?, ?, ?, ?)",
                    (
                        session_id,
                        MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                        preparation_key,
                        terminal_key,
                        settlement_key,
                        dispatch_key,
                    ),
                ).fetchall()
                records = {
                    row["idempotency_key"]: _decode_model_completion_stage_record(
                        row["record_json"]
                    )
                    for row in rows
                }
                _validate_model_completion_stage_for_dispatch(
                    session=loaded,
                    current_transcript_cursor=_transcript_cursor(connection, session_id),
                    stage=stage,
                    active_record=records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                    preparation_record=records.get(preparation_key),
                    terminal_record=records.get(terminal_key),
                    settlement_record=records.get(settlement_key),
                )
                published_at = _next_runtime_publication_timestamp(loaded)
                dispatch_record = records.get(dispatch_key)
                dispatch_is_new = dispatch_record is None
                if dispatch_is_new:
                    dispatch_record = _model_completion_stage_dispatch_record(
                        stage,
                        dispatched_at=published_at,
                    )
                    connection.execute(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record_json, updated_at) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            session_id,
                            dispatch_key,
                            sqlite_support.json_dumps(dispatch_record),
                            sqlite_support.format_datetime(published_at),
                        ),
                    )
                assert dispatch_record is not None
                dispatch = _reconstruct_model_completion_stage_dispatch(
                    dispatch_record,
                    session_id=session_id,
                    stage_id=stage.stage_id,
                    storage_key=dispatch_key,
                )
                _validate_model_completion_stage_dispatch(dispatch, stage)
                binding = child_session_notification_stage_binding(stage.intent)
                if binding is not None:
                    for claim in binding.claims:
                        child = _load_session(connection, claim.child_session_id)
                        event_row = (
                            None
                            if child is None
                            else connection.execute(
                                "SELECT * FROM cayu_events "
                                "WHERE session_id = ? "
                                "AND event_type IN (?, ?, ?, ?, ?, ?) "
                                "ORDER BY sequence DESC LIMIT 1",
                                (
                                    claim.child_session_id,
                                    str(EventType.SESSION_STARTED),
                                    str(EventType.SESSION_RESUMED),
                                    str(EventType.SESSION_FORKED),
                                    str(EventType.SESSION_COMPLETED),
                                    str(EventType.SESSION_FAILED),
                                    str(EventType.SESSION_INTERRUPTED),
                                ),
                            ).fetchone()
                        )
                        event_record = _event_record_from_row(event_row)
                        if child is None or event_record is None:
                            raise SessionModelCompletionStageConflict(
                                "Child-session notification occurrence is no longer canonical."
                            )
                        occurrence = ChildSessionLifecycleOccurrence(
                            source=ChildSessionLifecycleOccurrenceSource.EVENT,
                            source_id=event_record.event.id,
                            source_sequence=event_record.sequence,
                            source_type=str(event_record.event.type),
                            occurred_at=event_record.event.timestamp,
                        )
                        consumption = _child_session_notification_consumption_record(
                            parent=loaded,
                            child=child,
                            occurrence=occurrence,
                            stage=stage,
                            consumed_at=published_at,
                        )
                        consumption_key = child_session_notification_storage_key(
                            child.instance_id, occurrence.source_id
                        )
                        consumption_row = connection.execute(
                            "SELECT record_json FROM cayu_session_operations "
                            "WHERE session_id = ? AND idempotency_key = ?",
                            (session_id, consumption_key),
                        ).fetchone()
                        material = consumption.model_dump(mode="json")
                        if consumption_row is not None:
                            if not _child_session_notification_consumption_replays(
                                json.loads(consumption_row["record_json"]),
                                consumption,
                            ):
                                raise SessionModelCompletionStageConflict(
                                    "Child-session terminal notification was consumed by "
                                    "another stage."
                                )
                        elif consume_child_session_notifications:
                            connection.execute(
                                "INSERT INTO cayu_session_operations "
                                "(session_id, idempotency_key, record_json, updated_at) "
                                "VALUES (?, ?, ?, ?)",
                                (
                                    session_id,
                                    consumption_key,
                                    sqlite_support.json_dumps(material),
                                    sqlite_support.format_datetime(published_at),
                                ),
                            )
                formatted_at = sqlite_support.format_datetime(published_at)
                cursor = connection.execute(
                    "UPDATE cayu_sessions SET updated_at = ?, last_activity_at = ? WHERE id = ?",
                    (formatted_at, formatted_at, session_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"Session not found: {session_id}")
                connection.commit()
                return dispatch
            except BaseException:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def _prepare_model_completion_stage_atomic(
        self,
        prepared: _PreparedModelCompletionStage,
    ) -> ModelCompletionStageResult:
        session_id = prepared.session_id

        def statement(connection: sqlite3.Connection) -> ModelCompletionStageResult:
            try:
                connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                rows = connection.execute(
                    "SELECT idempotency_key, record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key IN (?, ?, ?)",
                    (
                        session_id,
                        prepared.preparation_storage_key,
                        prepared.terminal_storage_key,
                        prepared.abandonment_storage_key,
                    ),
                ).fetchall()
                records = {
                    row["idempotency_key"]: _decode_model_completion_stage_record(
                        row["record_json"]
                    )
                    for row in rows
                }
                stage = _reconstruct_model_completion_stage(
                    records.get(prepared.preparation_storage_key),
                    records.get(prepared.terminal_storage_key),
                    session_id=session_id,
                    stage_id=prepared.request.stage_id,
                    preparation_storage_key=prepared.preparation_storage_key,
                    terminal_storage_key=prepared.terminal_storage_key,
                )
                _validate_model_completion_stage_repreparation(
                    records.get(prepared.abandonment_storage_key),
                    prepared,
                    source_status=loaded.status if stage is None else stage.source_status,
                )
                if stage is not None:
                    _validate_model_completion_stage_preparation_replay(stage, prepared)

                active_row = connection.execute(
                    "SELECT record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                ).fetchone()
                active = None
                if active_row is not None:
                    active_record = _decode_model_completion_stage_record(active_row["record_json"])
                    marker = _reconstruct_active_model_completion_stage_record(
                        active_record,
                        session_id=session_id,
                    )
                    _, _, active_preparation_key, active_terminal_key = (
                        _model_completion_stage_storage_identity(
                            session_id,
                            marker.stage_id,
                        )
                    )
                    active_rows = connection.execute(
                        "SELECT idempotency_key, record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key IN (?, ?)",
                        (session_id, active_preparation_key, active_terminal_key),
                    ).fetchall()
                    active_records = {
                        row["idempotency_key"]: _decode_model_completion_stage_record(
                            row["record_json"]
                        )
                        for row in active_rows
                    }
                    active = _reconstruct_active_model_completion_stage(
                        active_record,
                        active_records.get(active_preparation_key),
                        active_records.get(active_terminal_key),
                        session_id=session_id,
                    )
                publication_rows = connection.execute(
                    "SELECT idempotency_key FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key IN (?, ?)",
                    (
                        session_id,
                        prepared.winner_storage_key,
                        prepared.publication_storage_key,
                    ),
                ).fetchall()
                publication_keys = {row["idempotency_key"] for row in publication_rows}
                winner_exists = prepared.winner_storage_key in publication_keys
                receipt_exists = prepared.publication_storage_key in publication_keys
                if stage is not None:
                    _validate_model_completion_preparation_replay_state(
                        stage,
                        active=active,
                        winner_exists=winner_exists,
                        receipt_exists=receipt_exists,
                    )
                    connection.rollback()
                    return ModelCompletionStageResult(
                        stage=stage,
                        replayed=True,
                        dispatch_authorized=False,
                    )
                _validate_model_completion_active_marker_for_preparation(
                    active,
                    prepared,
                    source_status=loaded.status,
                )
                retry_settlement_request = _model_completion_retry_settlement_request(
                    active,
                    prepared,
                )
                if winner_exists or receipt_exists:
                    raise SessionModelCompletionStageConflict(
                        "The logical model step already has durable publication state."
                    )

                _assert_session_run_epoch(session_id, loaded)
                if loaded.status not in prepared.expected_statuses:
                    raise SessionStatusConflict(
                        "Session status is not eligible for model-completion preparation: "
                        f"{loaded.status}"
                    )
                if loaded.run_epoch != prepared.expected_run_epoch:
                    raise SessionRunFenced(
                        "Session source run epoch is stale: expected "
                        f"{prepared.expected_run_epoch}, current {loaded.run_epoch}."
                    )
                current_cursor = _transcript_cursor(connection, session_id)
                if current_cursor != prepared.expected_transcript_cursor:
                    raise ValueError(
                        "Session source transcript cursor is stale: expected "
                        f"{prepared.expected_transcript_cursor}, current {current_cursor}."
                    )

                prepared_at = _next_runtime_publication_timestamp(loaded)
                record = _model_completion_stage_preparation_record(
                    prepared,
                    source_session=loaded,
                    prepared_at=prepared_at,
                )
                stage = _reconstruct_model_completion_stage(
                    record,
                    None,
                    session_id=session_id,
                    stage_id=prepared.request.stage_id,
                    preparation_storage_key=prepared.preparation_storage_key,
                    terminal_storage_key=prepared.terminal_storage_key,
                )
                assert stage is not None
                formatted_at = sqlite_support.format_datetime(prepared_at)
                connection.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record_json, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        session_id,
                        prepared.preparation_storage_key,
                        sqlite_support.json_dumps(record),
                        formatted_at,
                    ),
                )
                active_record = _active_model_completion_stage_record(
                    stage,
                    activated_at=prepared_at,
                )
                if retry_settlement_request is not None:
                    assert active is not None
                    retry_settlement_storage_key = _model_completion_stage_settlement_storage_key(
                        active.stage.stage_id
                    )
                    settlement_row = connection.execute(
                        "SELECT record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key = ?",
                        (session_id, retry_settlement_storage_key),
                    ).fetchone()
                    _validate_model_completion_stage_for_settlement(
                        session=loaded,
                        stage=active.stage,
                        active=active,
                        request=retry_settlement_request,
                        settlement_record=(
                            None
                            if settlement_row is None
                            else _decode_model_completion_stage_record(
                                settlement_row["record_json"]
                            )
                        ),
                        winner_exists=winner_exists,
                        receipt_exists=receipt_exists,
                    )
                    retry_settlement_record = _model_completion_stage_settlement_record(
                        active.stage,
                        request=retry_settlement_request,
                        settled_at=prepared_at,
                    )
                    connection.execute(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record_json, updated_at) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            session_id,
                            retry_settlement_storage_key,
                            sqlite_support.json_dumps(retry_settlement_record),
                            formatted_at,
                        ),
                    )
                connection.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record_json, updated_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                    "record_json = excluded.record_json, updated_at = excluded.updated_at",
                    (
                        session_id,
                        MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                        sqlite_support.json_dumps(active_record),
                        formatted_at,
                    ),
                )
                cursor = connection.execute(
                    "UPDATE cayu_sessions SET updated_at = ?, last_activity_at = ? WHERE id = ?",
                    (formatted_at, formatted_at, session_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"Session not found: {session_id}")
                connection.commit()
                return ModelCompletionStageResult(
                    stage=stage,
                    replayed=False,
                    dispatch_authorized=True,
                )
            except BaseException:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def _complete_model_completion_stage_atomic(
        self,
        prepared: _PreparedModelCompletionStageTerminal,
    ) -> ModelCompletionStageResult:
        session_id = prepared.session_id

        def statement(connection: sqlite3.Connection) -> ModelCompletionStageResult:
            try:
                connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                rows = connection.execute(
                    "SELECT idempotency_key, record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key IN (?, ?, ?)",
                    (
                        session_id,
                        prepared.preparation_storage_key,
                        prepared.terminal_storage_key,
                        prepared.settlement_storage_key,
                    ),
                ).fetchall()
                records = {
                    row["idempotency_key"]: _decode_model_completion_stage_record(
                        row["record_json"]
                    )
                    for row in rows
                }
                stage = _reconstruct_model_completion_stage(
                    records.get(prepared.preparation_storage_key),
                    records.get(prepared.terminal_storage_key),
                    session_id=session_id,
                    stage_id=prepared.stage_id,
                    preparation_storage_key=prepared.preparation_storage_key,
                    terminal_storage_key=prepared.terminal_storage_key,
                )
                if stage is None:
                    raise KeyError(f"Model-completion stage not found: {prepared.stage_id}")
                _reject_settled_model_completion_stage(
                    records.get(prepared.settlement_storage_key),
                    session_id=session_id,
                    stage_id=prepared.stage_id,
                    settlement_storage_key=prepared.settlement_storage_key,
                )
                if stage.state == "completed":
                    _validate_model_completion_stage_terminal_replay(stage, prepared)
                    connection.rollback()
                    return ModelCompletionStageResult(
                        stage=stage,
                        replayed=True,
                        dispatch_authorized=False,
                    )
                _validate_model_completion_stage_publication(
                    prepared.publication,
                    session_id=session_id,
                    stage=stage,
                )
                if not _runtime_publication_json_equal(prepared.publication.intent, stage.intent):
                    raise SessionModelCompletionStageConflict(
                        "The terminal model completion intent conflicts with its preparation."
                    )
                completed_at = _next_runtime_publication_timestamp(loaded)
                terminal_record = _model_completion_stage_terminal_record(
                    prepared,
                    stage=stage,
                    completed_at=completed_at,
                )
                completed_stage = _reconstruct_model_completion_stage(
                    records[prepared.preparation_storage_key],
                    terminal_record,
                    session_id=session_id,
                    stage_id=prepared.stage_id,
                    preparation_storage_key=prepared.preparation_storage_key,
                    terminal_storage_key=prepared.terminal_storage_key,
                )
                assert completed_stage is not None
                active_row = connection.execute(
                    "SELECT record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                ).fetchone()
                active_record = (
                    None
                    if active_row is None
                    else _decode_model_completion_stage_record(active_row["record_json"])
                )
                advances_last_activity = _model_completion_terminal_advances_last_activity(
                    active_record,
                    stage=stage,
                    current_run_epoch=loaded.run_epoch,
                )
                formatted_at = sqlite_support.format_datetime(completed_at)
                connection.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record_json, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        session_id,
                        prepared.terminal_storage_key,
                        sqlite_support.json_dumps(terminal_record),
                        formatted_at,
                    ),
                )
                cursor = connection.execute(
                    "UPDATE cayu_sessions SET updated_at = ?, "
                    "last_activity_at = CASE WHEN ? THEN ? ELSE last_activity_at END "
                    "WHERE id = ?",
                    (
                        formatted_at,
                        advances_last_activity,
                        formatted_at,
                        session_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"Session not found: {session_id}")
                connection.commit()
                return ModelCompletionStageResult(
                    stage=completed_stage,
                    replayed=False,
                    dispatch_authorized=False,
                )
            except BaseException:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def _abandon_model_completion_stage_atomic(
        self,
        prepared: _PreparedModelCompletionStageAbandonment,
    ) -> ModelCompletionStageAbandonmentResult:
        session_id = prepared.session_id

        def statement(
            connection: sqlite3.Connection,
        ) -> ModelCompletionStageAbandonmentResult:
            try:
                connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                rows = connection.execute(
                    "SELECT idempotency_key, record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key IN (?, ?, ?)",
                    (
                        session_id,
                        prepared.preparation_storage_key,
                        prepared.terminal_storage_key,
                        prepared.abandonment_storage_key,
                    ),
                ).fetchall()
                records = {
                    row["idempotency_key"]: _decode_model_completion_stage_record(
                        row["record_json"]
                    )
                    for row in rows
                }
                stage = _reconstruct_model_completion_stage(
                    records.get(prepared.preparation_storage_key),
                    records.get(prepared.terminal_storage_key),
                    session_id=session_id,
                    stage_id=prepared.stage_id,
                    preparation_storage_key=prepared.preparation_storage_key,
                    terminal_storage_key=prepared.terminal_storage_key,
                )
                active_row = connection.execute(
                    "SELECT record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                ).fetchone()
                active_record = (
                    None
                    if active_row is None
                    else _decode_model_completion_stage_record(active_row["record_json"])
                )
                if stage is None:
                    if active_record is not None:
                        active_marker = _reconstruct_active_model_completion_stage_record(
                            active_record,
                            session_id=session_id,
                        )
                        if active_marker.stage_id == prepared.stage_id:
                            raise SessionModelCompletionStageConflict(
                                "The active model-completion marker references a missing stage."
                            )
                    replayed = _replay_model_completion_stage_abandonment(
                        prepared,
                        records.get(prepared.abandonment_storage_key),
                    )
                    publication_rows = connection.execute(
                        "SELECT idempotency_key FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key IN (?, ?)",
                        (
                            session_id,
                            _model_completion_stage_winner_storage_key(
                                replayed.abandonment.logical_step_id
                            ),
                            _runtime_publication_storage_key(replayed.abandonment.logical_step_id),
                        ),
                    ).fetchall()
                    if publication_rows:
                        raise SessionModelCompletionStageConflict(
                            "An abandoned model-completion stage has durable publication state."
                        )
                    connection.rollback()
                    return replayed

                active = _reconstruct_active_model_completion_stage(
                    active_record,
                    records.get(prepared.preparation_storage_key),
                    records.get(prepared.terminal_storage_key),
                    session_id=session_id,
                )
                winner_storage_key = _model_completion_stage_winner_storage_key(
                    stage.logical_step_id
                )
                publication_storage_key = _runtime_publication_storage_key(stage.logical_step_id)
                dispatch_storage_key = _model_completion_stage_dispatch_storage_key(stage.stage_id)
                publication_rows = connection.execute(
                    "SELECT idempotency_key FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key IN (?, ?, ?)",
                    (
                        session_id,
                        winner_storage_key,
                        publication_storage_key,
                        dispatch_storage_key,
                    ),
                ).fetchall()
                publication_keys = {row["idempotency_key"] for row in publication_rows}
                _validate_model_completion_stage_for_abandonment(
                    session=loaded,
                    stage=stage,
                    active=active,
                    prepared=prepared,
                    abandonment_record=records.get(prepared.abandonment_storage_key),
                    dispatch_exists=dispatch_storage_key in publication_keys,
                    winner_exists=winner_storage_key in publication_keys,
                    receipt_exists=publication_storage_key in publication_keys,
                )
                assert active is not None
                abandoned_at = _next_runtime_publication_timestamp(loaded)
                abandonment_record = _model_completion_stage_abandonment_record(
                    stage,
                    active=active,
                    abandoned_at=abandoned_at,
                )
                abandonment = _reconstruct_model_completion_stage_abandonment(
                    abandonment_record,
                    session_id=session_id,
                    stage_id=prepared.stage_id,
                    storage_key=prepared.abandonment_storage_key,
                )
                formatted_at = sqlite_support.format_datetime(abandoned_at)
                connection.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record_json, updated_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                    "record_json = excluded.record_json, updated_at = excluded.updated_at",
                    (
                        session_id,
                        prepared.abandonment_storage_key,
                        sqlite_support.json_dumps(abandonment_record),
                        formatted_at,
                    ),
                )
                deleted_preparation = connection.execute(
                    "DELETE FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, prepared.preparation_storage_key),
                )
                if deleted_preparation.rowcount != 1:
                    raise SessionModelCompletionStageConflict(
                        "The model-completion preparation changed during abandonment."
                    )
                deleted_active = connection.execute(
                    "DELETE FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                )
                if deleted_active.rowcount != 1:
                    raise SessionModelCompletionStageConflict(
                        "The active model-completion marker changed during abandonment."
                    )
                cursor = connection.execute(
                    "UPDATE cayu_sessions SET updated_at = ?, last_activity_at = ? WHERE id = ?",
                    (formatted_at, formatted_at, session_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"Session not found: {session_id}")
                connection.commit()
                return ModelCompletionStageAbandonmentResult(
                    abandonment=abandonment,
                    replayed=False,
                )
            except BaseException:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def _promote_model_completion_stage_atomic(
        self,
        *,
        session_id: str,
        stage_id: str,
        preparation_storage_key: str,
        terminal_storage_key: str,
        expected_run_epoch: int,
    ) -> RuntimePublicationResult:
        stage = await self.load_model_completion_stage(session_id, stage_id)
        if stage is None:
            raise KeyError(f"Model-completion stage not found: {stage_id}")
        prepared = _prepare_model_completion_stage_promotion(
            stage,
            expected_run_epoch=expected_run_epoch,
        )
        assert stage.completion_digest is not None
        return await self._publish_runtime_publication_atomic(
            prepared,
            _model_completion_stage=_ModelCompletionStagePromotionContext(
                stage_id=stage_id,
                preparation_storage_key=preparation_storage_key,
                terminal_storage_key=terminal_storage_key,
                active_storage_key=MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                winner_storage_key=_model_completion_stage_winner_storage_key(
                    stage.logical_step_id
                ),
                completion_digest=stage.completion_digest,
            ),
        )

    async def _publish_runtime_publication_atomic(
        self,
        prepared: _PreparedRuntimePublication,
        *,
        _model_completion_stage: _ModelCompletionStagePromotionContext | None = None,
    ) -> RuntimePublicationResult:
        from cayu.runtime.pending_actions import (
            pending_action_event_storage_values,
            pending_action_lookup_key,
        )

        session_id = prepared.session_id
        request = prepared.request
        checkpoint_codec = prepared.checkpoint_codec
        checkpoint_decode = None if checkpoint_codec is None else checkpoint_codec.decode

        def statement(connection: sqlite3.Connection) -> RuntimePublicationResult:
            try:
                connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")

                locked_stage = None
                active_record = None
                winner_record = None
                if _model_completion_stage is not None:
                    stage_rows = connection.execute(
                        "SELECT idempotency_key, record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key IN (?, ?, ?, ?)",
                        (
                            session_id,
                            _model_completion_stage.preparation_storage_key,
                            _model_completion_stage.terminal_storage_key,
                            _model_completion_stage.active_storage_key,
                            _model_completion_stage.winner_storage_key,
                        ),
                    ).fetchall()
                    stage_records = {
                        row["idempotency_key"]: _decode_model_completion_stage_record(
                            row["record_json"]
                        )
                        for row in stage_rows
                    }
                    locked_stage = _reconstruct_model_completion_stage(
                        stage_records.get(_model_completion_stage.preparation_storage_key),
                        stage_records.get(_model_completion_stage.terminal_storage_key),
                        session_id=session_id,
                        stage_id=_model_completion_stage.stage_id,
                        preparation_storage_key=(_model_completion_stage.preparation_storage_key),
                        terminal_storage_key=_model_completion_stage.terminal_storage_key,
                    )
                    if locked_stage is None:
                        raise KeyError(
                            f"Model-completion stage not found: {_model_completion_stage.stage_id}"
                        )
                    if (
                        locked_stage.completion_digest != _model_completion_stage.completion_digest
                        or locked_stage.publication is None
                        or not _runtime_publication_json_equal(
                            locked_stage.publication.model_dump(mode="json"),
                            prepared.request.model_dump(mode="json"),
                        )
                    ):
                        raise SessionModelCompletionStageConflict(
                            "Model-completion stage changed before atomic promotion."
                        )
                    active_record = stage_records.get(_model_completion_stage.active_storage_key)
                    winner_record = stage_records.get(_model_completion_stage.winner_storage_key)

                receipt_row = connection.execute(
                    "SELECT record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, prepared.storage_key),
                ).fetchone()
                if receipt_row is not None:
                    receipt_record = _decode_runtime_publication_record(receipt_row["record_json"])
                    if locked_stage is not None:
                        replay_active = None
                        if active_record is not None:
                            active_marker = _reconstruct_active_model_completion_stage_record(
                                active_record,
                                session_id=session_id,
                            )
                            _, _, active_preparation_key, active_terminal_key = (
                                _model_completion_stage_storage_identity(
                                    session_id,
                                    active_marker.stage_id,
                                )
                            )
                            active_rows = connection.execute(
                                "SELECT idempotency_key, record_json "
                                "FROM cayu_session_operations "
                                "WHERE session_id = ? AND idempotency_key IN (?, ?)",
                                (
                                    session_id,
                                    active_preparation_key,
                                    active_terminal_key,
                                ),
                            ).fetchall()
                            active_records = {
                                row["idempotency_key"]: (
                                    _decode_model_completion_stage_record(row["record_json"])
                                )
                                for row in active_rows
                            }
                            replay_active = _reconstruct_active_model_completion_stage(
                                active_record,
                                active_records.get(active_preparation_key),
                                active_records.get(active_terminal_key),
                                session_id=session_id,
                            )
                        _validate_model_completion_promotion_replay_active_marker(
                            replay_active,
                            locked_stage,
                        )
                        receipt = _reconstruct_runtime_publication_receipt(
                            receipt_record,
                            storage_key=prepared.storage_key,
                            session_id=session_id,
                            publication_id=request.publication_id,
                        )
                        self._validate_runtime_publication_material(connection, receipt)
                        result = _replay_promoted_model_completion_stage(
                            session=loaded,
                            stage=locked_stage,
                            receipt_record=receipt_record,
                            winner_record=winner_record,
                        )
                        connection.rollback()
                        return result
                    receipt = _reconstruct_runtime_publication_receipt(
                        receipt_record,
                        storage_key=prepared.storage_key,
                        session_id=session_id,
                        publication_id=request.publication_id,
                        request_digest=prepared.request_digest,
                    )
                    _validate_runtime_publication_replay_receipt(receipt, prepared)
                    self._validate_runtime_publication_material(connection, receipt)
                    connection.rollback()
                    return RuntimePublicationResult(
                        session=loaded.model_copy(deep=True),
                        receipt=receipt,
                        replayed=True,
                    )

                operation_mutation_records: dict[str, dict[str, Any]] = {}
                if request.operation_record_mutations:
                    mutation_keys = tuple(
                        mutation.key for mutation in request.operation_record_mutations
                    )
                    placeholders = ", ".join("?" for _ in mutation_keys)
                    mutation_rows = connection.execute(
                        "SELECT idempotency_key, record_json FROM cayu_session_operations "
                        f"WHERE session_id = ? AND idempotency_key IN ({placeholders})",
                        (session_id, *mutation_keys),
                    ).fetchall()
                    current_mutation_records = {
                        row["idempotency_key"]: json.loads(row["record_json"])
                        for row in mutation_rows
                    }
                    operation_mutation_records = (
                        _apply_runtime_publication_operation_record_mutations(
                            request.operation_record_mutations,
                            current_mutation_records,
                        )
                    )

                if locked_stage is not None:
                    assert _model_completion_stage is not None
                    if winner_record is not None:
                        raise SessionModelCompletionStageConflict(
                            "A model-completion winner exists without its runtime publication "
                            "receipt."
                        )
                    if active_record is not None:
                        active_marker = _reconstruct_active_model_completion_stage_record(
                            active_record,
                            session_id=session_id,
                        )
                        if active_marker.stage_id != locked_stage.stage_id:
                            raise SessionModelCompletionStageConflict(
                                "The model-completion stage was superseded before promotion."
                            )
                    active = _reconstruct_active_model_completion_stage(
                        active_record,
                        stage_records.get(_model_completion_stage.preparation_storage_key),
                        stage_records.get(_model_completion_stage.terminal_storage_key),
                        session_id=session_id,
                    )
                    _validate_model_completion_active_marker_for_promotion(
                        active,
                        locked_stage,
                    )

                _assert_session_run_epoch(session_id, loaded)
                if (
                    prepared.expected_statuses is not None
                    and loaded.status not in prepared.expected_statuses
                ):
                    raise SessionStatusConflict(
                        f"Session status is not eligible for runtime publication: {loaded.status}"
                    )
                if (
                    prepared.expected_run_epoch is not None
                    and loaded.run_epoch != prepared.expected_run_epoch
                ):
                    raise SessionRunFenced(
                        "Session source run epoch is stale: expected "
                        f"{prepared.expected_run_epoch}, current {loaded.run_epoch}."
                    )
                transcript_start_cursor = _transcript_cursor(connection, session_id)
                if (
                    prepared.expected_transcript_cursor is not None
                    and transcript_start_cursor != prepared.expected_transcript_cursor
                ):
                    raise ValueError(
                        "Session source transcript cursor is stale: expected "
                        f"{prepared.expected_transcript_cursor}, "
                        f"current {transcript_start_cursor}."
                    )

                appended_event_ids = {event.id for event in request.events}
                referenced_event_ids = set(
                    _runtime_publication_referenced_event_ids(request.referenced_events)
                )
                if appended_event_ids & referenced_event_ids:
                    raise ValueError("Appended and referenced runtime publication events overlap.")
                durable_referenced_events: dict[str, Event] = {}
                if referenced_event_ids:
                    placeholders = ", ".join("?" for _ in referenced_event_ids)
                    rows = connection.execute(
                        f"SELECT {', '.join(_EVENT_COLUMN_NAMES)} FROM cayu_events "
                        f"WHERE session_id = ? AND event_id IN ({placeholders})",
                        (session_id, *referenced_event_ids),
                    ).fetchall()
                    durable_referenced_events = {
                        row["event_id"]: _event_from_row(row) for row in rows
                    }
                _validate_runtime_publication_event_references(
                    request.referenced_events,
                    durable_referenced_events,
                    interaction_id=request.interaction_id,
                )
                stored_checkpoint = self._load_checkpoint_unlocked(session_id)
                current_checkpoint = (
                    stored_checkpoint
                    if checkpoint_decode is None
                    else checkpoint_decode(loaded, stored_checkpoint)
                )
                _validate_user_input_checkpoint_mutation(
                    request,
                    current_checkpoint,
                    session_id=session_id,
                    session_instance_id=loaded.instance_id,
                    current_run_epoch=loaded.run_epoch,
                    durable_events_by_id=durable_referenced_events,
                )
                _validate_tool_round_checkpoint_mutation(
                    request,
                    current_checkpoint,
                )
                durable_tool_events: list[Event] = []
                tool_round_identity = _tool_lifecycle_publication_identity(request)
                if tool_round_identity is not None:
                    execution_identity, tool_call_ids = tool_round_identity
                    lookup_keys = tuple(
                        pending_action_lookup_key(tool_call_id) for tool_call_id in tool_call_ids
                    )
                    lifecycle_event_types = tuple(
                        sorted(str(event_type) for event_type in _TOOL_ROUND_LIFECYCLE_EVENT_TYPES)
                    )
                    lookup_placeholders = ", ".join("?" for _ in lookup_keys)
                    event_type_placeholders = ", ".join("?" for _ in lifecycle_event_types)
                    rows = connection.execute(
                        f"SELECT {', '.join(_EVENT_COLUMN_NAMES)} FROM cayu_events "
                        "INDEXED BY idx_cayu_events_pending_action_lookup "
                        f"WHERE session_id = ? AND pending_action_lookup_key IN "
                        f"({lookup_placeholders}) AND event_type IN "
                        f"({event_type_placeholders}) AND "
                        f"({_PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
                        "AND (json_extract(payload_json, '$.tool_round_id') = ? "
                        "OR (json_extract(payload_json, '$.model_step_id') = ? "
                        "AND json_extract(payload_json, '$.model_attempt_id') = ?) "
                        "OR cayu_is_execution_unit_id("
                        "json_extract(payload_json, '$.tool_round_id'), 'tool_round_id') = 0 "
                        "OR cayu_is_execution_unit_id("
                        "json_extract(payload_json, '$.model_step_id'), 'model_step_id') = 0 "
                        "OR cayu_is_execution_unit_id("
                        "json_extract(payload_json, '$.model_attempt_id'), 'model_attempt_id') = 0) "
                        "ORDER BY sequence ASC LIMIT ?",
                        (
                            session_id,
                            *lookup_keys,
                            *lifecycle_event_types,
                            execution_identity.tool_round_id,
                            execution_identity.model_step_id,
                            execution_identity.model_attempt_id,
                            RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS + 1,
                        ),
                    ).fetchall()
                    if len(rows) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
                        raise ValueError(
                            "Tool-round lifecycle evidence exceeds the publication limit."
                        )
                    durable_tool_events = [_event_from_row(row) for row in rows]
                _validate_tool_round_publication(
                    request,
                    durable_referenced_events,
                    durable_tool_events=durable_tool_events,
                )
                existing_event_id = _first_existing_event_id(
                    connection,
                    session_id,
                    [event.id for event in request.events],
                )
                if existing_event_id is not None:
                    raise ValueError(
                        f"Event already exists for session {session_id}: {existing_event_id}"
                    )

                if checkpoint_codec is None:
                    checkpoint = _apply_runtime_publication_checkpoint_mutation(
                        request.mutation,
                        current_checkpoint,
                    )
                    stored_target_checkpoint = checkpoint
                else:
                    checkpoint = checkpoint_codec.apply_mutation(
                        loaded,
                        stored_checkpoint,
                        request.mutation,
                    )
                    stored_target_checkpoint = checkpoint_codec.encode(loaded, checkpoint)

                transcript_rows = [
                    (
                        session_id,
                        str(message.role),
                        request.interaction_id,
                        sqlite_support.json_dumps(message_payload),
                        transcript_search_document(message),
                    )
                    for message, message_payload in zip(
                        request.transcript_messages,
                        prepared.transcript_payloads,
                        strict=True,
                    )
                ]
                event_rows = []
                for event, event_payload in zip(
                    request.events,
                    prepared.event_payloads,
                    strict=True,
                ):
                    lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                        event
                    )
                    event_rows.append(
                        (
                            session_id,
                            event.id,
                            event.interaction_id,
                            str(event.type),
                            sqlite_support.format_datetime(event.timestamp),
                            event.agent_name,
                            event.environment_name,
                            event.workflow_name,
                            event.tool_name,
                            sqlite_support.json_dumps(event_payload["payload"]),
                            lookup_key,
                            projection,
                            projection_bytes,
                        )
                    )

                published_at = _next_runtime_publication_timestamp(loaded)
                checkpoint_values = (
                    None
                    if stored_target_checkpoint is None or not request.mutation.operations
                    else sqlite_support.checkpoint_row_values(
                        session_id,
                        stored_target_checkpoint,
                        published_at,
                    )
                )
                receipt = _build_runtime_publication_receipt(
                    prepared,
                    source_session=loaded,
                    checkpoint=stored_target_checkpoint,
                    transcript_start_cursor=transcript_start_cursor,
                    published_at=published_at,
                )
                receipt_json = sqlite_support.json_dumps(
                    _runtime_publication_receipt_record(receipt)
                )
                formatted_published_at = sqlite_support.format_datetime(published_at)

                if transcript_rows:
                    connection.executemany(
                        """
                        INSERT INTO cayu_transcript_messages (
                            session_id, role, interaction_id, message_json,
                            transcript_search_document
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        transcript_rows,
                    )
                if checkpoint_values is not None:
                    connection.execute(
                        """
                        INSERT INTO cayu_checkpoints (
                            session_id, state_json, updated_at,
                            pending_action_source_bytes,
                            pending_action_tool_call_count,
                            pending_action_flags,
                            pending_action_metrics_ready
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(session_id) DO UPDATE SET
                            state_json = excluded.state_json,
                            updated_at = excluded.updated_at,
                            pending_action_source_bytes = excluded.pending_action_source_bytes,
                            pending_action_tool_call_count = excluded.pending_action_tool_call_count,
                            pending_action_flags = excluded.pending_action_flags,
                            pending_action_metrics_ready = excluded.pending_action_metrics_ready
                        """,
                        checkpoint_values,
                    )
                if event_rows:
                    connection.executemany(
                        """
                        INSERT INTO cayu_events (
                            session_id, event_id, interaction_id, event_type, timestamp,
                            agent_name, environment_name, workflow_name, tool_name,
                            payload_json, pending_action_lookup_key,
                            pending_action_projection_json, pending_action_projection_bytes
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        event_rows,
                    )
                    _enqueue_persisted_event_side_effects(
                        connection,
                        session_id,
                        request.events,
                    )
                if operation_mutation_records:
                    connection.executemany(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record_json, updated_at) "
                        "VALUES (?, ?, ?, ?) "
                        "ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                        "record_json = excluded.record_json, updated_at = excluded.updated_at",
                        [
                            (
                                session_id,
                                key,
                                sqlite_support.json_dumps(record),
                                formatted_published_at,
                            )
                            for key, record in operation_mutation_records.items()
                        ],
                    )
                connection.execute(
                    """
                    INSERT INTO cayu_session_operations (
                        session_id, idempotency_key, record_json, updated_at
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        prepared.storage_key,
                        receipt_json,
                        formatted_published_at,
                    ),
                )
                if locked_stage is not None and _model_completion_stage is not None:
                    winner = _model_completion_stage_winner_record(
                        locked_stage,
                        receipt=receipt,
                    )
                    connection.execute(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record_json, updated_at) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            session_id,
                            _model_completion_stage.winner_storage_key,
                            sqlite_support.json_dumps(winner),
                            formatted_published_at,
                        ),
                    )
                    active_delete = connection.execute(
                        "DELETE FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key = ?",
                        (session_id, _model_completion_stage.active_storage_key),
                    )
                    if active_delete.rowcount != 1:
                        raise SessionModelCompletionStageConflict(
                            "The active model-completion marker changed before commit."
                        )
                cursor = connection.execute(
                    """
                    UPDATE cayu_sessions
                    SET updated_at = ?, last_activity_at = ?
                    WHERE id = ?
                    """,
                    (formatted_published_at, formatted_published_at, session_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"Session not found: {session_id}")
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                existing_event_id = _first_existing_event_id(
                    connection,
                    session_id,
                    [event.id for event in request.events],
                )
                if existing_event_id is not None:
                    raise ValueError(
                        f"Event already exists for session {session_id}: {existing_event_id}"
                    ) from exc
                receipt_row = connection.execute(
                    "SELECT 1 FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, prepared.storage_key),
                ).fetchone()
                if receipt_row is not None:
                    raise SessionRuntimePublicationConflict(
                        "Runtime publication receipt was inserted concurrently."
                    ) from exc
                raise
            except BaseException:
                connection.rollback()
                raise
            updated = loaded.model_copy(
                update={
                    "updated_at": published_at,
                    "last_activity_at": published_at,
                },
                deep=True,
            )
            return RuntimePublicationResult(
                session=updated,
                receipt=receipt,
                replayed=False,
            )

        return await self._run_write(statement)

    async def publish_session_operation(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        return await self._publish_checkpoint_and_events(
            session_id,
            checkpoint_transform=None,
            operation_idempotency_key=_reject_reserved_runtime_publication_key(
                idempotency_key,
                "idempotency_key",
            ),
            operation_transform=operation_transform,
            store_time_operation_transform=None,
            operation_commit_guard=None,
            operation_commit_time_guard=None,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
            preserve_completion_result_publications=True,
        )

    async def publish_session_operation_guarded(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        commit_guard: Callable[[], None],
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        return await self._publish_checkpoint_and_events(
            session_id,
            checkpoint_transform=None,
            operation_idempotency_key=require_clean_nonblank(
                idempotency_key,
                "idempotency_key",
            ),
            operation_transform=operation_transform,
            store_time_operation_transform=None,
            operation_commit_guard=commit_guard,
            operation_commit_time_guard=None,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
            preserve_completion_result_publications=True,
        )

    async def publish_session_operation_guarded_with_store_time(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: StoreTimeSessionOperationTransform,
        commit_guard: Callable[[], None],
        commit_time_guard: Callable[[datetime], None],
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        return await self._publish_checkpoint_and_events(
            session_id,
            checkpoint_transform=None,
            operation_idempotency_key=require_clean_nonblank(
                idempotency_key,
                "idempotency_key",
            ),
            operation_transform=None,
            store_time_operation_transform=operation_transform,
            operation_commit_guard=commit_guard,
            operation_commit_time_guard=commit_time_guard,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
            preserve_completion_result_publications=True,
        )

    async def _publish_checkpoint_and_events(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform | None,
        store_time_checkpoint_transform: StoreTimeCheckpointTransform | None = None,
        operation_idempotency_key: str | None,
        operation_transform: SessionOperationTransform | None,
        store_time_operation_transform: StoreTimeSessionOperationTransform | None,
        operation_commit_guard: Callable[[], None] | None,
        operation_commit_time_guard: Callable[[datetime], None] | None,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None,
        expected_run_epoch: int | None,
        expected_transcript_cursor: int | None,
        preserve_completion_result_publications: bool,
    ) -> Session:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        session_id, copied_events = _copy_session_event_batch(session_id, events)
        transform_count = sum(
            transform is not None
            for transform in (
                checkpoint_transform,
                store_time_checkpoint_transform,
                operation_transform,
                store_time_operation_transform,
            )
        )
        if transform_count != 1:
            raise TypeError("Exactly one checkpoint publication transform is required.")
        if (
            operation_transform is not None or store_time_operation_transform is not None
        ) and operation_idempotency_key is None:
            raise TypeError("operation_idempotency_key is required.")
        allowed_statuses = (
            None
            if expected_statuses is None
            else _validate_status_set(expected_statuses, "expected_statuses")
        )

        def statement(connection: sqlite3.Connection) -> Session:
            try:
                connection.execute("BEGIN IMMEDIATE")
                updated_at = self._ownership_clock()
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, loaded)
                if allowed_statuses is not None and loaded.status not in allowed_statuses:
                    raise SessionStatusConflict(
                        "Session status is not eligible for checkpoint publication: "
                        f"{loaded.status}"
                    )
                if expected_run_epoch is not None and loaded.run_epoch != expected_run_epoch:
                    raise SessionRunFenced(
                        f"Session source run epoch is stale: expected {expected_run_epoch}, "
                        f"current {loaded.run_epoch}."
                    )
                current_cursor = _transcript_cursor(connection, session_id)
                if (
                    expected_transcript_cursor is not None
                    and current_cursor != expected_transcript_cursor
                ):
                    raise ValueError(
                        "Session source transcript cursor is stale: expected "
                        f"{expected_transcript_cursor}, current {current_cursor}."
                    )
                current_checkpoint = self._load_checkpoint_unlocked(session_id)
                callback_checkpoint = _copy_checkpoint_for_transform(
                    current_checkpoint,
                    session_id=session_id,
                )
                operation_records: dict[str, dict[str, Any]] = {}
                model_completion_stage_release = None
                if operation_transform is not None or store_time_operation_transform is not None:
                    operation_row = connection.execute(
                        "SELECT record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key = ?",
                        (session_id, operation_idempotency_key),
                    ).fetchone()
                    current_operation = (
                        None if operation_row is None else json.loads(operation_row["record_json"])
                    )
                    if operation_transform is not None:
                        publication = operation_transform(
                            loaded,
                            callback_checkpoint,
                            current_operation,
                        )
                    else:
                        assert store_time_operation_transform is not None
                        publication = store_time_operation_transform(
                            loaded,
                            callback_checkpoint,
                            current_operation,
                            updated_at,
                        )
                    if type(publication) is not SessionOperationPublication:
                        raise TypeError(
                            "Session operation transform must return a SessionOperationPublication."
                        )
                    transformed = copy_durable_json_object(
                        publication.checkpoint,
                        "checkpoint",
                    )
                    transformed = _checkpoint_transform_result_preserving_completion_result_event_publications(
                        current_checkpoint,
                        transformed,
                        session_id=session_id,
                    )
                    operation_records = copy_durable_json_object(
                        publication.operation_records,
                        "operation_records",
                    )
                    model_completion_stage_release = publication.model_completion_stage_release
                    _validate_session_operation_record_keys(operation_records)
                else:
                    if checkpoint_transform is not None:
                        transformed = checkpoint_transform(loaded, callback_checkpoint)
                    else:
                        assert store_time_checkpoint_transform is not None
                        transformed = store_time_checkpoint_transform(
                            loaded,
                            callback_checkpoint,
                            updated_at,
                        )
                    if transformed is None:
                        raise ValueError("Checkpoint transform must return a checkpoint.")
                    transformed = copy_durable_json_object(transformed, "checkpoint")
                    transformed = (
                        _replace_checkpoint_preserving_completion_result_event_publications(
                            current_checkpoint,
                            transformed,
                            preserve_completion_result_publications=(
                                preserve_completion_result_publications
                            ),
                            session_id=session_id,
                        )
                    )
                event_rows = []
                for event in copied_events:
                    lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                        event
                    )
                    event_rows.append(
                        (
                            session_id,
                            event.id,
                            event.interaction_id,
                            str(event.type),
                            sqlite_support.format_datetime(event.timestamp),
                            event.agent_name,
                            event.environment_name,
                            event.workflow_name,
                            event.tool_name,
                            sqlite_support.json_dumps(event.payload),
                            lookup_key,
                            projection,
                            projection_bytes,
                        )
                    )
                _publish_budget_reservation_identities(connection, copied_events)
                connection.execute(
                    """
                    INSERT INTO cayu_checkpoints (
                        session_id, state_json, updated_at,
                        pending_action_source_bytes,
                        pending_action_tool_call_count,
                        pending_action_flags,
                        pending_action_metrics_ready
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        state_json = excluded.state_json,
                        updated_at = excluded.updated_at,
                        pending_action_source_bytes = excluded.pending_action_source_bytes,
                        pending_action_tool_call_count = excluded.pending_action_tool_call_count,
                        pending_action_flags = excluded.pending_action_flags,
                        pending_action_metrics_ready = excluded.pending_action_metrics_ready
                    """,
                    sqlite_support.checkpoint_row_values(session_id, transformed, updated_at),
                )
                if operation_records:
                    connection.executemany(
                        """
                        INSERT INTO cayu_session_operations (
                            session_id, idempotency_key, record_json, updated_at
                        )
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(session_id, idempotency_key) DO UPDATE SET
                            record_json = excluded.record_json,
                            updated_at = excluded.updated_at
                        """,
                        [
                            (
                                session_id,
                                key,
                                sqlite_support.json_dumps(record),
                                sqlite_support.format_datetime(updated_at),
                            )
                            for key, record in operation_records.items()
                        ],
                    )
                if model_completion_stage_release is not None:
                    _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
                        session_id,
                        model_completion_stage_release.stage_id,
                    )
                    stage_rows = connection.execute(
                        "SELECT idempotency_key, record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key IN (?, ?, ?)",
                        (
                            session_id,
                            MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                            preparation_key,
                            terminal_key,
                        ),
                    ).fetchall()
                    stage_records = {
                        row["idempotency_key"]: _decode_model_completion_stage_record(
                            row["record_json"]
                        )
                        for row in stage_rows
                    }
                    _validate_model_completion_stage_release(
                        session=loaded,
                        active_record=stage_records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                        preparation_record=stage_records.get(preparation_key),
                        terminal_record=stage_records.get(terminal_key),
                        release=model_completion_stage_release,
                    )
                    deleted = connection.execute(
                        "DELETE FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key = ?",
                        (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                    )
                    if deleted.rowcount != 1:
                        raise SessionModelCompletionStageConflict(
                            "The active model-completion stage changed during disposition."
                        )
                if event_rows:
                    connection.executemany(
                        """
                        INSERT INTO cayu_events (
                            session_id, event_id, interaction_id, event_type, timestamp,
                            agent_name, environment_name, workflow_name, tool_name,
                            payload_json, pending_action_lookup_key,
                            pending_action_projection_json, pending_action_projection_bytes
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        event_rows,
                    )
                    _enqueue_persisted_event_side_effects(
                        connection,
                        session_id,
                        copied_events,
                    )
                if operation_commit_guard is not None:
                    operation_commit_guard()
                activity_at = (
                    self._ownership_clock() if operation_commit_guard is not None else updated_at
                )
                if operation_commit_time_guard is not None:
                    activity_at = self._ownership_clock()
                    operation_commit_time_guard(activity_at)
                _touch_session_activity(connection, session_id, activity_at)
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                existing_event_id = _first_existing_event_id(
                    connection,
                    session_id,
                    [event.id for event in copied_events],
                )
                if existing_event_id is not None:
                    raise ValueError(
                        f"Event already exists for session {session_id}: {existing_event_id}"
                    ) from exc
                raise
            except BaseException:
                connection.rollback()
                raise
            return loaded.model_copy(
                update={"updated_at": updated_at, "last_activity_at": activity_at}
            )

        return await self._run_write(statement)

    async def load_events(self, session_id: str) -> list[Event]:
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection: sqlite3.Connection) -> list[Event]:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            rows = connection.execute(
                f"""
                SELECT {", ".join(_EVENT_COLUMN_NAMES)}
                FROM cayu_events
                WHERE session_id = ?
                ORDER BY sequence ASC
                """,
                (session_id,),
            ).fetchall()
            return [_event_from_row(row) for row in rows]

        return await self._run_read(query)

    async def load_user_input_supersession_events(
        self,
        session_id: str,
        input_id: str,
    ) -> list[Event]:
        from cayu.runtime.pending_actions import pending_action_lookup_key

        session_id = require_clean_nonblank(session_id, "session_id")
        input_id = require_clean_nonblank(input_id, "input_id")
        lookup_key = pending_action_lookup_key(input_id)

        def query(connection: sqlite3.Connection) -> list[Event]:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            rows = connection.execute(
                f"SELECT {', '.join(_EVENT_COLUMN_NAMES)} FROM cayu_events "
                "INDEXED BY idx_cayu_events_pending_action_lookup "
                "WHERE session_id = ? AND pending_action_lookup_key = ? "
                "AND event_type = ? AND "
                f"({_PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
                "AND json_extract(payload_json, "
                "'$.user_input_supersession_intent.input_id') = ? "
                "ORDER BY sequence ASC LIMIT 2",
                (
                    session_id,
                    lookup_key,
                    str(EventType.SESSION_INTERRUPTED),
                    input_id,
                ),
            ).fetchall()
            return [_event_from_row(row) for row in rows]

        return await self._run_read(query)

    async def load_tool_round_lifecycle_events(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
    ) -> list[Event]:
        from cayu.runtime.pending_actions import pending_action_lookup_key

        session_id = require_clean_nonblank(session_id, "session_id")
        copied_ids = _validate_tool_round_call_ids(tool_call_ids, "tool_call_ids")
        lookup_keys = tuple(pending_action_lookup_key(call_id) for call_id in copied_ids)
        lifecycle_event_types = tuple(
            sorted(str(event_type) for event_type in _TOOL_ROUND_LIFECYCLE_EVENT_TYPES)
        )

        def query(connection: sqlite3.Connection) -> list[Event]:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            lookup_placeholders = ", ".join("?" for _ in lookup_keys)
            event_type_placeholders = ", ".join("?" for _ in lifecycle_event_types)
            rows = connection.execute(
                f"SELECT {', '.join(_EVENT_COLUMN_NAMES)} FROM cayu_events "
                "INDEXED BY idx_cayu_events_pending_action_lookup "
                f"WHERE session_id = ? AND pending_action_lookup_key IN "
                f"({lookup_placeholders}) AND event_type IN "
                f"({event_type_placeholders}) AND "
                f"({_PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
                "ORDER BY sequence ASC LIMIT ?",
                (
                    session_id,
                    *lookup_keys,
                    *lifecycle_event_types,
                    RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS + 1,
                ),
            ).fetchall()
            if len(rows) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
                raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
            return [_event_from_row(row) for row in rows]

        return await self._run_read(query)

    async def load_tool_round_lifecycle_events_for_round(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
        *,
        tool_round_identity: ToolRoundIdentity,
    ) -> list[Event]:
        from cayu.runtime.pending_actions import pending_action_lookup_key

        session_id = require_clean_nonblank(session_id, "session_id")
        copied_ids = _validate_tool_round_call_ids(tool_call_ids, "tool_call_ids")
        tool_round_identity = copy_tool_round_identity(tool_round_identity)
        lookup_keys = tuple(pending_action_lookup_key(call_id) for call_id in copied_ids)
        lifecycle_event_types = tuple(
            sorted(str(event_type) for event_type in _TOOL_ROUND_LIFECYCLE_EVENT_TYPES)
        )

        def query(connection: sqlite3.Connection) -> list[Event]:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            lookup_placeholders = ", ".join("?" for _ in lookup_keys)
            event_type_placeholders = ", ".join("?" for _ in lifecycle_event_types)
            rows = connection.execute(
                f"SELECT {', '.join(_EVENT_COLUMN_NAMES)} FROM cayu_events "
                "INDEXED BY idx_cayu_events_pending_action_lookup "
                f"WHERE session_id = ? AND pending_action_lookup_key IN "
                f"({lookup_placeholders}) AND event_type IN "
                f"({event_type_placeholders}) AND "
                f"({_PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
                "AND (json_extract(payload_json, '$.tool_round_id') = ? "
                "OR (json_extract(payload_json, '$.model_step_id') = ? "
                "AND json_extract(payload_json, '$.model_attempt_id') = ?) "
                "OR cayu_is_execution_unit_id("
                "json_extract(payload_json, '$.tool_round_id'), 'tool_round_id') = 0 "
                "OR cayu_is_execution_unit_id("
                "json_extract(payload_json, '$.model_step_id'), 'model_step_id') = 0 "
                "OR cayu_is_execution_unit_id("
                "json_extract(payload_json, '$.model_attempt_id'), 'model_attempt_id') = 0) "
                "ORDER BY sequence ASC LIMIT ?",
                (
                    session_id,
                    *lookup_keys,
                    *lifecycle_event_types,
                    tool_round_identity.tool_round_id,
                    tool_round_identity.model_step_id,
                    tool_round_identity.model_attempt_id,
                    RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS + 1,
                ),
            ).fetchall()
            if len(rows) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
                raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
            return [_event_from_row(row) for row in rows]

        return await self._run_read(query)

    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        query = copy_event_query(query)
        if len(query.session_ids) > _EVENT_QUERY_SESSION_IDS_BATCH_SIZE:
            return await self._query_events_by_session_id_batches(query)

        plan = session_store_sql.build_event_query_sql(query, dialect=_SQL_DIALECT)
        params = [*plan.params, query.limit]

        def run_query(connection: sqlite3.Connection) -> list[EventRecord]:
            event_columns = ", ".join(f"cayu_events.{name}" for name in _EVENT_COLUMN_NAMES)
            rows = connection.execute(
                f"""
                SELECT cayu_events.sequence, {event_columns}
                FROM cayu_events
                JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
                {plan.where_sql}
                ORDER BY cayu_events.sequence {plan.order_direction}
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [
                EventRecord(sequence=row["sequence"], event=_event_from_row(row)) for row in rows
            ]

        return await self._run_read(run_query)

    async def query_events_bounded(
        self,
        query: EventQuery,
        *,
        max_bytes: int,
    ) -> list[EventRecord]:
        query = copy_event_query(query)
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer.")
        if len(query.session_ids) > _EVENT_QUERY_SESSION_IDS_BATCH_SIZE:
            raise ValueError("Byte-bounded event queries require one bounded SQL batch.")
        plan = session_store_sql.build_event_query_sql(query, dialect=_SQL_DIALECT)
        params = [*plan.params, query.limit]

        def run_query(connection: sqlite3.Connection) -> list[EventRecord]:
            event_columns = ", ".join(f"cayu_events.{name}" for name in _EVENT_COLUMN_NAMES)
            serialized_bytes = " + ".join(
                [
                    "256",
                    *(
                        f"COALESCE(length(CAST(cayu_events.{name} AS BLOB)), 0)"
                        for name in _EVENT_COLUMN_NAMES
                    ),
                ]
            )
            connection.execute("BEGIN")
            try:
                size_row = connection.execute(
                    f"""
                    WITH bounded_candidates AS (
                        SELECT {serialized_bytes} AS serialized_bytes
                        FROM cayu_events
                        JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
                        {plan.where_sql}
                        ORDER BY cayu_events.sequence {plan.order_direction}
                        LIMIT ?
                    )
                    SELECT COALESCE(SUM(serialized_bytes), 0)
                    FROM bounded_candidates
                    """,
                    params,
                ).fetchone()
                if size_row is None or int(size_row[0]) > max_bytes:
                    raise EventQueryResultTooLarge(max_bytes)
                rows = connection.execute(
                    f"""
                    SELECT cayu_events.sequence, {event_columns}
                    FROM cayu_events
                    JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
                    {plan.where_sql}
                    ORDER BY cayu_events.sequence {plan.order_direction}
                    LIMIT ?
                    """,
                    params,
                ).fetchall()
                return [
                    EventRecord(sequence=row["sequence"], event=_event_from_row(row))
                    for row in rows
                ]
            finally:
                connection.rollback()

        return await self._run_read(run_query)

    async def load_terminal_session_evidence(
        self,
        session_id: str,
        *,
        limits: TerminalSessionEvidenceLimits | None = None,
    ) -> TerminalSessionEvidence:
        return await self._load_terminal_session_evidence(
            session_id,
            limits=limits,
            observed_interrupted_events=None,
            expected_interrupted_parent_session_id=None,
            require_interrupted_proof=False,
        )

    async def load_runner_owned_interrupted_evidence(
        self,
        session_id: str,
        *,
        observed_events: tuple[RunnerObservedEventIdentity, ...] | None = None,
        expected_parent_session_id: str | None = None,
        limits: TerminalSessionEvidenceLimits | None = None,
    ) -> TerminalSessionEvidence:
        return await self._load_terminal_session_evidence(
            session_id,
            limits=limits,
            observed_interrupted_events=observed_events,
            expected_interrupted_parent_session_id=expected_parent_session_id,
            require_interrupted_proof=True,
        )

    async def _load_terminal_session_evidence(
        self,
        session_id: str,
        *,
        limits: TerminalSessionEvidenceLimits | None,
        observed_interrupted_events: tuple[RunnerObservedEventIdentity, ...] | None,
        expected_interrupted_parent_session_id: str | None,
        require_interrupted_proof: bool,
    ) -> TerminalSessionEvidence:
        session_id = require_clean_nonblank(session_id, "session_id")
        resolved_limits = _copy_terminal_session_evidence_limits(limits)
        observed, expected_parent_session_id = _copy_runner_owned_interruption_proof(
            session_id,
            observed_events=observed_interrupted_events,
            expected_parent_session_id=expected_interrupted_parent_session_id,
            limits=resolved_limits,
            required=require_interrupted_proof,
        )
        allow_interrupted = observed is not None or expected_parent_session_id is not None
        evidence_event_types = tuple(
            str(event_type) for event_type in _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES
        )
        evidence_type_placeholders = ", ".join("?" for _ in evidence_event_types)
        event_columns = ", ".join(_EVENT_COLUMN_NAMES)
        event_stored_bytes = " + ".join(
            [
                "length(CAST(sequence AS TEXT))",
                *(f"COALESCE(length(CAST({column} AS BLOB)), 0)" for column in _EVENT_COLUMN_NAMES),
            ]
        )
        session_stored_bytes = " + ".join(
            f"COALESCE(length(CAST(session.{column} AS BLOB)), 0)"
            for column in (
                "id",
                "agent_name",
                "provider_name",
                "model",
                "parent_session_id",
                "causal_budget_id",
                "runtime_name",
                "runtime_version",
                "environment_name",
                "status",
                "created_at",
                "updated_at",
                "last_activity_at",
                "run_epoch",
                "metadata_json",
            )
        )
        transcript_stored_bytes = " + ".join(
            (
                "length(CAST(session_order AS TEXT))",
                "COALESCE(length(CAST(interaction_id AS BLOB)), 0)",
                "length(CAST(message_json AS BLOB))",
            )
        )

        def run_query(connection: sqlite3.Connection) -> TerminalSessionEvidence:
            limits = resolved_limits
            connection.execute("BEGIN")
            try:
                session_preflight = connection.execute(
                    f"""
                    SELECT session.status, session.run_epoch, session.parent_session_id,
                           ({session_stored_bytes})
                           + COALESCE((
                               SELECT SUM(
                                   length(CAST(label.key AS BLOB))
                                   + length(CAST(label.value AS BLOB))
                               )
                               FROM cayu_session_labels AS label
                               WHERE label.session_id = session.id
                           ), 0) AS stored_bytes
                    FROM cayu_sessions AS session
                    WHERE session.id = ?
                    """,
                    (session_id,),
                ).fetchone()
                if session_preflight is None:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.SESSION_NOT_FOUND
                    )
                session_status = SessionStatus(session_preflight["status"])
                session_run_epoch = session_preflight["run_epoch"]
                if type(session_run_epoch) is not int:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                    )
                _terminal_session_evidence_expected_event_type(
                    session_status,
                    allow_interrupted=allow_interrupted,
                )
                if allow_interrupted and session_status != SessionStatus.INTERRUPTED:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                    )
                if (
                    expected_parent_session_id is not None
                    and session_preflight["parent_session_id"] != expected_parent_session_id
                ):
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                    )
                if (
                    int(session_preflight["stored_bytes"])
                    > TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES
                ):
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
                        limit=limits.max_record_bytes,
                    )

                if observed is not None:
                    identity_preflight = connection.execute(
                        """
                        WITH bounded_identities AS (
                            SELECT length(CAST(event_type AS BLOB))
                                       + length(CAST(sequence AS TEXT)) AS stored_bytes
                            FROM cayu_events
                            WHERE session_id = ?
                            ORDER BY sequence ASC
                            LIMIT ?
                        )
                        SELECT COUNT(*) AS record_count,
                               COALESCE(MAX(stored_bytes), 0) AS largest_record_bytes,
                               COALESCE(SUM(stored_bytes), 0) AS total_bytes
                        FROM bounded_identities
                        """,
                        (session_id, limits.max_events + 1),
                    ).fetchone()
                    if identity_preflight is None:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                        )
                    identity_count = int(identity_preflight["record_count"])
                    if identity_count > limits.max_events:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
                            limit=limits.max_events,
                            observed=identity_count,
                        )
                    if identity_count != len(observed):
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                        )
                    identity_largest_bytes = int(identity_preflight["largest_record_bytes"])
                    if identity_largest_bytes > limits.max_record_bytes:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
                            limit=limits.max_record_bytes,
                            observed=identity_largest_bytes,
                        )
                    identity_total_bytes = int(identity_preflight["total_bytes"])
                    if identity_total_bytes > limits.max_total_bytes:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED,
                            limit=limits.max_total_bytes,
                            observed=identity_total_bytes,
                        )
                    identity_rows = connection.execute(
                        """
                        SELECT sequence, event_type
                        FROM cayu_events
                        WHERE session_id = ?
                        ORDER BY sequence ASC
                        LIMIT ?
                        """,
                        (session_id, identity_count),
                    ).fetchall()
                    _validate_runner_observed_event_identity_snapshot(
                        observed,
                        tuple(
                            RunnerObservedEventIdentity(
                                session_id=session_id,
                                sequence=row["sequence"],
                                event_type=row["event_type"],
                            )
                            for row in identity_rows
                        ),
                    )

                checkpoint_projection = connection.execute(
                    """
                    SELECT
                        json_type(state_json, '$.session_run_operation') AS marker_type,
                        json_type(
                            state_json,
                            '$.session_run_operation.version'
                        ) AS version_type,
                        json_extract(
                            state_json,
                            '$.session_run_operation.version'
                        ) AS version_value,
                        json_type(
                            state_json,
                            '$.session_run_operation.operation_id'
                        ) AS operation_id_type,
                        length(CAST(json_extract(
                            state_json,
                            '$.session_run_operation.operation_id'
                        ) AS BLOB)) AS operation_id_bytes,
                        length(trim(COALESCE(json_extract(
                            state_json,
                            '$.session_run_operation.operation_id'
                        ), ''))) > 0 AS operation_id_nonblank,
                        json_type(
                            state_json,
                            '$.session_run_operation.run_epoch'
                        ) AS run_epoch_type,
                        json_extract(
                            state_json,
                            '$.session_run_operation.run_epoch'
                        ) AS run_epoch_value,
                        json_type(
                            state_json,
                            '$.initial_transcript_pending'
                        ) IS NOT NULL AS initial_transcript_pending,
                        json_type(
                            state_json,
                            '$.pending_session_interrupt'
                        ) IS NOT NULL AS pending_session_interrupt
                    FROM cayu_checkpoints
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
                marker: TerminalPublicationMarker | None = None
                initial_transcript_pending = False
                pending_session_interrupt = False
                marker_stored_bytes = 0
                if checkpoint_projection is not None:
                    initial_transcript_pending = bool(
                        checkpoint_projection["initial_transcript_pending"]
                    )
                    pending_session_interrupt = bool(
                        checkpoint_projection["pending_session_interrupt"]
                    )
                    marker_type = checkpoint_projection["marker_type"]
                    if marker_type is not None:
                        marker_valid = (
                            marker_type == "object"
                            and checkpoint_projection["version_type"] == "integer"
                            and checkpoint_projection["version_value"] == 1
                            and checkpoint_projection["operation_id_type"] == "text"
                            and bool(checkpoint_projection["operation_id_nonblank"])
                            and checkpoint_projection["run_epoch_type"] == "integer"
                            and type(checkpoint_projection["run_epoch_value"]) is int
                            and 1
                            <= checkpoint_projection["run_epoch_value"]
                            <= MAX_DURABLE_JSON_INTEGER
                        )
                        if not marker_valid:
                            raise TerminalSessionEvidenceError(
                                TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_INVALID
                            )
                        operation_id_bytes = int(checkpoint_projection["operation_id_bytes"])
                        if operation_id_bytes > TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES:
                            raise TerminalSessionEvidenceError(
                                TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
                                limit=limits.max_record_bytes,
                            )
                        marker_stored_bytes = operation_id_bytes + len(
                            str(checkpoint_projection["run_epoch_value"]).encode("utf-8")
                        )
                        if marker_stored_bytes > limits.max_record_bytes:
                            raise TerminalSessionEvidenceError(
                                TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
                                limit=limits.max_record_bytes,
                            )
                        operation_row = connection.execute(
                            """
                            SELECT json_extract(
                                state_json,
                                '$.session_run_operation.operation_id'
                            ) AS operation_id
                            FROM cayu_checkpoints
                            WHERE session_id = ?
                            """,
                            (session_id,),
                        ).fetchone()
                        if operation_row is None:
                            raise TerminalSessionEvidenceError(
                                TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                            )
                        try:
                            marker = TerminalPublicationMarker(
                                operation_id=operation_row["operation_id"],
                                run_epoch=checkpoint_projection["run_epoch_value"],
                            )
                        except (TypeError, ValueError) as exc:
                            raise TerminalSessionEvidenceError(
                                TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_INVALID
                            ) from exc

                newest_preflight_rows = connection.execute(
                    f"""
                    SELECT sequence, event_type,
                           ({event_stored_bytes}) AS stored_bytes,
                           json_type(
                               payload_json,
                               '$.session_run_operation_id'
                           ) AS operation_id_type,
                           length(trim(COALESCE(json_extract(
                               payload_json,
                               '$.session_run_operation_id'
                           ), ''))) > 0 AS operation_id_nonblank
                    FROM cayu_events
                    WHERE session_id = ?
                      AND event_type IN ({evidence_type_placeholders})
                    ORDER BY sequence DESC
                    LIMIT ?
                    """,
                    (
                        session_id,
                        *evidence_event_types,
                        _TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT,
                    ),
                ).fetchall()
                if any(
                    int(row["stored_bytes"]) > limits.max_record_bytes
                    for row in newest_preflight_rows
                ):
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
                        limit=limits.max_record_bytes,
                    )
                if any(
                    row["operation_id_type"] not in {None, "text"}
                    or (
                        row["operation_id_type"] == "text"
                        and not bool(row["operation_id_nonblank"])
                    )
                    for row in newest_preflight_rows
                ):
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                    )
                newest_sequences = tuple(row["sequence"] for row in newest_preflight_rows)
                newest_evidence_records: tuple[EventRecord, ...]
                if newest_sequences:
                    sequence_placeholders = ", ".join("?" for _ in newest_sequences)
                    newest_rows = connection.execute(
                        f"""
                        SELECT sequence, event_id, event_type,
                               json_extract(
                                   payload_json,
                                   '$.session_run_operation_id'
                               ) AS operation_id
                        FROM cayu_events
                        WHERE sequence IN ({sequence_placeholders})
                        ORDER BY sequence DESC
                        """,
                        newest_sequences,
                    ).fetchall()
                    newest_evidence_records = tuple(
                        EventRecord(
                            sequence=row["sequence"],
                            event=Event(
                                id=row["event_id"],
                                type=row["event_type"],
                                session_id=session_id,
                                payload=(
                                    {}
                                    if row["operation_id"] is None
                                    else {"session_run_operation_id": row["operation_id"]}
                                ),
                            ),
                        )
                        for row in newest_rows
                    )
                else:
                    newest_evidence_records = ()

                terminal_record = _classify_terminal_session_evidence_records(
                    session_id=session_id,
                    status=session_status,
                    run_epoch=session_run_epoch,
                    marker=marker,
                    newest_evidence_records=newest_evidence_records,
                    initial_transcript_pending=initial_transcript_pending,
                    pending_session_interrupt=pending_session_interrupt,
                    allow_interrupted=allow_interrupted,
                )

                event_preflight = connection.execute(
                    f"""
                    WITH bounded_events AS (
                        SELECT ({event_stored_bytes}) AS stored_bytes
                        FROM cayu_events
                        WHERE session_id = ? AND sequence <= ?
                        ORDER BY sequence ASC
                        LIMIT ?
                    )
                    SELECT COUNT(*) AS record_count,
                           COALESCE(MAX(stored_bytes), 0) AS largest_record_bytes,
                           COALESCE(SUM(stored_bytes), 0) AS total_bytes
                    FROM bounded_events
                    """,
                    (
                        session_id,
                        terminal_record.sequence,
                        limits.max_events + 1,
                    ),
                ).fetchone()
                transcript_preflight = connection.execute(
                    f"""
                    WITH bounded_transcript AS (
                        SELECT ({transcript_stored_bytes}) AS stored_bytes
                        FROM cayu_transcript_messages
                        WHERE session_id = ?
                        ORDER BY session_order ASC
                        LIMIT ?
                    )
                    SELECT COUNT(*) AS record_count,
                           COALESCE(MAX(stored_bytes), 0) AS largest_record_bytes,
                           COALESCE(SUM(stored_bytes), 0) AS total_bytes
                    FROM bounded_transcript
                    """,
                    (session_id, limits.max_transcript_records + 1),
                ).fetchone()
                if event_preflight is None or transcript_preflight is None:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                    )
                event_count = int(event_preflight["record_count"])
                transcript_count = int(transcript_preflight["record_count"])
                if event_count > limits.max_events:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
                        limit=limits.max_events,
                        observed=event_count,
                    )
                if transcript_count > limits.max_transcript_records:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.TRANSCRIPT_LIMIT_EXCEEDED,
                        limit=limits.max_transcript_records,
                        observed=transcript_count,
                    )
                session_lower_bytes = int(session_preflight["stored_bytes"])
                largest_lower_bytes = max(
                    session_lower_bytes,
                    int(event_preflight["largest_record_bytes"]),
                    int(transcript_preflight["largest_record_bytes"]),
                    marker_stored_bytes,
                )
                if largest_lower_bytes > limits.max_record_bytes:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
                        limit=limits.max_record_bytes,
                    )
                total_lower_bytes = (
                    session_lower_bytes
                    + int(event_preflight["total_bytes"])
                    + int(transcript_preflight["total_bytes"])
                    + marker_stored_bytes
                )
                if total_lower_bytes > limits.max_total_bytes:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED,
                        limit=limits.max_total_bytes,
                    )

                session = _load_session(connection, session_id)
                if session is None:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                    )
                event_rows = connection.execute(
                    f"""
                    SELECT sequence, {event_columns}
                    FROM cayu_events
                    WHERE session_id = ? AND sequence <= ?
                    ORDER BY sequence ASC
                    """,
                    (session_id, terminal_record.sequence),
                ).fetchall()
                transcript_rows = connection.execute(
                    """
                    SELECT session_order - 1 AS transcript_index,
                           interaction_id,
                           message_json
                    FROM cayu_transcript_messages
                    WHERE session_id = ?
                    ORDER BY session_order ASC
                    """,
                    (session_id,),
                ).fetchall()
                if len(event_rows) != event_count or len(transcript_rows) != transcript_count:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                    )
                events = tuple(
                    EventRecord(sequence=row["sequence"], event=_event_from_row(row))
                    for row in event_rows
                )
                transcript = tuple(
                    TranscriptRecord(
                        index=row["transcript_index"],
                        interaction_id=row["interaction_id"],
                        message=Message(**json.loads(row["message_json"])),
                    )
                    for row in transcript_rows
                )
                return _assemble_terminal_session_evidence(
                    session=session,
                    marker=marker,
                    terminal_record=terminal_record,
                    events=events,
                    transcript=transcript,
                    limits=limits,
                    allow_interrupted=allow_interrupted,
                )
            except TerminalSessionEvidenceError:
                raise
            except (json.JSONDecodeError, sqlite3.DatabaseError, TypeError, ValueError) as exc:
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                ) from exc
            finally:
                connection.rollback()

        return await self._run_read(run_query)

    async def query_latest_interaction_events(
        self,
        session_id: str,
        *,
        before_sequence: int | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        session_id = require_clean_nonblank(session_id, "session_id")
        before_sequence, limit = _validate_interaction_page(before_sequence, limit)
        cursor_clause = "" if before_sequence is None else "AND latest.latest_event_sequence < ?"
        params: list[object] = [session_id]
        if before_sequence is not None:
            params.append(before_sequence)
        params.append(limit)

        def run_query(connection: sqlite3.Connection) -> list[EventRecord]:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            event_columns = ", ".join(f"event.{name}" for name in _EVENT_COLUMN_NAMES)
            rows = connection.execute(
                f"""
                SELECT event.sequence, {event_columns}
                FROM cayu_interaction_latest_events AS latest
                JOIN cayu_events AS event
                  ON event.sequence = latest.latest_event_sequence
                WHERE latest.session_id = ? {cursor_clause}
                ORDER BY latest.latest_event_sequence DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [
                EventRecord(sequence=row["sequence"], event=_event_from_row(row)) for row in rows
            ]

        return await self._run_read(run_query)

    async def _query_events_by_session_id_batches(self, query: EventQuery) -> list[EventRecord]:
        records: list[EventRecord] = []
        for batch in _event_query_session_id_batches(query.session_ids):
            records.extend(
                await self.query_events(
                    session_store_sql.event_query_with_session_ids(
                        query,
                        session_ids=batch,
                    ),
                )
            )
        records.sort(
            key=lambda record: record.sequence,
            reverse=query.order_by.value == "sequence_desc",
        )
        return records[: query.limit]

    async def summarize_events(self, session_id: str) -> EventSummary:
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection: sqlite3.Connection) -> EventSummary:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")

            total_row = connection.execute(
                """
                SELECT COUNT(*) AS total_events
                FROM cayu_events
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            count_rows = connection.execute(
                """
                SELECT event_type, COUNT(*) AS count
                FROM cayu_events
                WHERE session_id = ?
                GROUP BY event_type
                ORDER BY event_type ASC
                """,
                (session_id,),
            ).fetchall()
            latest_row = connection.execute(
                f"""
                SELECT sequence, {", ".join(_EVENT_COLUMN_NAMES)}
                FROM cayu_events
                WHERE session_id = ?
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()

            return EventSummary(
                session_id=session_id,
                total_events=int(total_row["total_events"]),
                counts_by_type={row["event_type"]: int(row["count"]) for row in count_rows},
                latest_event=_event_record_from_row(latest_row),
            )

        return await self._run_read(query)

    async def summarize_outcome(self, session_id: str) -> SessionOutcome:
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection: sqlite3.Connection) -> SessionOutcome:
            session = _load_session(connection, session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")

            terminal_row = connection.execute(
                f"""
                SELECT sequence, {", ".join(_EVENT_COLUMN_NAMES)}
                FROM cayu_events
                WHERE session_id = ?
                  AND event_type IN ('session.completed', 'session.failed', 'session.interrupted')
                  AND sequence > COALESCE(
                      (
                          SELECT MAX(sequence)
                          FROM cayu_events
                          WHERE session_id = ?
                            AND event_type IN ('session.started', 'session.resumed')
                      ),
                      0
                  )
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (session_id, session_id),
            ).fetchone()
            retry_row = connection.execute(
                f"""
                SELECT sequence, {", ".join(_EVENT_COLUMN_NAMES)}
                FROM cayu_events
                WHERE session_id = ?
                  AND event_type = 'model.retry'
                  AND sequence > COALESCE(
                      (
                          SELECT MAX(sequence)
                          FROM cayu_events
                          WHERE session_id = ?
                            AND event_type IN ('session.started', 'session.resumed')
                      ),
                      0
                  )
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (session_id, session_id),
            ).fetchone()

            return session_outcome(
                session,
                terminal_event=_event_record_from_row(terminal_row),
                latest_retry_event=_event_record_from_row(retry_row),
            )

        return await self._run_read(query)

    async def prune_events(
        self,
        *,
        before: datetime,
        session_id: str | None = None,
    ) -> int:
        """Delete events older than ``before`` to bound unbounded event growth.

        ``before`` is compared against each event's timestamp (events strictly
        older are removed). When ``session_id`` is given the prune is scoped to
        that session (which must exist); otherwise every session is pruned.
        The latest active or paused interaction lifecycle event is retained
        until a terminal event replaces it. Sessions with an active
        model-completion stage, pending tool round, or immutable
        runtime-publication receipt, or unacknowledged queued-dispatch terminal
        receipt are retained because deleting their evidence would make exact
        recovery or receipt replay impossible. Immutable profiled-fork decision
        and fork events are likewise retained while their child relationship
        exists. Targeted-grant issuance, accepted-consumption, revocation, and
        fork-reset evidence is retained because the durable grant state uses it
        to validate exact retry and negative inheritance authority.
        Returns the number of events deleted.
        """
        if not isinstance(before, datetime):
            raise TypeError("prune_events 'before' must be a datetime.")
        cutoff = sqlite_support.format_datetime(before)
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")

        def statement(connection: sqlite3.Connection) -> int:
            if session_id is not None and not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            publication_key_pattern = RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX + "*"
            with connection:
                if session_id is None:
                    cursor = connection.execute(
                        """
                        DELETE FROM cayu_events
                        WHERE timestamp < ?
                          AND event_type NOT IN (?, ?, ?, ?)
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_persisted_event_side_effects AS delivery
                              WHERE delivery.session_id = cayu_events.session_id
                                AND delivery.event_id = cayu_events.event_id
                                AND delivery.status <> 'delivered'
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_session_operations AS operation
                              WHERE operation.session_id = cayu_events.session_id
                                AND operation.idempotency_key GLOB ?
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_session_operations AS active_stage
                              WHERE active_stage.session_id = cayu_events.session_id
                                AND active_stage.idempotency_key = ?
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_checkpoints AS checkpoint
                              WHERE checkpoint.session_id = cayu_events.session_id
                                AND json_type(
                                    checkpoint.state_json,
                                    '$.pending_tool_round'
                                ) IS NOT NULL
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_checkpoints AS checkpoint
                              WHERE checkpoint.session_id = cayu_events.session_id
                                AND (
                                    json_extract(
                                        checkpoint.state_json,
                                        '$.session_run_operation.terminal_event_id'
                                    ) = cayu_events.event_id
                                    OR EXISTS (
                                        SELECT 1
                                        FROM json_each(
                                            checkpoint.state_json,
                                            '$.queued_dispatch_terminal_receipts.receipts'
                                        ) AS receipt
                                        WHERE json_extract(
                                            receipt.value,
                                            '$.terminal_event_id'
                                        ) = cayu_events.event_id
                                    )
                                )
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_sessions AS profiled_fork
                              WHERE profiled_fork.id = cayu_events.session_id
                                AND (
                                    json_extract(
                                        profiled_fork.metadata_json,
                                        ?
                                    ) = cayu_events.event_id
                                    OR json_extract(
                                        profiled_fork.metadata_json,
                                        ?
                                    ) = cayu_events.event_id
                                )
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_interaction_latest_events AS latest
                              JOIN cayu_events AS latest_event
                                ON latest_event.sequence = latest.latest_event_sequence
                              WHERE latest.session_id = cayu_events.session_id
                                AND latest.latest_event_sequence = cayu_events.sequence
                                AND latest_event.event_type IN (
                                    'interaction.started',
                                    'interaction.resumed',
                                    'interaction.paused'
                                )
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_targeted_tool_grants AS targeted_grant
                              WHERE targeted_grant.session_id = cayu_events.session_id
                                AND targeted_grant.interaction_id = cayu_events.interaction_id
                                AND cayu_events.event_type IN (
                                    'interaction.started',
                                    'interaction.completed',
                                    'interaction.failed',
                                    'interaction.interrupted'
                                )
                          )
                        """,
                        (
                            cutoff,
                            str(EventType.TARGETED_TOOL_GRANT_ISSUED),
                            str(EventType.TARGETED_TOOL_REFERENCE_CONSUMED),
                            str(EventType.TARGETED_TOOL_GRANT_REVOKED),
                            str(EventType.TARGETED_TOOL_GRANT_FORK_RESET),
                            publication_key_pattern,
                            MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                            f'$."{FORK_EXECUTION_PROFILE_METADATA_KEY}".fork_event_id',
                            f'$."{FORK_EXECUTION_PROFILE_METADATA_KEY}".decision.event_id',
                        ),
                    )
                else:
                    cursor = connection.execute(
                        """
                        DELETE FROM cayu_events
                        WHERE session_id = ? AND timestamp < ?
                          AND event_type NOT IN (?, ?, ?, ?)
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_persisted_event_side_effects AS delivery
                              WHERE delivery.session_id = cayu_events.session_id
                                AND delivery.event_id = cayu_events.event_id
                                AND delivery.status <> 'delivered'
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_session_operations AS operation
                              WHERE operation.session_id = cayu_events.session_id
                                AND operation.idempotency_key GLOB ?
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_session_operations AS active_stage
                              WHERE active_stage.session_id = cayu_events.session_id
                                AND active_stage.idempotency_key = ?
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_checkpoints AS checkpoint
                              WHERE checkpoint.session_id = cayu_events.session_id
                                AND json_type(
                                    checkpoint.state_json,
                                    '$.pending_tool_round'
                                ) IS NOT NULL
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_checkpoints AS checkpoint
                              WHERE checkpoint.session_id = cayu_events.session_id
                                AND (
                                    json_extract(
                                        checkpoint.state_json,
                                        '$.session_run_operation.terminal_event_id'
                                    ) = cayu_events.event_id
                                    OR EXISTS (
                                        SELECT 1
                                        FROM json_each(
                                            checkpoint.state_json,
                                            '$.queued_dispatch_terminal_receipts.receipts'
                                        ) AS receipt
                                        WHERE json_extract(
                                            receipt.value,
                                            '$.terminal_event_id'
                                        ) = cayu_events.event_id
                                    )
                                )
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_sessions AS profiled_fork
                              WHERE profiled_fork.id = cayu_events.session_id
                                AND (
                                    json_extract(
                                        profiled_fork.metadata_json,
                                        ?
                                    ) = cayu_events.event_id
                                    OR json_extract(
                                        profiled_fork.metadata_json,
                                        ?
                                    ) = cayu_events.event_id
                                )
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_interaction_latest_events AS latest
                              JOIN cayu_events AS latest_event
                                ON latest_event.sequence = latest.latest_event_sequence
                              WHERE latest.session_id = cayu_events.session_id
                                AND latest.latest_event_sequence = cayu_events.sequence
                                AND latest_event.event_type IN (
                                    'interaction.started',
                                    'interaction.resumed',
                                    'interaction.paused'
                                )
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_targeted_tool_grants AS targeted_grant
                              WHERE targeted_grant.session_id = cayu_events.session_id
                                AND targeted_grant.interaction_id = cayu_events.interaction_id
                                AND cayu_events.event_type IN (
                                    'interaction.started',
                                    'interaction.completed',
                                    'interaction.failed',
                                    'interaction.interrupted'
                                )
                          )
                        """,
                        (
                            session_id,
                            cutoff,
                            str(EventType.TARGETED_TOOL_GRANT_ISSUED),
                            str(EventType.TARGETED_TOOL_REFERENCE_CONSUMED),
                            str(EventType.TARGETED_TOOL_GRANT_REVOKED),
                            str(EventType.TARGETED_TOOL_GRANT_FORK_RESET),
                            publication_key_pattern,
                            MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                            f'$."{FORK_EXECUTION_PROFILE_METADATA_KEY}".fork_event_id',
                            f'$."{FORK_EXECUTION_PROFILE_METADATA_KEY}".decision.event_id',
                        ),
                    )
            return cursor.rowcount

        return await self._run_write(statement)

    async def compact_transcript(self, session_id: str, *, keep_last: int) -> int:
        """Compact a session's transcript, keeping only its most recent messages.

        Retains the ``keep_last`` newest transcript messages (by insertion order)
        for ``session_id`` and deletes the rest, bounding transcript growth for
        long-lived sessions. Active model stages, pending tool rounds, and
        immutable publication receipts pin their recovery material. Active
        model-target projection runs also pin the transcript; their permanent
        absolute cursor makes retention safe again at a terminal boundary.
        Returns the number of messages deleted.
        """
        if type(keep_last) is not int:
            raise TypeError("compact_transcript 'keep_last' must be an int.")
        if keep_last < 0:
            raise ValueError("compact_transcript 'keep_last' must be >= 0.")
        session_id = require_clean_nonblank(session_id, "session_id")

        def statement(connection: sqlite3.Connection) -> int:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            with connection:
                durability_guard = connection.execute(
                    """
                    SELECT 1
                    FROM cayu_session_operations
                    WHERE session_id = ?
                      AND (
                          idempotency_key GLOB ?
                          OR idempotency_key = ?
                      )
                    UNION ALL
                    SELECT 1
                    FROM cayu_checkpoints
                    WHERE session_id = ?
                      AND json_type(state_json, '$.pending_tool_round') IS NOT NULL
                    UNION ALL
                    SELECT 1
                    FROM cayu_sessions
                    WHERE id = ?
                      AND status IN (?, ?, ?)
                      AND json_type(metadata_json, ?) IS NOT NULL
                    LIMIT 1
                    """,
                    (
                        session_id,
                        RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX + "*",
                        MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                        session_id,
                        session_id,
                        str(SessionStatus.PENDING),
                        str(SessionStatus.RUNNING),
                        str(SessionStatus.INTERRUPTING),
                        f'$."{MODEL_TARGET_PROJECTION_METADATA_KEY}"',
                    ),
                ).fetchone()
                if durability_guard is not None:
                    return 0
                cursor = connection.execute(
                    """
                    DELETE FROM cayu_transcript_messages
                    WHERE session_id = ?
                      AND sequence NOT IN (
                          SELECT sequence
                          FROM cayu_transcript_messages
                          WHERE session_id = ?
                          ORDER BY sequence DESC
                          LIMIT ?
                      )
                    """,
                    (session_id, session_id, keep_last),
                )
                deleted = cursor.rowcount
            return deleted

        return await self._run_write(statement)

    async def list_sessions(self, query: SessionQuery | None = None) -> SessionListResult:
        return await self._list_sessions(query, pending_interruption_cascade_only=False)

    async def query_session_topology(
        self,
        query: SessionTopologyQuery,
    ) -> SessionTopologyStoreResult:
        if type(query) is not SessionTopologyQuery:
            raise TypeError("Session topology queries must be SessionTopologyQuery instances.")
        query = query.model_copy(deep=True)

        def read_topology_snapshot(
            connection: sqlite3.Connection,
        ) -> SessionTopologyStoreResult:
            focus_row = connection.execute(
                f"""
                SELECT {_SESSION_TOPOLOGY_COLUMNS}
                FROM cayu_sessions
                WHERE id = ?
                """,
                (query.focus_session_id,),
            ).fetchone()
            if focus_row is None:
                raise KeyError(f"Session not found: {query.focus_session_id}")
            focus = _session_topology_node_from_sqlite_row(focus_row)

            ancestors: list[SessionTopologyNode] = []
            seen_ids = {focus.id}
            parent_session_id = focus.parent_session_id
            while parent_session_id is not None:
                if parent_session_id in seen_ids:
                    raise SessionTopologyCycle(
                        f"Session topology contains a parent cycle at {parent_session_id}."
                    )
                if len(ancestors) >= query.ancestor_depth_limit:
                    raise SessionTopologyDepthExceeded(
                        f"Session topology exceeds the {query.ancestor_depth_limit}-ancestor limit."
                    )
                parent_row = connection.execute(
                    f"""
                    SELECT {_SESSION_TOPOLOGY_COLUMNS}
                    FROM cayu_sessions
                    WHERE id = ?
                    """,
                    (parent_session_id,),
                ).fetchone()
                if parent_row is None:
                    raise ValueError(
                        f"Session topology references missing parent {parent_session_id}."
                    )
                parent = _session_topology_node_from_sqlite_row(parent_row)
                ancestors.append(parent)
                seen_ids.add(parent.id)
                parent_session_id = parent.parent_session_id
            ancestors.reverse()

            expanded_parents: list[SessionTopologyNode] = []
            if query.expanded_parent_ids:
                placeholders = ", ".join("?" for _ in query.expanded_parent_ids)
                parent_rows = connection.execute(
                    f"""
                    SELECT {_SESSION_TOPOLOGY_COLUMNS}
                    FROM cayu_sessions
                    WHERE id IN ({placeholders})
                    """,
                    query.expanded_parent_ids,
                ).fetchall()
                parents_by_id = {
                    row["id"]: _session_topology_node_from_sqlite_row(row) for row in parent_rows
                }
                for parent_id in query.expanded_parent_ids:
                    parent = parents_by_id.get(parent_id)
                    if parent is None:
                        raise KeyError(f"Session not found: {parent_id}")
                    expanded_parents.append(parent)

            candidates_by_parent: dict[str, list[SessionTopologyNode]] = {
                parent.id: [] for parent in expanded_parents
            }
            if expanded_parents:
                branch_queries: list[str] = []
                branch_params: list[object] = []
                for branch_order, parent in enumerate(expanded_parents):
                    cursor = query.child_cursors.get(parent.id)
                    if cursor is None:
                        cursor_clause = ""
                        cursor_params: list[object] = []
                    else:
                        cursor_created_at, cursor_id = decode_session_topology_cursor(
                            cursor,
                            parent_session_id=parent.id,
                        )
                        cursor_clause = "AND (created_at > ? OR (created_at = ? AND id > ?))"
                        formatted_cursor = sqlite_support.format_datetime(cursor_created_at)
                        cursor_params = [formatted_cursor, formatted_cursor, cursor_id]
                    branch_queries.append(
                        f"""
                        SELECT branch_order, {_SESSION_TOPOLOGY_PROJECTED_COLUMNS}
                        FROM (
                            SELECT ? AS branch_order, {_SESSION_TOPOLOGY_COLUMNS}
                            FROM cayu_sessions
                            WHERE parent_session_id = ?
                              {cursor_clause}
                            ORDER BY created_at ASC, id ASC
                            LIMIT ?
                        )
                        """
                    )
                    branch_params.extend(
                        [
                            branch_order,
                            parent.id,
                            *cursor_params,
                            query.child_limit + 1,
                        ]
                    )
                candidate_rows = connection.execute(
                    f"""
                    {" UNION ALL ".join(branch_queries)}
                    ORDER BY branch_order ASC, created_at ASC, id ASC
                    """,
                    branch_params,
                ).fetchall()
                for row in candidate_rows:
                    candidates_by_parent[row["parent_session_id"]].append(
                        _session_topology_node_from_sqlite_row(row)
                    )

            result = build_session_topology_result(
                focus=focus,
                ancestors=ancestors,
                expanded_parents=expanded_parents,
                branch_candidates=(candidates_by_parent[parent.id] for parent in expanded_parents),
                child_limit=query.child_limit,
            )
            return result

        def read_topology(connection: sqlite3.Connection) -> SessionTopologyStoreResult:
            # Multiple point reads plus the batched child query must describe one
            # SQLite snapshot. A plain sequence of SELECT statements in Python's
            # legacy transaction mode would otherwise observe commits made
            # between statements.
            connection.execute("BEGIN")
            try:
                return read_topology_snapshot(connection)
            finally:
                connection.rollback()

        return await self._run_read(read_topology)

    async def query_session_lineage(
        self,
        query: SessionLineageQuery,
    ) -> SessionLineageResult:
        query = copy_session_lineage_query(query)

        def read_lineage_snapshot(connection: sqlite3.Connection) -> SessionLineageResult:
            parent_exists = connection.execute(
                "SELECT 1 FROM cayu_sessions WHERE id = ?",
                (query.parent_session_id,),
            ).fetchone()
            if parent_exists is None:
                raise KeyError(f"Session not found: {query.parent_session_id}")

            cursor_clause = ""
            params: list[object] = [
                SESSION_LINEAGE_MAX_IDENTIFIER_BYTES,
                SESSION_LINEAGE_MAX_TIMESTAMP_BYTES,
                query.parent_session_id,
            ]
            if query.cursor is not None:
                cursor_created_at, cursor_id = decode_session_lineage_cursor(
                    query.cursor,
                    parent_session_id=query.parent_session_id,
                )
                formatted_cursor = sqlite_support.format_datetime(cursor_created_at)
                cursor_clause = "AND (created_at > ? OR (created_at = ? AND id > ?))"
                params.extend((formatted_cursor, formatted_cursor, cursor_id))
            params.append(query.limit + 1)
            rows = connection.execute(
                f"""
                SELECT CASE
                           WHEN length(CAST(id AS BLOB)) <= ? THEN id
                       END AS id,
                       CASE
                           WHEN length(CAST(created_at AS BLOB)) <= ? THEN created_at
                       END AS created_at
                FROM cayu_sessions
                WHERE parent_session_id = ?
                  {cursor_clause}
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
            retained_rows = rows[: query.limit]
            children: list[SessionLineageNode] = []
            for row in retained_rows:
                base = SessionLineageNode(
                    id=row["id"],
                    parent_session_id=query.parent_session_id,
                    created_at=sqlite_support.parse_datetime(row["created_at"]),
                )
                origin_rows = connection.execute(
                    """
                    SELECT sequence,
                           CASE
                               WHEN length(event_id) <= ?
                                AND length(CAST(event_id AS BLOB)) <= ?
                               THEN event_id
                           END AS event_id,
                           event_type
                    FROM cayu_events
                    WHERE session_id = ?
                      AND event_type IN (?, ?)
                    ORDER BY sequence ASC
                    LIMIT ?
                    """,
                    (
                        EVENT_ID_MAX_CHARS,
                        SESSION_LINEAGE_MAX_EVENT_ID_BYTES,
                        base.id,
                        str(EventType.SESSION_STARTED),
                        str(EventType.SESSION_FORKED),
                        SESSION_LINEAGE_MAX_ORIGIN_EVENTS,
                    ),
                ).fetchall()
                children.append(
                    SessionLineageNode(
                        id=base.id,
                        parent_session_id=base.parent_session_id,
                        created_at=base.created_at,
                        origin_events=tuple(
                            SessionLineageOrigin(
                                sequence=origin_row["sequence"],
                                event_id=origin_row["event_id"],
                                event_type=EventType(origin_row["event_type"]),
                            )
                            for origin_row in origin_rows
                        ),
                    )
                )
            has_more = len(rows) > len(retained_rows)
            return SessionLineageResult(
                parent_session_id=query.parent_session_id,
                children=tuple(children),
                next_cursor=(
                    encode_session_lineage_cursor(query.parent_session_id, children[-1])
                    if has_more and children
                    else None
                ),
                has_more=has_more,
            )

        def read_lineage(connection: sqlite3.Connection) -> SessionLineageResult:
            connection.execute("BEGIN")
            try:
                return read_lineage_snapshot(connection)
            finally:
                connection.rollback()

        return await self._run_read(read_lineage)

    async def query_child_session_lifecycle(
        self,
        query: ChildSessionLifecycleQuery,
    ) -> ChildSessionLifecyclePage:
        query = ChildSessionLifecycleQuery.model_validate(query)

        def read_snapshot(connection: sqlite3.Connection) -> ChildSessionLifecyclePage:
            parent = _load_session(connection, query.parent_session_id)
            if parent is None:
                raise KeyError(f"Session not found: {query.parent_session_id}")
            rows = connection.execute(
                "SELECT child_session_id FROM cayu_child_session_lifecycle_candidates "
                "WHERE parent_session_id = ? "
                "ORDER BY priority, sort_at, child_session_id "
                "LIMIT ?",
                (parent.id, query.max_children_inspected + 1),
            ).fetchall()
            retained_rows = rows[: query.max_children_inspected]
            retained_ids = [str(row["child_session_id"]) for row in retained_rows]
            entries = []
            unavailable_count = 0
            lifecycle_types = (
                str(EventType.SESSION_STARTED),
                str(EventType.SESSION_RESUMED),
                str(EventType.SESSION_FORKED),
                str(EventType.SESSION_COMPLETED),
                str(EventType.SESSION_FAILED),
                str(EventType.SESSION_INTERRUPTED),
            )
            children_by_id: dict[str, Session] = {}
            records_by_child: dict[str, dict[EventType, EventRecord]] = {
                child_id: {} for child_id in retained_ids
            }
            if retained_ids:
                placeholders = ", ".join("?" for _child_id in retained_ids)
                child_rows = connection.execute(
                    "SELECT id, instance_id, agent_name, provider_name, model, "
                    "parent_session_id, causal_budget_id, runtime_name, runtime_version, "
                    "environment_name, status, created_at, updated_at, last_activity_at, "
                    "run_epoch, invocation_json, metadata_json FROM cayu_sessions "
                    f"WHERE id IN ({placeholders})",
                    retained_ids,
                ).fetchall()
                children_by_id = {
                    str(child_row["id"]): sqlite_support.session_from_row(
                        child_row,
                        labels={},
                    )
                    for child_row in child_rows
                }
                event_rows = connection.execute(
                    """
                    SELECT event.*
                    FROM cayu_events AS event
                    JOIN (
                        SELECT session_id, event_type, MAX(sequence) AS sequence
                        FROM cayu_events
                        WHERE session_id IN ("""
                    + placeholders
                    + """)
                          AND event_type IN (?, ?, ?, ?, ?, ?)
                        GROUP BY session_id, event_type
                    ) AS latest ON latest.sequence = event.sequence
                    ORDER BY event.session_id, event.sequence ASC
                    """,
                    (*retained_ids, *lifecycle_types),
                ).fetchall()
                for event_row in event_rows:
                    event_record = _event_record_from_row(event_row)
                    if event_record is None:  # pragma: no cover - row is present
                        raise RuntimeError("SQLite lifecycle event row disappeared.")
                    records_by_child[str(event_row["session_id"])][
                        EventType(event_row["event_type"])
                    ] = event_record

            consumption_key_by_child: dict[str, str] = {}
            for child_id in retained_ids:
                child = children_by_id.get(child_id)
                if child is None or child.parent_session_id != parent.id:
                    raise RuntimeError("SQLite child-session lifecycle index is inconsistent.")
                occurrence_source = _child_session_lifecycle_occurrence(
                    child,
                    records_by_child[child_id],
                )
                if occurrence_source is not None:
                    _relationship, occurrence = occurrence_source
                    consumption_key_by_child[child_id] = child_session_notification_storage_key(
                        child.instance_id,
                        occurrence.source_id,
                    )
            consumption_by_key: dict[str, dict[str, Any]] = {}
            if consumption_key_by_child:
                consumption_keys = tuple(consumption_key_by_child.values())
                placeholders = ", ".join("?" for _key in consumption_keys)
                operation_rows = connection.execute(
                    "SELECT idempotency_key, record_json FROM cayu_session_operations "
                    f"WHERE session_id = ? AND idempotency_key IN ({placeholders})",
                    (parent.id, *consumption_keys),
                ).fetchall()
                consumption_by_key = {
                    str(operation_row["idempotency_key"]): json.loads(operation_row["record_json"])
                    for operation_row in operation_rows
                }

            for child_id in retained_ids:
                child = children_by_id[child_id]
                consumption_key = consumption_key_by_child.get(child_id)
                entry = _child_session_lifecycle_entry(
                    parent=parent,
                    child=child,
                    records_by_type=records_by_child[child_id],
                    consumption_record=(
                        None if consumption_key is None else consumption_by_key.get(consumption_key)
                    ),
                )
                if entry is None:
                    unavailable_count += 1
                else:
                    entries.append(entry)
            entries.sort(key=_child_session_lifecycle_entry_sort_key)
            return ChildSessionLifecyclePage(
                parent_session_id=parent.id,
                parent_session_instance_id=parent.instance_id,
                entries=tuple(entries),
                inspected_child_count=len(retained_rows),
                unavailable_child_count=unavailable_count,
                has_more=len(rows) > len(retained_rows),
            )

        def read(connection: sqlite3.Connection) -> ChildSessionLifecyclePage:
            connection.execute("BEGIN")
            try:
                return read_snapshot(connection)
            finally:
                connection.rollback()

        return await self._run_read(read)

    async def aggregate_operational_snapshot(
        self,
        filters: SessionAggregateFilter | None = None,
    ) -> SessionOperationalSnapshot:
        filters = copy_session_aggregate_filter(filters)
        plan = session_store_sql.build_session_query_sql(
            session_query_from_aggregate_filter(filters),
            dialect=_SQL_DIALECT,
        )

        def query_snapshot(connection: sqlite3.Connection) -> SessionOperationalSnapshot:
            rows = connection.execute(
                f"""
                WITH
                snapshot(as_of) AS (
                    SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                ),
                status_counts AS (
                    SELECT status, COUNT(*) AS status_count
                    FROM cayu_sessions
                    {plan.filter_where_sql}
                    GROUP BY status
                )
                SELECT snapshot.as_of, status_counts.status, status_counts.status_count
                FROM snapshot
                LEFT JOIN status_counts ON TRUE
                """,
                plan.filter_params,
            ).fetchall()
            counts = {status: 0 for status in SessionStatus}
            for row in rows:
                if row["status"] is not None:
                    status = SessionStatus(row["status"])
                    counts[status] = row["status_count"]
            return SessionOperationalSnapshot(
                as_of=sqlite_support.parse_datetime(rows[0]["as_of"]),
                total_count=sum(counts.values()),
                counts_by_status=SessionStatusCounts.model_validate(counts),
                accuracy=EXACT_AGGREGATE.model_copy(),
            )

        return await self._run_read(query_snapshot)

    async def aggregate_usage(self, query: UsageRollupQuery) -> UsageRollupStoreResult:
        query = copy_usage_rollup_query(query)
        plan = session_store_sql.build_session_query_sql(
            session_query_from_aggregate_filter(query.sessions),
            dialect=_SQL_DIALECT,
        )

        def query_aggregate(connection: sqlite3.Connection) -> UsageRollupStoreResult:
            return sqlite_aggregates.aggregate_session_usage(
                connection,
                session_plan=plan,
                query=query,
            )

        return await self._run_read(query_aggregate)

    async def list_sessions_with_pending_interruption_cascade(
        self,
        query: SessionQuery | None = None,
    ) -> SessionListResult:
        return await self._list_sessions(query, pending_interruption_cascade_only=True)

    async def list_queued_dispatch_terminal_receipts(
        self,
        query: QueuedDispatchTerminalReceiptQuery | None = None,
    ) -> list[QueuedDispatchTerminalReceipt]:
        if query is None:
            query = QueuedDispatchTerminalReceiptQuery()
        elif type(query) is not QueuedDispatchTerminalReceiptQuery:
            raise TypeError(
                "Queued dispatch receipt queries must be "
                "QueuedDispatchTerminalReceiptQuery instances."
            )
        else:
            query = QueuedDispatchTerminalReceiptQuery(
                after_session_id=query.after_session_id,
                after_operation_id=query.after_operation_id,
                limit=query.limit,
            )
        cursor_sql = ""
        params: list[Any] = []
        if query.after_session_id is not None:
            assert query.after_operation_id is not None
            cursor_sql = "WHERE session_id > ? OR (session_id = ? AND operation_id > ?)"
            params.extend(
                [
                    query.after_session_id,
                    query.after_session_id,
                    query.after_operation_id,
                ]
            )
        params.append(query.limit)

        def run_query(connection: sqlite3.Connection) -> list[QueuedDispatchTerminalReceipt]:
            rows = connection.execute(
                f"""
                WITH queued_dispatch_receipts AS (
                    SELECT
                        checkpoint.session_id,
                        json_extract(
                            checkpoint.state_json,
                            '$.session_run_operation.queue_task_id'
                        ) AS queue_task_id,
                        json_extract(
                            checkpoint.state_json,
                            '$.session_run_operation.operation_id'
                        ) AS operation_id,
                        json_extract(
                            checkpoint.state_json,
                            '$.session_run_operation.terminal_event_id'
                        ) AS terminal_event_id
                    FROM cayu_checkpoints AS checkpoint
                    INNER JOIN cayu_events AS terminal_event
                        ON terminal_event.session_id = checkpoint.session_id
                       AND terminal_event.event_id = json_extract(
                            checkpoint.state_json,
                            '$.session_run_operation.terminal_event_id'
                       )
                    WHERE json_type(
                        checkpoint.state_json,
                        '$.session_run_operation.queue_task_id'
                    ) IS NOT NULL

                    UNION

                    SELECT
                        checkpoint.session_id,
                        json_extract(receipt.value, '$.queue_task_id') AS queue_task_id,
                        receipt.key AS operation_id,
                        json_extract(
                            receipt.value,
                            '$.terminal_event_id'
                        ) AS terminal_event_id
                    FROM cayu_checkpoints AS checkpoint,
                         json_each(
                             checkpoint.state_json,
                             '$.queued_dispatch_terminal_receipts.receipts'
                         ) AS receipt
                    WHERE json_type(
                        checkpoint.state_json,
                        '$.queued_dispatch_terminal_receipts.receipts'
                    ) IS NOT NULL
                )
                SELECT session_id, queue_task_id, operation_id, terminal_event_id
                FROM queued_dispatch_receipts
                {cursor_sql}
                ORDER BY session_id ASC, operation_id ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [
                QueuedDispatchTerminalReceipt(
                    session_id=row["session_id"],
                    queue_task_id=row["queue_task_id"],
                    operation_id=row["operation_id"],
                    terminal_event_id=row["terminal_event_id"],
                )
                for row in rows
            ]

        return await self._run_read(run_query)

    async def query_pending_actions(
        self,
        query: PendingActionQuery | None = None,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> PendingActionListResult:
        from cayu.runtime.pending_actions import (
            pending_action_from_records,
            pending_action_matches_query,
            pending_action_source_is_invalid,
        )

        if query is None:
            query = PendingActionQuery()
        elif type(query) is not PendingActionQuery:
            raise TypeError("Pending-action queries must be PendingActionQuery instances.")
        else:
            query = query.model_copy(deep=True)

        inspected_candidate_limit = min(query.limit * 4, 800)
        candidate_limit = inspected_candidate_limit + 1
        filters = [
            "cayu_sessions.status IN ('interrupted', 'failed', 'completed')",
            "cayu_checkpoints.pending_action_metrics_ready = 1",
            "cayu_checkpoints.pending_action_flags <> 0",
        ]
        params: list[Any] = []
        if query.session_id is not None:
            filters.append("cayu_sessions.id = ?")
            params.append(query.session_id)
        if query.agent_name is not None:
            filters.append("cayu_sessions.agent_name = ?")
            params.append(query.agent_name)
        if query.environment_name is not None:
            filters.append("cayu_sessions.environment_name = ?")
            params.append(query.environment_name)
        if query.kind == PendingActionKind.TOOL_APPROVAL:
            filters.append("(cayu_checkpoints.pending_action_flags & 1) <> 0")
        elif query.kind == PendingActionKind.USER_INPUT:
            filters.append("(cayu_checkpoints.pending_action_flags & 2) <> 0")
        if query.cursor is not None:
            cursor_dt, cursor_id = decode_session_cursor(query.cursor)
            cursor_value = sqlite_support.format_datetime(cursor_dt)
            filters.append(
                """
                (
                    cayu_sessions.updated_at < ?
                    OR (cayu_sessions.updated_at = ? AND cayu_sessions.id > ?)
                )
                """
            )
            params.extend((cursor_value, cursor_value, cursor_id))

        where_sql = " AND ".join(f"({clause.strip()})" for clause in filters)
        candidate_select_sql = f"""
            SELECT
                cayu_sessions.id,
                cayu_sessions.agent_name,
                cayu_sessions.provider_name,
                cayu_sessions.model,
                cayu_sessions.parent_session_id,
                cayu_sessions.causal_budget_id,
                cayu_sessions.runtime_name,
                cayu_sessions.runtime_version,
                cayu_sessions.environment_name,
                cayu_sessions.status,
                cayu_sessions.created_at,
                cayu_sessions.updated_at,
                json_extract(
                    cayu_sessions.metadata_json,
                    '$."cayu:runtime_build_provenance"'
                ) AS runtime_build_provenance_json
            FROM cayu_checkpoints
                INDEXED BY idx_cayu_checkpoints_pending_control_action
            JOIN cayu_sessions ON cayu_sessions.id = cayu_checkpoints.session_id
            WHERE {where_sql}
            ORDER BY cayu_sessions.updated_at DESC, cayu_sessions.id ASC
            LIMIT ?
        """
        selected_candidate_sql = """
            SELECT
                cayu_checkpoints.session_id AS id,
                json_object(
                    'pending_tool_approval',
                    json_extract(
                        cayu_checkpoints.state_json,
                        '$.pending_tool_approval'
                    ),
                    'pending_user_input',
                    json_extract(
                        cayu_checkpoints.state_json,
                        '$.pending_user_input'
                    ),
                    'pending_tool_round',
                    json_extract(
                        cayu_checkpoints.state_json,
                        '$.pending_tool_round'
                    )
                ) AS pending_state_json
            FROM cayu_checkpoints
            WHERE cayu_checkpoints.session_id IN (
                SELECT CAST(value AS TEXT) FROM json_each(?)
            )
        """
        checkpoint_root_key = (
            "__cayu_no_checkpoint_root_guard__"
            if checkpoint_root_guard is None
            else checkpoint_root_guard.key
        )
        checkpoint_root_path = f"$.{checkpoint_root_key}"
        checkpoint_preflight_sql = f"""
            SELECT
                cayu_checkpoints.session_id,
                cayu_checkpoints.pending_action_source_bytes AS pending_state_bytes,
                cayu_checkpoints.pending_action_tool_call_count AS pending_tool_call_count,
                json_type(
                    cayu_checkpoints.state_json,
                    '{checkpoint_root_path}'
                ) AS checkpoint_root_field_type,
                CASE
                    WHEN json_type(
                        cayu_checkpoints.state_json,
                        '{checkpoint_root_path}'
                    ) = 'integer'
                    THEN substr(
                        CAST(json_extract(
                            cayu_checkpoints.state_json,
                            '{checkpoint_root_path}'
                        ) AS TEXT),
                        1,
                        {CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS + 1}
                    )
                END AS checkpoint_root_field_scalar
            FROM cayu_checkpoints
            WHERE cayu_checkpoints.session_id IN (
                SELECT CAST(value AS TEXT) FROM json_each(?)
            )
        """
        projected_event_sql = "json(source_event.pending_action_projection_json)"
        pending_action_ctes = f"""
            WITH candidates AS ({selected_candidate_sql}),
            candidate_tool_scopes AS (
                SELECT candidates.id AS session_id,
                    CASE
                        WHEN json_type(
                            candidates.pending_state_json,
                            '$.pending_tool_approval'
                        ) = 'object'
                        THEN json_extract(
                            candidates.pending_state_json,
                            '$.pending_tool_approval'
                        )
                        WHEN json_type(
                            candidates.pending_state_json,
                            '$.pending_user_input'
                        ) = 'object'
                        THEN json_extract(
                            candidates.pending_state_json,
                            '$.pending_user_input'
                        )
                        WHEN json_type(
                            candidates.pending_state_json,
                            '$.pending_tool_round'
                        ) = 'object'
                        THEN json_extract(
                            candidates.pending_state_json,
                            '$.pending_tool_round'
                        )
                        ELSE NULL
                    END AS pending_tool_state_json
                FROM candidates
            ),
            candidate_tool_calls AS (
                SELECT
                    tool_scope.session_id,
                    json_extract(pending_call.value, '$.tool_call_id') AS tool_call_id
                FROM candidate_tool_scopes AS tool_scope
                JOIN json_each(
                    CASE
                        WHEN json_type(
                            tool_scope.pending_tool_state_json,
                            '$.tool_calls'
                        ) = 'array'
                        THEN json_extract(
                            tool_scope.pending_tool_state_json,
                            '$.tool_calls'
                        )
                        ELSE json('[]')
                    END
                ) AS pending_call
                WHERE json_type(pending_call.value, '$.tool_call_id') = 'text'
            ),
            candidate_action_keys AS (
                SELECT id AS session_id,
                    cayu_pending_action_lookup_key(json_extract(
                        pending_state_json,
                        '$.pending_tool_approval.approval_id'
                    )) AS action_key
                FROM candidates
                WHERE json_type(
                    pending_state_json,
                    '$.pending_tool_approval.approval_id'
                ) = 'text'
                UNION
                SELECT id,
                    cayu_pending_action_lookup_key(
                        json_extract(pending_state_json, '$.pending_user_input.input_id')
                    )
                FROM candidates
                WHERE json_type(
                    pending_state_json,
                    '$.pending_user_input.input_id'
                ) = 'text'
                UNION
                SELECT tool_scope.session_id,
                    cayu_pending_action_lookup_key(
                        json_extract(
                            tool_scope.pending_tool_state_json,
                            '$.tool_round_id'
                        )
                    )
                FROM candidate_tool_scopes AS tool_scope
                WHERE json_type(
                    tool_scope.pending_tool_state_json,
                    '$.tool_round_id'
                ) = 'text'
                UNION
                SELECT pending_call.session_id,
                    cayu_pending_action_lookup_key(pending_call.tool_call_id)
                FROM candidate_tool_calls AS pending_call
            ),
            pending_action_event_types(event_type) AS (
                VALUES
                    ('tool.call.approval_requested'),
                    ('session.awaiting_user_input'),
                    ('session.interrupted')
            ),
            latest_barriers AS (
                SELECT candidates.id AS session_id,
                    COALESCE((
                        SELECT MAX(event.sequence)
                        FROM cayu_events AS event
                            INDEXED BY idx_cayu_events_pending_action_barrier
                        WHERE event.session_id = candidates.id
                          AND (
                              event.event_type = 'session.resumed'
                              OR event.event_type = 'session.completed'
                              OR event.event_type = 'session.failed'
                          )
                    ), 0) AS sequence
                FROM candidates
            ),
            matched_action_sequences AS (
                SELECT
                    action_keys.session_id AS candidate_session_id,
                    (
                        SELECT MAX(candidate_event.sequence)
                        FROM cayu_events AS candidate_event
                            INDEXED BY idx_cayu_events_pending_action_lookup
                        WHERE candidate_event.session_id = action_keys.session_id
                          AND candidate_event.event_type = action_type.event_type
                          AND candidate_event.event_type IN (
                              'tool.call.approval_requested',
                              'session.awaiting_user_input',
                              'session.interrupted',
                              'tool.call.started',
                              'tool.call.completed',
                              'tool.call.failed',
                              'tool.call.blocked',
                              'tool.call.approval_denied'
                          )
                          AND candidate_event.pending_action_lookup_key IS NOT NULL
                          AND candidate_event.pending_action_lookup_key = action_keys.action_key
                    ) AS sequence
                FROM candidate_action_keys AS action_keys
                CROSS JOIN pending_action_event_types AS action_type
            ),
            matched_ledger_sequences AS (
                SELECT
                    action_keys.session_id AS candidate_session_id,
                    action_keys.action_key,
                    candidate_event.sequence
                FROM candidate_action_keys AS action_keys
                JOIN candidates ON candidates.id = action_keys.session_id
                JOIN candidate_tool_scopes AS tool_scope
                    ON tool_scope.session_id = action_keys.session_id
                JOIN cayu_events AS candidate_event
                    ON candidate_event.sequence IN (
                        SELECT scoped_event.sequence
                        FROM cayu_events AS scoped_event
                            INDEXED BY idx_cayu_events_pending_action_lookup
                        WHERE scoped_event.session_id = action_keys.session_id
                          AND scoped_event.pending_action_lookup_key
                              = action_keys.action_key
                          AND scoped_event.event_type IN (
                              'tool.call.approval_requested',
                              'session.awaiting_user_input',
                              'session.interrupted',
                              'tool.call.started',
                              'tool.call.completed',
                              'tool.call.failed',
                              'tool.call.blocked',
                              'tool.call.approval_denied'
                          )
                          AND scoped_event.event_type IN (
                              'tool.call.started',
                              'tool.call.completed',
                              'tool.call.failed',
                              'tool.call.blocked',
                              'tool.call.approval_denied'
                          )
                          AND scoped_event.pending_action_lookup_key IS NOT NULL
                          AND (
                              json_extract(
                                  scoped_event.pending_action_projection_json,
                                  '$.payload.tool_round_id'
                              ) = json_extract(
                                  tool_scope.pending_tool_state_json,
                                  '$.tool_round_id'
                              )
                              OR (
                                  json_extract(
                                      scoped_event.pending_action_projection_json,
                                      '$.payload.model_step_id'
                                  ) = json_extract(
                                      tool_scope.pending_tool_state_json,
                                      '$.model_step_id'
                                  )
                                  AND json_extract(
                                      scoped_event.pending_action_projection_json,
                                      '$.payload.model_attempt_id'
                                  ) = json_extract(
                                      tool_scope.pending_tool_state_json,
                                      '$.model_attempt_id'
                                  )
                              )
                          )
                        LIMIT {MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL + 1}
                    )
                WHERE json_type(
                    tool_scope.pending_tool_state_json
                ) = 'object'
            ),
            scope_conflict_sequences AS (
                SELECT tool_scope.session_id AS candidate_session_id,
                    COALESCE(
                    (
                        SELECT scoped_event.sequence
                        FROM cayu_events AS scoped_event
                            INDEXED BY idx_cayu_events_pending_action_round_scope
                        WHERE scoped_event.session_id = tool_scope.session_id
                          AND scoped_event.event_type IN (
                              'tool.call.started',
                              'tool.call.completed',
                              'tool.call.failed',
                              'tool.call.blocked',
                              'tool.call.approval_denied'
                          )
                          AND json_type(
                              scoped_event.pending_action_projection_json,
                              '$.payload.tool_round_id'
                          ) = 'text'
                          AND length(json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.tool_round_id'
                          )) = 39
                          AND substr(json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.tool_round_id'
                          ), 1, 7) = 'tround_'
                          AND substr(json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.tool_round_id'
                          ), 8) NOT GLOB '*[^0-9a-f]*'
                          AND json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.tool_round_id'
                          ) = json_extract(
                              tool_scope.pending_tool_state_json,
                              '$.tool_round_id'
                          )
                          AND COALESCE(
                              json_extract(
                                  scoped_event.pending_action_projection_json,
                                  '$.payload.tool_round_id'
                              ) = json_extract(
                                  tool_scope.pending_tool_state_json,
                                  '$.tool_round_id'
                              )
                              AND json_extract(
                                  scoped_event.pending_action_projection_json,
                                  '$.payload.model_step_id'
                              ) = json_extract(
                                  tool_scope.pending_tool_state_json,
                                  '$.model_step_id'
                              )
                              AND json_extract(
                                  scoped_event.pending_action_projection_json,
                                  '$.payload.model_attempt_id'
                              ) = json_extract(
                                  tool_scope.pending_tool_state_json,
                                  '$.model_attempt_id'
                              )
                              AND EXISTS (
                                  SELECT 1
                                  FROM candidate_tool_calls AS pending_call
                                  WHERE pending_call.session_id = tool_scope.session_id
                                    AND pending_call.tool_call_id = json_extract(
                                        scoped_event.pending_action_projection_json,
                                        '$.payload.tool_call_id'
                                    )
                              ),
                              0
                          ) = 0
                        LIMIT 1
                    ),
                    (
                        SELECT scoped_event.sequence
                        FROM cayu_events AS scoped_event
                            INDEXED BY idx_cayu_events_pending_action_attempt_scope
                        WHERE scoped_event.session_id = tool_scope.session_id
                          AND scoped_event.event_type IN (
                              'tool.call.started',
                              'tool.call.completed',
                              'tool.call.failed',
                              'tool.call.blocked',
                              'tool.call.approval_denied'
                          )
                          AND json_type(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_step_id'
                          ) = 'text'
                          AND json_type(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_attempt_id'
                          ) = 'text'
                          AND length(json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_step_id'
                          )) = 38
                          AND substr(json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_step_id'
                          ), 1, 6) = 'mstep_'
                          AND substr(json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_step_id'
                          ), 7) NOT GLOB '*[^0-9a-f]*'
                          AND length(json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_attempt_id'
                          )) = 37
                          AND substr(json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_attempt_id'
                          ), 1, 5) = 'matt_'
                          AND substr(json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_attempt_id'
                          ), 6) NOT GLOB '*[^0-9a-f]*'
                          AND json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_step_id'
                          ) = json_extract(
                              tool_scope.pending_tool_state_json,
                              '$.model_step_id'
                          )
                          AND json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_attempt_id'
                          ) = json_extract(
                              tool_scope.pending_tool_state_json,
                              '$.model_attempt_id'
                          )
                          AND COALESCE(
                              json_extract(
                                  scoped_event.pending_action_projection_json,
                                  '$.payload.tool_round_id'
                              ) = json_extract(
                                  tool_scope.pending_tool_state_json,
                                  '$.tool_round_id'
                              )
                              AND json_extract(
                                  scoped_event.pending_action_projection_json,
                                  '$.payload.model_step_id'
                              ) = json_extract(
                                  tool_scope.pending_tool_state_json,
                                  '$.model_step_id'
                              )
                              AND json_extract(
                                  scoped_event.pending_action_projection_json,
                                  '$.payload.model_attempt_id'
                              ) = json_extract(
                                  tool_scope.pending_tool_state_json,
                                  '$.model_attempt_id'
                              )
                              AND EXISTS (
                                  SELECT 1
                                  FROM candidate_tool_calls AS pending_call
                                  WHERE pending_call.session_id = tool_scope.session_id
                                    AND pending_call.tool_call_id = json_extract(
                                        scoped_event.pending_action_projection_json,
                                        '$.payload.tool_call_id'
                                    )
                              ),
                              0
                          ) = 0
                        LIMIT 1
                    )
                    ) AS sequence
                FROM candidate_tool_scopes AS tool_scope
                WHERE json_type(tool_scope.pending_tool_state_json) = 'object'
            ),
            matched_event_sequences AS (
                SELECT
                    matched_action.candidate_session_id,
                    matched_action.sequence
                FROM matched_action_sequences AS matched_action
                WHERE matched_action.sequence IS NOT NULL
                UNION
                SELECT
                    matched_ledger.candidate_session_id,
                    matched_ledger.sequence
                FROM matched_ledger_sequences AS matched_ledger
                UNION
                SELECT
                    scope_conflict.candidate_session_id,
                    scope_conflict.sequence
                FROM scope_conflict_sequences AS scope_conflict
                WHERE scope_conflict.sequence IS NOT NULL
                UNION
                SELECT
                    candidates.id,
                    event.sequence
                FROM candidates
                JOIN latest_barriers ON latest_barriers.session_id = candidates.id
                JOIN cayu_events AS event ON event.sequence = latest_barriers.sequence
            ),
            matched_events AS (
                SELECT
                    matched_event_sequences.candidate_session_id,
                    source_event.sequence,
                    source_event.pending_action_projection_bytes AS event_bytes,
                    source_event.pending_action_projection_bytes IS NOT NULL
                        AND source_event.pending_action_projection_json IS NOT NULL
                        AS projection_ready
                FROM matched_event_sequences
                JOIN cayu_events AS source_event
                    ON source_event.sequence = matched_event_sequences.sequence
            )
        """
        source_size_sql = f"""
            {pending_action_ctes}
            SELECT candidates.id,
                length(CAST(candidates.pending_state_json AS BLOB))
                + COALESCE((
                    SELECT SUM(length(CAST(json_object(
                        'key', label.key,
                        'value', label.value
                    ) AS BLOB)))
                    FROM cayu_session_labels AS label
                    WHERE label.session_id = candidates.id
                ), 0)
                + COALESCE((
                    SELECT SUM(
                        matched_event.event_bytes
                        + length(CAST(matched_event.sequence AS TEXT))
                        + 22
                    )
                    FROM matched_events AS matched_event
                    WHERE matched_event.candidate_session_id = candidates.id
                ), 0) AS source_bytes,
                COALESCE((
                    SELECT MIN(CASE WHEN matched_event.projection_ready THEN 1 ELSE 0 END)
                    FROM matched_events AS matched_event
                    WHERE matched_event.candidate_session_id = candidates.id
                ), 1) AS projections_ready,
                EXISTS (
                    SELECT 1
                    FROM matched_ledger_sequences AS matched_ledger
                    WHERE matched_ledger.candidate_session_id = candidates.id
                    GROUP BY matched_ledger.action_key
                    HAVING COUNT(*) > {MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL}
                ) AS ledger_too_complex,
                COALESCE((
                    SELECT json_group_array(ordered_sequence.sequence)
                    FROM (
                        SELECT matched_event.sequence
                        FROM matched_events AS matched_event
                        WHERE matched_event.candidate_session_id = candidates.id
                        ORDER BY matched_event.sequence DESC
                    ) AS ordered_sequence
                ), json('[]')) AS matched_event_sequences_json
            FROM candidates
        """
        materialize_sql = f"""
            WITH candidates AS ({selected_candidate_sql}),
            matched_events AS (
                SELECT
                    source_event.session_id AS candidate_session_id,
                    source_event.sequence,
                    {projected_event_sql} AS event_json
                FROM cayu_events AS source_event
                WHERE source_event.sequence IN (
                    SELECT CAST(value AS INTEGER) FROM json_each(?)
                )
            )
            SELECT
                candidates.id,
                candidates.pending_state_json,
                COALESCE((
                    SELECT json_group_array(json_object(
                        'sequence', ordered_event.sequence,
                        'event', json(ordered_event.event_json)
                    ))
                    FROM (
                        SELECT *
                        FROM matched_events
                        WHERE candidate_session_id = candidates.id
                        ORDER BY sequence DESC
                    ) AS ordered_event
                ), json('[]')) AS pending_events_json
            FROM candidates
        """

        def run_query(connection: sqlite3.Connection) -> PendingActionListResult:
            connection.execute("BEGIN")
            try:
                candidate_rows = connection.execute(
                    candidate_select_sql,
                    [*params, candidate_limit],
                ).fetchall()
                has_more_candidates = len(candidate_rows) > inspected_candidate_limit
                inspected_rows = candidate_rows[:inspected_candidate_limit]
                candidate_sessions = {
                    row["id"]: sqlite_support.pending_action_session_from_row(row, labels={})
                    for row in inspected_rows
                }
                inspected_ids = [row["id"] for row in inspected_rows]
                selected_ids_json = sqlite_support.json_dumps(inspected_ids)

                checkpoint_preflight_by_session_id: dict[str, tuple[int, int]] = {}
                if inspected_ids:
                    for row in connection.execute(
                        checkpoint_preflight_sql,
                        (selected_ids_json,),
                    ).fetchall():
                        version_type = row["checkpoint_root_field_type"]
                        scalar_text = row["checkpoint_root_field_scalar"]
                        if checkpoint_root_guard is not None:
                            checkpoint_root_guard.validate(
                                row["session_id"],
                                checkpoint_root_field_projection_from_storage(
                                    json_type=version_type,
                                    scalar_text=scalar_text,
                                ),
                            )
                        if row["pending_state_bytes"] is not None:
                            checkpoint_preflight_by_session_id[row["session_id"]] = (
                                int(row["pending_state_bytes"]),
                                int(row["pending_tool_call_count"]),
                            )

                oversized_ids: set[str] = set()
                overcomplex_ids: set[str] = set()
                preflight_eligible_ids: list[str] = []
                preflight_processable_ids: list[str] = []
                preflight_source_bytes = 0
                preflight_stopped_for_bytes = False
                for session_id in inspected_ids:
                    checkpoint_preflight = checkpoint_preflight_by_session_id.get(session_id)
                    if checkpoint_preflight is None:
                        oversized_ids.add(session_id)
                        preflight_processable_ids.append(session_id)
                        continue
                    pending_state_bytes, pending_tool_call_count = checkpoint_preflight
                    if pending_state_bytes > query.max_result_bytes:
                        oversized_ids.add(session_id)
                        preflight_processable_ids.append(session_id)
                        continue
                    if pending_tool_call_count > MAX_PENDING_ACTION_TOOL_CALLS:
                        overcomplex_ids.add(session_id)
                        preflight_processable_ids.append(session_id)
                        continue
                    if preflight_source_bytes + pending_state_bytes > query.max_result_bytes:
                        preflight_stopped_for_bytes = True
                        break
                    preflight_source_bytes += pending_state_bytes
                    preflight_eligible_ids.append(session_id)
                    preflight_processable_ids.append(session_id)

                source_metadata_by_session_id: dict[str, tuple[int, list[int]]] = {}
                invalid_ids: set[str] = set()
                ledger_overcomplex_ids: set[str] = set()
                if preflight_eligible_ids:
                    for row in connection.execute(
                        source_size_sql,
                        (sqlite_support.json_dumps(preflight_eligible_ids),),
                    ).fetchall():
                        sequence_values = json.loads(row["matched_event_sequences_json"])
                        if type(sequence_values) is not list or any(
                            type(sequence) is not int for sequence in sequence_values
                        ):
                            raise ValueError(
                                "SQLite pending event sequence projection must be an integer array."
                            )
                        source_metadata_by_session_id[row["id"]] = (
                            int(row["source_bytes"]),
                            sequence_values,
                        )
                        if not bool(row["projections_ready"]):
                            invalid_ids.add(row["id"])
                        if bool(row["ledger_too_complex"]):
                            ledger_overcomplex_ids.add(row["id"])

                processable_ids: list[str] = []
                materializable_ids: list[str] = []
                materialized_source_bytes = 0
                stopped_for_bytes = preflight_stopped_for_bytes
                for session_id in preflight_processable_ids:
                    session = candidate_sessions[session_id]
                    if (
                        session_id in oversized_ids
                        or session_id in overcomplex_ids
                        or session_id in ledger_overcomplex_ids
                        or session_id in invalid_ids
                    ):
                        processable_ids.append(session_id)
                        continue
                    session_size = JsonUtf8SizeCounter(query.max_result_bytes)
                    session_fits = session_size.value(session)
                    source_metadata = source_metadata_by_session_id.get(session_id)
                    if not session_fits or source_metadata is None:
                        oversized_ids.add(session_id)
                        processable_ids.append(session_id)
                        continue
                    stored_source_bytes = source_metadata[0]
                    candidate_bytes = (
                        query.max_result_bytes - session_size.remaining + stored_source_bytes
                    )
                    if candidate_bytes > query.max_result_bytes:
                        oversized_ids.add(session_id)
                        processable_ids.append(session_id)
                        continue
                    if materialized_source_bytes + candidate_bytes > query.max_result_bytes:
                        stopped_for_bytes = True
                        break
                    materialized_source_bytes += candidate_bytes
                    materializable_ids.append(session_id)
                    processable_ids.append(session_id)

                grouped: dict[str, tuple[dict[str, Any], list[EventRecord]]] = {}
                if materializable_ids:
                    materializable_sequences = sorted(
                        {
                            sequence
                            for session_id in materializable_ids
                            for sequence in source_metadata_by_session_id[session_id][1]
                        }
                    )
                    rows = connection.execute(
                        materialize_sql,
                        (
                            sqlite_support.json_dumps(materializable_ids),
                            sqlite_support.json_dumps(materializable_sequences),
                        ),
                    ).fetchall()
                    for row in rows:
                        session_id = row["id"]
                        pending_events = json.loads(row["pending_events_json"])
                        if type(pending_events) is not list:
                            raise ValueError("SQLite pending events projection must be an array.")
                        records: list[EventRecord] = []
                        for pending_event in pending_events:
                            if type(pending_event) is not dict:
                                raise ValueError(
                                    "SQLite pending event projections must be objects."
                                )
                            event_value = pending_event.get("event")
                            if type(event_value) is not dict:
                                raise ValueError(
                                    "SQLite pending event values must be event objects."
                                )
                            records.append(
                                EventRecord(
                                    sequence=pending_event.get("sequence"),
                                    event=Event(**event_value),
                                )
                            )
                        grouped[session_id] = (
                            copy_durable_json_object(
                                json.loads(row["pending_state_json"]),
                                "checkpoint",
                            ),
                            records,
                        )

                labels_by_session_id = self._load_labels_for_sessions_unlocked(
                    materializable_ids,
                    connection=connection,
                )
                actions = []
                issues: list[PendingActionIssue] = []
                inspected_count = 0
                more_matching = False
                last_inspected_session: PendingActionSession | None = None
                for session_id in processable_ids:
                    session = candidate_sessions[session_id]
                    if session_id in oversized_ids:
                        if len(actions) + len(issues) == query.limit:
                            more_matching = True
                            break
                        issues.append(
                            PendingActionIssue.source_too_large(
                                session,
                                max_bytes=query.max_result_bytes,
                            )
                        )
                        inspected_count += 1
                        last_inspected_session = session
                        continue
                    if session_id in overcomplex_ids:
                        if len(actions) + len(issues) == query.limit:
                            more_matching = True
                            break
                        issues.append(
                            PendingActionIssue.source_too_complex(
                                session,
                                max_tool_calls=MAX_PENDING_ACTION_TOOL_CALLS,
                            )
                        )
                        inspected_count += 1
                        last_inspected_session = session
                        continue
                    if session_id in ledger_overcomplex_ids:
                        if len(actions) + len(issues) == query.limit:
                            more_matching = True
                            break
                        issues.append(
                            PendingActionIssue.ledger_too_complex(
                                session,
                                max_events_per_call=(MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL),
                            )
                        )
                        inspected_count += 1
                        last_inspected_session = session
                        continue
                    if session_id in invalid_ids:
                        if len(actions) + len(issues) == query.limit:
                            more_matching = True
                            break
                        issues.append(PendingActionIssue.source_invalid(session))
                        inspected_count += 1
                        last_inspected_session = session
                        continue

                    checkpoint, records = grouped[session_id]
                    session = session.model_copy(
                        update={"labels": labels_by_session_id.get(session_id, {})},
                        deep=True,
                    )
                    action = pending_action_from_records(session, records, checkpoint)
                    if pending_action_source_is_invalid(session, checkpoint, action, records):
                        if len(actions) + len(issues) == query.limit:
                            more_matching = True
                            break
                        issues.append(PendingActionIssue.source_invalid(session))
                        inspected_count += 1
                        last_inspected_session = session
                        continue
                    if action is None or (query.kind is not None and action.kind != query.kind):
                        inspected_count += 1
                        last_inspected_session = session
                        continue
                    if not pending_action_matches_query(action, query.q):
                        inspected_count += 1
                        last_inspected_session = session
                        continue
                    if len(actions) + len(issues) == query.limit:
                        more_matching = True
                        break
                    actions.append(action)
                    inspected_count += 1
                    last_inspected_session = session

                has_more = more_matching or has_more_candidates or stopped_for_bytes
                next_cursor = (
                    encode_session_cursor(
                        last_inspected_session,
                        SessionOrder.UPDATED_AT_DESC,
                    )
                    if has_more and last_inspected_session is not None
                    else None
                )
                return enforce_pending_action_result_size(
                    PendingActionListResult(
                        actions=actions,
                        issues=issues,
                        next_cursor=next_cursor,
                        has_more=has_more,
                        total_count=None,
                        inspected_candidate_count=inspected_count,
                    ),
                    max_bytes=query.max_result_bytes,
                )
            finally:
                # End the pinned WAL snapshot on success and on every failure.
                connection.rollback()

        return await self._run_read(run_query)

    async def _list_sessions(
        self,
        query: SessionQuery | None,
        *,
        pending_interruption_cascade_only: bool,
    ) -> SessionListResult:
        query = copy_session_query(query)
        session_source_sql = (
            """
            (
                SELECT session_id
                FROM cayu_checkpoints
                    INDEXED BY idx_cayu_checkpoints_pending_interruption_cascade
                WHERE json_type(
                    state_json,
                    '$.pending_interruption_cascade'
                ) IS NOT NULL
            ) AS pending_interruption_cascades
            CROSS JOIN cayu_sessions
                ON cayu_sessions.id = pending_interruption_cascades.session_id
            """
            if pending_interruption_cascade_only
            else "cayu_sessions"
        )

        def run_query(connection: sqlite3.Connection) -> SessionListResult:
            inactive_before = (
                query.last_activity_before
                if query.inactive_for_seconds is None
                else utc_duration_cutoff(
                    self._ownership_clock(),
                    query.inactive_for_seconds,
                )
            )
            if query.inactive_for_seconds is not None and inactive_before is None:
                return SessionListResult(
                    sessions=[],
                    next_cursor=None,
                    total_count=0 if query.include_total_count else None,
                )
            resolved_query = query.model_copy(
                update={
                    "last_activity_before": inactive_before,
                    "inactive_for_seconds": None,
                }
            )
            plan = session_store_sql.build_session_query_sql(
                resolved_query,
                dialect=_SQL_DIALECT,
            )
            total_count: int | None = None
            if query.include_total_count:
                total_count = connection.execute(
                    f"SELECT COUNT(*) FROM {session_source_sql} {plan.filter_where_sql}",
                    plan.filter_params,
                ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT id, instance_id, agent_name, provider_name, model, parent_session_id,
                       causal_budget_id, runtime_name, runtime_version, environment_name,
                       status, created_at, updated_at, last_activity_at, run_epoch,
                       invocation_json, metadata_json
                FROM {session_source_sql}
                {plan.page_where_sql}
                ORDER BY {plan.order_sql}
                {plan.pagination_sql}
                """,
                plan.page_params,
            ).fetchall()
            has_more = len(rows) > query.limit
            rows = rows[: query.limit]
            labels_by_session_id = self._load_labels_for_sessions_unlocked(
                [row["id"] for row in rows],
                connection=connection,
            )
            sessions = [
                sqlite_support.session_from_row(
                    row,
                    labels=labels_by_session_id.get(row["id"], {}),
                )
                for row in rows
            ]
            next_cursor = session_next_cursor(sessions, has_more, query.order_by)
            return SessionListResult(
                sessions=sessions,
                next_cursor=next_cursor,
                total_count=total_count,
            )

        return await self._run_read(run_query)

    async def append_transcript_messages(
        self,
        session_id: str,
        messages: list[Message],
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        copied_messages = copy_transcript_messages(messages)

        def statement(connection: sqlite3.Connection) -> None:
            if not copied_messages:
                if not _session_exists(connection, session_id):
                    raise KeyError(f"Session not found: {session_id}")
                return
            try:
                connection.execute("BEGIN IMMEDIATE")
                if not _session_exists(connection, session_id):
                    raise KeyError(f"Session not found: {session_id}")
                activity_at = self._ownership_clock()
                _touch_session_activity(connection, session_id, activity_at)
                connection.executemany(
                    """
                    INSERT INTO cayu_transcript_messages (
                        session_id,
                        role,
                        interaction_id,
                        message_json,
                        transcript_search_document
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            session_id,
                            str(message.role),
                            interaction_id,
                            sqlite_support.json_dumps(message.model_dump(mode="json")),
                            transcript_search_document(message),
                        )
                        for message in copied_messages
                    ],
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

        await self._run_write(statement)

    async def replace_initial_transcript_messages(
        self,
        session_id: str,
        expected_messages: list[Message],
        replacement_messages: list[Message],
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
        checkpoint_transform: CheckpointTransform | None = None,
        runtime_suffix_count: int = 0,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        if interaction_id is None:
            raise ValueError("Initial transcript publication requires an interaction identity.")
        expected = copy_transcript_messages(expected_messages)
        replacement = copy_transcript_messages(replacement_messages)

        def statement(connection: sqlite3.Connection) -> None:
            try:
                connection.execute("BEGIN IMMEDIATE")
                updated_at = self._ownership_clock()
                session = self._load_unlocked(session_id)
                if session is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, session)
                row = connection.execute(
                    "SELECT interaction_id, source_messages_json "
                    "FROM cayu_deferred_interaction_inputs WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if row is None or row["interaction_id"] != interaction_id:
                    raise RuntimeError("Deferred interaction input changed before finalization.")
                stored = deferred_interaction_input_from_storage_payload(
                    row["interaction_id"],
                    json.loads(row["source_messages_json"]),
                )
                require_deferred_initial_transcript_replacement(
                    stored,
                    expected_messages=expected,
                    replacement_messages=replacement,
                )
                existing = connection.execute(
                    "SELECT 1 FROM cayu_transcript_messages WHERE session_id = ? LIMIT 1",
                    (session_id,),
                ).fetchone()
                if existing is not None:
                    raise RuntimeError("Initial transcript changed before finalization.")
                prefix_count = _initial_transcript_prefix_count(
                    expected,
                    replacement,
                    runtime_suffix_count=runtime_suffix_count,
                )
                current_checkpoint = self._load_checkpoint_unlocked(session_id)
                if checkpoint_transform is not None:
                    transformed = checkpoint_transform(
                        session,
                        _copy_checkpoint_for_transform(
                            current_checkpoint,
                            session_id=session_id,
                        ),
                    )
                    if transformed is not None:
                        current_checkpoint = _checkpoint_transform_result_preserving_completion_result_event_publications(
                            current_checkpoint,
                            transformed,
                            session_id=session_id,
                        )
                checkpoint = _checkpoint_after_initial_transcript_publication(
                    current_checkpoint,
                    interaction_id=interaction_id,
                )
                connection.executemany(
                    "INSERT INTO cayu_transcript_messages "
                    "(session_id, role, interaction_id, message_json, "
                    "transcript_search_document) VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            session_id,
                            str(message.role),
                            None if index < prefix_count else interaction_id,
                            sqlite_support.json_dumps(message.model_dump(mode="json")),
                            transcript_search_document(message),
                        )
                        for index, message in enumerate(replacement)
                    ],
                )
                connection.execute(
                    "DELETE FROM cayu_deferred_interaction_inputs WHERE session_id = ?",
                    (session_id,),
                )
                if checkpoint is None:
                    connection.execute(
                        "DELETE FROM cayu_checkpoints WHERE session_id = ?",
                        (session_id,),
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO cayu_checkpoints (
                            session_id, state_json, updated_at,
                            pending_action_source_bytes,
                            pending_action_tool_call_count,
                            pending_action_flags,
                            pending_action_metrics_ready
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(session_id) DO UPDATE SET
                            state_json = excluded.state_json,
                            updated_at = excluded.updated_at,
                            pending_action_source_bytes =
                                excluded.pending_action_source_bytes,
                            pending_action_tool_call_count =
                                excluded.pending_action_tool_call_count,
                            pending_action_flags = excluded.pending_action_flags,
                            pending_action_metrics_ready =
                                excluded.pending_action_metrics_ready
                        """,
                        sqlite_support.checkpoint_row_values(
                            session_id,
                            checkpoint,
                            updated_at,
                        ),
                    )
                _touch_session_activity(connection, session_id, updated_at)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

        await self._run_write(statement)

    async def materialize_deferred_interaction_input(
        self,
        session_id: str,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> bool:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)

        def statement(connection: sqlite3.Connection) -> bool:
            try:
                connection.execute("BEGIN IMMEDIATE")
                session = self._load_unlocked(session_id)
                if session is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, session)
                row = connection.execute(
                    "SELECT interaction_id, source_messages_json "
                    "FROM cayu_deferred_interaction_inputs WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if row is None:
                    connection.commit()
                    return False
                if row["interaction_id"] != interaction_id:
                    raise RuntimeError("Deferred interaction input belongs to another interaction.")
                deferred = deferred_interaction_input_from_storage_payload(
                    row["interaction_id"],
                    json.loads(row["source_messages_json"]),
                )
                messages = deferred.source_messages
                connection.executemany(
                    "INSERT INTO cayu_transcript_messages "
                    "(session_id, role, interaction_id, message_json, "
                    "transcript_search_document) VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            session_id,
                            str(message.role),
                            interaction_id,
                            sqlite_support.json_dumps(message.model_dump(mode="json")),
                            transcript_search_document(message),
                        )
                        for message in messages
                    ],
                )
                connection.execute(
                    "DELETE FROM cayu_deferred_interaction_inputs WHERE session_id = ?",
                    (session_id,),
                )
                _touch_session_activity(connection, session_id, self._ownership_clock())
                connection.commit()
                return True
            except Exception:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def load_deferred_interaction_input(
        self,
        session_id: str,
    ) -> DeferredInteractionInput | None:
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection: sqlite3.Connection) -> DeferredInteractionInput | None:
            if (
                connection.execute(
                    "SELECT 1 FROM cayu_sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                is None
            ):
                raise KeyError(f"Session not found: {session_id}")
            row = connection.execute(
                "SELECT interaction_id, source_messages_json "
                "FROM cayu_deferred_interaction_inputs WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return deferred_interaction_input_from_storage_payload(
                row["interaction_id"],
                json.loads(row["source_messages_json"]),
            )

        return await self._run_read(query)

    async def append_transcript_messages_and_transform_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint_transform: CheckpointTransform,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        copied_messages = copy_transcript_messages(messages)
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")

        def statement(connection: sqlite3.Connection) -> None:
            try:
                connection.execute("BEGIN IMMEDIATE")
                updated_at = self._ownership_clock()
                session = self._load_unlocked(session_id)
                if session is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, session)
                current_checkpoint = self._load_checkpoint_unlocked(session_id)
                transformed = checkpoint_transform(
                    session,
                    _copy_checkpoint_for_transform(
                        current_checkpoint,
                        session_id=session_id,
                    ),
                )
                if transformed is None:
                    raise ValueError("Checkpoint transform must return a checkpoint.")
                transformed = (
                    _checkpoint_transform_result_preserving_completion_result_event_publications(
                        current_checkpoint,
                        transformed,
                        session_id=session_id,
                    )
                )
                _touch_session_activity(connection, session_id, updated_at)
                if copied_messages:
                    connection.executemany(
                        """
                        INSERT INTO cayu_transcript_messages (
                            session_id,
                            role,
                            interaction_id,
                            message_json,
                            transcript_search_document
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                session_id,
                                str(message.role),
                                interaction_id,
                                sqlite_support.json_dumps(message.model_dump(mode="json")),
                                transcript_search_document(message),
                            )
                            for message in copied_messages
                        ],
                    )
                connection.execute(
                    """
                    INSERT INTO cayu_checkpoints (
                        session_id, state_json, updated_at,
                        pending_action_source_bytes,
                        pending_action_tool_call_count,
                        pending_action_flags,
                        pending_action_metrics_ready
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        state_json = excluded.state_json,
                        updated_at = excluded.updated_at,
                        pending_action_source_bytes = excluded.pending_action_source_bytes,
                        pending_action_tool_call_count = excluded.pending_action_tool_call_count,
                        pending_action_flags = excluded.pending_action_flags,
                        pending_action_metrics_ready = excluded.pending_action_metrics_ready
                    """,
                    sqlite_support.checkpoint_row_values(session_id, transformed, updated_at),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

        await self._run_write(statement)

    async def load_transcript(self, session_id: str) -> list[Message]:
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection: sqlite3.Connection) -> list[Message]:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            rows = connection.execute(
                """
                SELECT message_json
                FROM cayu_transcript_messages
                WHERE session_id = ?
                ORDER BY sequence ASC
                """,
                (session_id,),
            ).fetchall()
            return [Message(**json.loads(row["message_json"])) for row in rows]

        return await self._run_read(query)

    async def load_transcript_snapshot(self, session_id: str) -> TranscriptSnapshot:
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection: sqlite3.Connection) -> TranscriptSnapshot:
            rows = connection.execute(
                """
                SELECT session.transcript_seq,
                       transcript.session_order - 1 AS transcript_index,
                       transcript.interaction_id,
                       transcript.message_json
                FROM cayu_sessions AS session
                LEFT JOIN cayu_transcript_messages AS transcript
                  ON transcript.session_id = session.id
                WHERE session.id = ?
                ORDER BY transcript.session_order ASC
                """,
                (session_id,),
            ).fetchall()
            if not rows:
                raise KeyError(f"Session not found: {session_id}")
            return TranscriptSnapshot(
                records=[
                    TranscriptRecord(
                        index=row["transcript_index"],
                        interaction_id=row["interaction_id"],
                        message=Message(**json.loads(row["message_json"])),
                    )
                    for row in rows
                    if row["transcript_index"] is not None
                ],
                cursor=int(rows[0]["transcript_seq"]),
            )

        return await self._run_read(query)

    async def load_transcript_cursor(self, session_id: str) -> int:
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection: sqlite3.Connection) -> int:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            return _transcript_cursor(connection, session_id)

        return await self._run_read(query)

    async def load_latest_transcript_message(
        self,
        session_id: str,
        *,
        role: MessageRole,
    ) -> TranscriptRecord | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(role, MessageRole):
            raise TypeError("role must be a MessageRole.")

        def query_latest(connection: sqlite3.Connection) -> TranscriptRecord | None:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            row = connection.execute(
                "SELECT session_order - 1 AS transcript_index, interaction_id, message_json "
                "FROM cayu_transcript_messages "
                "WHERE session_id = ? AND role = ? "
                "ORDER BY session_order DESC LIMIT 1",
                (session_id, str(role)),
            ).fetchone()
            if row is None:
                return None
            return TranscriptRecord(
                index=row["transcript_index"],
                interaction_id=row["interaction_id"],
                message=Message(**json.loads(row["message_json"])),
            )

        return await self._run_read(query_latest)

    async def load_latest_transcript_text(
        self,
        session_id: str,
        *,
        role: MessageRole,
        max_chars: int,
    ) -> tuple[str, bool] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(role, MessageRole):
            raise TypeError("role must be a MessageRole.")
        if type(max_chars) is not int:
            raise TypeError("max_chars must be an integer.")
        if not 1 <= max_chars <= LATEST_TRANSCRIPT_TEXT_MAX_CHARS:
            raise ValueError(f"max_chars must be between 1 and {LATEST_TRANSCRIPT_TEXT_MAX_CHARS}.")

        def query_latest(connection: sqlite3.Connection) -> tuple[str, bool] | None:
            row = connection.execute(
                """
                SELECT session.id,
                       transcript.sequence,
                       length(CAST(transcript.message_json AS BLOB))
                FROM cayu_sessions AS session
                LEFT JOIN cayu_transcript_messages AS transcript
                  ON transcript.sequence = (
                      SELECT candidate.sequence
                      FROM cayu_transcript_messages AS candidate
                      WHERE candidate.session_id = session.id
                        AND candidate.role = ?
                      ORDER BY candidate.session_order DESC
                      LIMIT 1
                  )
                WHERE session.id = ?
                """,
                (str(role), session_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"Session not found: {session_id}")
            sequence = row[1]
            if sequence is None:
                return None
            if int(row[2]) > LATEST_TRANSCRIPT_TEXT_MAX_SOURCE_BYTES:
                raise TranscriptTextReadLimitExceeded(
                    "Transcript message exceeds the bounded serialized-source limit."
                )
            projection = connection.execute(
                """
                WITH RECURSIVE
                source(message_json, part_count) AS (
                    SELECT
                        message_json,
                        json_array_length(message_json, '$.content')
                    FROM cayu_transcript_messages
                    WHERE sequence = ?
                ),
                prefix(part_index, text_value, part_count) AS (
                    SELECT 0, '', part_count
                    FROM source
                    UNION ALL
                    SELECT
                        prefix.part_index + 1,
                        substr(
                            prefix.text_value ||
                            CASE
                                WHEN json_extract(
                                    source.message_json,
                                    '$.content[' || prefix.part_index || '].type'
                                ) = 'text'
                                THEN COALESCE(
                                    json_extract(
                                        source.message_json,
                                        '$.content[' || prefix.part_index || '].text'
                                    ),
                                    ''
                                )
                                ELSE ''
                            END,
                            1,
                            ?
                        ),
                        prefix.part_count
                    FROM prefix
                    CROSS JOIN source
                    WHERE prefix.part_index < prefix.part_count
                      AND prefix.part_index < ?
                      AND length(prefix.text_value) <= ?
                )
                SELECT text_value, part_index, part_count
                FROM prefix
                ORDER BY part_index DESC
                LIMIT 1
                """,
                (
                    int(sequence),
                    max_chars + 1,
                    LATEST_TRANSCRIPT_TEXT_MAX_PARTS,
                    max_chars,
                ),
            ).fetchone()
            if projection is None:
                raise TranscriptTextReadLimitExceeded(
                    "Transcript message changed during its bounded text projection."
                )
            text_value = str(projection[0])
            if int(projection[1]) < int(projection[2]) and len(text_value) <= max_chars:
                raise TranscriptTextReadLimitExceeded(
                    "Transcript message exceeds the bounded content-part inspection limit."
                )
            return text_value[:max_chars], len(text_value) > max_chars

        return await self._run_read(query_latest)

    async def load_transcript_window(
        self,
        session_id: str,
        *,
        start_index: int,
        limit: int,
    ) -> TranscriptSnapshot:
        session_id = require_clean_nonblank(session_id, "session_id")
        if type(start_index) is not int:
            raise TypeError("start_index must be an integer.")
        if not 0 <= start_index <= MAX_DURABLE_JSON_INTEGER:
            raise ValueError("start_index exceeds the durable integer limit.")
        if type(limit) is not int:
            raise TypeError("limit must be an integer.")
        if not 1 <= limit <= 5000:
            raise ValueError("limit must be between 1 and 5000.")

        def query(connection: sqlite3.Connection) -> TranscriptSnapshot:
            rows = connection.execute(
                """
                SELECT session.transcript_seq,
                       transcript.session_order - 1 AS transcript_index,
                       transcript.interaction_id,
                       transcript.message_json
                FROM cayu_sessions AS session
                LEFT JOIN cayu_transcript_messages AS transcript
                  ON transcript.session_id = session.id
                 AND transcript.session_order > ?
                WHERE session.id = ?
                ORDER BY transcript.session_order ASC
                LIMIT ?
                """,
                (start_index, session_id, limit),
            ).fetchall()
            if not rows:
                raise KeyError(f"Session not found: {session_id}")
            return TranscriptSnapshot(
                records=[
                    TranscriptRecord(
                        index=row["transcript_index"],
                        interaction_id=row["interaction_id"],
                        message=Message(**json.loads(row["message_json"])),
                    )
                    for row in rows
                    if row["transcript_index"] is not None
                ],
                cursor=int(rows[0]["transcript_seq"]),
            )

        return await self._run_read(query)

    async def query_transcript(self, query: TranscriptQuery) -> TranscriptPage:
        query = copy_transcript_query(query)
        filters: list[str] = []
        filter_params: list[object] = []
        if query.role is not None:
            filters.append("role = ?")
            filter_params.append(str(query.role))
        if query.interaction_id is not None:
            filters.append("interaction_id = ?")
            filter_params.append(query.interaction_id)
        filter_clause = " AND " + " AND ".join(filters) if filters else ""

        def run_query(connection: sqlite3.Connection) -> TranscriptPage:
            if not _session_exists(connection, query.session_id):
                raise KeyError(f"Session not found: {query.session_id}")

            total_row = connection.execute(
                f"""
                SELECT COUNT(*) AS total_records
                FROM cayu_transcript_messages
                WHERE session_id = ?
                {filter_clause}
                """,
                [query.session_id, *filter_params],
            ).fetchone()
            total_records = int(total_row["total_records"])

            page_params: list[object] = [
                query.session_id,
                *filter_params,
                query.limit,
                query.offset,
            ]
            rows = connection.execute(
                f"""
                SELECT session_order - 1 AS transcript_index, interaction_id, message_json
                FROM cayu_transcript_messages
                WHERE session_id = ?
                {filter_clause}
                ORDER BY session_order ASC
                LIMIT ? OFFSET ?
                """,
                page_params,
            ).fetchall()
            records = [
                TranscriptRecord(
                    index=row["transcript_index"],
                    interaction_id=row["interaction_id"],
                    message=Message(**json.loads(row["message_json"])),
                )
                for row in rows
            ]
            return TranscriptPage(
                records=filter_transcript_records(records, include_thinking=query.include_thinking),
                total_records=total_records,
            )

        return await self._run_read(run_query)

    async def search_transcript(
        self,
        query: TranscriptSearchQuery,
    ) -> TranscriptSearchResult:
        query = copy_transcript_search_query(query)
        cursor = decode_transcript_search_cursor(query)
        roles = tuple(str(role) for role in query.roles)
        role_placeholders = ", ".join("?" for _ in roles)
        session_placeholders = ", ".join("?" for _ in query.session_ids)
        before_filter = "".join(
            " AND (transcript.session_id <> ? OR transcript.session_order <= ?)"
            for _ in query.before_transcript_indexes
        )
        before_params = [
            value
            for session_id, before_index in query.before_transcript_indexes.items()
            for value in (session_id, before_index)
        ]
        query_document = transcript_search_query_document(query.text)
        fetch_limit = query.max_records_scanned + 1

        def run_search(connection: sqlite3.Connection) -> TranscriptSearchResult:
            rows = connection.execute(
                f"""
                SELECT
                    transcript.session_id,
                    transcript.session_order - 1 AS transcript_index,
                    transcript.interaction_id,
                    transcript.message_json,
                    transcript.transcript_search_document
                FROM cayu_transcript_messages_fts
                JOIN cayu_transcript_messages AS transcript
                  ON transcript.sequence = cayu_transcript_messages_fts.rowid
                WHERE cayu_transcript_messages_fts MATCH ?
                  AND transcript.session_id IN ({session_placeholders})
                  AND transcript.role IN ({role_placeholders})
                  {before_filter}
                LIMIT ?
                """,
                [
                    _sqlite_transcript_search_expression(query),
                    *query.session_ids,
                    *roles,
                    *before_params,
                    fetch_limit,
                ],
            ).fetchall()
            if len(rows) > query.max_records_scanned:
                return TranscriptSearchResult(
                    query=query,
                    matched_records_examined=query.max_records_scanned,
                    truncated=True,
                    coverage_complete=False,
                )

            candidates: list[tuple[int, sqlite3.Row, Message]] = []
            for row in rows:
                message = Message.model_validate(json.loads(row["message_json"]))
                document = transcript_search_document(message)
                if row["transcript_search_document"] != document:
                    raise RuntimeError("SQLite transcript search document is inconsistent.")
                score = transcript_search_document_score(document, query_document)
                if score <= 0:
                    raise RuntimeError("SQLite transcript search index is inconsistent.")
                candidates.append((score, row, message))
            candidates.sort(
                key=lambda item: (
                    -item[0],
                    item[1]["session_id"],
                    -item[1]["transcript_index"],
                )
            )
            if cursor is not None:
                candidates = [
                    candidate
                    for candidate in candidates
                    if transcript_search_position_after_cursor(
                        raw_score=candidate[0],
                        session_id=candidate[1]["session_id"],
                        transcript_index=candidate[1]["transcript_index"],
                        cursor=cursor,
                    )
                ]

            hits: list[TranscriptSearchHit] = []
            remaining_bytes = query.max_bytes
            truncated = False
            continuation_available = False
            for candidate_index, (score, row, message) in enumerate(candidates):
                if len(hits) >= query.limit:
                    truncated = True
                    continuation_available = True
                    break
                hit = transcript_search_hit_from_message(
                    session_id=row["session_id"],
                    transcript_index=row["transcript_index"],
                    interaction_id=row["interaction_id"],
                    message=message,
                    max_text_bytes=remaining_bytes,
                    raw_score=float(score),
                )
                if hit is None:
                    truncated = True
                    continuation_available = bool(hits)
                    break
                hits.append(hit)
                remaining_bytes -= len(hit.text.encode("utf-8"))
                has_remaining_candidate = candidate_index + 1 < len(candidates)
                if not hit.text_complete:
                    truncated = True
                    continuation_available = has_remaining_candidate
                    break
                if remaining_bytes == 0:
                    truncated = has_remaining_candidate
                    continuation_available = truncated
                    break
            next_cursor = (
                encode_transcript_search_cursor(
                    query,
                    raw_score=int(hits[-1].raw_score or 0),
                    session_id=hits[-1].session_id,
                    transcript_index=hits[-1].transcript_index,
                )
                if continuation_available and hits
                else None
            )
            return TranscriptSearchResult(
                query=query,
                hits=tuple(hits),
                matched_records_examined=len(rows),
                truncated=truncated,
                coverage_complete=True,
                next_cursor=next_cursor,
            )

        return await self._run_read(run_search)

    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state must be a dictionary.")
        checkpoint = copy_durable_json_object(state, "checkpoint")

        def statement(connection: sqlite3.Connection) -> None:
            try:
                connection.execute("BEGIN IMMEDIATE")
                updated_at = self._ownership_clock()
                if not _session_exists(connection, session_id):
                    raise KeyError(f"Session not found: {session_id}")
                replacement = _replace_checkpoint_preserving_completion_result_event_publications(
                    self._load_checkpoint_unlocked(session_id),
                    checkpoint,
                    session_id=session_id,
                )
                _touch_session_activity(connection, session_id, updated_at)
                connection.execute(
                    """
                    INSERT INTO cayu_checkpoints (
                        session_id, state_json, updated_at,
                        pending_action_source_bytes,
                        pending_action_tool_call_count,
                        pending_action_flags,
                        pending_action_metrics_ready
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        state_json = excluded.state_json,
                        updated_at = excluded.updated_at,
                        pending_action_source_bytes = excluded.pending_action_source_bytes,
                        pending_action_tool_call_count = excluded.pending_action_tool_call_count,
                        pending_action_flags = excluded.pending_action_flags,
                        pending_action_metrics_ready = excluded.pending_action_metrics_ready
                    """,
                    sqlite_support.checkpoint_row_values(session_id, replacement, updated_at),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

        await self._run_write(statement)

    async def transform_checkpoint(
        self,
        session_id: str,
        checkpoint_transform: CheckpointTransform,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")

        def statement(connection: sqlite3.Connection) -> None:
            try:
                connection.execute("BEGIN IMMEDIATE")
                updated_at = self._ownership_clock()
                session = self._load_unlocked(session_id)
                if session is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, session)
                current = self._load_checkpoint_unlocked(session_id)
                transformed = checkpoint_transform(
                    session,
                    _copy_checkpoint_for_transform(current, session_id=session_id),
                )
                if transformed is not None:
                    transformed = (
                        _replace_checkpoint_preserving_completion_result_event_publications(
                            current,
                            copy_durable_json_object(transformed, "checkpoint"),
                            session_id=session_id,
                        )
                    )
                    _touch_session_activity(connection, session_id, updated_at)
                    connection.execute(
                        """
                        INSERT INTO cayu_checkpoints (
                            session_id, state_json, updated_at,
                            pending_action_source_bytes,
                            pending_action_tool_call_count,
                            pending_action_flags,
                            pending_action_metrics_ready
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(session_id) DO UPDATE SET
                            state_json = excluded.state_json,
                            updated_at = excluded.updated_at,
                            pending_action_source_bytes = excluded.pending_action_source_bytes,
                            pending_action_tool_call_count = excluded.pending_action_tool_call_count,
                            pending_action_flags = excluded.pending_action_flags,
                            pending_action_metrics_ready = excluded.pending_action_metrics_ready
                        """,
                        sqlite_support.checkpoint_row_values(session_id, transformed, updated_at),
                    )
                connection.commit()
            except BaseException as primary:
                transaction_failure = sqlite_support._settle_failed_transaction(
                    connection,
                    primary,
                )
                if transaction_failure is not primary:
                    raise transaction_failure from None
                raise

        await self._run_write(statement)

    async def transform_checkpoint_with_store_time(
        self,
        session_id: str,
        checkpoint_transform: StoreTimeCheckpointTransform,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")

        def statement(connection: sqlite3.Connection) -> None:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = self._ownership_clock()
                session = self._load_unlocked(session_id)
                if session is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, session)
                current = self._load_checkpoint_unlocked(session_id)
                transformed = checkpoint_transform(
                    session,
                    _copy_checkpoint_for_transform(current, session_id=session_id),
                    now,
                )
                if transformed is not None:
                    transformed = (
                        _replace_checkpoint_preserving_completion_result_event_publications(
                            current,
                            copy_durable_json_object(transformed, "checkpoint"),
                            session_id=session_id,
                        )
                    )
                    _touch_session_activity(connection, session_id, now)
                    connection.execute(
                        """
                        INSERT INTO cayu_checkpoints (
                            session_id, state_json, updated_at,
                            pending_action_source_bytes,
                            pending_action_tool_call_count,
                            pending_action_flags,
                            pending_action_metrics_ready
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(session_id) DO UPDATE SET
                            state_json = excluded.state_json,
                            updated_at = excluded.updated_at,
                            pending_action_source_bytes = excluded.pending_action_source_bytes,
                            pending_action_tool_call_count = excluded.pending_action_tool_call_count,
                            pending_action_flags = excluded.pending_action_flags,
                            pending_action_metrics_ready = excluded.pending_action_metrics_ready
                        """,
                        sqlite_support.checkpoint_row_values(session_id, transformed, now),
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

        await self._run_write(statement)

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        return await self._run_read(
            lambda connection: _load_checkpoint_state(connection, session_id)
        )

    async def load_interruption_cascade_marker(
        self,
        session_id: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        return await self._run_read(
            lambda connection: _load_interruption_cascade_marker(
                connection,
                session_id,
                checkpoint_root_guard,
            )
        )

    def _load_checkpoint_unlocked(self, session_id: str) -> dict[str, Any] | None:
        return _load_checkpoint_state(self._connection, session_id)

    async def close(self) -> None:
        async with self._lock:
            if self._read_connection is not self._connection:
                async with self._read_lock:
                    self._read_connection.close()
            self._connection.close()

    def _connect(self, path: Path) -> sqlite3.Connection:
        return sqlite_support.connect(path)

    def _connect_read_only(self, path: Path) -> sqlite3.Connection:
        if sqlite_support.current_diagnostic_store_inspection() is not None:
            return sqlite_support.connect_read_only_inspection(path)
        return sqlite_support.connect(path, read_only=True)

    def _initialize_schema(self) -> None:
        sqlite_support.reconcile_schema(
            self._connection,
            self._schema_mode,
            app_min_supported=_SQLITE_SESSION_MIN_REQUIRED_REVISION,
        )
        state = sqlite_support.read_schema_state(self._connection)
        if state.revision < _SQLITE_SESSION_MIN_REQUIRED_REVISION:
            raise schema.SchemaTooOld(
                f"SQLite session schema is at revision {state.revision}; this build requires "
                f">= {_SQLITE_SESSION_MIN_REQUIRED_REVISION}. Run `cayu storage migrate` before "
                "starting."
            )

    def _load_unlocked(self, session_id: str) -> Session | None:
        return _load_session(self._connection, session_id)

    def _load_labels_unlocked(self, session_id: str) -> dict[str, str]:
        return _load_labels(self._connection, session_id)

    def _load_labels_for_sessions_unlocked(
        self,
        session_ids: list[str],
        *,
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, dict[str, str]]:
        if not session_ids:
            return {}
        source = self._connection if connection is None else connection
        placeholders = ", ".join("?" for _ in session_ids)
        rows = source.execute(
            f"""
            SELECT session_id, key, value
            FROM cayu_session_labels
            WHERE session_id IN ({placeholders})
            ORDER BY session_id ASC, key ASC
            """,
            session_ids,
        ).fetchall()
        labels_by_session_id: dict[str, dict[str, str]] = {
            session_id: {} for session_id in session_ids
        }
        for row in rows:
            labels_by_session_id[row["session_id"]][row["key"]] = row["value"]
        return labels_by_session_id

    def _session_exists_unlocked(self, session_id: str) -> bool:
        return _session_exists(self._connection, session_id)

    def _first_existing_event_id_unlocked(
        self,
        session_id: str,
        event_ids: list[str],
    ) -> str | None:
        return _first_existing_event_id(self._connection, session_id, event_ids)


class SQLiteTaskStore(TaskStore):
    """SQLite-backed task store for durable local work items."""

    supports_delayed_availability: ClassVar[bool] = True
    supports_task_topology: ClassVar[bool] = True
    supports_idempotent_terminalization: ClassVar[bool] = True
    supports_attached_task_recovery_terminalization: ClassVar[bool] = True
    supports_interrupted_task_handoffs: ClassVar[bool] = True
    supports_task_cancellation_reconciliation: ClassVar[bool] = True
    supports_task_retry_series: ClassVar[bool] = True
    supports_verified_work_contracts: ClassVar[bool] = True
    supports_work_attempt_admission: ClassVar[bool] = True
    supports_local_execution_attempts: ClassVar[bool] = True
    verified_work_mutations_are_cancellation_quiescent: ClassVar[bool] = True

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
        ownership_clock: Callable[[], datetime] | None = None,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
    ) -> None:
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str:
            db_path = Path(require_nonblank(path, "path"))
        else:
            raise TypeError("SQLiteTaskStore path must be a string or Path.")
        self.service_durability = (
            RuntimeStoreDurability.DEVELOPMENT
            if str(db_path) == ":memory:"
            else RuntimeStoreDurability.DURABLE
        )
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")

        self.path = db_path
        diagnostic_source_missing = sqlite_support.diagnostic_sqlite_source_missing(db_path)
        self._diagnostic_source_missing = diagnostic_source_missing
        self._schema_mode = schema.SchemaMode.CREATE if diagnostic_source_missing else schema_mode
        self._clock = utc_clock(clock)
        self._enable_task_admission_wakeups()
        self._ownership_clock = utc_clock(ownership_clock)
        self._lock = asyncio.Lock()
        effective_db_path = Path(":memory:") if diagnostic_source_missing else db_path
        self._connection = self._connect(effective_db_path)
        self._initialize_schema()
        if diagnostic_source_missing:
            self._connection.execute("PRAGMA query_only = ON")

    def _verified_transaction_unlocked(self):
        return sqlite_support._transaction(self._connection)

    def _load_local_execution_attempt_unlocked(
        self,
        attempt_id: str,
    ) -> LocalExecutionAttemptRecord | None:
        row = self._connection.execute(
            "SELECT attempt_id, task_id, retry_series_id, effect_lineage_id, "
            "request_sha256, phase, quiescence, retry_admissible, "
            "recovery_generation, recovery_owner_id, recovery_owner_expires_at, "
            "record_json, created_at, updated_at "
            "FROM cayu_local_execution_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            record = LocalExecutionAttemptRecord.model_validate(json.loads(row["record_json"]))
        except (json.JSONDecodeError, ValidationError):
            raise LocalExecutionAttemptConflict(
                "Stored local execution attempt content is malformed."
            ) from None
        if (
            record.authority.attempt_id != row["attempt_id"]
            or record.authority.task_id != row["task_id"]
            or record.authority.retry_series_id != row["retry_series_id"]
            or record.authority.effect_lineage_id != row["effect_lineage_id"]
            or record.authority.request_sha256 != row["request_sha256"]
            or record.phase.value != row["phase"]
            or record.quiescence.value != row["quiescence"]
            or int(record.retry_admissible) != row["retry_admissible"]
            or record.recovery_generation != row["recovery_generation"]
            or record.recovery_owner_id != row["recovery_owner_id"]
            or record.recovery_owner_expires_at
            != sqlite_support.parse_optional_datetime(row["recovery_owner_expires_at"])
            or record.created_at != sqlite_support.parse_datetime(row["created_at"])
            or record.updated_at != sqlite_support.parse_datetime(row["updated_at"])
        ):
            raise LocalExecutionAttemptConflict(
                "Stored local execution attempt indexes conflict with canonical content."
            )
        return record

    def _latest_local_execution_attempt_unlocked(
        self,
        authority: LocalExecutionAttemptAuthority,
    ) -> LocalExecutionAttemptRecord | None:
        if authority.retry_series_id is None:
            row = self._connection.execute(
                "SELECT attempt_id FROM cayu_local_execution_attempts "
                "WHERE retry_series_id IS NULL AND task_id = ? AND effect_lineage_id = ? "
                "ORDER BY retry_admissible ASC, created_at DESC, attempt_id DESC LIMIT 1",
                (authority.task_id, authority.effect_lineage_id),
            ).fetchone()
        else:
            row = self._connection.execute(
                "SELECT attempt_id FROM cayu_local_execution_attempts "
                "WHERE retry_series_id = ? AND effect_lineage_id = ? "
                "ORDER BY retry_admissible ASC, created_at DESC, attempt_id DESC LIMIT 1",
                (authority.retry_series_id, authority.effect_lineage_id),
            ).fetchone()
        return (
            None if row is None else self._load_local_execution_attempt_unlocked(row["attempt_id"])
        )

    def _store_local_execution_attempt_unlocked(
        self,
        record: LocalExecutionAttemptRecord,
        *,
        insert: bool,
    ) -> None:
        values = (
            record.authority.attempt_id,
            record.authority.task_id,
            record.authority.retry_series_id,
            record.authority.effect_lineage_id,
            record.authority.request_sha256,
            record.phase.value,
            record.quiescence.value,
            int(record.retry_admissible),
            record.recovery_generation,
            record.recovery_owner_id,
            sqlite_support.format_optional_datetime(record.recovery_owner_expires_at),
            sqlite_support.json_dumps(record.model_dump(mode="json", warnings=False)),
            sqlite_support.format_datetime(record.created_at),
            sqlite_support.format_datetime(record.updated_at),
        )
        if insert:
            self._connection.execute(
                "INSERT INTO cayu_local_execution_attempts ("
                "attempt_id, task_id, retry_series_id, effect_lineage_id, request_sha256, "
                "phase, quiescence, retry_admissible, recovery_generation, "
                "recovery_owner_id, recovery_owner_expires_at, record_json, created_at, "
                "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
            return
        cursor = self._connection.execute(
            "UPDATE cayu_local_execution_attempts SET task_id = ?, retry_series_id = ?, "
            "effect_lineage_id = ?, request_sha256 = ?, phase = ?, quiescence = ?, "
            "retry_admissible = ?, recovery_generation = ?, recovery_owner_id = ?, "
            "recovery_owner_expires_at = ?, record_json = ?, created_at = ?, updated_at = ? "
            "WHERE attempt_id = ? AND request_sha256 = ?",
            (*values[1:], values[0], values[4]),
        )
        if cursor.rowcount != 1:
            raise LocalExecutionAttemptConflict(
                "Local execution attempt changed during durable publication."
            )

    async def prepare_local_execution_attempt(
        self,
        authority: LocalExecutionAttemptAuthority,
    ) -> LocalExecutionAttemptRecord:
        authority = _copy_local_execution_attempt_authority(authority)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                existing = self._load_local_execution_attempt_unlocked(authority.attempt_id)
                task = (
                    None if existing is not None else self._require_task_unlocked(authority.task_id)
                )
                prior = self._latest_local_execution_attempt_unlocked(authority)
                evidence_now = self._clock()
                lease_now = self._ownership_clock()
                record = prepare_local_execution_attempt_record(
                    authority=authority,
                    task=task,
                    existing=existing,
                    prior=prior,
                    evidence_now=evidence_now,
                    lease_now=lease_now,
                )
                if existing is None:
                    self._store_local_execution_attempt_unlocked(record, insert=True)
                self._connection.commit()
                return record.model_copy(deep=True)
            except BaseException:
                self._connection.rollback()
                raise

    async def start_local_execution_attempt(
        self,
        start: LocalExecutionAttemptStart,
    ) -> LocalExecutionAttemptRecord:
        start = _copy_local_execution_attempt_start(start)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                record = self._load_local_execution_attempt_unlocked(start.attempt_id)
                if record is None:
                    raise LocalExecutionAttemptConflict(
                        "Local execution start has no prepared attempt."
                    )
                evidence_now = self._clock()
                lease_now = self._ownership_clock()
                if record.start is None:
                    require_local_execution_task_authority(
                        self._require_task_unlocked(record.authority.task_id),
                        record.authority,
                        now=lease_now,
                    )
                updated = advance_local_execution_attempt_start(
                    record,
                    start,
                    evidence_now=evidence_now,
                    lease_now=lease_now,
                )
                if updated != record:
                    self._store_local_execution_attempt_unlocked(updated, insert=False)
                self._connection.commit()
                return updated.model_copy(deep=True)
            except BaseException:
                self._connection.rollback()
                raise

    async def settle_local_execution_attempt(
        self,
        settlement: LocalExecutionAttemptSettlement,
    ) -> LocalExecutionAttemptRecord:
        settlement = _copy_authenticated_local_execution_attempt_settlement(settlement)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                record = self._load_local_execution_attempt_unlocked(settlement.attempt_id)
                if record is None:
                    raise LocalExecutionAttemptConflict(
                        "Local execution settlement has no prepared attempt."
                    )
                updated = settle_local_execution_attempt_record(
                    record,
                    settlement,
                    evidence_now=self._clock(),
                    lease_now=self._ownership_clock(),
                )
                if updated != record:
                    self._store_local_execution_attempt_unlocked(updated, insert=False)
                self._connection.commit()
                return updated.model_copy(deep=True)
            except BaseException:
                self._connection.rollback()
                raise

    async def load_local_execution_attempt(
        self,
        attempt_id: str,
    ) -> LocalExecutionAttemptRecord | None:
        attempt_id = require_clean_nonblank(attempt_id, "attempt_id")
        async with self._lock:
            record = self._load_local_execution_attempt_unlocked(attempt_id)
            return None if record is None else record.model_copy(deep=True)

    async def list_unsettled_local_execution_attempts(
        self,
        *,
        limit: int = 100,
        after: LocalExecutionAttemptListCursor | None = None,
    ) -> tuple[LocalExecutionAttemptRecord, ...]:
        limit = _validate_task_positive_int(limit, "limit")
        after = _copy_local_execution_attempt_list_cursor(after)
        async with self._lock:
            predicate = "(phase <> ? OR quiescence IN (?, ?))"
            parameters: list[Any] = [
                "terminal",
                "terminal_not_quiescent",
                "unavailable",
            ]
            if after is not None:
                predicate += " AND (created_at > ? OR (created_at = ? AND attempt_id > ?))"
                created_at = sqlite_support.format_datetime(after.created_at)
                parameters.extend(
                    (
                        created_at,
                        created_at,
                        after.attempt_id,
                    )
                )
            parameters.append(limit)
            rows = self._connection.execute(
                "SELECT attempt_id FROM cayu_local_execution_attempts WHERE "
                f"{predicate} "
                "ORDER BY created_at ASC, attempt_id ASC LIMIT ?",
                parameters,
            ).fetchall()
            records = [
                self._load_local_execution_attempt_unlocked(row["attempt_id"]) for row in rows
            ]
            return tuple(record.model_copy(deep=True) for record in records if record is not None)

    async def claim_local_execution_attempt_recovery(
        self,
        claim: LocalExecutionAttemptRecoveryClaim,
    ) -> LocalExecutionAttemptRecord:
        claim = _copy_local_execution_attempt_recovery_claim(claim)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                record = self._load_local_execution_attempt_unlocked(claim.attempt_id)
                if record is None:
                    raise LocalExecutionAttemptConflict(
                        "Local execution recovery attempt was not found."
                    )
                evidence_now = self._clock()
                lease_now = self._ownership_clock()
                task = self._require_task_unlocked(record.authority.task_id)
                require_local_execution_recovery_eligible(
                    task,
                    record,
                    now=lease_now,
                )
                updated = claim_local_execution_attempt_recovery_record(
                    record,
                    claim,
                    evidence_now=evidence_now,
                    lease_now=lease_now,
                )
                if updated != record:
                    self._store_local_execution_attempt_unlocked(updated, insert=False)
                self._connection.commit()
                return updated.model_copy(deep=True)
            except BaseException:
                self._connection.rollback()
                raise

    def _load_work_contract_unlocked(
        self,
        reference: WorkContractRef,
    ) -> WorkContract | None:
        row = self._connection.execute(
            "SELECT contract_id, version, fingerprint, contract_json "
            "FROM cayu_work_contracts WHERE contract_id = ? AND version = ?",
            (reference.contract_id, reference.version),
        ).fetchone()
        if row is None:
            return None
        contract = WorkContract.model_validate(json.loads(row["contract_json"]))
        if (
            contract.contract_id != row["contract_id"]
            or contract.version != row["version"]
            or contract.fingerprint != row["fingerprint"]
        ):
            raise WorkContractConflict(
                "Stored work-contract indexes conflict with canonical content."
            )
        return verified_work_support.require_contract_reference(contract, reference)

    def _load_work_attempt_unlocked(self, attempt_id: str) -> WorkAttempt | None:
        row = self._connection.execute(
            "SELECT attempt_id, task_id, ordinal, request_sha256, started_at, attempt_json "
            "FROM cayu_work_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        if row is None:
            return None
        attempt = WorkAttempt.model_validate(json.loads(row["attempt_json"]))
        if (
            attempt.attempt_id != row["attempt_id"]
            or attempt.task_id != row["task_id"]
            or attempt.ordinal != row["ordinal"]
            or attempt.request_sha256 != row["request_sha256"]
            or attempt.started_at != sqlite_support.parse_datetime(row["started_at"])
        ):
            raise WorkCompletionConflict(
                "Stored work-attempt indexes conflict with canonical content."
            )
        return attempt

    def _load_work_attempt_admission_unlocked(
        self,
        admission_id: str,
    ) -> WorkAttemptAdmission | None:
        row = self._connection.execute(
            "SELECT admission.admission_id, admission.attempt_id, admission.task_id, "
            "admission.session_id, admission.interaction_id, admission.state, "
            "admission.prepare_request_sha256, admission.current_claim_id, "
            "admission.current_generation, admission.lease_expires_at, "
            "admission.admission_json, claim.claim_id AS durable_claim_id, "
            "claim.admission_id AS durable_claim_admission_id, "
            "claim.generation AS durable_claim_generation, "
            "claim.request_sha256 AS durable_claim_request_sha256, "
            "claim.lease_expires_at AS durable_claim_lease_expires_at, "
            "claim.is_current AS durable_claim_is_current, "
            "claim.claim_json AS durable_claim_json "
            "FROM cayu_work_attempt_admissions AS admission "
            "LEFT JOIN cayu_work_attempt_execution_claims AS claim "
            "ON claim.admission_id = admission.admission_id AND claim.is_current = 1 "
            "WHERE admission.admission_id = ?",
            (admission_id,),
        ).fetchone()
        if row is None:
            return None
        admission = WorkAttemptAdmission.model_validate(json.loads(row["admission_json"]))
        if (
            admission.admission_id != row["admission_id"]
            or admission.attempt_id != row["attempt_id"]
            or admission.task_id != row["task_id"]
            or admission.session_id != row["session_id"]
            or admission.interaction_id != row["interaction_id"]
            or admission.state.value != row["state"]
            or admission.prepare_request_sha256 != row["prepare_request_sha256"]
            or admission.claim.claim_id != row["current_claim_id"]
            or admission.claim.generation != row["current_generation"]
            or admission.claim.lease_expires_at
            != sqlite_support.parse_datetime(row["lease_expires_at"])
        ):
            raise WorkAttemptAdmissionConflict(
                "Stored work-attempt admission indexes conflict with canonical content."
            )
        if row["durable_claim_json"] is None:
            raise WorkAttemptAdmissionConflict(
                "Stored admission has no durable current execution claim."
            )
        durable_claim = WorkAttemptExecutionClaim.model_validate(
            json.loads(row["durable_claim_json"])
        )
        if (
            durable_claim != admission.claim
            or durable_claim.claim_id != row["durable_claim_id"]
            or durable_claim.admission_id != row["durable_claim_admission_id"]
            or durable_claim.generation != row["durable_claim_generation"]
            or durable_claim.request_sha256 != row["durable_claim_request_sha256"]
            or durable_claim.lease_expires_at
            != sqlite_support.parse_datetime(row["durable_claim_lease_expires_at"])
            or row["durable_claim_is_current"] != 1
        ):
            raise WorkAttemptAdmissionConflict(
                "Stored execution-claim authority conflicts with its admission."
            )
        return admission

    def _load_work_attempt_admission_for_attempt_unlocked(
        self,
        attempt_id: str,
    ) -> WorkAttemptAdmission | None:
        row = self._connection.execute(
            "SELECT admission_id FROM cayu_work_attempt_admissions WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        return (
            None if row is None else self._load_work_attempt_admission_unlocked(row["admission_id"])
        )

    def _load_work_attempt_execution_claim_unlocked(
        self,
        claim_id: str,
    ) -> WorkAttemptExecutionClaim | None:
        row = self._connection.execute(
            "SELECT claim_id, admission_id, generation, request_sha256, "
            "lease_expires_at, claim_json FROM cayu_work_attempt_execution_claims "
            "WHERE claim_id = ?",
            (claim_id,),
        ).fetchone()
        if row is None:
            return None
        claim = WorkAttemptExecutionClaim.model_validate(json.loads(row["claim_json"]))
        if (
            claim.claim_id != row["claim_id"]
            or claim.admission_id != row["admission_id"]
            or claim.generation != row["generation"]
            or claim.request_sha256 != row["request_sha256"]
            or claim.lease_expires_at != sqlite_support.parse_datetime(row["lease_expires_at"])
        ):
            raise WorkAttemptAdmissionConflict(
                "Stored execution-claim indexes conflict with canonical content."
            )
        return claim

    def _insert_work_attempt_execution_claim_unlocked(
        self,
        claim: WorkAttemptExecutionClaim,
    ) -> None:
        self._connection.execute(
            "INSERT INTO cayu_work_attempt_execution_claims "
            "(claim_id, admission_id, generation, request_sha256, lease_expires_at, "
            "is_current, claim_json) VALUES (?, ?, ?, ?, ?, 1, ?)",
            (
                claim.claim_id,
                claim.admission_id,
                claim.generation,
                claim.request_sha256,
                sqlite_support.format_datetime(claim.lease_expires_at),
                sqlite_support.json_dumps(claim.model_dump(mode="json", warnings=False)),
            ),
        )

    def _update_work_attempt_admission_unlocked(
        self,
        admission: WorkAttemptAdmission,
    ) -> None:
        cursor = self._connection.execute(
            "UPDATE cayu_work_attempt_admissions SET state = ?, current_claim_id = ?, "
            "current_generation = ?, lease_expires_at = ?, admission_json = ? "
            "WHERE admission_id = ?",
            (
                admission.state.value,
                admission.claim.claim_id,
                admission.claim.generation,
                sqlite_support.format_datetime(admission.claim.lease_expires_at),
                sqlite_support.json_dumps(admission.model_dump(mode="json", warnings=False)),
                admission.admission_id,
            ),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"Work-attempt admission not found: {admission.admission_id}")

    @staticmethod
    def _ensure_live_work_attempt_admission_claim(
        admission: WorkAttemptAdmission,
        *,
        now: datetime,
    ) -> None:
        if admission.claim.lease_expires_at <= now:
            raise WorkAttemptExecutionClaimLost("Work-attempt execution claim has expired.")

    def _latest_work_attempt_id_unlocked(self, task_id: str) -> str | None:
        row = self._connection.execute(
            "SELECT attempt_id FROM cayu_work_attempts "
            "WHERE task_id = ? ORDER BY ordinal DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        return None if row is None else row["attempt_id"]

    def _load_completion_proposal_unlocked(
        self,
        proposal_id: str,
    ) -> CompletionProposal | None:
        row = self._connection.execute(
            "SELECT proposal_id, attempt_id, task_id, request_sha256, proposed_at, "
            "proposal_json FROM cayu_completion_proposals WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        if row is None:
            return None
        proposal = CompletionProposal.model_validate(json.loads(row["proposal_json"]))
        if (
            proposal.proposal_id != row["proposal_id"]
            or proposal.attempt_id != row["attempt_id"]
            or proposal.task_id != row["task_id"]
            or proposal.request_sha256 != row["request_sha256"]
            or proposal.proposed_at != sqlite_support.parse_datetime(row["proposed_at"])
        ):
            raise WorkCompletionConflict(
                "Stored completion-proposal indexes conflict with canonical content."
            )
        return proposal

    def _load_completion_claim_unlocked(
        self,
        proposal_id: str,
    ) -> CompletionVerificationClaim | None:
        row = self._connection.execute(
            "SELECT claim_id, proposal_id, attempt_number, verifier_profile_fingerprint, request_sha256, "
            "lease_expires_at, claim_json FROM cayu_completion_verification_claims "
            "WHERE proposal_id = ? AND is_current = 1",
            (proposal_id,),
        ).fetchone()
        return None if row is None else self._completion_claim_from_row(row)

    def _load_completion_verifier_profile_unlocked(
        self,
        proposal_id: str,
    ) -> CompletionVerifierProfileRecord | None:
        row = self._connection.execute(
            "SELECT proposal_id, task_id, attempt_id, profile_fingerprint, "
            "request_sha256, prepared_at, profile_json "
            "FROM cayu_completion_verifier_profiles WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        if row is None:
            return None
        profile = completion_verifier_profile_record_from_document(json.loads(row["profile_json"]))
        if (
            profile.proposal_id != row["proposal_id"]
            or profile.task_id != row["task_id"]
            or profile.attempt_id != row["attempt_id"]
            or profile.profile.fingerprint != row["profile_fingerprint"]
            or profile.request_sha256 != row["request_sha256"]
            or profile.prepared_at != sqlite_support.parse_datetime(row["prepared_at"])
        ):
            raise WorkCompletionConflict(
                "Stored completion-verifier profile indexes conflict with canonical content."
            )
        return profile

    def _load_completion_verifier_adoption_unlocked(
        self,
        *,
        task_id: str,
        idempotency_key: str,
    ) -> CompletionVerifierProfileRecord | None:
        rows = self._connection.execute(
            "SELECT proposal_id FROM cayu_completion_verifier_profiles "
            "WHERE task_id = ? "
            "AND json_extract(profile_json, '$.adoption.idempotency_key') = ? "
            "ORDER BY proposal_id LIMIT 2",
            (task_id, idempotency_key),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise WorkCompletionConflict(
                "Stored completion-verifier adoption idempotency authority is ambiguous."
            )
        profile = self._load_completion_verifier_profile_unlocked(rows[0]["proposal_id"])
        if (
            profile is None
            or profile.task_id != task_id
            or profile.adoption is None
            or profile.adoption.idempotency_key != idempotency_key
        ):
            raise WorkCompletionConflict(
                "Stored completion-verifier adoption idempotency authority is invalid."
            )
        return profile

    def _load_completion_claim_by_id_unlocked(
        self,
        claim_id: str,
    ) -> CompletionVerificationClaim | None:
        row = self._connection.execute(
            "SELECT claim_id, proposal_id, attempt_number, verifier_profile_fingerprint, request_sha256, "
            "lease_expires_at, claim_json FROM cayu_completion_verification_claims "
            "WHERE claim_id = ?",
            (claim_id,),
        ).fetchone()
        return None if row is None else self._completion_claim_from_row(row)

    @staticmethod
    def _completion_claim_from_row(row: sqlite3.Row) -> CompletionVerificationClaim:
        claim = CompletionVerificationClaim.model_validate(json.loads(row["claim_json"]))
        if (
            claim.claim_id != row["claim_id"]
            or claim.proposal_id != row["proposal_id"]
            or claim.attempt_number != row["attempt_number"]
            or claim.verifier_profile_fingerprint != row["verifier_profile_fingerprint"]
            or claim.request_sha256 != row["request_sha256"]
            or claim.lease_expires_at != sqlite_support.parse_datetime(row["lease_expires_at"])
        ):
            raise WorkCompletionConflict(
                "Stored verification-claim indexes conflict with canonical content."
            )
        return claim

    def _load_completion_decision_unlocked(
        self,
        decision_id: str,
    ) -> CompletionDecision | None:
        row = self._connection.execute(
            "SELECT decision_id, proposal_id, task_id, attempt_id, claim_id, verifier_profile_fingerprint, verdict, "
            "gap_fingerprint, request_sha256, decided_at, decision_json "
            "FROM cayu_completion_decisions WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()
        return None if row is None else self._completion_decision_from_row(row)

    def _load_completion_decision_for_proposal_unlocked(
        self,
        proposal_id: str,
    ) -> CompletionDecision | None:
        row = self._connection.execute(
            "SELECT decision_id, proposal_id, task_id, attempt_id, claim_id, verifier_profile_fingerprint, verdict, "
            "gap_fingerprint, request_sha256, decided_at, decision_json "
            "FROM cayu_completion_decisions WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        return None if row is None else self._completion_decision_from_row(row)

    @staticmethod
    def _completion_decision_from_row(row: sqlite3.Row) -> CompletionDecision:
        decision = CompletionDecision.model_validate(json.loads(row["decision_json"]))
        if (
            decision.decision_id != row["decision_id"]
            or decision.proposal_id != row["proposal_id"]
            or decision.task_id != row["task_id"]
            or decision.attempt_id != row["attempt_id"]
            or decision.claim_id != row["claim_id"]
            or decision.verifier_profile_fingerprint != row["verifier_profile_fingerprint"]
            or decision.verdict.value != row["verdict"]
            or decision.gap_fingerprint != row["gap_fingerprint"]
            or decision.request_sha256 != row["request_sha256"]
            or decision.decided_at != sqlite_support.parse_datetime(row["decided_at"])
        ):
            raise WorkCompletionConflict(
                "Stored completion-decision indexes conflict with canonical content."
            )
        return decision

    def _load_decision_application_receipt_unlocked(
        self,
        task_id: str,
        idempotency_key: str,
    ) -> CompletionDecisionApplicationReceipt | None:
        row = self._connection.execute(
            "SELECT task_id, idempotency_key, decision_id, request_sha256, applied_at, "
            "receipt_json FROM cayu_completion_decision_application_receipts "
            "WHERE task_id = ? AND idempotency_key = ?",
            (task_id, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        receipt = CompletionDecisionApplicationReceipt.model_validate(
            json.loads(row["receipt_json"])
        )
        if (
            receipt.task_id != row["task_id"]
            or receipt.idempotency_key != row["idempotency_key"]
            or receipt.decision_id != row["decision_id"]
            or receipt.request_sha256 != row["request_sha256"]
            or receipt.applied_at != sqlite_support.parse_datetime(row["applied_at"])
        ):
            raise WorkCompletionConflict(
                "Stored decision-application receipt indexes conflict with canonical content."
            )
        return receipt

    def _ensure_session_execution_authority_unlocked(
        self,
        session_id: str,
        authority_kind: Literal["ordinary", "contracted"],
    ) -> None:
        now = self._ownership_clock()
        self._connection.execute(
            "INSERT OR IGNORE INTO cayu_task_session_execution_authority "
            "(session_id, authority_kind, committed_at) VALUES (?, ?, ?)",
            (session_id, authority_kind, sqlite_support.format_datetime(now)),
        )
        row = self._connection.execute(
            "SELECT authority_kind FROM cayu_task_session_execution_authority WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise TaskTopologyInconsistent("Session execution authority was not persisted.")
        if row["authority_kind"] != authority_kind:
            if authority_kind == "ordinary":
                raise TaskCompletionDecisionRequired(
                    "Contracted tasks require the verifier-aware execution entrance."
                )
            raise WorkCompletionConflict(
                "Work-contract attachment conflicts with prior ordinary session execution."
            )

    def _require_task_contract_unlocked(
        self,
        task: Task,
        reference: WorkContractRef,
    ) -> WorkContract:
        return verified_work_support.require_task_contract(
            task,
            reference,
            self._load_work_contract_unlocked(reference),
        )

    def _update_task_snapshot_unlocked(self, task: Task) -> None:
        if task.work_contract is not None:
            task = copy_task(task)
        cursor = self._connection.execute(
            """
            UPDATE cayu_tasks
            SET status = ?, session_id = ?, session_instance_id = ?,
                worker_id = ?, lease_expires_at = ?,
                status_reason = ?, status_payload_json = ?, result_json = ?, error_json = ?,
                updated_at = ?, started_at = ?, completed_at = ?, retry_series_json = ?,
                work_contract_json = ?
            WHERE id = ?
            """,
            (
                str(task.status),
                task.session_id,
                task.session_instance_id,
                task.worker_id,
                sqlite_support.format_optional_datetime(task.lease_expires_at),
                task.status_reason,
                None
                if task.status_payload is None
                else sqlite_support.json_dumps(task.status_payload),
                None if task.result is None else sqlite_support.json_dumps(task.result),
                None if task.error is None else sqlite_support.json_dumps(task.error),
                sqlite_support.format_datetime(task.updated_at),
                sqlite_support.format_optional_datetime(task.started_at),
                sqlite_support.format_optional_datetime(task.completed_at),
                None
                if task.retry_series is None
                else sqlite_support.json_dumps(task.retry_series.model_dump(mode="json")),
                None
                if task.work_contract is None
                else sqlite_support.json_dumps(
                    task.work_contract.model_dump(mode="json", warnings=False)
                ),
                task.id,
            ),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"Task not found: {task.id}")

    async def publish_work_contract(self, contract: WorkContract) -> WorkContract:
        contract = copy_work_contract(contract)
        async with self._lock:
            with self._verified_transaction_unlocked():
                existing = self._load_work_contract_unlocked(contract.reference())
                if existing is not None:
                    if existing != contract:
                        raise WorkContractConflict(
                            "Work-contract identity is already bound to different content."
                        )
                    return copy_work_contract(existing)
                if contract.supersedes is not None:
                    predecessor = self._load_work_contract_unlocked(contract.supersedes)
                    verified_work_support.require_contract_reference(
                        predecessor,
                        contract.supersedes,
                    )
                self._connection.execute(
                    "INSERT INTO cayu_work_contracts "
                    "(contract_id, version, fingerprint, contract_json) VALUES (?, ?, ?, ?)",
                    (
                        contract.contract_id,
                        contract.version,
                        contract.fingerprint,
                        sqlite_support.json_dumps(contract.model_dump(mode="json", warnings=False)),
                    ),
                )
                return copy_work_contract(contract)

    async def load_work_contract(self, reference: WorkContractRef) -> WorkContract | None:
        copied = copy_work_contract_ref(reference)
        if copied is None:
            raise TypeError("reference must be a WorkContractRef.")
        async with self._lock:
            contract = self._load_work_contract_unlocked(copied)
            return None if contract is None else copy_work_contract(contract)

    async def load_active_work_contract_task_for_session(
        self,
        session_id: str,
    ) -> Task | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            authority = self._connection.execute(
                "SELECT authority_kind FROM cayu_task_session_execution_authority "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if authority is None or authority["authority_kind"] == "ordinary":
                return None
            row = self._connection.execute(
                "SELECT * FROM cayu_tasks WHERE session_id = ? "
                "AND work_contract_json IS NOT NULL ORDER BY created_at, id LIMIT 1",
                (session_id,),
            ).fetchone()
            if row is None:
                raise TaskTopologyInconsistent(
                    "Contracted session authority has no matching durable task."
                )
            return sqlite_support.task_from_row(row)

    async def admit_ordinary_session_execution(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            with self._verified_transaction_unlocked():
                self._ensure_session_execution_authority_unlocked(session_id, "ordinary")

    async def hold_claimed_work_contract_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        lease_expires_at: datetime | None = None,
        contract: WorkContractRef,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        expected_lease = (
            None
            if lease_expires_at is None
            else normalize_utc_datetime(lease_expires_at, "lease_expires_at")
        )
        copied_contract = copy_work_contract_ref(contract)
        if copied_contract is None:
            raise TypeError("contract must be a WorkContractRef.")
        async with self._lock:
            with self._verified_transaction_unlocked():
                task = self._require_task_unlocked(task_id)
                now = self._ownership_clock()
                if expected_lease is None:
                    raise TaskClaimLost("Contracted task parking requires its exact worker lease.")
                _ensure_exact_owned_active_task_lease(
                    task,
                    worker_id,
                    expected_lease,
                    now=now,
                )
                if task.status is not TaskStatus.CLAIMED or task.session_id is not None:
                    raise TaskClaimLost(
                        "Only the current worker may park its unattached claimed task."
                    )
                self._require_task_contract_unlocked(task, copied_contract)
                updated = task.model_copy(
                    update={
                        "status": TaskStatus.NEEDS_ATTENTION,
                        "status_reason": "verified_work_contract_runner_required",
                        "status_payload": {
                            "contract_id": copied_contract.contract_id,
                            "contract_version": copied_contract.version,
                        },
                        "worker_id": None,
                        "lease_expires_at": None,
                        "updated_at": now,
                    }
                )
                self._update_task_snapshot_unlocked(updated)
                return updated.model_copy(deep=True)

    def _work_attempt_continuation_context_unlocked(
        self,
        task: Task,
        contract: WorkContract,
    ) -> WorkAttemptContinuationContext | None:
        prior_attempt_id = self._latest_work_attempt_id_unlocked(task.id)
        if prior_attempt_id is None:
            return None
        prior_admission = self._load_work_attempt_admission_for_attempt_unlocked(prior_attempt_id)
        if (
            prior_admission is None
            or prior_admission.state is not WorkAttemptAdmissionState.RELEASED
            or prior_admission.task_id != task.id
            or prior_admission.session_id != task.session_id
        ):
            raise WorkAttemptAdmissionConflict(
                "The latest work attempt has no exact released admission authority."
            )
        row = self._connection.execute(
            "SELECT proposal.proposal_id, decision.decision_id, "
            "receipt.idempotency_key FROM cayu_completion_proposals AS proposal "
            "LEFT JOIN cayu_completion_decisions AS decision "
            "ON decision.proposal_id = proposal.proposal_id "
            "LEFT JOIN cayu_completion_decision_application_receipts AS receipt "
            "ON receipt.decision_id = decision.decision_id "
            "WHERE proposal.attempt_id = ?",
            (prior_attempt_id,),
        ).fetchone()
        if row is None:
            raise WorkAttemptAdmissionConflict(
                "The latest work attempt has no durable completion proposal."
            )
        if row["decision_id"] is None:
            raise WorkAttemptAdmissionConflict(
                "The latest work attempt has no durable completion decision."
            )
        if row["idempotency_key"] is None:
            raise WorkAttemptAdmissionConflict(
                "The latest completion decision has not been applied durably."
            )
        decision = self._load_completion_decision_unlocked(row["decision_id"])
        if decision is None:
            raise WorkAttemptAdmissionConflict(
                "The latest work-attempt decision index is incomplete."
            )
        receipt = self._load_decision_application_receipt_unlocked(
            task.id,
            row["idempotency_key"],
        )
        if receipt is None or receipt.decision_id != decision.decision_id or receipt.task != task:
            raise WorkAttemptAdmissionConflict(
                "The latest decision application conflicts with continuation authority."
            )
        if (
            decision.verdict is not CompletionVerdict.REJECTED
            or contract.continuation_policy.rejection_action
            is not CompletionRejectionAction.CONTINUE
        ):
            raise WorkAttemptAdmissionConflict(
                "The latest completion decision does not authorize continuation."
            )
        return WorkAttemptContinuationContext(
            prior_admission_id=prior_admission.admission_id,
            prior_attempt_id=prior_attempt_id,
            proposal_id=row["proposal_id"],
            decision=decision,
            application_idempotency_key=row["idempotency_key"],
            gap_fingerprint=decision.gap_fingerprint,
        )

    async def prepare_work_attempt_admission(
        self,
        request: WorkAttemptAdmissionPrepare,
    ) -> WorkAttemptAdmission:
        request = copy_work_attempt_admission_prepare(request)
        if request.generation != 1:
            raise WorkAttemptAdmissionConflict(
                "A new work-attempt admission must start at execution generation 1."
            )
        request_sha256 = work_attempt_admission_prepare_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                existing = self._load_work_attempt_admission_unlocked(request.admission_id)
                if existing is not None:
                    if not work_attempt_admission_prepare_matches_sha256(
                        request,
                        existing.prepare_request_sha256,
                    ):
                        raise WorkAttemptAdmissionConflict(
                            "Work-attempt admission identity is bound to another request."
                        )
                    return existing.model_copy(deep=True)
                if (
                    self._load_work_attempt_admission_for_attempt_unlocked(request.attempt_id)
                    is not None
                ):
                    raise WorkAttemptAdmissionConflict(
                        "Work-attempt identity is already bound to another admission."
                    )
                if self._load_work_attempt_unlocked(request.attempt_id) is not None:
                    raise WorkAttemptAdmissionConflict(
                        "Work-attempt identity already exists without this admission."
                    )
                occupied_claim = self._connection.execute(
                    "SELECT admission_id FROM cayu_work_attempt_execution_claims "
                    "WHERE claim_id = ?",
                    (request.claim_id,),
                ).fetchone()
                if occupied_claim is not None:
                    raise WorkAttemptAdmissionConflict(
                        "Execution-claim identity is bound to another admission."
                    )
                occupied_interaction = self._connection.execute(
                    "SELECT admission_id FROM cayu_work_attempt_admissions "
                    "WHERE session_id = ? AND interaction_id = ?",
                    (request.session_id, request.interaction_id),
                ).fetchone()
                if occupied_interaction is not None:
                    raise WorkAttemptAdmissionConflict(
                        "Session interaction is already bound to another admission."
                    )
                occupied_session = self._connection.execute(
                    "SELECT admission_id FROM cayu_work_attempt_admissions "
                    "WHERE session_id = ? AND state != 'released' LIMIT 1",
                    (request.session_id,),
                ).fetchone()
                if occupied_session is not None:
                    raise WorkAttemptAdmissionConflict(
                        "Session already has an unreleased work-attempt admission."
                    )

                task = self._require_task_unlocked(request.task_id)
                unreleased_admission_row = self._connection.execute(
                    "SELECT admission_id FROM cayu_work_attempt_admissions "
                    "WHERE task_id = ? AND state != 'released' LIMIT 1",
                    (request.task_id,),
                ).fetchone()
                if unreleased_admission_row is not None:
                    raise WorkAttemptAdmissionConflict(
                        "Task already has an unreleased work-attempt admission."
                    )
                contract = self._require_task_contract_unlocked(task, request.contract)
                lease_now = self._ownership_clock()
                availability_now = self._clock()
                continuation = self._work_attempt_continuation_context_unlocked(
                    task,
                    contract,
                )
                if continuation is None:
                    if request.kind != "initial":
                        raise WorkAttemptAdmissionConflict(
                            "Continuation admission requires a rejected/continue decision."
                        )
                    if task.status not in {TaskStatus.PENDING, TaskStatus.CLAIMED}:
                        raise WorkAttemptAdmissionConflict(
                            "Initial admission requires a pending or claimed contracted task."
                        )
                    if task.available_at is not None and task.available_at > availability_now:
                        raise WorkAttemptAdmissionConflict(
                            "Contracted task is not yet available for admission."
                        )
                    if task.status is TaskStatus.CLAIMED:
                        if request.task_lease_expires_at is None:
                            raise TaskClaimLost(
                                "Work-attempt admission requires the claimed task's exact lease."
                            )
                        _ensure_exact_owned_active_task_lease(
                            task,
                            request.worker_id,
                            request.task_lease_expires_at,
                            now=lease_now,
                        )
                    elif task.worker_id is not None or task.lease_expires_at is not None:
                        raise WorkAttemptAdmissionConflict(
                            "Pending contracted task has conflicting worker ownership."
                        )
                    elif request.task_lease_expires_at is not None:
                        raise WorkAttemptAdmissionConflict(
                            "Pending work-attempt admission cannot consume worker lease authority."
                        )
                    if task.session_id not in {None, request.session_id}:
                        raise WorkAttemptAdmissionConflict(
                            "Initial admission conflicts with the task's session."
                        )
                else:
                    if request.task_lease_expires_at is not None:
                        raise WorkAttemptAdmissionConflict(
                            "Continuation admission cannot consume prior worker lease authority."
                        )
                    if request.kind != "continuation":
                        raise WorkAttemptAdmissionConflict(
                            "Initial admission cannot consume continuation authority."
                        )
                    if continuation.prior_admission_id != request.predecessor_admission_id:
                        raise WorkAttemptAdmissionConflict(
                            "Continuation admission selected another predecessor admission."
                        )
                    if (
                        task.status is not TaskStatus.RUNNING
                        or task.session_id != request.session_id
                        or task.worker_id is not None
                        or task.lease_expires_at is not None
                    ):
                        raise WorkAttemptAdmissionConflict(
                            "Continuation admission requires an unowned running task on its exact session."
                        )
                    if (
                        contract.continuation_policy.rejection_action
                        is not CompletionRejectionAction.CONTINUE
                    ):
                        raise WorkAttemptAdmissionConflict(
                            "The frozen contract does not authorize continuation."
                        )

                self._ensure_session_execution_authority_unlocked(
                    request.session_id,
                    "contracted",
                )
                _task_invocation_for_attachment(
                    task.invocation,
                    session_id=request.session_id,
                    session_binding=request.session_invocation,
                )
                session_instance_id = _task_session_instance_for_attachment(
                    stored_session_instance_id=task.session_instance_id,
                    session_id=request.session_id,
                    session_binding=request.session_invocation,
                )
                claim_request = WorkAttemptExecutionClaimRequest(
                    admission_id=request.admission_id,
                    claim_id=request.claim_id,
                    worker_id=request.worker_id,
                    execution_owner_id=request.execution_owner_id,
                    generation=request.generation,
                    lease_seconds=request.lease_seconds,
                )
                claim = WorkAttemptExecutionClaim(
                    admission_id=request.admission_id,
                    claim_id=request.claim_id,
                    worker_id=request.worker_id,
                    execution_owner_id=request.execution_owner_id,
                    generation=request.generation,
                    request_sha256=work_attempt_execution_claim_request_sha256(claim_request),
                    claimed_at=lease_now,
                    lease_expires_at=lease_now + timedelta(seconds=request.lease_seconds),
                )
                admission = WorkAttemptAdmission(
                    admission_id=request.admission_id,
                    prepare_request_sha256=request_sha256,
                    state=WorkAttemptAdmissionState.PREPARING,
                    attempt_id=request.attempt_id,
                    task_id=request.task_id,
                    session_id=request.session_id,
                    interaction_id=request.interaction_id,
                    kind=request.kind,
                    source_request_sha256=request.source_request_sha256,
                    contract=request.contract,
                    session_invocation=request.session_invocation,
                    source_execution_profile_fingerprint=(
                        request.source_execution_profile_fingerprint
                    ),
                    claim=claim,
                    continuation=continuation,
                    prepared_at=lease_now,
                )
                updated_task = task.model_copy(
                    update={
                        "status": TaskStatus.RUNNING,
                        "session_id": request.session_id,
                        "session_instance_id": session_instance_id,
                        "worker_id": request.worker_id,
                        "lease_expires_at": claim.lease_expires_at,
                        "started_at": task.started_at or lease_now,
                        "updated_at": lease_now,
                    }
                )
                self._update_task_snapshot_unlocked(updated_task)
                self._connection.execute(
                    "INSERT INTO cayu_work_attempt_admissions "
                    "(admission_id, attempt_id, task_id, session_id, interaction_id, state, "
                    "prepare_request_sha256, current_claim_id, current_generation, "
                    "lease_expires_at, admission_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        admission.admission_id,
                        admission.attempt_id,
                        admission.task_id,
                        admission.session_id,
                        admission.interaction_id,
                        admission.state.value,
                        admission.prepare_request_sha256,
                        admission.claim.claim_id,
                        admission.claim.generation,
                        sqlite_support.format_datetime(admission.claim.lease_expires_at),
                        sqlite_support.json_dumps(
                            admission.model_dump(mode="json", warnings=False)
                        ),
                    ),
                )
                self._insert_work_attempt_execution_claim_unlocked(claim)
                return admission.model_copy(deep=True)

    async def activate_work_attempt_admission(
        self,
        request: WorkAttemptAdmissionActivate,
    ) -> WorkAttemptAdmission:
        request = copy_work_attempt_admission_activate(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                admission = self._load_work_attempt_admission_unlocked(request.admission_id)
                if admission is None:
                    raise KeyError(f"Work-attempt admission not found: {request.admission_id}")
                if admission.prepare_request_sha256 != request.prepare_request_sha256:
                    raise WorkAttemptAdmissionConflict(
                        "Admission activation conflicts with its prepared request."
                    )
                if admission.claim.claim_id != request.claim_id:
                    raise WorkAttemptExecutionClaimLost(
                        "Admission activation no longer owns the prepared execution claim."
                    )
                task = self._require_task_unlocked(admission.task_id)
                if (
                    task.session_id != admission.session_id
                    or task.session_instance_id != admission.session_invocation.session_instance_id
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Prepared admission conflicts with exact task-session authority."
                    )
                if admission.state is WorkAttemptAdmissionState.ACTIVE:
                    if admission.session_evidence_sha256 != request.session_evidence_sha256:
                        raise WorkAttemptAdmissionConflict(
                            "Admission activation conflicts with durable session evidence."
                        )
                    return admission.model_copy(deep=True)
                if admission.state is not WorkAttemptAdmissionState.PREPARING:
                    raise WorkAttemptAdmissionConflict(
                        "Only a prepared admission can publish its work attempt."
                    )
                lease_now = self._ownership_clock()
                self._ensure_live_work_attempt_admission_claim(admission, now=lease_now)
                if (
                    task.status is not TaskStatus.RUNNING
                    or task.worker_id != admission.claim.worker_id
                    or task.lease_expires_at != admission.claim.lease_expires_at
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Prepared admission conflicts with current task ownership."
                    )
                contract = self._require_task_contract_unlocked(task, admission.contract)
                prior_id = self._latest_work_attempt_id_unlocked(task.id)
                prior = None if prior_id is None else self._load_work_attempt_unlocked(prior_id)
                ordinal = 1 if prior is None else prior.ordinal + 1
                if ordinal > contract.continuation_policy.max_attempts:
                    raise WorkAttemptAdmissionConflict(
                        "Work-contract attempt limit forbids activation."
                    )
                attempt_request = WorkAttemptCreate(
                    attempt_id=admission.attempt_id,
                    task_id=admission.task_id,
                    session_id=admission.session_id,
                    contract=admission.contract,
                    execution_profile_fingerprint=(admission.source_execution_profile_fingerprint),
                    worker_id=admission.claim.worker_id,
                )
                attempt = WorkAttempt(
                    **attempt_request.model_dump(mode="python"),
                    ordinal=ordinal,
                    request_sha256=work_attempt_request_sha256(attempt_request),
                    started_at=(evidence_now := self._clock()),
                )
                activated = WorkAttemptAdmission.model_validate(
                    admission.model_copy(
                        update={
                            "state": WorkAttemptAdmissionState.ACTIVE,
                            "attempt": attempt,
                            "session_evidence_sha256": request.session_evidence_sha256,
                            "activated_at": evidence_now,
                        }
                    ).model_dump(mode="python", warnings=False)
                )
                self._connection.execute(
                    "INSERT INTO cayu_work_attempts "
                    "(attempt_id, task_id, ordinal, request_sha256, started_at, attempt_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        attempt.attempt_id,
                        attempt.task_id,
                        attempt.ordinal,
                        attempt.request_sha256,
                        sqlite_support.format_datetime(attempt.started_at),
                        sqlite_support.json_dumps(attempt.model_dump(mode="json", warnings=False)),
                    ),
                )
                self._update_work_attempt_admission_unlocked(activated)
                return activated.model_copy(deep=True)

    async def load_work_attempt_admission(
        self,
        admission_id: str,
    ) -> WorkAttemptAdmission | None:
        admission_id = require_clean_nonblank(admission_id, "admission_id")
        async with self._lock:
            admission = self._load_work_attempt_admission_unlocked(admission_id)
            return None if admission is None else admission.model_copy(deep=True)

    async def load_work_attempt_execution_claim(
        self,
        claim_id: str,
    ) -> WorkAttemptExecutionClaim | None:
        claim_id = require_clean_nonblank(claim_id, "claim_id")
        async with self._lock:
            claim = self._load_work_attempt_execution_claim_unlocked(claim_id)
            return None if claim is None else claim.model_copy(deep=True)

    async def renew_work_attempt_execution_claim(
        self,
        request: WorkAttemptExecutionClaimRequest,
    ) -> WorkAttemptAdmission:
        request = copy_work_attempt_execution_claim_request(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                admission = self._load_work_attempt_admission_unlocked(request.admission_id)
                if admission is None:
                    raise KeyError(f"Work-attempt admission not found: {request.admission_id}")
                claim = admission.claim
                if (
                    admission.state is not WorkAttemptAdmissionState.ACTIVE
                    or claim.claim_id != request.claim_id
                    or claim.worker_id != request.worker_id
                    or claim.execution_owner_id != request.execution_owner_id
                    or claim.generation != request.generation
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Execution-claim renewal conflicts with current authority."
                    )
                now = self._ownership_clock()
                self._ensure_live_work_attempt_admission_claim(admission, now=now)
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_completion_proposals WHERE attempt_id = ?",
                        (admission.attempt_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Completion proposal has already closed execution authority."
                    )
                task = self._require_task_unlocked(admission.task_id)
                if (
                    task.status is not TaskStatus.RUNNING
                    or task.session_id != admission.session_id
                    or task.session_instance_id != admission.session_invocation.session_instance_id
                    or task.worker_id != claim.worker_id
                    or task.lease_expires_at != claim.lease_expires_at
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Execution-claim renewal lost exact task ownership."
                    )
                renewed_claim = WorkAttemptExecutionClaim.model_validate(
                    claim.model_copy(
                        update={
                            "lease_expires_at": max(
                                claim.lease_expires_at,
                                now + timedelta(seconds=request.lease_seconds),
                            )
                        }
                    ).model_dump(mode="python", warnings=False)
                )
                renewed = WorkAttemptAdmission.model_validate(
                    admission.model_copy(update={"claim": renewed_claim}).model_dump(
                        mode="python", warnings=False
                    )
                )
                self._connection.execute(
                    "UPDATE cayu_work_attempt_execution_claims "
                    "SET lease_expires_at = ?, claim_json = ? "
                    "WHERE claim_id = ? AND is_current = 1",
                    (
                        sqlite_support.format_datetime(renewed_claim.lease_expires_at),
                        sqlite_support.json_dumps(
                            renewed_claim.model_dump(mode="json", warnings=False)
                        ),
                        renewed_claim.claim_id,
                    ),
                )
                self._update_work_attempt_admission_unlocked(renewed)
                self._update_task_snapshot_unlocked(
                    task.model_copy(
                        update={
                            "lease_expires_at": renewed_claim.lease_expires_at,
                            "updated_at": now,
                        }
                    )
                )
                return renewed.model_copy(deep=True)

    async def claim_work_attempt_recovery(
        self,
        request: WorkAttemptExecutionClaimRequest,
    ) -> WorkAttemptAdmission:
        request = copy_work_attempt_execution_claim_request(request)
        request_sha256 = work_attempt_execution_claim_request_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                admission = self._load_work_attempt_admission_unlocked(request.admission_id)
                if admission is None:
                    raise KeyError(f"Work-attempt admission not found: {request.admission_id}")
                preparing = admission.state is WorkAttemptAdmissionState.PREPARING
                if not preparing and (
                    admission.attempt is None
                    or admission.state
                    not in {
                        WorkAttemptAdmissionState.ACTIVE,
                        WorkAttemptAdmissionState.RECOVERING,
                    }
                ):
                    raise WorkAttemptAdmissionConflict(
                        "Only a prepared or published admission can enter recovery."
                    )
                current = admission.claim
                now = self._ownership_clock()
                exact_current_request = (
                    current.claim_id == request.claim_id
                    and current.worker_id == request.worker_id
                    and current.execution_owner_id == request.execution_owner_id
                    and current.generation == request.generation
                    and current.request_sha256 == request_sha256
                )
                task = self._require_task_unlocked(admission.task_id)
                if (
                    task.status is not TaskStatus.RUNNING
                    or task.session_id != admission.session_id
                    or task.session_instance_id != admission.session_invocation.session_instance_id
                    or task.worker_id != current.worker_id
                    or task.lease_expires_at != current.lease_expires_at
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Recovery conflicts with current task ownership."
                    )
                if preparing and exact_current_request:
                    if current.lease_expires_at <= now:
                        raise WorkAttemptExecutionClaimLost(
                            "The prepared execution claim expired and must be replaced."
                        )
                    return admission.model_copy(deep=True)
                if admission.state is WorkAttemptAdmissionState.ACTIVE and exact_current_request:
                    if current.lease_expires_at <= now:
                        raise WorkAttemptExecutionClaimLost(
                            "The active execution claim expired and must be replaced."
                        )
                    return admission.model_copy(deep=True)
                if admission.state is WorkAttemptAdmissionState.RECOVERING:
                    if exact_current_request:
                        if current.lease_expires_at <= now:
                            raise WorkAttemptExecutionClaimLost(
                                "The recovery claim expired and must be replaced."
                            )
                        return admission.model_copy(deep=True)
                    if current.lease_expires_at > now:
                        raise WorkAttemptExecutionClaimLost(
                            "Another execution generation already owns live recovery."
                        )
                if current.lease_expires_at > now:
                    raise WorkAttemptExecutionClaimLost(
                        "The prior execution generation still owns a live lease."
                    )
                if request.generation != current.generation + 1:
                    raise WorkAttemptAdmissionConflict(
                        "Recovery must advance the execution generation exactly once."
                    )
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_completion_proposals WHERE attempt_id = ?",
                        (admission.attempt_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkAttemptAdmissionConflict(
                        "A proposed attempt cannot acquire replacement execution authority."
                    )
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_work_attempt_execution_claims WHERE claim_id = ?",
                        (request.claim_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkAttemptAdmissionConflict("Execution-claim identity is already bound.")
                replacement = WorkAttemptExecutionClaim(
                    admission_id=request.admission_id,
                    claim_id=request.claim_id,
                    worker_id=request.worker_id,
                    execution_owner_id=request.execution_owner_id,
                    generation=request.generation,
                    request_sha256=request_sha256,
                    claimed_at=now,
                    lease_expires_at=now + timedelta(seconds=request.lease_seconds),
                )
                recovering = WorkAttemptAdmission.model_validate(
                    admission.model_copy(
                        update={
                            "state": (
                                WorkAttemptAdmissionState.PREPARING
                                if preparing
                                else WorkAttemptAdmissionState.RECOVERING
                            ),
                            "claim": replacement,
                            "recovery_evidence_sha256": None,
                        }
                    ).model_dump(mode="python", warnings=False)
                )
                retired = self._connection.execute(
                    "UPDATE cayu_work_attempt_execution_claims SET is_current = 0 "
                    "WHERE admission_id = ? AND is_current = 1",
                    (admission.admission_id,),
                )
                if retired.rowcount != 1:
                    raise WorkAttemptAdmissionConflict(
                        "Recovery could not retire the prior execution claim."
                    )
                self._insert_work_attempt_execution_claim_unlocked(replacement)
                self._update_work_attempt_admission_unlocked(recovering)
                self._update_task_snapshot_unlocked(
                    task.model_copy(
                        update={
                            "worker_id": request.worker_id,
                            "lease_expires_at": replacement.lease_expires_at,
                            "updated_at": now,
                        }
                    )
                )
                return recovering.model_copy(deep=True)

    async def activate_work_attempt_recovery(
        self,
        request: WorkAttemptRecoveryActivate,
    ) -> WorkAttemptAdmission:
        request = copy_work_attempt_recovery_activate(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                admission = self._load_work_attempt_admission_unlocked(request.admission_id)
                if admission is None:
                    raise KeyError(f"Work-attempt admission not found: {request.admission_id}")
                if admission.state is WorkAttemptAdmissionState.ACTIVE:
                    if not (
                        admission.claim.claim_id == request.claim_id
                        and admission.claim.generation == request.generation
                        and admission.recovery_evidence_sha256 == request.recovery_evidence_sha256
                    ):
                        raise WorkAttemptAdmissionConflict(
                            "Recovery activation conflicts with current active authority."
                        )
                elif (
                    admission.state is not WorkAttemptAdmissionState.RECOVERING
                    or admission.claim.claim_id != request.claim_id
                    or admission.claim.generation != request.generation
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Recovery activation no longer owns the replacement claim."
                    )
                task = self._require_task_unlocked(admission.task_id)
                if (
                    task.session_id != admission.session_id
                    or task.session_instance_id != admission.session_invocation.session_instance_id
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Recovery activation conflicts with exact task-session authority."
                    )
                if admission.state is WorkAttemptAdmissionState.ACTIVE:
                    return admission.model_copy(deep=True)
                now = self._ownership_clock()
                self._ensure_live_work_attempt_admission_claim(admission, now=now)
                if (
                    task.status is not TaskStatus.RUNNING
                    or task.worker_id != admission.claim.worker_id
                    or task.lease_expires_at != admission.claim.lease_expires_at
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Recovery activation conflicts with current task ownership."
                    )
                if admission.attempt is None:
                    raise WorkAttemptAdmissionConflict(
                        "Recovery activation requires a published work attempt."
                    )
                contract = self._load_work_contract_unlocked(admission.contract)
                verified_work_support.require_attempt_state_current(
                    task,
                    admission.attempt,
                    latest_attempt_id=self._latest_work_attempt_id_unlocked(task.id),
                    contract=contract,
                )
                active = WorkAttemptAdmission.model_validate(
                    admission.model_copy(
                        update={
                            "state": WorkAttemptAdmissionState.ACTIVE,
                            "recovery_evidence_sha256": request.recovery_evidence_sha256,
                        }
                    ).model_dump(mode="python", warnings=False)
                )
                self._update_work_attempt_admission_unlocked(active)
                return active.model_copy(deep=True)

    async def begin_work_attempt(self, request: WorkAttemptCreate) -> WorkAttempt:
        request = copy_work_attempt_create(request)
        request_sha256 = work_attempt_request_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                if (
                    self._load_work_attempt_admission_for_attempt_unlocked(request.attempt_id)
                    is not None
                ):
                    raise WorkAttemptAdmissionConflict(
                        "Admitted work attempts are published only by admission activation."
                    )
                existing = self._load_work_attempt_unlocked(request.attempt_id)
                if existing is not None:
                    if existing.request_sha256 != request_sha256:
                        raise WorkCompletionConflict(
                            "Work-attempt identity is already bound to another request."
                        )
                    return existing.model_copy(deep=True)
                task = self._require_task_unlocked(request.task_id)
                governed_admission = self._connection.execute(
                    "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                    (task.id,),
                ).fetchone()
                if governed_admission is not None:
                    raise WorkAttemptAdmissionConflict(
                        "Task is permanently governed by runtime-owned work-attempt admission."
                    )
                contract = self._require_task_contract_unlocked(task, request.contract)
                if task.status is not TaskStatus.RUNNING:
                    raise ValueError("Work attempts require a running contracted task.")
                if task.session_id != request.session_id:
                    raise WorkCompletionConflict(
                        "Work attempt is bound to a different task session."
                    )
                verified_work_support.require_attempt_worker(
                    task,
                    request.worker_id,
                    now=self._ownership_clock(),
                )
                prior_id = self._latest_work_attempt_id_unlocked(task.id)
                prior = None if prior_id is None else self._load_work_attempt_unlocked(prior_id)
                ordinal = 1 if prior is None else prior.ordinal + 1
                if ordinal > contract.continuation_policy.max_attempts:
                    raise WorkCompletionConflict(
                        "Work-contract attempt limit forbids another work attempt."
                    )
                if prior is not None:
                    row = self._connection.execute(
                        "SELECT decision.decision_id, receipt.decision_id AS applied_decision_id "
                        "FROM cayu_completion_proposals AS proposal "
                        "LEFT JOIN cayu_completion_decisions AS decision "
                        "ON decision.proposal_id = proposal.proposal_id "
                        "LEFT JOIN cayu_completion_decision_application_receipts AS receipt "
                        "ON receipt.decision_id = decision.decision_id "
                        "WHERE proposal.attempt_id = ?",
                        (prior.attempt_id,),
                    ).fetchone()
                    if row is None or row["decision_id"] is None:
                        raise WorkCompletionConflict(
                            "A prior work attempt has not reached a durable decision."
                        )
                    if row["applied_decision_id"] is None:
                        raise WorkCompletionConflict(
                            "A prior verifier decision has not reached durable task application."
                        )
                attempt = WorkAttempt(
                    attempt_id=request.attempt_id,
                    task_id=request.task_id,
                    session_id=request.session_id,
                    contract=request.contract,
                    execution_profile_fingerprint=request.execution_profile_fingerprint,
                    worker_id=request.worker_id,
                    ordinal=ordinal,
                    request_sha256=request_sha256,
                    started_at=self._clock(),
                )
                self._connection.execute(
                    "INSERT INTO cayu_work_attempts "
                    "(attempt_id, task_id, ordinal, request_sha256, started_at, attempt_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        attempt.attempt_id,
                        attempt.task_id,
                        attempt.ordinal,
                        attempt.request_sha256,
                        sqlite_support.format_datetime(attempt.started_at),
                        sqlite_support.json_dumps(attempt.model_dump(mode="json", warnings=False)),
                    ),
                )
                return attempt.model_copy(deep=True)

    async def load_work_attempt(self, attempt_id: str) -> WorkAttempt | None:
        attempt_id = require_clean_nonblank(attempt_id, "attempt_id")
        async with self._lock:
            attempt = self._load_work_attempt_unlocked(attempt_id)
            return None if attempt is None else attempt.model_copy(deep=True)

    async def submit_completion_proposal(
        self,
        request: CompletionProposalCreate,
    ) -> CompletionProposal:
        request = copy_completion_proposal_create(request)
        request_sha256 = completion_proposal_request_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                if (
                    self._load_work_attempt_admission_for_attempt_unlocked(request.attempt_id)
                    is not None
                ):
                    raise WorkAttemptAdmissionConflict(
                        "Admitted work attempts require claim-fenced proposal publication."
                    )
                existing = self._load_completion_proposal_unlocked(request.proposal_id)
                if existing is not None:
                    if existing.request_sha256 != request_sha256:
                        raise WorkCompletionConflict(
                            "Completion-proposal identity is already bound to another request."
                        )
                    return existing.model_copy(deep=True)
                occupied = self._connection.execute(
                    "SELECT proposal_id FROM cayu_completion_proposals WHERE attempt_id = ?",
                    (request.attempt_id,),
                ).fetchone()
                if occupied is not None:
                    raise WorkCompletionConflict(
                        "Work attempt already has a different completion proposal."
                    )
                attempt = self._load_work_attempt_unlocked(request.attempt_id)
                if attempt is None:
                    raise KeyError(f"Work attempt not found: {request.attempt_id}")
                task = self._require_task_unlocked(attempt.task_id)
                contract = self._load_work_contract_unlocked(attempt.contract)
                verified_work_support.require_attempt_current(
                    task,
                    attempt,
                    latest_attempt_id=self._latest_work_attempt_id_unlocked(task.id),
                    contract=contract,
                    now=self._ownership_clock(),
                )
                proposal = CompletionProposal(
                    proposal_id=request.proposal_id,
                    attempt_id=request.attempt_id,
                    result=request.result,
                    evidence_references=request.evidence_references,
                    task_id=attempt.task_id,
                    contract=attempt.contract,
                    request_sha256=request_sha256,
                    proposed_at=self._clock(),
                )
                self._connection.execute(
                    "INSERT INTO cayu_completion_proposals "
                    "(proposal_id, attempt_id, task_id, request_sha256, proposed_at, "
                    "proposal_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        proposal.proposal_id,
                        proposal.attempt_id,
                        proposal.task_id,
                        proposal.request_sha256,
                        sqlite_support.format_datetime(proposal.proposed_at),
                        sqlite_support.json_dumps(proposal.model_dump(mode="json", warnings=False)),
                    ),
                )
                return proposal.model_copy(deep=True)

    async def submit_admitted_completion_proposal(
        self,
        request: AdmittedCompletionProposalRequest,
    ) -> CompletionProposal:
        request = copy_admitted_completion_proposal_request(request)
        proposal_request = request.proposal
        proposal_sha256 = completion_proposal_request_sha256(proposal_request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                admission = self._load_work_attempt_admission_unlocked(request.admission_id)
                if admission is None:
                    raise KeyError(f"Work-attempt admission not found: {request.admission_id}")
                exact_claim_authority = (
                    admission.attempt is not None
                    and admission.attempt_id == proposal_request.attempt_id
                    and admission.claim.claim_id == request.claim_id
                    and admission.claim.generation == request.generation
                )
                if not exact_claim_authority:
                    raise WorkAttemptExecutionClaimLost(
                        "Completion proposal no longer owns the exact active admission."
                    )
                existing = self._load_completion_proposal_unlocked(proposal_request.proposal_id)
                if admission.state is WorkAttemptAdmissionState.RELEASED:
                    prior = self._connection.execute(
                        "SELECT proposal_id FROM cayu_completion_proposals WHERE attempt_id = ?",
                        (admission.attempt_id,),
                    ).fetchone()
                    if (
                        existing is None
                        or existing.request_sha256 != proposal_sha256
                        or prior is None
                        or prior["proposal_id"] != proposal_request.proposal_id
                    ):
                        raise WorkCompletionConflict(
                            "Released admission conflicts with the requested proposal replay."
                        )
                    return existing.model_copy(deep=True)
                if (
                    admission.state is not WorkAttemptAdmissionState.ACTIVE
                    or admission.claim.execution_owner_id != request.execution_owner_id
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Completion proposal no longer owns the exact active admission."
                    )
                lease_now = self._ownership_clock()
                self._ensure_live_work_attempt_admission_claim(admission, now=lease_now)
                if existing is not None:
                    if existing.request_sha256 != proposal_sha256:
                        raise WorkCompletionConflict(
                            "Completion-proposal identity is bound to another request."
                        )
                    return existing.model_copy(deep=True)
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_completion_proposals WHERE attempt_id = ?",
                        (admission.attempt_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkCompletionConflict(
                        "Work attempt already has a different completion proposal."
                    )
                task = self._require_task_unlocked(admission.task_id)
                contract = self._load_work_contract_unlocked(admission.contract)
                verified_work_support.require_attempt_state_current(
                    task,
                    admission.attempt,
                    latest_attempt_id=self._latest_work_attempt_id_unlocked(task.id),
                    contract=contract,
                )
                if (
                    task.session_instance_id != admission.session_invocation.session_instance_id
                    or task.worker_id != admission.claim.worker_id
                    or task.lease_expires_at != admission.claim.lease_expires_at
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Completion proposal lost exact task-worker lease ownership."
                    )
                proposal = CompletionProposal(
                    proposal_id=proposal_request.proposal_id,
                    attempt_id=proposal_request.attempt_id,
                    result=proposal_request.result,
                    evidence_references=proposal_request.evidence_references,
                    task_id=admission.task_id,
                    contract=admission.contract,
                    request_sha256=proposal_sha256,
                    proposed_at=self._clock(),
                )
                released = WorkAttemptAdmission.model_validate(
                    admission.model_copy(
                        update={"state": WorkAttemptAdmissionState.RELEASED}
                    ).model_dump(mode="python", warnings=False)
                )
                self._connection.execute(
                    "INSERT INTO cayu_completion_proposals "
                    "(proposal_id, attempt_id, task_id, request_sha256, proposed_at, "
                    "proposal_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        proposal.proposal_id,
                        proposal.attempt_id,
                        proposal.task_id,
                        proposal.request_sha256,
                        sqlite_support.format_datetime(proposal.proposed_at),
                        sqlite_support.json_dumps(proposal.model_dump(mode="json", warnings=False)),
                    ),
                )
                self._update_work_attempt_admission_unlocked(released)
                self._update_task_snapshot_unlocked(
                    task.model_copy(
                        update={
                            "worker_id": None,
                            "lease_expires_at": None,
                            "updated_at": lease_now,
                        }
                    )
                )
                return proposal.model_copy(deep=True)

    async def load_completion_proposal(self, proposal_id: str) -> CompletionProposal | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            proposal = self._load_completion_proposal_unlocked(proposal_id)
            return None if proposal is None else proposal.model_copy(deep=True)

    def _load_prior_completion_verifier_profile_unlocked(
        self,
        proposal: CompletionProposal,
    ) -> CompletionVerifierProfileRecord | None:
        row = self._connection.execute(
            "SELECT prior_proposal.proposal_id "
            "FROM cayu_work_attempts AS current_attempt "
            "JOIN cayu_work_attempts AS prior_attempt "
            "ON prior_attempt.task_id = current_attempt.task_id "
            "AND prior_attempt.ordinal = current_attempt.ordinal - 1 "
            "LEFT JOIN cayu_completion_proposals AS prior_proposal "
            "ON prior_proposal.attempt_id = prior_attempt.attempt_id "
            "WHERE current_attempt.attempt_id = ?",
            (proposal.attempt_id,),
        ).fetchone()
        if row is None:
            return None
        if row["proposal_id"] is None:
            raise WorkCompletionConflict("Prior work attempt has no completion proposal authority.")
        profile = self._load_completion_verifier_profile_unlocked(row["proposal_id"])
        if profile is None:
            raise WorkCompletionConflict("Prior work attempt has no verifier-profile authority.")
        return profile

    async def prepare_completion_verifier_profile(
        self,
        request: CompletionVerifierProfilePreparationRequest,
    ) -> CompletionVerifierProfileRecord:
        request = copy_completion_verifier_profile_preparation_request(request)
        request_sha256 = completion_verifier_profile_preparation_request_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                existing = self._load_completion_verifier_profile_unlocked(request.proposal_id)
                if existing is not None:
                    if existing.request_sha256 != request_sha256:
                        raise WorkCompletionConflict(
                            "Completion-verifier profile is already bound to another request."
                        )
                    return copy_completion_verifier_profile_record(existing)
                proposal = self._load_completion_proposal_unlocked(request.proposal_id)
                if proposal is None:
                    raise KeyError(f"Completion proposal not found: {request.proposal_id}")
                attempt = self._load_work_attempt_unlocked(proposal.attempt_id)
                if attempt is None:
                    raise WorkCompletionConflict("Completion proposal has no durable work attempt.")
                contract = verified_work_support.require_contract_reference(
                    self._load_work_contract_unlocked(proposal.contract),
                    proposal.contract,
                )
                if (
                    request.task_id != proposal.task_id
                    or request.attempt_id != attempt.attempt_id
                    or request.attempt_request_sha256 != attempt.request_sha256
                    or request.source_execution_profile_fingerprint
                    != attempt.execution_profile_fingerprint
                    or request.proposal_request_sha256 != proposal.request_sha256
                    or request.contract != contract.reference()
                    or request.profile.verifier != contract.verifier
                ):
                    raise WorkCompletionConflict(
                        "Completion-verifier profile conflicts with its durable proposal authority."
                    )
                prior = self._load_prior_completion_verifier_profile_unlocked(proposal)
                require_completion_verifier_profile_transition(request, prior)
                adoption = request.adoption
                if (
                    adoption is not None
                    and self._load_completion_verifier_adoption_unlocked(
                        task_id=request.task_id,
                        idempotency_key=adoption.idempotency_key,
                    )
                    is not None
                ):
                    raise WorkCompletionConflict(
                        "Completion-verifier profile adoption idempotency key is already "
                        "bound to another proposal."
                    )
                record = completion_verifier_profile_record_from_preparation(
                    request,
                    request_sha256=request_sha256,
                    prepared_at=self._clock(),
                )
                self._connection.execute(
                    "INSERT INTO cayu_completion_verifier_profiles "
                    "(proposal_id, task_id, attempt_id, profile_fingerprint, "
                    "request_sha256, prepared_at, profile_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.proposal_id,
                        record.task_id,
                        record.attempt_id,
                        record.profile.fingerprint,
                        record.request_sha256,
                        sqlite_support.format_datetime(record.prepared_at),
                        sqlite_support.json_dumps(record.model_dump(mode="json", warnings=False)),
                    ),
                )
                return copy_completion_verifier_profile_record(record)

    async def load_completion_verifier_profile(
        self,
        proposal_id: str,
    ) -> CompletionVerifierProfileRecord | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            profile = self._load_completion_verifier_profile_unlocked(proposal_id)
            return None if profile is None else copy_completion_verifier_profile_record(profile)

    async def load_prior_completion_verifier_profile(
        self,
        proposal_id: str,
    ) -> CompletionVerifierProfileRecord | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            proposal = self._load_completion_proposal_unlocked(proposal_id)
            if proposal is None:
                raise KeyError(f"Completion proposal not found: {proposal_id}")
            profile = self._load_prior_completion_verifier_profile_unlocked(proposal)
            return None if profile is None else copy_completion_verifier_profile_record(profile)

    async def claim_completion_verification(
        self,
        request: CompletionVerificationClaimRequest,
    ) -> CompletionVerificationClaim:
        request = copy_completion_verification_claim_request(request)
        request_sha256 = completion_verification_claim_request_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                claim_by_id = self._load_completion_claim_by_id_unlocked(request.claim_id)
                if claim_by_id is not None and (
                    claim_by_id.proposal_id != request.proposal_id
                    or claim_by_id.request_sha256 != request_sha256
                ):
                    raise WorkCompletionConflict(
                        "Verification-claim identity is already bound to another request."
                    )
                proposal = self._load_completion_proposal_unlocked(request.proposal_id)
                if proposal is None:
                    raise KeyError(f"Completion proposal not found: {request.proposal_id}")
                contract = self._load_work_contract_unlocked(proposal.contract)
                contract = verified_work_support.require_contract_reference(
                    contract,
                    proposal.contract,
                )
                if request.verifier != contract.verifier:
                    raise WorkCompletionConflict(
                        "Verification claim uses a verifier other than the frozen contract verifier."
                    )
                profile = self._load_completion_verifier_profile_unlocked(request.proposal_id)
                if (
                    profile is None
                    or profile.profile.fingerprint != request.verifier_profile_fingerprint
                ):
                    raise WorkCompletionConflict(
                        "Verification claim requires the exact prepared verifier profile."
                    )
                now = self._ownership_clock()
                current = self._load_completion_claim_unlocked(request.proposal_id)
                decision = self._load_completion_decision_for_proposal_unlocked(request.proposal_id)
                if (
                    current is not None
                    and current.claim_id == request.claim_id
                    and current.request_sha256 == request_sha256
                ):
                    if current.lease_expires_at > now or decision is not None:
                        return current.model_copy(deep=True)
                    raise CompletionVerificationClaimLost(
                        "Verification claim expired and cannot regain authority by replay."
                    )
                if decision is not None:
                    raise WorkCompletionConflict(
                        "Completion proposal already has a durable decision."
                    )
                if current is not None and current.lease_expires_at > now:
                    raise CompletionVerificationClaimLost(
                        "Completion proposal is owned by another live verifier claim."
                    )
                if claim_by_id is not None:
                    raise CompletionVerificationClaimLost(
                        "Verification claim expired and cannot regain authority by replay."
                    )
                attempt = self._load_work_attempt_unlocked(proposal.attempt_id)
                if attempt is None:
                    raise WorkCompletionConflict("Completion proposal has no durable work attempt.")
                task = self._require_task_unlocked(proposal.task_id)
                verified_work_support.require_proposal_chain(
                    proposal,
                    attempt,
                    task,
                    latest_attempt_id=self._latest_work_attempt_id_unlocked(task.id),
                    contract=contract,
                )
                attempt_number = 1 if current is None else current.attempt_number + 1
                claim = CompletionVerificationClaim(
                    claim_id=request.claim_id,
                    proposal_id=request.proposal_id,
                    worker_id=request.worker_id,
                    execution_owner_id=request.execution_owner_id,
                    execution_timeout_seconds=request.execution_timeout_seconds,
                    verifier=request.verifier,
                    verifier_profile_fingerprint=request.verifier_profile_fingerprint,
                    attempt_number=attempt_number,
                    request_sha256=request_sha256,
                    claimed_at=now,
                    lease_expires_at=now + timedelta(seconds=request.lease_seconds),
                )
                self._connection.execute(
                    "UPDATE cayu_completion_verification_claims SET is_current = 0 "
                    "WHERE proposal_id = ? AND is_current = 1",
                    (request.proposal_id,),
                )
                self._connection.execute(
                    "INSERT INTO cayu_completion_verification_claims "
                    "(claim_id, proposal_id, attempt_number, verifier_profile_fingerprint, "
                    "request_sha256, lease_expires_at, is_current, claim_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                    (
                        claim.claim_id,
                        claim.proposal_id,
                        claim.attempt_number,
                        claim.verifier_profile_fingerprint,
                        claim.request_sha256,
                        sqlite_support.format_datetime(claim.lease_expires_at),
                        sqlite_support.json_dumps(claim.model_dump(mode="json", warnings=False)),
                    ),
                )
                return claim.model_copy(deep=True)

    async def load_completion_verification_claim(
        self,
        proposal_id: str,
    ) -> CompletionVerificationClaim | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            claim = self._load_completion_claim_unlocked(proposal_id)
            return None if claim is None else claim.model_copy(deep=True)

    async def renew_completion_verification_claim(
        self,
        request: CompletionVerificationClaimRequest,
    ) -> CompletionVerificationClaim:
        request = copy_completion_verification_claim_request(request)
        request_sha256 = completion_verification_claim_request_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                proposal = self._load_completion_proposal_unlocked(request.proposal_id)
                if proposal is None:
                    raise KeyError(f"Completion proposal not found: {request.proposal_id}")
                current = self._load_completion_claim_unlocked(request.proposal_id)
                now = self._ownership_clock()
                if (
                    current is None
                    or current.claim_id != request.claim_id
                    or current.worker_id != request.worker_id
                    or current.execution_owner_id != request.execution_owner_id
                    or current.execution_timeout_seconds != request.execution_timeout_seconds
                    or current.verifier != request.verifier
                    or current.verifier_profile_fingerprint != request.verifier_profile_fingerprint
                    or current.request_sha256 != request_sha256
                    or current.lease_expires_at <= now
                    or self._load_completion_decision_for_proposal_unlocked(proposal.proposal_id)
                    is not None
                ):
                    raise CompletionVerificationClaimLost(
                        "Verification claim cannot be renewed without exact current live authority."
                    )
                attempt = self._load_work_attempt_unlocked(proposal.attempt_id)
                if attempt is None:
                    raise WorkCompletionConflict("Completion proposal has no durable work attempt.")
                task = self._require_task_unlocked(proposal.task_id)
                contract = self._load_work_contract_unlocked(proposal.contract)
                verified_work_support.require_proposal_chain(
                    proposal,
                    attempt,
                    task,
                    latest_attempt_id=self._latest_work_attempt_id_unlocked(task.id),
                    contract=contract,
                )
                renewed = current.model_copy(
                    update={
                        "lease_expires_at": max(
                            current.lease_expires_at,
                            now + timedelta(seconds=request.lease_seconds),
                        )
                    }
                )
                self._connection.execute(
                    "UPDATE cayu_completion_verification_claims "
                    "SET lease_expires_at = ?, claim_json = ? "
                    "WHERE claim_id = ? AND is_current = 1",
                    (
                        sqlite_support.format_datetime(renewed.lease_expires_at),
                        sqlite_support.json_dumps(renewed.model_dump(mode="json", warnings=False)),
                        renewed.claim_id,
                    ),
                )
                return renewed.model_copy(deep=True)

    async def record_completion_decision(
        self,
        request: CompletionDecisionCreate,
    ) -> CompletionDecision:
        request = copy_completion_decision_create(request)
        request_sha256 = completion_decision_request_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                existing = self._load_completion_decision_unlocked(request.decision_id)
                if existing is not None:
                    if existing.request_sha256 != request_sha256:
                        raise WorkCompletionConflict(
                            "Completion-decision identity is already bound to another request."
                        )
                    return existing.model_copy(deep=True)
                prior = self._load_completion_decision_for_proposal_unlocked(request.proposal_id)
                if prior is not None:
                    raise WorkCompletionConflict(
                        "Completion proposal already has a different durable decision."
                    )
                proposal = self._load_completion_proposal_unlocked(request.proposal_id)
                if proposal is None:
                    raise KeyError(f"Completion proposal not found: {request.proposal_id}")
                claim = self._load_completion_claim_unlocked(proposal.proposal_id)
                profile = self._load_completion_verifier_profile_unlocked(proposal.proposal_id)
                lease_now = self._ownership_clock()
                evidence_now = self._clock()
                if (
                    claim is None
                    or claim.claim_id != request.claim_id
                    or claim.worker_id != request.worker_id
                    or claim.verifier != request.verifier
                    or claim.verifier_profile_fingerprint != request.verifier_profile_fingerprint
                    or profile is None
                    or profile.profile.fingerprint != request.verifier_profile_fingerprint
                    or claim.lease_expires_at <= lease_now
                ):
                    raise CompletionVerificationClaimLost(
                        "Completion decision requires the current live verifier claim."
                    )
                attempt = self._load_work_attempt_unlocked(proposal.attempt_id)
                if attempt is None:
                    raise WorkCompletionConflict("Completion proposal has no durable work attempt.")
                task = self._require_task_unlocked(proposal.task_id)
                contract = self._load_work_contract_unlocked(proposal.contract)
                contract = verified_work_support.require_proposal_chain(
                    proposal,
                    attempt,
                    task,
                    latest_attempt_id=self._latest_work_attempt_id_unlocked(task.id),
                    contract=contract,
                )
                validate_completion_decision_contract(contract, request)
                decision = CompletionDecision(
                    decision_id=request.decision_id,
                    proposal_id=request.proposal_id,
                    claim_id=request.claim_id,
                    worker_id=request.worker_id,
                    verifier=request.verifier,
                    verifier_profile_fingerprint=request.verifier_profile_fingerprint,
                    decision_version=request.decision_version,
                    verdict=request.verdict,
                    criterion_outcomes=request.criterion_outcomes,
                    constraint_outcomes=request.constraint_outcomes,
                    gaps=request.gaps,
                    evidence_references=request.evidence_references,
                    task_id=proposal.task_id,
                    attempt_id=proposal.attempt_id,
                    contract=proposal.contract,
                    claim_authority_sha256=completion_verification_claim_authority_sha256(claim),
                    request_sha256=request_sha256,
                    gap_fingerprint=completion_gap_fingerprint(request),
                    decided_at=evidence_now,
                )
                self._connection.execute(
                    "INSERT INTO cayu_completion_decisions "
                    "(decision_id, proposal_id, task_id, attempt_id, claim_id, "
                    "verifier_profile_fingerprint, verdict, "
                    "gap_fingerprint, request_sha256, decided_at, decision_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        decision.decision_id,
                        decision.proposal_id,
                        decision.task_id,
                        decision.attempt_id,
                        decision.claim_id,
                        decision.verifier_profile_fingerprint,
                        decision.verdict.value,
                        decision.gap_fingerprint,
                        decision.request_sha256,
                        sqlite_support.format_datetime(decision.decided_at),
                        sqlite_support.json_dumps(decision.model_dump(mode="json", warnings=False)),
                    ),
                )
                return decision.model_copy(deep=True)

    async def load_completion_decision(
        self,
        decision_id: str,
    ) -> CompletionDecision | None:
        decision_id = require_clean_nonblank(decision_id, "decision_id")
        async with self._lock:
            decision = self._load_completion_decision_unlocked(decision_id)
            return None if decision is None else decision.model_copy(deep=True)

    async def load_completion_decision_for_proposal(
        self,
        proposal_id: str,
    ) -> CompletionDecision | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            decision = self._load_completion_decision_for_proposal_unlocked(proposal_id)
            return None if decision is None else decision.model_copy(deep=True)

    async def apply_completion_decision(
        self,
        request: CompletionDecisionApplicationRequest,
    ) -> Task:
        try:
            copied_request = copy_completion_decision_application_request(request)
        except BaseException:
            del request
            raise
        request = copied_request
        del copied_request
        request_sha256 = completion_decision_application_request_sha256(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                receipt = self._load_decision_application_receipt_unlocked(
                    request.task_id,
                    request.idempotency_key,
                )
                if receipt is not None:
                    if receipt.request_sha256 != request_sha256:
                        raise WorkCompletionConflict(
                            "Decision-application identity is already bound to another request."
                        )
                    return receipt.task.model_copy(deep=True)
                prior = self._connection.execute(
                    "SELECT task_id, idempotency_key FROM "
                    "cayu_completion_decision_application_receipts WHERE decision_id = ?",
                    (request.decision_id,),
                ).fetchone()
                if prior is not None:
                    raise WorkCompletionConflict(
                        "Completion decision was already applied under another identity."
                    )
                task = self._require_task_unlocked(request.task_id)
                decision = self._load_completion_decision_unlocked(request.decision_id)
                if decision is None:
                    raise KeyError(f"Completion decision not found: {request.decision_id}")
                if decision.task_id != task.id:
                    raise WorkCompletionConflict("Completion decision belongs to another task.")
                contract = self._require_task_contract_unlocked(task, decision.contract)
                attempt = self._load_work_attempt_unlocked(decision.attempt_id)
                if attempt is None:
                    raise WorkCompletionConflict("Completion decision has no work attempt.")
                verified_work_support.require_decision_attempt_current(
                    task,
                    attempt,
                    latest_attempt_id=self._latest_work_attempt_id_unlocked(task.id),
                    contract=contract,
                )
                proposal = self._load_completion_proposal_unlocked(decision.proposal_id)
                if proposal is None:
                    raise WorkCompletionConflict("Completion decision has no completion proposal.")
                profile = self._load_completion_verifier_profile_unlocked(proposal.proposal_id)
                if (
                    profile is None
                    or profile.profile.fingerprint != decision.verifier_profile_fingerprint
                ):
                    raise WorkCompletionConflict(
                        "Completion decision has no exact verifier-profile authority."
                    )
                row = self._connection.execute(
                    "SELECT COUNT(*) AS matching FROM cayu_completion_decisions "
                    "WHERE task_id = ? AND verdict = ? AND gap_fingerprint = ?",
                    (task.id, CompletionVerdict.REJECTED.value, decision.gap_fingerprint),
                ).fetchone()
                matching_gap_count = 0 if row is None else int(row["matching"])
                updated, receipt = verified_work_support.plan_decision_application(
                    request,
                    request_sha256=request_sha256,
                    task=task,
                    decision=decision,
                    proposal=proposal,
                    attempt=attempt,
                    contract=contract,
                    matching_gap_count=matching_gap_count,
                    now=self._ownership_clock(),
                )
                if updated != task:
                    self._update_task_snapshot_unlocked(updated)
                self._connection.execute(
                    "INSERT INTO cayu_completion_decision_application_receipts "
                    "(task_id, idempotency_key, decision_id, request_sha256, applied_at, "
                    "receipt_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        receipt.task_id,
                        receipt.idempotency_key,
                        receipt.decision_id,
                        receipt.request_sha256,
                        sqlite_support.format_datetime(receipt.applied_at),
                        sqlite_support.json_dumps(receipt.model_dump(mode="json", warnings=False)),
                    ),
                )
                return updated.model_copy(deep=True)

    async def load_completion_decision_application_receipt(
        self,
        task_id: str,
        idempotency_key: str,
    ) -> CompletionDecisionApplicationReceipt | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        idempotency_key = validate_work_completion_idempotency_key(idempotency_key)
        async with self._lock:
            receipt = self._load_decision_application_receipt_unlocked(
                task_id,
                idempotency_key,
            )
            return None if receipt is None else receipt.model_copy(deep=True)

    async def create_task(self, request: TaskCreate) -> Task:
        request = copy_task_create(request)
        async with self._lock:
            with self._verified_transaction_unlocked():
                task_id = request.task_id or str(uuid4())
                parent = self._task_parent_for_create_unlocked(request, task_id=task_id)
                if request.work_contract is not None:
                    contract = self._load_work_contract_unlocked(request.work_contract)
                    verified_work_support.require_contract_reference(
                        contract,
                        request.work_contract,
                    )
                    if request.session_id is not None:
                        self._ensure_session_execution_authority_unlocked(
                            request.session_id,
                            "contracted",
                        )
                admission_now = self._clock()
                task = _task_from_create(
                    request,
                    task_id=task_id,
                    parent_task=parent,
                    retry_started_at=admission_now,
                    supports_verified_work_contracts=True,
                )
                self._insert_task_unlocked(task)
                created = task.model_copy(deep=True)
        self._publish_task_admission_wakeup(task, now=admission_now)
        return created

    async def create_running_task(
        self,
        request: TaskCreate,
        *,
        session_invocation: SessionInvocationBinding,
    ) -> Task:
        request = copy_task_create(request)
        session_binding = _copy_required_session_binding(session_invocation)
        async with self._lock:
            with self._verified_transaction_unlocked():
                task_id = request.task_id or str(uuid4())
                parent = self._task_parent_for_create_unlocked(request, task_id=task_id)
                if request.work_contract is not None:
                    contract = self._load_work_contract_unlocked(request.work_contract)
                    verified_work_support.require_contract_reference(
                        contract,
                        request.work_contract,
                    )
                    if request.session_id is not None:
                        self._ensure_session_execution_authority_unlocked(
                            request.session_id,
                            "contracted",
                        )
                task = _running_task_from_create(
                    request,
                    task_id=task_id,
                    parent_task=parent,
                    session_invocation=session_binding,
                    retry_started_at=self._clock(),
                    supports_verified_work_contracts=True,
                )
                self._insert_task_unlocked(task)
                return task.model_copy(deep=True)

    def _insert_task_unlocked(self, task: Task) -> None:
        try:
            self._connection.execute(
                """
                INSERT INTO cayu_tasks (
                    id,
                    type,
                    title,
                    description,
                    status,
                    session_id,
                    session_instance_id,
                    parent_task_id,
                    assigned_agent_name,
                    available_at,
                    worker_id,
                    lease_expires_at,
                    interrupted_handoff_id,
                    status_reason,
                    status_payload_json,
                    input_json,
                    result_json,
                    error_json,
                    metadata_json,
                    created_at,
                    updated_at,
                    started_at,
                    completed_at,
                    invocation_json,
                    retry_series_json,
                    work_contract_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                sqlite_support.task_to_row_values(task),
            )
        except sqlite3.IntegrityError as exc:
            if self._task_exists_unlocked(task.id):
                raise ValueError(f"Task already exists: {task.id}") from exc
            raise

    async def load_task(self, task_id: str) -> Task | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        async with self._lock:
            return self._load_task_unlocked(task_id)

    async def load_active_attached_task_worker(
        self,
        task_id: str,
        worker_id: str,
        *,
        session_id: str,
        session_instance_id: str,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        session_id = require_clean_nonblank(session_id, "session_id")
        session_instance_id = require_clean_nonblank(
            session_instance_id,
            "session_instance_id",
        )
        async with self._lock:
            return _require_active_attached_task_worker(
                self._require_task_unlocked(task_id),
                worker_id=worker_id,
                session_id=session_id,
                session_instance_id=session_instance_id,
                now=self._ownership_clock(),
            )

    async def load_direct_attached_task_resume(
        self,
        task_id: str,
        *,
        session_id: str,
        session_instance_id: str,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        session_id = require_clean_nonblank(session_id, "session_id")
        session_instance_id = require_clean_nonblank(
            session_instance_id,
            "session_instance_id",
        )
        async with self._lock:
            return _require_direct_attached_task_resume(
                self._require_task_unlocked(task_id),
                session_id=session_id,
                session_instance_id=session_instance_id,
            )

    async def load_invocation_snapshot(
        self,
        task_id: str,
    ) -> TaskInvocationSnapshot | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        async with self._lock:
            row = self._connection.execute(
                "SELECT id, session_id, session_instance_id, invocation_json "
                "FROM cayu_tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            return TaskInvocationSnapshot(
                id=row["id"],
                session_id=row["session_id"],
                session_instance_id=row["session_instance_id"],
                invocation=TaskInvocation.model_validate(json.loads(row["invocation_json"])),
            )

    async def list_tasks(self, query: TaskQuery | None = None) -> list[Task]:
        query = copy_task_query(query)
        clauses: list[str] = []
        params: list[object] = []

        if query.q is not None:
            like = _like_contains_pattern(query.q)
            clauses.append(
                """
                (
                    id COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR type COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR title COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR description COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR status COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR session_id COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR parent_task_id COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR assigned_agent_name COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR worker_id COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR status_reason COLLATE NOCASE LIKE ? ESCAPE '\\'
                )
                """
            )
            params.extend([like] * 10)
        if query.status is not None:
            clauses.append("status = ?")
            params.append(str(query.status))
        if query.type is not None:
            clauses.append("type = ?")
            params.append(query.type)
        if query.session_id is not None:
            clauses.append("session_id = ?")
            params.append(query.session_id)
        if query.parent_task_id is not None:
            clauses.append("parent_task_id = ?")
            params.append(query.parent_task_id)
        if query.assigned_agent_name is not None:
            clauses.append("assigned_agent_name = ?")
            params.append(query.assigned_agent_name)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_sql = sqlite_support.task_order_sql(query.order_by)
        params.extend([query.limit, query.offset])

        async with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT *
                FROM cayu_tasks
                {where_sql}
                ORDER BY {order_sql}, id ASC
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
            return [sqlite_support.task_from_row(row) for row in rows]

    async def query_task_topology(
        self,
        query: TaskTopologyQuery,
    ) -> TaskTopologyStoreResult:
        if type(query) is not TaskTopologyQuery:
            raise TypeError("Task topology queries must be TaskTopologyQuery instances.")
        query = TaskTopologyQuery.model_validate(query.model_dump(mode="python"))
        session_branch_limits, child_branch_limits = _allocate_task_topology_branch_limits(query)

        def read_branch_candidates(
            *,
            branch_ids: tuple[str, ...],
            cursors: dict[str, str],
            scope_kind: Literal["session", "parent_task"],
            scope_column: Literal["session_id", "parent_task_id"],
            branch_limits: tuple[int, ...],
        ) -> list[list[TaskTopologyNode]]:
            candidates: list[list[TaskTopologyNode]] = [[] for _ in branch_ids]
            if not branch_ids:
                return candidates
            branch_queries: list[str] = []
            branch_params: list[object] = []
            for branch_order, (branch_id, branch_limit) in enumerate(
                zip(branch_ids, branch_limits, strict=True)
            ):
                cursor = cursors.get(branch_id)
                if cursor is None:
                    cursor_clause = ""
                    cursor_params: list[object] = []
                else:
                    cursor_created_at, cursor_id = decode_task_topology_cursor(
                        cursor,
                        scope_kind=scope_kind,
                        scope_id=branch_id,
                    )
                    cursor_clause = "AND (created_at > ? OR (created_at = ? AND id > ?))"
                    formatted = sqlite_support.format_datetime(cursor_created_at)
                    cursor_params = [formatted, formatted, cursor_id]
                branch_queries.append(
                    f"""
                    SELECT branch_order, candidate.*
                    FROM (
                        SELECT ? AS branch_order, {sqlite_support.TASK_TOPOLOGY_COLUMNS}
                        FROM cayu_tasks
                        WHERE {scope_column} = ?
                          {cursor_clause}
                        ORDER BY created_at ASC, id ASC
                        LIMIT ?
                    ) AS candidate
                    """
                )
                branch_params.extend(
                    [
                        branch_order,
                        branch_id,
                        *cursor_params,
                        branch_limit + 1,
                    ]
                )
            rows = self._connection.execute(
                f"""
                {" UNION ALL ".join(branch_queries)}
                ORDER BY branch_order ASC, topology_created_at ASC, topology_id ASC
                """,
                branch_params,
            ).fetchall()
            for row in rows:
                candidates[row["branch_order"]].append(
                    sqlite_support.task_topology_node_from_row(row)
                )
            return candidates

        async with self._lock:
            self._connection.execute("BEGIN")
            try:
                observed_row = self._connection.execute(
                    "SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
                ).fetchone()
                if observed_row is None:
                    raise RuntimeError("SQLite did not return a topology snapshot timestamp.")
                observed_at = sqlite_support.parse_datetime(observed_row[0])

                expanded_parents: list[TaskTopologyNode] = []
                if query.expanded_parent_ids:
                    placeholders = ", ".join("?" for _ in query.expanded_parent_ids)
                    rows = self._connection.execute(
                        f"""
                        SELECT {sqlite_support.TASK_TOPOLOGY_COLUMNS}
                        FROM cayu_tasks
                        WHERE id IN ({placeholders})
                        """,
                        query.expanded_parent_ids,
                    ).fetchall()
                    parents_by_id = {
                        row["topology_id"]: sqlite_support.task_topology_node_from_row(row)
                        for row in rows
                    }
                    for parent_id in query.expanded_parent_ids:
                        parent = parents_by_id.get(parent_id)
                        if parent is None:
                            raise KeyError(f"Task not found: {parent_id}")
                        expanded_parents.append(parent)

                session_candidates = read_branch_candidates(
                    branch_ids=query.linked_session_ids,
                    cursors=query.session_cursors,
                    scope_kind="session",
                    scope_column="session_id",
                    branch_limits=session_branch_limits,
                )
                child_candidates = read_branch_candidates(
                    branch_ids=query.expanded_parent_ids,
                    cursors=query.child_cursors,
                    scope_kind="parent_task",
                    scope_column="parent_task_id",
                    branch_limits=child_branch_limits,
                )

                async def load_parent_links(
                    task_ids: tuple[str, ...],
                ) -> dict[str, str | None]:
                    links: dict[str, str | None] = {}
                    for index in range(0, len(task_ids), 500):
                        batch = task_ids[index : index + 500]
                        placeholders = ", ".join("?" for _ in batch)
                        rows = self._connection.execute(
                            f"""
                            SELECT
                                id,
                                CASE
                                    WHEN parent_task_id IS NULL
                                      OR length(CAST(parent_task_id AS BLOB))
                                         <= {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
                                    THEN parent_task_id
                                END AS topology_parent_task_id,
                                parent_task_id IS NOT NULL
                                  AND length(CAST(parent_task_id AS BLOB))
                                      > {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
                                    AS topology_parent_task_id_oversized
                            FROM cayu_tasks
                            WHERE id IN ({placeholders})
                            """,
                            batch,
                        ).fetchall()
                        for row in rows:
                            if row["topology_parent_task_id_oversized"]:
                                raise TaskTopologyInconsistent(
                                    "A task topology ancestor contains an oversized "
                                    "parent identifier."
                                )
                            links[row["id"]] = _bounded_optional_task_topology_parent_id(
                                row["topology_parent_task_id"]
                            )
                    return links

                await _validate_task_topology_ancestry(
                    (
                        *expanded_parents,
                        *(task for branch in session_candidates for task in branch),
                        *(task for branch in child_candidates for task in branch),
                    ),
                    load_parent_links,
                )
                result = build_task_topology_result(
                    observed_at=observed_at,
                    linked_session_ids=query.linked_session_ids,
                    session_branch_candidates=session_candidates,
                    session_branch_limits=session_branch_limits,
                    expanded_parents=expanded_parents,
                    child_branch_candidates=child_candidates,
                    child_branch_limits=child_branch_limits,
                    session_task_limit=query.session_task_limit,
                    child_limit=query.child_limit,
                )
                self._connection.commit()
                return result
            except Exception:
                self._connection.rollback()
                raise

    async def aggregate_operational_snapshot(
        self,
        filters: TaskAggregateFilter | None = None,
    ) -> TaskOperationalSnapshot:
        filters = copy_task_aggregate_filter(filters)
        clauses, params = self._task_filter_clauses(task_query_from_aggregate_filter(filters))
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        def query_snapshot(connection: sqlite3.Connection) -> TaskOperationalSnapshot:
            snapshot_as_of = sqlite_support.format_datetime(self._clock())
            rows = connection.execute(
                f"""
                WITH
                snapshot(as_of) AS (
                    SELECT ?
                ),
                matching_tasks AS (
                    SELECT id, status, session_id, available_at, retry_series_json
                    FROM cayu_tasks
                    {where_sql}
                ),
                status_counts AS (
                    SELECT status, COUNT(*) AS status_count
                    FROM matching_tasks
                    GROUP BY status
                ),
                pending_counts AS (
                    SELECT
                        COALESCE(SUM(
                            CASE
                                 WHEN status = 'pending'
                                 AND session_id IS NULL
                                 AND (available_at IS NULL OR available_at <= snapshot.as_of)
                                 AND NOT EXISTS (
                                     SELECT 1
                                     FROM cayu_local_execution_attempts AS attempt
                                     WHERE attempt.retry_admissible = 0
                                       AND (
                                           attempt.task_id = matching_tasks.id
                                           OR (
                                               matching_tasks.retry_series_json IS NOT NULL
                                               AND attempt.retry_series_id = json_extract(
                                                   matching_tasks.retry_series_json,
                                                   '$.series_id'
                                               )
                                           )
                                       )
                                 )
                                THEN 1 ELSE 0
                            END
                        ), 0) AS claimable_pending_count,
                        COALESCE(SUM(
                            CASE
                                WHEN status = 'pending'
                                 AND available_at > snapshot.as_of
                                THEN 1 ELSE 0
                            END
                        ), 0) AS scheduled_pending_count
                    FROM matching_tasks
                    CROSS JOIN snapshot
                )
                SELECT
                    snapshot.as_of,
                    status_counts.status,
                    status_counts.status_count,
                    pending_counts.claimable_pending_count,
                    pending_counts.scheduled_pending_count
                FROM snapshot
                CROSS JOIN pending_counts
                LEFT JOIN status_counts ON TRUE
                """,
                (snapshot_as_of, *params),
            ).fetchall()
            counts = {status: 0 for status in TaskStatus}
            for row in rows:
                if row["status"] is not None:
                    status = TaskStatus(row["status"])
                    counts[status] = row["status_count"]
            return TaskOperationalSnapshot(
                as_of=sqlite_support.parse_datetime(rows[0]["as_of"]),
                total_count=sum(counts.values()),
                counts_by_status=TaskStatusCounts.model_validate(counts),
                claimable_pending_count=rows[0]["claimable_pending_count"],
                scheduled_pending_count=rows[0]["scheduled_pending_count"],
                accuracy=EXACT_AGGREGATE.model_copy(),
            )

        return await _run_off_thread_with_connection_ownership(
            self._lock,
            self._connection,
            query_snapshot,
            interrupt_on_cancellation=True,
        )

    async def start_task(
        self,
        task_id: str,
        *,
        session_id: str | None = None,
        session_invocation: SessionInvocationBinding | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        session_binding = _copy_optional_session_binding(session_invocation)
        async with self._lock:
            with self._verified_transaction_unlocked():
                task = self._require_task_unlocked(task_id)
                _ensure_retry_series_queue_attempt(task.retry_series)
                _ensure_can_transition(task, TaskStatus.RUNNING)
                effective_session_id = _task_session_id_for_start(
                    task_id=task_id,
                    stored_session_id=task.session_id,
                    requested_session_id=session_id,
                )
                if task.work_contract is not None:
                    self._require_task_contract_unlocked(task, task.work_contract)
                    if effective_session_id is None:
                        raise WorkCompletionConflict(
                            "Contracted tasks require a session binding before starting."
                        )
                    self._ensure_session_execution_authority_unlocked(
                        effective_session_id,
                        "contracted",
                    )
                _task_invocation_for_attachment(
                    task.invocation,
                    session_id=effective_session_id,
                    session_binding=session_binding,
                )
                session_instance_id = _task_session_instance_for_attachment(
                    stored_session_instance_id=task.session_instance_id,
                    session_id=effective_session_id,
                    session_binding=session_binding,
                )
                now = self._ownership_clock()
                updated = task.model_copy(
                    update={
                        "status": TaskStatus.RUNNING,
                        "session_id": effective_session_id,
                        "session_instance_id": session_instance_id,
                        "started_at": task.started_at or now,
                        "updated_at": now,
                    }
                )
                self._update_task_snapshot_unlocked(updated)
                return updated.model_copy(deep=True)

    async def attach_task(
        self,
        task_id: str,
        *,
        session_id: str,
        session_invocation: SessionInvocationBinding,
        worker_id: str,
        lease_expires_at: datetime | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        session_id = require_clean_nonblank(session_id, "session_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        expected_lease = (
            None
            if lease_expires_at is None
            else normalize_utc_datetime(lease_expires_at, "lease_expires_at")
        )
        session_binding = _copy_required_session_binding(session_invocation)
        async with self._lock:
            with self._verified_transaction_unlocked():
                # Lease authority must be sampled only after BEGIN IMMEDIATE
                # has acquired SQLite's cross-process writer lock. A timestamp
                # captured before that wait could outlive the worker lease.
                now = self._ownership_clock()
                task = self._require_task_unlocked(task_id)
                _ensure_retry_series_queue_attempt(task.retry_series)
                if not _can_attach_claimed_task_state(
                    status=task.status,
                    session_id=task.session_id,
                    worker_id=task.worker_id,
                    lease_expires_at=task.lease_expires_at,
                    expected_worker_id=worker_id,
                    now=now,
                ):
                    self._raise_task_claim_attach_error(task_id, worker_id, now=now)
                if expected_lease is None:
                    raise TaskClaimLost("Task attachment requires its exact worker lease.")
                _ensure_exact_owned_active_task_lease(
                    task,
                    worker_id,
                    expected_lease,
                    now=now,
                )
                if task.work_contract is not None:
                    self._require_task_contract_unlocked(task, task.work_contract)
                    self._ensure_session_execution_authority_unlocked(
                        session_id,
                        "contracted",
                    )
                _task_invocation_for_attachment(
                    task.invocation,
                    session_id=session_id,
                    session_binding=session_binding,
                )
                session_instance_id = _task_session_instance_for_attachment(
                    stored_session_instance_id=task.session_instance_id,
                    session_id=session_id,
                    session_binding=session_binding,
                )
                updated = task.model_copy(
                    update={
                        "status": TaskStatus.RUNNING,
                        "session_id": session_id,
                        "session_instance_id": session_instance_id,
                        "started_at": task.started_at or now,
                        "updated_at": now,
                    }
                )
                self._update_task_snapshot_unlocked(updated)
                return updated.model_copy(deep=True)

    async def complete_task(
        self,
        task_id: str,
        result: dict[str, Any],
        *,
        worker_id: str | None = None,
        lease_expires_at: datetime | None = None,
        handoff_id: str | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        result = copy_durable_json_object(result, "result")
        if worker_id is not None and lease_expires_at is None:
            raise TaskClaimLost(
                "Worker-owned task terminalization requires its exact lease generation."
            )
        async with (
            managed_task_lease_mutation(
                task_id=task_id,
                worker_id=worker_id,
                handoff_id=handoff_id,
                presented_lease_expires_at=lease_expires_at,
            ) as effective_lease,
            self._lock,
        ):
            return self._finish_task_unlocked(
                task_id,
                TaskStatus.COMPLETED,
                result=result,
                error=None,
                worker_id=worker_id,
                expected_lease_expires_at=effective_lease,
                handoff_id=handoff_id,
            )

    async def fail_task(
        self,
        task_id: str,
        error: dict[str, Any],
        *,
        worker_id: str | None = None,
        lease_expires_at: datetime | None = None,
        handoff_id: str | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        error = copy_durable_json_object(error, "error")
        if worker_id is not None and lease_expires_at is None:
            raise TaskClaimLost(
                "Worker-owned task terminalization requires its exact lease generation."
            )
        async with (
            managed_task_lease_mutation(
                task_id=task_id,
                worker_id=worker_id,
                handoff_id=handoff_id,
                presented_lease_expires_at=lease_expires_at,
            ) as effective_lease,
            self._lock,
        ):
            return self._finish_task_unlocked(
                task_id,
                TaskStatus.FAILED,
                result=None,
                error=error,
                worker_id=worker_id,
                expected_lease_expires_at=effective_lease,
                handoff_id=handoff_id,
            )

    async def terminalize_task(self, request: TaskTerminalizationRequest) -> Task:
        request, request_sha256 = prepare_task_terminalization(request)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                receipt_row = self._connection.execute(
                    "SELECT request_sha256, worker_id, terminal_kind, task_json, committed_at "
                    "FROM cayu_task_terminalization_receipts "
                    "WHERE task_id = ? AND idempotency_key = ?",
                    (request.task_id, request.idempotency_key),
                ).fetchone()
                if receipt_row is not None:
                    receipt = _sqlite_task_terminalization_receipt(
                        task_id=request.task_id,
                        idempotency_key=request.idempotency_key,
                        row=receipt_row,
                    )
                    replayed = _replay_task_terminalization_receipt(
                        request=request,
                        request_sha256=request_sha256,
                        receipt=receipt,
                        current_task=self._load_task_unlocked(request.task_id),
                    )
                    self._connection.commit()
                    return replayed

                task = self._require_task_unlocked(request.task_id)
                self._raise_if_governed_work_attempt_admission(
                    request.task_id,
                    "Admitted work attempts cannot use ordinary terminalization.",
                )
                if task.retry_series is not None:
                    raise ValueError(
                        "Retry-series tasks require settle_task_retry_attempt for "
                        "completion or failure."
                    )
                if task.status in {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                }:
                    raise TaskTerminalizationConflict(
                        "Task is terminal without the matching terminalization receipt."
                    )
                now = self._ownership_clock()
                _ensure_task_terminalization_lease_authority(task, request, now=now)
                _ensure_task_handoff_authority(task, request.handoff_id)

                _validate_ordinary_task_terminalization_against_cancellation(task, request)
                status = TaskStatus(request.kind.value)
                verified_work_support.require_contracted_completion_authority(
                    task,
                    status,
                )
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?,
                        status_reason = NULL,
                        status_payload_json = NULL,
                        result_json = ?,
                        error_json = ?,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        interrupted_handoff_id = NULL,
                        started_at = COALESCE(started_at, ?),
                        completed_at = ?,
                        updated_at = ?
                    WHERE id = ?
                      AND status IN (?, ?)
                      AND worker_id = ?
                      AND interrupted_handoff_id IS ?
                      AND lease_expires_at IS NOT NULL
                      AND lease_expires_at > ?
                    """,
                    (
                        str(status),
                        (
                            None
                            if request.result is None
                            else sqlite_support.json_dumps(request.result)
                        ),
                        (
                            None
                            if request.error is None
                            else sqlite_support.json_dumps(request.error)
                        ),
                        sqlite_support.format_datetime(now),
                        sqlite_support.format_datetime(now),
                        sqlite_support.format_datetime(now),
                        request.task_id,
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.RUNNING),
                        request.worker_id,
                        request.handoff_id,
                        sqlite_support.format_datetime(now),
                    ),
                )
                if cursor.rowcount != 1:
                    self._raise_task_active_lease_error(
                        request.task_id,
                        request.worker_id,
                        now=now,
                    )
                terminal_task = self._require_task_unlocked(request.task_id)
                self._connection.execute(
                    "INSERT INTO cayu_task_terminalization_receipts "
                    "(task_id, idempotency_key, request_sha256, worker_id, "
                    "terminal_kind, task_json, committed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        request.task_id,
                        request.idempotency_key,
                        request_sha256,
                        request.worker_id,
                        request.kind.value,
                        sqlite_support.json_dumps(terminal_task.model_dump(mode="json")),
                        sqlite_support.format_datetime(now),
                    ),
                )
                self._connection.commit()
                return terminal_task.model_copy(deep=True)
            except BaseException:
                self._connection.rollback()
                raise

    async def load_task_terminalization_receipt(
        self,
        task_id: str,
        idempotency_key: str,
    ) -> TaskTerminalizationReceipt | None:
        task_id, idempotency_key = prepare_task_terminalization_receipt_lookup(
            task_id,
            idempotency_key,
        )
        async with self._lock:
            row = self._connection.execute(
                "SELECT request_sha256, worker_id, terminal_kind, task_json, committed_at "
                "FROM cayu_task_terminalization_receipts "
                "WHERE task_id = ? AND idempotency_key = ?",
                (task_id, idempotency_key),
            ).fetchone()
            if row is None:
                return None
            return _sqlite_task_terminalization_receipt(
                task_id=task_id,
                idempotency_key=idempotency_key,
                row=row,
            )

    async def recover_attached_task_failure(
        self,
        request: TaskTerminalizationRequest,
        *,
        session_id: str,
        session_instance_id: str,
    ) -> Task:
        request, request_sha256 = prepare_task_terminalization(request)
        session_id = require_clean_nonblank(session_id, "session_id")
        session_instance_id = require_clean_nonblank(
            session_instance_id,
            "session_instance_id",
        )
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                receipt_row = self._connection.execute(
                    "SELECT request_sha256, worker_id, terminal_kind, task_json, committed_at "
                    "FROM cayu_task_terminalization_receipts "
                    "WHERE task_id = ? AND idempotency_key = ?",
                    (request.task_id, request.idempotency_key),
                ).fetchone()
                if receipt_row is not None:
                    receipt = _sqlite_task_terminalization_receipt(
                        task_id=request.task_id,
                        idempotency_key=request.idempotency_key,
                        row=receipt_row,
                    )
                    replayed = _replay_task_terminalization_receipt(
                        request=request,
                        request_sha256=request_sha256,
                        receipt=receipt,
                        current_task=self._load_task_unlocked(request.task_id),
                    )
                    _ensure_recovered_attached_task_session(
                        replayed,
                        session_id=session_id,
                        session_instance_id=session_instance_id,
                    )
                    self._connection.commit()
                    return replayed

                task = self._require_task_unlocked(request.task_id)
                self._raise_if_governed_work_attempt_admission(
                    request.task_id,
                    "Admitted work attempts cannot use attached-task recovery terminalization.",
                )
                now = self._ownership_clock()
                _ensure_recovered_attached_task_failure_authority(
                    task,
                    request,
                    session_id=session_id,
                    session_instance_id=session_instance_id,
                    now=now,
                )
                verified_work_support.require_contracted_completion_authority(
                    task,
                    TaskStatus.FAILED,
                )
                terminal_task = self._finish_task_in_transaction_unlocked(
                    request.task_id,
                    TaskStatus.FAILED,
                    result=None,
                    error=request.error,
                    worker_id=None,
                )
                self._connection.execute(
                    "INSERT INTO cayu_task_terminalization_receipts "
                    "(task_id, idempotency_key, request_sha256, worker_id, "
                    "terminal_kind, task_json, committed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        request.task_id,
                        request.idempotency_key,
                        request_sha256,
                        request.worker_id,
                        request.kind.value,
                        sqlite_support.json_dumps(terminal_task.model_dump(mode="json")),
                        sqlite_support.format_datetime(now),
                    ),
                )
                self._connection.commit()
                return terminal_task.model_copy(deep=True)
            except BaseException:
                self._connection.rollback()
                raise

    async def release_interrupted_task_worker(
        self,
        request: TaskInterruptedHandoffRequest,
    ) -> TaskInterruptedHandoffReceipt:
        return await self._settle_interrupted_task_handoff(
            request,
            recover_expired=False,
        )

    async def recover_interrupted_task_worker(
        self,
        request: TaskInterruptedHandoffRequest,
    ) -> TaskInterruptedHandoffReceipt:
        return await self._settle_interrupted_task_handoff(
            request,
            recover_expired=True,
        )

    async def _settle_interrupted_task_handoff(
        self,
        request: TaskInterruptedHandoffRequest,
        *,
        recover_expired: bool,
    ) -> TaskInterruptedHandoffReceipt:
        request, request_sha256 = prepare_interrupted_task_handoff(request)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                receipt_row = self._connection.execute(
                    "SELECT request_sha256, request_json, task_json, committed_at "
                    "FROM cayu_task_interrupted_handoff_receipts "
                    "WHERE task_id = ? AND handoff_id = ?",
                    (request.task_id, request.handoff_id),
                ).fetchone()
                if receipt_row is not None:
                    receipt = _sqlite_interrupted_task_handoff_receipt(
                        task_id=request.task_id,
                        handoff_id=request.handoff_id,
                        row=receipt_row,
                    )
                    replayed = _replay_interrupted_task_handoff_receipt(
                        request=request,
                        request_sha256=request_sha256,
                        receipt=receipt,
                    )
                    self._connection.commit()
                    return replayed

                task = self._require_task_unlocked(request.task_id)
                self._raise_if_governed_work_attempt_admission(
                    request.task_id,
                    "Admitted work attempts do not use interrupted-task handoff release.",
                )
                now = self._ownership_clock()
                _require_interrupted_task_handoff_authority(
                    task,
                    request,
                    now=now,
                    recover_expired=recover_expired,
                )
                lease_comparison = "<=" if recover_expired else ">"
                cursor = self._connection.execute(
                    f"""
                    UPDATE cayu_tasks
                    SET worker_id = NULL, lease_expires_at = NULL,
                        interrupted_handoff_id = ?, updated_at = ?
                    WHERE id = ?
                      AND status = ?
                      AND session_id = ?
                      AND session_instance_id = ?
                      AND worker_id = ?
                      AND lease_expires_at = ?
                      AND lease_expires_at {lease_comparison} ?
                    """,
                    (
                        request.handoff_id,
                        sqlite_support.format_datetime(now),
                        request.task_id,
                        str(TaskStatus.RUNNING),
                        request.session_id,
                        request.session_instance_id,
                        request.worker_id,
                        sqlite_support.format_datetime(request.lease_expires_at),
                        sqlite_support.format_datetime(now),
                    ),
                )
                if cursor.rowcount != 1:
                    raise TaskInterruptedHandoffConflict(
                        "Interrupted-task handoff lost its exact durable authority."
                    )
                released = self._require_task_unlocked(request.task_id)
                receipt = TaskInterruptedHandoffReceipt(
                    request=request,
                    request_sha256=request_sha256,
                    task=released,
                    committed_at=now,
                )
                self._connection.execute(
                    "INSERT INTO cayu_task_interrupted_handoff_receipts "
                    "(task_id, handoff_id, request_sha256, request_json, "
                    "task_json, committed_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        request.task_id,
                        request.handoff_id,
                        request_sha256,
                        sqlite_support.json_dumps(request.model_dump(mode="json")),
                        sqlite_support.json_dumps(released.model_dump(mode="json")),
                        sqlite_support.format_datetime(now),
                    ),
                )
                self._connection.commit()
                return receipt
            except BaseException:
                self._connection.rollback()
                raise

    async def load_interrupted_task_handoff_receipt(
        self,
        task_id: str,
        handoff_id: str,
    ) -> TaskInterruptedHandoffReceipt | None:
        task_id, handoff_id = prepare_interrupted_task_handoff_receipt_lookup(
            task_id,
            handoff_id,
        )
        async with self._lock:
            row = self._connection.execute(
                "SELECT request_sha256, request_json, task_json, committed_at "
                "FROM cayu_task_interrupted_handoff_receipts "
                "WHERE task_id = ? AND handoff_id = ?",
                (task_id, handoff_id),
            ).fetchone()
            return (
                None
                if row is None
                else _sqlite_interrupted_task_handoff_receipt(
                    task_id=task_id,
                    handoff_id=handoff_id,
                    row=row,
                )
            )

    async def list_expired_interrupted_task_handoff_candidates(
        self,
        *,
        after: tuple[datetime, str] | None = None,
        limit: int = 100,
    ) -> list[Task]:
        after, limit = prepare_interrupted_task_handoff_candidate_page(
            after=after,
            limit=limit,
        )
        after_clause = ""
        after_params: tuple[str, ...] = ()
        if after is not None:
            after_timestamp = sqlite_support.format_datetime(after[0])
            after_clause = "AND (lease_expires_at > ? OR (lease_expires_at = ? AND id > ?)) "
            after_params = (after_timestamp, after_timestamp, after[1])
        async with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM cayu_tasks WHERE status = ? "
                "AND session_id IS NOT NULL AND session_instance_id IS NOT NULL "
                "AND worker_id IS NOT NULL AND lease_expires_at IS NOT NULL "
                "AND lease_expires_at <= ? AND status_reason IS NULL "
                "AND NOT EXISTS ("
                "SELECT 1 FROM cayu_work_attempt_admissions "
                "WHERE cayu_work_attempt_admissions.task_id = cayu_tasks.id"
                ") "
                f"{after_clause}"
                "ORDER BY lease_expires_at ASC, id ASC LIMIT ?",
                (
                    str(TaskStatus.RUNNING),
                    sqlite_support.format_datetime(self._ownership_clock()),
                    *after_params,
                    limit,
                ),
            ).fetchall()
            return [sqlite_support.task_from_row(row) for row in rows]

    async def claim_interrupted_task_continuation(
        self,
        worker_id: str,
        query: TaskQuery | None = None,
        *,
        handoff_id: str,
        lease_seconds: int = 300,
        after: tuple[datetime, str] | None = None,
        scan_limit: int = _TASK_INTERRUPTED_HANDOFF_RECOVERY_MAX_PAGE_SIZE,
    ) -> InterruptedTaskContinuationClaimPage:
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        handoff_id = require_clean_nonblank(handoff_id, "handoff_id")
        handoff_id_sha256 = _interrupted_task_continuation_handoff_id_sha256(handoff_id)
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        lease_seconds = _validate_task_positive_int(lease_seconds, "lease_seconds")
        after, scan_limit = prepare_interrupted_task_continuation_claim_page(
            after=after,
            limit=scan_limit,
        )
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                now = self._ownership_clock()
                lease_expires_at = now + timedelta(seconds=lease_seconds)
                prior_claim_row = self._connection.execute(
                    "SELECT task_id, worker_id "
                    "FROM cayu_task_interrupted_continuation_claims "
                    "WHERE handoff_id_sha256 = ?",
                    (handoff_id_sha256,),
                ).fetchone()
                if prior_claim_row is not None:
                    existing_row = self._connection.execute(
                        "SELECT * FROM cayu_tasks WHERE id = ?",
                        (prior_claim_row["task_id"],),
                    ).fetchone()
                    existing = (
                        None if existing_row is None else sqlite_support.task_from_row(existing_row)
                    )
                    if (
                        prior_claim_row["worker_id"] != worker_id
                        or existing is None
                        or existing.interrupted_handoff_id != handoff_id
                        or existing.worker_id != worker_id
                        or existing.status is not TaskStatus.RUNNING
                        or existing.session_id is None
                        or existing.session_instance_id is None
                        or existing.lease_expires_at is None
                        or existing.lease_expires_at <= now
                        or not _task_matches_claim_filter(existing, query)
                    ):
                        raise TaskClaimLost(
                            "Interrupted-task continuation claim generation is no longer live."
                        )
                    result = InterruptedTaskContinuationClaimPage(
                        task=existing,
                        next_after=(existing.created_at, existing.id),
                        scanned_candidates=0,
                        rejected_candidates=0,
                        replayed=True,
                        exhausted=False,
                    )
                    self._connection.commit()
                    return result
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_tasks WHERE interrupted_handoff_id = ? LIMIT 1",
                        (handoff_id,),
                    ).fetchone()
                    is not None
                ):
                    raise TaskClaimLost(
                        "Interrupted-task continuation claim generation is already in use."
                    )
                if query.status is not None and query.status is not TaskStatus.RUNNING:
                    result = InterruptedTaskContinuationClaimPage(
                        scanned_candidates=0,
                        rejected_candidates=0,
                        exhausted=True,
                    )
                    self._connection.commit()
                    return result
                cursor = (
                    None if after is None else (sqlite_support.format_datetime(after[0]), after[1])
                )
                after_sql = ""
                after_params: tuple[str, ...] = ()
                if cursor is not None:
                    after_sql = "AND (created_at > ? OR (created_at = ? AND id > ?)) "
                    after_params = (cursor[0], cursor[0], cursor[1])
                rows = self._connection.execute(
                    "SELECT * FROM cayu_tasks WHERE status = ? "
                    "AND session_id IS NOT NULL AND session_instance_id IS NOT NULL "
                    "AND status_reason IS NULL "
                    "AND worker_id IS NULL AND lease_expires_at IS NULL "
                    "AND interrupted_handoff_id IS NOT NULL "
                    f"{after_sql}"
                    "ORDER BY created_at ASC, id ASC LIMIT ?",
                    (
                        str(TaskStatus.RUNNING),
                        *after_params,
                        scan_limit,
                    ),
                ).fetchall()
                rejected = 0
                filtered = 0
                last_observed: Task | None = None
                for index, row in enumerate(rows):
                    observed = sqlite_support.task_from_row(row)
                    last_observed = observed
                    if not _task_matches_claim_filter(observed, query):
                        filtered += 1
                        continue
                    candidate_handoff_id = observed.interrupted_handoff_id
                    if candidate_handoff_id is None:
                        raise AssertionError("Continuation candidate lost its handoff generation.")
                    if (
                        self._connection.execute(
                            "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                            (observed.id,),
                        ).fetchone()
                        is not None
                    ):
                        rejected += 1
                        continue
                    receipt_row = self._connection.execute(
                        "SELECT request_sha256, request_json, task_json, committed_at "
                        "FROM cayu_task_interrupted_handoff_receipts "
                        "WHERE task_id = ? AND handoff_id = ?",
                        (observed.id, candidate_handoff_id),
                    ).fetchone()
                    try:
                        receipt = (
                            None
                            if receipt_row is None
                            else _sqlite_interrupted_task_handoff_receipt(
                                task_id=observed.id,
                                handoff_id=candidate_handoff_id,
                                row=receipt_row,
                            )
                        )
                    except TaskInterruptedHandoffConflict:
                        receipt = None
                    if receipt is None or receipt.task != observed:
                        rejected += 1
                        continue
                    self._connection.execute(
                        "INSERT INTO cayu_task_interrupted_continuation_claims ("
                        "handoff_id_sha256, task_id, worker_id, claimed_at"
                        ") VALUES (?, ?, ?, ?)",
                        (
                            handoff_id_sha256,
                            observed.id,
                            worker_id,
                            sqlite_support.format_datetime(now),
                        ),
                    )
                    update_cursor = self._connection.execute(
                        "UPDATE cayu_tasks SET worker_id = ?, lease_expires_at = ?, "
                        "interrupted_handoff_id = ?, updated_at = ? "
                        "WHERE id = ? AND status = ? AND worker_id IS NULL "
                        "AND lease_expires_at IS NULL AND interrupted_handoff_id = ?",
                        (
                            worker_id,
                            sqlite_support.format_datetime(lease_expires_at),
                            handoff_id,
                            sqlite_support.format_datetime(now),
                            observed.id,
                            str(TaskStatus.RUNNING),
                            observed.interrupted_handoff_id,
                        ),
                    )
                    if update_cursor.rowcount != 1:
                        raise RuntimeError("SQLite continuation claim lost its locked candidate.")
                    claimed = self._require_task_unlocked(observed.id)
                    result = InterruptedTaskContinuationClaimPage(
                        task=claimed,
                        next_after=(observed.created_at, observed.id),
                        scanned_candidates=index + 1,
                        rejected_candidates=rejected,
                        filtered_candidates=filtered,
                        exhausted=index == len(rows) - 1 and len(rows) < scan_limit,
                    )
                    self._connection.commit()
                    return result
                result = InterruptedTaskContinuationClaimPage(
                    next_after=(
                        (last_observed.created_at, last_observed.id)
                        if last_observed is not None
                        else None
                    ),
                    scanned_candidates=len(rows),
                    rejected_candidates=rejected,
                    filtered_candidates=filtered,
                    exhausted=len(rows) < scan_limit,
                )
                self._connection.commit()
                return result
            except BaseException:
                self._connection.rollback()
                raise

    async def reconcile_task_cancellation(
        self,
        request: TaskCancellationReconciliationRequest,
    ) -> TaskCancellationReconciliationResult:
        request, request_sha256 = prepare_task_cancellation_reconciliation(request)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                now = self._ownership_clock()
                rejection_row = self._connection.execute(
                    "SELECT request_sha256, record_json "
                    "FROM cayu_task_retry_reconciliation_rejections "
                    "WHERE task_id = ? AND reconciliation_idempotency_key = ?",
                    (request.task_id, request.reconciliation_idempotency_key),
                ).fetchone()
                if rejection_row is not None:
                    rejection = _TaskCancellationReconciliationRejectionRecord.model_validate(
                        json.loads(rejection_row["record_json"])
                    )
                    if rejection.request_sha256 != rejection_row["request_sha256"]:
                        raise RuntimeError(
                            "SQLite cancellation reconciliation rejection contains invalid "
                            "durable material."
                        )
                    raise _replay_task_cancellation_reconciliation_rejection(
                        request,
                        request_sha256=request_sha256,
                        record=rejection,
                    )

                receipt_row = self._connection.execute(
                    "SELECT request_sha256, worker_id, terminal_kind, task_json, committed_at "
                    "FROM cayu_task_terminalization_receipts "
                    "WHERE task_id = ? AND idempotency_key = ?",
                    (request.task_id, request.cancellation_idempotency_key),
                ).fetchone()
                if receipt_row is not None:
                    receipt = _sqlite_task_terminalization_receipt(
                        task_id=request.task_id,
                        idempotency_key=request.cancellation_idempotency_key,
                        row=receipt_row,
                    )
                    replayed = _replay_task_cancellation_reconciliation(
                        request=request,
                        request_sha256=request_sha256,
                        receipt=receipt,
                        current_task=self._load_task_unlocked(request.task_id),
                    )
                    self._connection.commit()
                    return replayed

                task = self._load_task_unlocked(request.task_id)
                if task is None:
                    raise _task_cancellation_reconciliation_conflict(
                        request,
                        "Task cancellation reconciliation task was not found.",
                    )
                self._raise_if_governed_work_attempt_admission(
                    request.task_id,
                    "Admitted work attempts cannot use ordinary cancellation reconciliation.",
                )
                rejection = _task_cancellation_reconciliation_rejection_record(
                    request,
                    request_sha256=request_sha256,
                    recorded_at=now,
                )
                if rejection is not None:
                    _validated_task_cancellation(
                        task,
                        request,
                        now=now,
                        require_owner_lost=False,
                    )
                    self._connection.execute(
                        "INSERT INTO cayu_task_retry_reconciliation_rejections "
                        "(task_id, reconciliation_idempotency_key, request_sha256, "
                        "record_json, recorded_at) VALUES (?, ?, ?, ?, ?)",
                        (
                            rejection.task_id,
                            rejection.reconciliation_idempotency_key,
                            rejection.request_sha256,
                            sqlite_support.json_dumps(rejection.model_dump(mode="json")),
                            sqlite_support.format_datetime(rejection.recorded_at),
                        ),
                    )
                    self._connection.commit()
                    raise _rejected_task_cancellation_reconciliation(rejection)

                result = _reconciled_task_cancellation(
                    task,
                    request,
                    request_sha256=request_sha256,
                    committed_at=now,
                )
                settled = result.task
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?, status_reason = NULL, status_payload_json = ?,
                        result_json = NULL, error_json = ?, worker_id = NULL,
                        lease_expires_at = NULL, interrupted_handoff_id = NULL,
                        started_at = ?, completed_at = ?, updated_at = ?
                    WHERE id = ? AND status IN (?, ?) AND status_reason = ?
                      AND worker_id = ? AND lease_expires_at = ?
                      AND lease_expires_at <= ? AND retry_series_json IS NULL
                    """,
                    (
                        str(settled.status),
                        sqlite_support.json_dumps(settled.status_payload),
                        sqlite_support.json_dumps(settled.error),
                        sqlite_support.format_optional_datetime(settled.started_at),
                        sqlite_support.format_optional_datetime(settled.completed_at),
                        sqlite_support.format_datetime(settled.updated_at),
                        request.task_id,
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.RUNNING),
                        request.expected_status_reason,
                        request.original_worker_id,
                        sqlite_support.format_datetime(request.original_lease_expires_at),
                        sqlite_support.format_datetime(now),
                    ),
                )
                if cursor.rowcount != 1:
                    raise _task_cancellation_reconciliation_conflict(
                        request,
                        "Task cancellation reconciliation lost its fenced transition.",
                    )
                durable_task = self._require_task_unlocked(request.task_id)
                receipt = result.terminalization_receipt.model_copy(
                    update={"task": durable_task},
                    deep=True,
                )
                durable_result = TaskCancellationReconciliationResult(
                    request_sha256=request_sha256,
                    task=durable_task,
                    terminalization_receipt=receipt,
                    reconciliation=result.reconciliation,
                    committed_at=now,
                )
                self._connection.execute(
                    "INSERT INTO cayu_task_terminalization_receipts "
                    "(task_id, idempotency_key, request_sha256, worker_id, "
                    "terminal_kind, task_json, committed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        receipt.task_id,
                        receipt.idempotency_key,
                        receipt.request_sha256,
                        receipt.worker_id,
                        receipt.kind.value,
                        sqlite_support.json_dumps(receipt.task.model_dump(mode="json")),
                        sqlite_support.format_datetime(receipt.committed_at),
                    ),
                )
                self._connection.commit()
                return _copy_task_cancellation_reconciliation_result(durable_result)
            except BaseException:
                self._connection.rollback()
                raise

    async def settle_task_retry_attempt(
        self,
        request: TaskRetrySettlementRequest,
    ) -> TaskRetrySettlementResult:
        request, request_sha256 = prepare_task_retry_settlement(request)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    "SELECT request_sha256, receipt_json "
                    "FROM cayu_task_retry_settlements "
                    "WHERE task_id = ? AND idempotency_key = ?",
                    (request.task_id, request.idempotency_key),
                ).fetchone()
                if row is not None:
                    receipt = TaskRetrySettlementResult.model_validate(
                        json.loads(row["receipt_json"])
                    )
                    replayed = _replay_task_retry_settlement(
                        request=request,
                        request_sha256=request_sha256,
                        receipt=receipt,
                        current_task=self._load_task_unlocked(request.task_id),
                    )
                    self._connection.commit()
                    return replayed

                task = self._require_task_unlocked(request.task_id)
                now = self._ownership_clock()
                if request.lease_expires_at is None:
                    raise TaskClaimLost("Task retry settlement requires its exact worker lease.")
                _ensure_exact_owned_active_task_lease(
                    task,
                    request.worker_id,
                    request.lease_expires_at,
                    now=now,
                )
                series_now = self._clock()
                settled, successor = _settled_task_retry_attempt(
                    task,
                    request,
                    now=now,
                    series_now=series_now,
                )
                assert settled.retry_series is not None
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?, status_reason = ?, status_payload_json = ?,
                        result_json = ?, error_json = ?, worker_id = NULL,
                        lease_expires_at = NULL, started_at = ?, completed_at = ?,
                        updated_at = ?, retry_series_json = ?
                    WHERE id = ? AND status IN (?, ?) AND worker_id = ?
                      AND lease_expires_at IS NOT NULL AND lease_expires_at > ?
                    """,
                    (
                        str(settled.status),
                        settled.status_reason,
                        sqlite_support.json_dumps(settled.status_payload),
                        (
                            None
                            if settled.result is None
                            else sqlite_support.json_dumps(settled.result)
                        ),
                        (
                            None
                            if settled.error is None
                            else sqlite_support.json_dumps(settled.error)
                        ),
                        sqlite_support.format_optional_datetime(settled.started_at),
                        sqlite_support.format_optional_datetime(settled.completed_at),
                        sqlite_support.format_datetime(settled.updated_at),
                        sqlite_support.json_dumps(settled.retry_series.model_dump(mode="json")),
                        request.task_id,
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.RUNNING),
                        request.worker_id,
                        sqlite_support.format_datetime(now),
                    ),
                )
                if cursor.rowcount != 1:
                    self._raise_task_active_lease_error(
                        request.task_id,
                        request.worker_id,
                        now=now,
                    )
                if successor is not None:
                    self._connection.execute(
                        """
                        INSERT INTO cayu_tasks (
                            id, type, title, description, status, session_id,
                            session_instance_id, parent_task_id, assigned_agent_name, available_at, worker_id,
                            lease_expires_at, interrupted_handoff_id, status_reason,
                            status_payload_json, input_json,
                            result_json, error_json, metadata_json, created_at, updated_at,
                            started_at, completed_at, invocation_json, retry_series_json,
                            work_contract_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        sqlite_support.task_to_row_values(successor),
                    )
                receipt = TaskRetrySettlementResult(
                    task_id=request.task_id,
                    idempotency_key=request.idempotency_key,
                    request_sha256=request_sha256,
                    task=settled,
                    successor=successor,
                    events=_task_retry_events(settled, occurred_at=now),
                    committed_at=now,
                )
                self._connection.execute(
                    "INSERT INTO cayu_task_retry_settlements "
                    "(task_id, idempotency_key, request_sha256, receipt_json, committed_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        request.task_id,
                        request.idempotency_key,
                        request_sha256,
                        sqlite_support.json_dumps(receipt.model_dump(mode="json")),
                        sqlite_support.format_datetime(receipt.committed_at),
                    ),
                )
                self._connection.commit()
                committed = receipt.model_copy(deep=True)
            except BaseException:
                self._connection.rollback()
                raise
        if successor is not None:
            self._publish_task_admission_wakeup(successor, now=series_now)
        return committed

    async def load_task_retry_settlement(
        self,
        task_id: str,
        idempotency_key: str,
    ) -> TaskRetrySettlementResult | None:
        task_id, idempotency_key = prepare_task_terminalization_receipt_lookup(
            task_id,
            idempotency_key,
        )
        async with self._lock:
            row = self._connection.execute(
                "SELECT receipt_json FROM cayu_task_retry_settlements "
                "WHERE task_id = ? AND idempotency_key = ?",
                (task_id, idempotency_key),
            ).fetchone()
            if row is None:
                return None
            return TaskRetrySettlementResult.model_validate(json.loads(row["receipt_json"]))

    async def reconcile_task_retry_cancellation(
        self,
        request: TaskRetryCancellationReconciliationRequest,
    ) -> TaskRetrySettlementResult:
        request, request_sha256 = prepare_task_retry_cancellation_reconciliation(request)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                now = self._ownership_clock()
                rejection_row = self._connection.execute(
                    "SELECT request_sha256, record_json "
                    "FROM cayu_task_retry_reconciliation_rejections "
                    "WHERE task_id = ? AND reconciliation_idempotency_key = ?",
                    (request.task_id, request.reconciliation_idempotency_key),
                ).fetchone()
                if rejection_row is not None:
                    rejection = _TaskRetryCancellationReconciliationRejectionRecord.model_validate(
                        json.loads(rejection_row["record_json"])
                    )
                    if rejection.request_sha256 != rejection_row["request_sha256"]:
                        raise RuntimeError(
                            "SQLite retry reconciliation rejection contains invalid "
                            "durable material."
                        )
                    raise _replay_task_retry_cancellation_reconciliation_rejection(
                        request,
                        request_sha256=request_sha256,
                        record=rejection,
                    )
                row = self._connection.execute(
                    "SELECT request_sha256, receipt_json "
                    "FROM cayu_task_retry_settlements "
                    "WHERE task_id = ? AND idempotency_key = ?",
                    (request.task_id, request.cancellation_idempotency_key),
                ).fetchone()
                if row is not None:
                    receipt = TaskRetrySettlementResult.model_validate(
                        json.loads(row["receipt_json"])
                    )
                    replayed = _replay_task_retry_cancellation_reconciliation(
                        request=request,
                        request_sha256=request_sha256,
                        receipt=receipt,
                        current_task=self._load_task_unlocked(request.task_id),
                    )
                    self._connection.commit()
                    return replayed

                task = self._load_task_unlocked(request.task_id)
                if task is None:
                    raise _task_retry_cancellation_reconciliation_conflict(
                        request,
                        "Task retry cancellation reconciliation task was not found.",
                    )
                rejection = _task_retry_cancellation_reconciliation_rejection_record(
                    request,
                    request_sha256=request_sha256,
                    recorded_at=now,
                )
                if rejection is not None:
                    _validated_task_retry_cancellation(
                        task,
                        request,
                        now=now,
                        require_owner_lost=False,
                    )
                    self._connection.execute(
                        "INSERT INTO cayu_task_retry_reconciliation_rejections "
                        "(task_id, reconciliation_idempotency_key, request_sha256, "
                        "record_json, recorded_at) VALUES (?, ?, ?, ?, ?)",
                        (
                            rejection.task_id,
                            rejection.reconciliation_idempotency_key,
                            rejection.request_sha256,
                            sqlite_support.json_dumps(rejection.model_dump(mode="json")),
                            sqlite_support.format_datetime(rejection.recorded_at),
                        ),
                    )
                    self._connection.commit()
                    raise _rejected_task_retry_cancellation_reconciliation(rejection)
                receipt = _reconciled_task_retry_cancellation(
                    task,
                    request,
                    request_sha256=request_sha256,
                    committed_at=now,
                )
                settled = receipt.task
                assert settled.retry_series is not None
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?, status_reason = ?, status_payload_json = ?,
                        result_json = NULL, error_json = ?, worker_id = NULL,
                        lease_expires_at = NULL, started_at = ?, completed_at = ?,
                        updated_at = ?, retry_series_json = ?
                    WHERE id = ? AND status IN (?, ?) AND status_reason = ?
                      AND worker_id = ? AND lease_expires_at = ?
                      AND lease_expires_at <= ?
                    """,
                    (
                        str(settled.status),
                        settled.status_reason,
                        sqlite_support.json_dumps(settled.status_payload),
                        sqlite_support.json_dumps(settled.error),
                        sqlite_support.format_optional_datetime(settled.started_at),
                        sqlite_support.format_optional_datetime(settled.completed_at),
                        sqlite_support.format_datetime(settled.updated_at),
                        sqlite_support.json_dumps(settled.retry_series.model_dump(mode="json")),
                        request.task_id,
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.RUNNING),
                        request.expected_status_reason,
                        request.original_worker_id,
                        sqlite_support.format_datetime(request.original_lease_expires_at),
                        sqlite_support.format_datetime(now),
                    ),
                )
                if cursor.rowcount != 1:
                    raise _task_retry_cancellation_reconciliation_conflict(
                        request,
                        "Task retry cancellation reconciliation lost its fenced transition.",
                    )
                self._connection.execute(
                    "INSERT INTO cayu_task_retry_settlements "
                    "(task_id, idempotency_key, request_sha256, receipt_json, committed_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        request.task_id,
                        request.cancellation_idempotency_key,
                        request_sha256,
                        sqlite_support.json_dumps(receipt.model_dump(mode="json")),
                        sqlite_support.format_datetime(receipt.committed_at),
                    ),
                )
                self._connection.commit()
                return receipt.model_copy(deep=True)
            except BaseException:
                self._connection.rollback()
                raise

    async def enforce_task_retry_deadline(
        self,
        task_id: str,
        worker_id: str,
        *,
        lease_expires_at: datetime,
        token_count: int = 0,
        estimated_cost: Decimal = Decimal(0),
    ) -> TaskRetrySettlementResult | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        expected_lease = normalize_utc_datetime(lease_expires_at, "lease_expires_at")
        token_count, estimated_cost = _validated_task_retry_terminal_accounting(
            token_count=token_count,
            estimated_cost=estimated_cost,
        )
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                task = self._require_task_unlocked(task_id)
                lease_now = self._ownership_clock()
                _ensure_exact_owned_active_task_lease(
                    task,
                    worker_id,
                    expected_lease,
                    now=lease_now,
                )
                if not _claimed_task_retry_attempt_elapsed(task, series_now=self._clock()):
                    self._connection.commit()
                    return None
                receipt = _elapsed_claimed_task_retry_settlement(
                    task,
                    committed_at=lease_now,
                    token_count=token_count,
                    estimated_cost=estimated_cost,
                )
                settled = receipt.task
                assert settled.retry_series is not None
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?, status_reason = ?, status_payload_json = ?,
                        result_json = NULL, error_json = ?, worker_id = NULL,
                        lease_expires_at = NULL, started_at = ?, completed_at = ?,
                        updated_at = ?, retry_series_json = ?
                    WHERE id = ? AND status IN (?, ?) AND worker_id = ?
                      AND lease_expires_at = ? AND lease_expires_at > ?
                    """,
                    (
                        str(settled.status),
                        settled.status_reason,
                        sqlite_support.json_dumps(settled.status_payload),
                        sqlite_support.json_dumps(settled.error),
                        sqlite_support.format_optional_datetime(settled.started_at),
                        sqlite_support.format_optional_datetime(settled.completed_at),
                        sqlite_support.format_datetime(settled.updated_at),
                        sqlite_support.json_dumps(settled.retry_series.model_dump(mode="json")),
                        task_id,
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.RUNNING),
                        worker_id,
                        sqlite_support.format_datetime(expected_lease),
                        sqlite_support.format_datetime(lease_now),
                    ),
                )
                if cursor.rowcount != 1:
                    self._raise_task_active_lease_error(task_id, worker_id, now=lease_now)
                self._connection.execute(
                    "INSERT INTO cayu_task_retry_settlements "
                    "(task_id, idempotency_key, request_sha256, receipt_json, committed_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        receipt.task_id,
                        receipt.idempotency_key,
                        receipt.request_sha256,
                        sqlite_support.json_dumps(receipt.model_dump(mode="json")),
                        sqlite_support.format_datetime(receipt.committed_at),
                    ),
                )
                self._connection.commit()
                return receipt.model_copy(deep=True)
            except BaseException:
                self._connection.rollback()
                raise

    async def task_retry_deadline_elapsed(
        self,
        task_id: str,
        worker_id: str,
        *,
        lease_expires_at: datetime,
    ) -> bool:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        expected_lease = normalize_utc_datetime(lease_expires_at, "lease_expires_at")
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                task = self._require_task_unlocked(task_id)
                _ensure_exact_owned_active_task_lease(
                    task,
                    worker_id,
                    expected_lease,
                    now=self._ownership_clock(),
                )
                elapsed = _claimed_task_retry_attempt_elapsed(
                    task,
                    series_now=self._clock(),
                )
                self._connection.commit()
                return elapsed
            except BaseException:
                self._connection.rollback()
                raise

    async def cancel_task(
        self,
        task_id: str,
        error: dict[str, Any] | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        copied_error = None if error is None else copy_durable_json_object(error, "error")
        async with self._lock:
            return self._finish_task_unlocked(
                task_id,
                TaskStatus.CANCELLED,
                result=None,
                error=copied_error,
            )

    async def request_claimed_task_cancellation(
        self,
        task_id: str,
        worker_id: str,
        lease_expires_at: datetime,
        error: dict[str, Any] | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        expected_lease = normalize_utc_datetime(lease_expires_at, "lease_expires_at")
        copied_error = None if error is None else copy_durable_json_object(error, "error")
        async with self._lock:
            with self._verified_transaction_unlocked():
                current = self._require_task_unlocked(task_id)
                if current.worker_id != worker_id or current.lease_expires_at != expected_lease:
                    raise TaskClaimLost(
                        "Claimed-task cancellation no longer owns the expected worker lease."
                    )
                if _task_cancellation_requested(current) or (
                    current.status_reason == _TASK_RETRY_CANCELLATION_REQUESTED_REASON
                ):
                    return current.model_copy(deep=True)
                if current.started_at is not None:
                    requested = _expired_dispatched_task_cancellation(
                        current,
                        updated_at=self._ownership_clock(),
                        error=copied_error,
                    )
                    self._update_task_snapshot_unlocked(requested)
                    return requested.model_copy(deep=True)
                return self._finish_task_in_transaction_unlocked(
                    task_id,
                    TaskStatus.CANCELLED,
                    result=None,
                    error=copied_error,
                    worker_id=worker_id,
                    handoff_id=current.interrupted_handoff_id,
                    expected_lease_expires_at=expected_lease,
                )

    async def mark_claimed_task_execution_started(
        self,
        task_id: str,
        worker_id: str,
        lease_expires_at: datetime,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        expected_lease = normalize_utc_datetime(lease_expires_at, "lease_expires_at")
        async with self._lock:
            with self._verified_transaction_unlocked():
                now = self._ownership_clock()
                current = self._require_task_unlocked(task_id)
                if current.worker_id != worker_id or current.lease_expires_at != expected_lease:
                    raise TaskClaimLost(
                        "Claimed-task execution no longer owns the expected worker lease."
                    )
                _ensure_owned_active_task_lease(current, worker_id, now=now)
                if (
                    current.status is not TaskStatus.CLAIMED
                    or current.session_id is not None
                    or _task_cancellation_requested(current)
                    or current.status_reason == _TASK_RETRY_CANCELLATION_REQUESTED_REASON
                ):
                    raise TaskTerminalizationConflict(
                        "Claimed task cannot begin ordinary worker execution."
                    )
                if current.started_at is not None:
                    return current.model_copy(deep=True)
                started = current.model_copy(update={"started_at": now, "updated_at": now})
                self._update_task_snapshot_unlocked(started)
                return started.model_copy(deep=True)

    async def pause_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        return await self._hold_task(
            task_id,
            TaskStatus.PAUSED,
            reason=reason,
            payload=payload,
        )

    async def block_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        return await self._hold_task(
            task_id,
            TaskStatus.BLOCKED,
            reason=reason,
            payload=payload,
        )

    async def mark_task_needs_attention(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        return await self._hold_task(
            task_id,
            TaskStatus.NEEDS_ATTENTION,
            reason=reason,
            payload=payload,
        )

    async def resume_task(self, task_id: str) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        async with self._lock:
            with self._verified_transaction_unlocked():
                now = self._ownership_clock()
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                        (task_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Admitted work attempts cannot use ordinary task resumption."
                    )
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?,
                        status_reason = NULL,
                        status_payload_json = NULL,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?
                    WHERE id = ?
                      AND status IN (?, ?, ?)
                      AND NOT EXISTS (
                          SELECT 1 FROM cayu_work_attempt_admissions
                          WHERE task_id = ?
                      )
                    """,
                    (
                        str(TaskStatus.PENDING),
                        sqlite_support.format_datetime(now),
                        task_id,
                        str(TaskStatus.PAUSED),
                        str(TaskStatus.BLOCKED),
                        str(TaskStatus.NEEDS_ATTENTION),
                        task_id,
                    ),
                )
            if cursor.rowcount != 1:
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                        (task_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Admitted work attempts cannot use ordinary task resumption."
                    )
                task = self._require_task_unlocked(task_id)
                _ensure_can_resume_task(task)
                raise ValueError(f"Task {task.id} cannot resume from {task.status}")
            updated = self._require_task_unlocked(task_id)
            return updated.model_copy(deep=True)

    async def claim_task(
        self,
        worker_id: str,
        query: TaskQuery | None = None,
        *,
        lease_seconds: int = 300,
    ) -> Task | None:
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        retry_worker_id_is_bounded = _task_retry_reconciliation_identity_is_bounded(worker_id)
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        lease_seconds = _validate_task_positive_int(lease_seconds, "lease_seconds")
        if query.status is not None and query.status is not TaskStatus.PENDING:
            return None
        clauses, params = self._task_filter_clauses(query)
        retry_deadline_clause = (
            (
                "(retry_series_json IS NULL "
                "OR json_extract(retry_series_json, '$.elapsed_deadline') IS NULL "
                "OR julianday(json_extract(retry_series_json, '$.elapsed_deadline')) "
                "> julianday(?))"
            )
            if retry_worker_id_is_bounded
            else "retry_series_json IS NULL"
        )
        where_sql = " AND ".join(
            [
                "status = ?",
                "session_id IS NULL",
                "(available_at IS NULL OR available_at <= ?)",
                retry_deadline_clause,
                "NOT EXISTS (SELECT 1 FROM cayu_local_execution_attempts AS attempt "
                "WHERE attempt.retry_admissible = 0 AND ("
                "attempt.task_id = cayu_tasks.id OR ("
                "cayu_tasks.retry_series_json IS NOT NULL AND "
                "attempt.retry_series_id = json_extract("
                "cayu_tasks.retry_series_json, '$.series_id'))))",
                *clauses,
            ]
        )
        # Claiming is always FIFO by creation time, independent of the query's
        # display ordering, so the oldest pending task is dispatched first.
        order_sql = sqlite_support.task_order_sql(TaskOrder.CREATED_AT_ASC)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                availability_now = self._clock()
                now = self._ownership_clock()
                lease_expires_at = now + timedelta(seconds=lease_seconds)
                expired_rows = self._connection.execute(
                    """
                    SELECT id
                    FROM cayu_tasks
                    WHERE status = ?
                      AND session_id IS NULL
                      AND retry_series_json IS NOT NULL
                      AND json_extract(retry_series_json, '$.disposition') = ?
                      AND json_extract(retry_series_json, '$.elapsed_deadline') IS NOT NULL
                      AND julianday(json_extract(retry_series_json, '$.elapsed_deadline'))
                          <= julianday(?)
                      AND NOT EXISTS (
                          SELECT 1 FROM cayu_local_execution_attempts AS attempt
                          WHERE attempt.retry_admissible = 0 AND (
                              attempt.task_id = cayu_tasks.id OR
                              attempt.retry_series_id = json_extract(
                                  cayu_tasks.retry_series_json, '$.series_id'
                              )
                          )
                      )
                    ORDER BY created_at ASC, id ASC
                    LIMIT 100
                    """,
                    (
                        str(TaskStatus.PENDING),
                        str(TaskRetrySeriesDisposition.ACTIVE),
                        sqlite_support.format_datetime(availability_now),
                    ),
                ).fetchall()
                for expired_row in expired_rows:
                    expired_task = self._require_task_unlocked(expired_row["id"])
                    expiration = _expired_task_retry_settlement(
                        expired_task,
                        committed_at=now,
                        series_now=availability_now,
                    )
                    assert expiration.task.retry_series is not None
                    cursor = self._connection.execute(
                        """
                        UPDATE cayu_tasks
                        SET status = ?, status_reason = ?, status_payload_json = ?,
                            result_json = NULL, error_json = ?, worker_id = NULL,
                            lease_expires_at = NULL, started_at = ?, completed_at = ?,
                            updated_at = ?, retry_series_json = ?
                        WHERE id = ? AND status = ? AND session_id IS NULL
                        """,
                        (
                            str(expiration.task.status),
                            expiration.task.status_reason,
                            sqlite_support.json_dumps(expiration.task.status_payload),
                            sqlite_support.json_dumps(expiration.task.error),
                            sqlite_support.format_optional_datetime(expiration.task.started_at),
                            sqlite_support.format_optional_datetime(expiration.task.completed_at),
                            sqlite_support.format_datetime(expiration.task.updated_at),
                            sqlite_support.json_dumps(
                                expiration.task.retry_series.model_dump(mode="json")
                            ),
                            expiration.task_id,
                            str(TaskStatus.PENDING),
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise TaskTerminalizationConflict(
                            "Elapsed task retry attempt changed during claim admission."
                        )
                    self._connection.execute(
                        "INSERT INTO cayu_task_retry_settlements "
                        "(task_id, idempotency_key, request_sha256, receipt_json, committed_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            expiration.task_id,
                            expiration.idempotency_key,
                            expiration.request_sha256,
                            sqlite_support.json_dumps(expiration.model_dump(mode="json")),
                            sqlite_support.format_datetime(expiration.committed_at),
                        ),
                    )
                row = self._connection.execute(
                    f"""
                    SELECT id
                    FROM cayu_tasks
                    WHERE {where_sql}
                    ORDER BY {order_sql}, id ASC
                    LIMIT 1
                    """,
                    [
                        str(TaskStatus.PENDING),
                        sqlite_support.format_datetime(availability_now),
                        *(
                            [sqlite_support.format_datetime(availability_now)]
                            if retry_worker_id_is_bounded
                            else []
                        ),
                        *params,
                    ],
                ).fetchone()
                if row is None:
                    self._connection.commit()
                    return None
                task_id = row["id"]
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?,
                        worker_id = ?,
                        lease_expires_at = ?,
                        updated_at = ?
                    WHERE id = ? AND status = ?
                    """,
                    (
                        str(TaskStatus.CLAIMED),
                        worker_id,
                        sqlite_support.format_datetime(lease_expires_at),
                        sqlite_support.format_datetime(now),
                        task_id,
                        str(TaskStatus.PENDING),
                    ),
                )
                if cursor.rowcount != 1:
                    self._connection.rollback()
                    return None
                updated = self._require_task_unlocked(task_id)
                self._connection.commit()
                return updated.model_copy(deep=True)
            except Exception:
                self._connection.rollback()
                raise

    async def heartbeat(
        self,
        task_id: str,
        worker_id: str,
        *,
        lease_expires_at: datetime,
        handoff_id: str | None = None,
        extend_seconds: int = 300,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        expected_lease = normalize_utc_datetime(lease_expires_at, "lease_expires_at")
        extend_seconds = _validate_task_positive_int(extend_seconds, "extend_seconds")
        async with self._lock:
            with self._verified_transaction_unlocked():
                now = self._ownership_clock()
                lease_expires_at = now + timedelta(seconds=extend_seconds)
                task = self._require_task_unlocked(task_id)
                _ensure_owned_active_task_lease(task, worker_id, now=now)
                if task.lease_expires_at != expected_lease:
                    raise TaskClaimLost("Task heartbeat no longer owns the expected worker lease.")
                _ensure_task_handoff_authority(task, handoff_id)
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                        (task_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Admitted work attempts require claim-fenced lease renewal."
                    )
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET lease_expires_at = ?,
                        updated_at = ?
                    WHERE id = ? AND worker_id = ? AND interrupted_handoff_id IS ?
                      AND status IN (?, ?)
                      AND lease_expires_at = ? AND lease_expires_at > ?
                      AND NOT EXISTS (
                          SELECT 1 FROM cayu_work_attempt_admissions
                          WHERE task_id = ?
                      )
                    """,
                    (
                        sqlite_support.format_datetime(lease_expires_at),
                        sqlite_support.format_datetime(now),
                        task_id,
                        worker_id,
                        handoff_id,
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.RUNNING),
                        sqlite_support.format_datetime(expected_lease),
                        sqlite_support.format_datetime(now),
                        task_id,
                    ),
                )
                if cursor.rowcount != 1:
                    self._raise_if_governed_work_attempt_admission(
                        task_id,
                        "Admitted work attempts require claim-fenced lease renewal.",
                    )
                    self._raise_task_active_lease_error(task_id, worker_id, now=now)
                updated = self._require_task_unlocked(task_id)
                return updated.model_copy(deep=True)

    async def release_task(
        self,
        task_id: str,
        worker_id: str,
        *,
        lease_expires_at: datetime,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        expected_lease = normalize_utc_datetime(lease_expires_at, "lease_expires_at")
        async with self._lock:
            with self._verified_transaction_unlocked():
                now = self._ownership_clock()
                task = self._require_task_unlocked(task_id)
                _ensure_owned_active_task_lease(task, worker_id, now=now)
                if task.lease_expires_at != expected_lease:
                    raise TaskClaimLost("Task release no longer owns the expected worker lease.")
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                        (task_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Admitted work attempts release ownership through proposal publication."
                    )
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        started_at = NULL,
                        updated_at = ?
                    WHERE id = ? AND worker_id = ? AND status = ?
                      AND session_id IS NULL
                      AND (status_reason IS NULL OR status_reason NOT IN (?, ?))
                      AND lease_expires_at = ? AND lease_expires_at > ?
                      AND NOT EXISTS (
                          SELECT 1 FROM cayu_work_attempt_admissions
                          WHERE task_id = ?
                      )
                    """,
                    (
                        str(TaskStatus.PENDING),
                        sqlite_support.format_datetime(now),
                        task_id,
                        worker_id,
                        str(TaskStatus.CLAIMED),
                        _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
                        _TASK_CANCELLATION_REQUESTED_REASON,
                        sqlite_support.format_datetime(expected_lease),
                        sqlite_support.format_datetime(now),
                        task_id,
                    ),
                )
                if cursor.rowcount != 1:
                    self._raise_if_governed_work_attempt_admission(
                        task_id,
                        "Admitted work attempts release ownership through proposal publication.",
                    )
                    self._raise_task_release_error(task_id, worker_id, now=now)
                updated = self._require_task_unlocked(task_id)
                return updated.model_copy(deep=True)

    async def release_attached_task_worker(
        self,
        task_id: str,
        worker_id: str,
        *,
        lease_expires_at: datetime,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        expected_lease = normalize_utc_datetime(lease_expires_at, "lease_expires_at")
        async with self._lock:
            with self._verified_transaction_unlocked():
                now = self._ownership_clock()
                task = self._require_task_unlocked(task_id)
                _ensure_owned_active_task_lease(task, worker_id, now=now)
                if task.lease_expires_at != expected_lease:
                    raise TaskClaimLost(
                        "Attached-task release no longer owns the expected worker lease."
                    )
                if task.interrupted_handoff_id is not None:
                    raise TaskInterruptedHandoffConflict(
                        "Recovery-owned attached tasks must publish an interrupted handoff."
                    )
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                        (task_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Admitted work attempts release ownership through proposal publication."
                    )
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET worker_id = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?
                    WHERE id = ? AND worker_id = ? AND status = ?
                      AND session_id IS NOT NULL
                      AND (status_reason IS NULL OR status_reason != ?)
                      AND lease_expires_at = ? AND lease_expires_at > ?
                      AND NOT EXISTS (
                          SELECT 1 FROM cayu_work_attempt_admissions
                          WHERE task_id = ?
                      )
                    """,
                    (
                        sqlite_support.format_datetime(now),
                        task_id,
                        worker_id,
                        str(TaskStatus.RUNNING),
                        _TASK_CANCELLATION_REQUESTED_REASON,
                        sqlite_support.format_datetime(expected_lease),
                        sqlite_support.format_datetime(now),
                        task_id,
                    ),
                )
                if cursor.rowcount != 1:
                    self._raise_if_governed_work_attempt_admission(
                        task_id,
                        "Admitted work attempts release ownership through proposal publication.",
                    )
                    self._raise_attached_task_worker_release_error(
                        task_id,
                        worker_id,
                        now=now,
                    )
                updated = self._require_task_unlocked(task_id)
                return updated.model_copy(deep=True)

    async def reclaim_expired(
        self,
        *,
        query: TaskQuery | None = None,
        max_reclaims: int = 100,
    ) -> list[Task]:
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        max_reclaims = _validate_task_positive_int(max_reclaims, "max_reclaims")
        if query.status is not None and query.status is not TaskStatus.CLAIMED:
            return []
        clauses, params = self._task_filter_clauses(query)
        where_sql = " AND ".join(
            [
                "status = ?",
                "session_id IS NULL",
                "lease_expires_at IS NOT NULL",
                "lease_expires_at <= ?",
                "(status_reason IS NULL OR status_reason NOT IN (?, ?))",
                "NOT EXISTS (SELECT 1 FROM cayu_local_execution_attempts AS attempt "
                "WHERE attempt.retry_admissible = 0 AND ("
                "attempt.task_id = cayu_tasks.id OR ("
                "cayu_tasks.retry_series_json IS NOT NULL AND "
                "attempt.retry_series_id = json_extract("
                "cayu_tasks.retry_series_json, '$.series_id'))))",
                *clauses,
            ]
        )
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                now = self._ownership_clock()
                rows = self._connection.execute(
                    f"""
                    SELECT id
                    FROM cayu_tasks
                    WHERE {where_sql}
                    ORDER BY lease_expires_at ASC, id ASC
                    LIMIT ?
                    """,
                    [
                        str(TaskStatus.CLAIMED),
                        sqlite_support.format_datetime(now),
                        _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
                        _TASK_CANCELLATION_REQUESTED_REASON,
                        *params,
                        max_reclaims,
                    ],
                ).fetchall()
                task_ids = [row["id"] for row in rows]
                reclaimed: list[Task] = []
                for task_id in task_ids:
                    task = self._require_task_unlocked(task_id)
                    if task.started_at is not None:
                        requested = _expired_dispatched_task_cancellation(
                            task,
                            updated_at=now,
                        )
                        self._update_task_snapshot_unlocked(requested)
                        continue
                    updated = task.model_copy(
                        update={
                            "status": TaskStatus.PENDING,
                            "worker_id": None,
                            "lease_expires_at": None,
                            "updated_at": now,
                        }
                    )
                    self._update_task_snapshot_unlocked(updated)
                    reclaimed.append(updated)
                self._connection.commit()
                return [task.model_copy(deep=True) for task in reclaimed]
            except Exception:
                self._connection.rollback()
                raise

    async def close(self) -> None:
        async with self._lock:
            self._connection.close()

    def _connect(self, path: Path) -> sqlite3.Connection:
        return sqlite_support.connect(path)

    def _initialize_schema(self) -> None:
        sqlite_support.reconcile_schema(
            self._connection,
            self._schema_mode,
            app_min_supported=_SQLITE_TASK_MIN_REQUIRED_REVISION,
        )

    def _load_task_unlocked(self, task_id: str) -> Task | None:
        row = self._connection.execute(
            "SELECT * FROM cayu_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return sqlite_support.task_from_row(row)

    def _task_parent_for_create_unlocked(
        self,
        request: TaskCreate,
        *,
        task_id: str,
    ) -> TaskInvocationSnapshot | None:
        parent_task_id = request.parent_task_id
        if parent_task_id is None:
            return None
        if parent_task_id == task_id:
            raise ValueError("Task cannot be its own parent.")
        row = self._connection.execute(
            "SELECT id, session_id, session_instance_id, invocation_json "
            "FROM cayu_tasks WHERE id = ?",
            (parent_task_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Parent task not found: {parent_task_id}")
        return TaskInvocationSnapshot(
            id=row["id"],
            session_id=row["session_id"],
            session_instance_id=row["session_instance_id"],
            invocation=TaskInvocation.model_validate(json.loads(row["invocation_json"])),
        )

    def _require_task_unlocked(self, task_id: str) -> Task:
        task = self._load_task_unlocked(task_id)
        if task is None:
            raise KeyError(f"Task not found: {task_id}")
        return task

    def _task_exists_unlocked(self, task_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM cayu_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        return row is not None

    def _raise_if_governed_work_attempt_admission(
        self,
        task_id: str,
        message: str,
    ) -> None:
        if (
            self._connection.execute(
                "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                (task_id,),
            ).fetchone()
            is not None
        ):
            raise WorkAttemptExecutionClaimLost(message)

    def _finish_task_unlocked(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
        worker_id: str | None = None,
        handoff_id: str | None = None,
        expected_lease_expires_at: datetime | None = None,
    ) -> Task:
        with self._verified_transaction_unlocked():
            return self._finish_task_in_transaction_unlocked(
                task_id,
                status,
                result=result,
                error=error,
                worker_id=worker_id,
                handoff_id=handoff_id,
                expected_lease_expires_at=expected_lease_expires_at,
            )

    def _finish_task_in_transaction_unlocked(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
        worker_id: str | None = None,
        handoff_id: str | None = None,
        expected_lease_expires_at: datetime | None = None,
    ) -> Task:
        now = self._ownership_clock()
        task = self._require_task_unlocked(task_id)
        if worker_id is not None:
            if expected_lease_expires_at is not None:
                expected_lease_expires_at = normalize_utc_datetime(
                    expected_lease_expires_at,
                    "lease_expires_at",
                )
                if task.lease_expires_at != expected_lease_expires_at:
                    raise TaskClaimLost(
                        "Task terminalization no longer owns the expected worker lease."
                    )
            _ensure_owned_active_task_lease(task, worker_id, now=now)
            _ensure_task_handoff_authority(task, handoff_id)
        cancellation_owner_clause = ""
        cancellation_owner_params: list[str | None] = []
        if worker_id is not None and expected_lease_expires_at is not None:
            cancellation_owner_clause = (
                "\n                          AND worker_id = ?"
                "\n                          AND lease_expires_at = ?"
                "\n                          AND interrupted_handoff_id IS ?"
            )
            cancellation_owner_params = [
                worker_id,
                sqlite_support.format_datetime(expected_lease_expires_at),
                handoff_id,
            ]
        if (
            self._connection.execute(
                "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                (task_id,),
            ).fetchone()
            is not None
        ):
            raise WorkAttemptExecutionClaimLost(
                "Admitted work attempts cannot use ordinary terminalization."
            )
        verified_work_support.require_contracted_completion_authority(task, status)
        cancellation = None
        if task.retry_series is not None:
            if status is not TaskStatus.CANCELLED:
                raise ValueError(
                    "Retry-series tasks require settle_task_retry_attempt for "
                    "completion or failure."
                )
            if task.status in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
                cancellation_requested = _task_retry_cancellation_requested_task(
                    task,
                    error=error,
                    updated_at=now,
                )
                cursor = self._connection.execute(
                    f"""
                    UPDATE cayu_tasks
                    SET status_reason = ?, status_payload_json = ?, updated_at = ?
                    WHERE id = ? AND status IN (?, ?){cancellation_owner_clause}
                      AND NOT EXISTS (
                          SELECT 1 FROM cayu_work_attempt_admissions
                          WHERE task_id = ?
                      )
                    """,
                    (
                        cancellation_requested.status_reason,
                        sqlite_support.json_dumps(cancellation_requested.status_payload),
                        sqlite_support.format_datetime(cancellation_requested.updated_at),
                        task_id,
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.RUNNING),
                        *cancellation_owner_params,
                        task_id,
                    ),
                )
                if cursor.rowcount != 1:
                    self._raise_if_governed_work_attempt_admission(
                        task_id,
                        "Admitted work attempts cannot use ordinary terminalization.",
                    )
                    raise TaskTerminalizationConflict(
                        "Task retry cancellation lost active ownership."
                    )
                return self._require_task_unlocked(task_id).model_copy(deep=True)
            cancellation = _cancelled_task_retry_settlement(
                task,
                error=error,
                committed_at=now,
            )
            terminal_task = cancellation.task
        elif (
            task.status in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
            and status is TaskStatus.CANCELLED
            and task.worker_id is not None
            and task.lease_expires_at is not None
            and not _task_cancellation_requested(task)
        ):
            cancellation_requested = _task_cancellation_requested_task(
                task,
                error=error,
                updated_at=now,
            )
            cursor = self._connection.execute(
                f"""
                UPDATE cayu_tasks
                SET status_reason = ?, status_payload_json = ?, updated_at = ?
                WHERE id = ? AND status IN (?, ?){cancellation_owner_clause}
                  AND NOT EXISTS (
                      SELECT 1 FROM cayu_work_attempt_admissions
                      WHERE task_id = ?
                  )
                """,
                (
                    cancellation_requested.status_reason,
                    sqlite_support.json_dumps(cancellation_requested.status_payload),
                    sqlite_support.format_datetime(cancellation_requested.updated_at),
                    task_id,
                    str(TaskStatus.CLAIMED),
                    str(TaskStatus.RUNNING),
                    *cancellation_owner_params,
                    task_id,
                ),
            )
            if cursor.rowcount != 1:
                self._raise_if_governed_work_attempt_admission(
                    task_id,
                    "Admitted work attempts cannot use ordinary terminalization.",
                )
                raise TaskTerminalizationConflict("Task cancellation lost active ownership.")
            return self._require_task_unlocked(task_id).model_copy(deep=True)
        else:
            if _task_cancellation_requested(task):
                raise TaskTerminalizationConflict(
                    "Task cancellation is still draining under its current owner."
                )
            terminal_task = task.model_copy(
                update={
                    "status": status,
                    "status_reason": None,
                    "status_payload": None,
                    "result": result,
                    "error": error,
                    "started_at": task.started_at or now,
                    "completed_at": now,
                    "updated_at": now,
                    "interrupted_handoff_id": None,
                }
            )
        # When a worker_id is given, only terminalize if that worker still owns an active
        # lease — a worker that lost its lease must not clobber a task another has reclaimed.
        owner_clause = ""
        owner_params: list[str | None] = []
        if worker_id is not None:
            owner_clause = (
                "\n                  AND worker_id = ?"
                "\n                  AND lease_expires_at IS NOT NULL AND lease_expires_at > ?"
                "\n                  AND interrupted_handoff_id IS ?"
            )
            owner_params = [worker_id, sqlite_support.format_datetime(now), handoff_id]
        cursor = self._connection.execute(
            f"""
            UPDATE cayu_tasks
            SET status = ?,
                status_reason = ?,
                status_payload_json = ?,
                result_json = ?,
                error_json = ?,
                worker_id = NULL,
                lease_expires_at = NULL,
                interrupted_handoff_id = NULL,
                started_at = COALESCE(started_at, ?),
                completed_at = ?,
                updated_at = ?,
                retry_series_json = ?
            WHERE id = ?
              AND status NOT IN (?, ?, ?)
              AND NOT EXISTS (
                  SELECT 1 FROM cayu_work_attempt_admissions
                  WHERE task_id = ?
              ){owner_clause}
            """,
            (
                str(status),
                terminal_task.status_reason,
                (
                    None
                    if terminal_task.status_payload is None
                    else sqlite_support.json_dumps(terminal_task.status_payload)
                ),
                (
                    None
                    if terminal_task.result is None
                    else sqlite_support.json_dumps(terminal_task.result)
                ),
                (
                    None
                    if terminal_task.error is None
                    else sqlite_support.json_dumps(terminal_task.error)
                ),
                sqlite_support.format_optional_datetime(terminal_task.started_at),
                sqlite_support.format_optional_datetime(terminal_task.completed_at),
                sqlite_support.format_datetime(terminal_task.updated_at),
                (
                    None
                    if terminal_task.retry_series is None
                    else sqlite_support.json_dumps(
                        terminal_task.retry_series.model_dump(mode="json")
                    )
                ),
                task_id,
                str(TaskStatus.COMPLETED),
                str(TaskStatus.FAILED),
                str(TaskStatus.CANCELLED),
                task_id,
                *owner_params,
            ),
        )
        if cursor.rowcount == 1 and cancellation is not None:
            self._connection.execute(
                "INSERT INTO cayu_task_retry_settlements "
                "(task_id, idempotency_key, request_sha256, receipt_json, committed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    cancellation.task_id,
                    cancellation.idempotency_key,
                    cancellation.request_sha256,
                    sqlite_support.json_dumps(cancellation.model_dump(mode="json")),
                    sqlite_support.format_datetime(cancellation.committed_at),
                ),
            )
        if cursor.rowcount != 1:
            if worker_id is not None:
                current = self._require_task_unlocked(task_id)
                _ensure_owned_active_task_lease(current, worker_id, now=now)
                _ensure_task_handoff_authority(current, handoff_id)
            self._raise_if_governed_work_attempt_admission(
                task_id,
                "Admitted work attempts cannot use ordinary terminalization.",
            )
            if worker_id is not None:
                raise RuntimeError(f"Task {task_id} active-lease mutation did not update a row.")
            task = self._require_task_unlocked(task_id)
            _ensure_can_transition(task, status)
            raise ValueError(f"Task {task.id} cannot transition from {task.status}")
        updated = self._require_task_unlocked(task_id)
        return updated.model_copy(deep=True)

    async def _hold_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        reason: str | None,
        payload: dict[str, Any] | None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        reason = _copy_optional_status_reason(reason)
        payload = _copy_optional_status_payload(payload)
        async with self._lock:
            with self._verified_transaction_unlocked():
                now = self._ownership_clock()
                if (
                    self._connection.execute(
                        "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                        (task_id,),
                    ).fetchone()
                    is not None
                ):
                    raise WorkAttemptExecutionClaimLost(
                        "Admitted work attempts cannot use ordinary task holds."
                    )
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?,
                        status_reason = ?,
                        status_payload_json = ?,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?
                    WHERE id = ?
                      AND (
                        status = ?
                        OR status = ?
                        OR status = ?
                        OR status = ?
                        OR status = ?
                        OR (status = ? AND session_id IS NULL)
                      )
                      AND (status_reason IS NULL OR status_reason NOT IN (?, ?))
                      AND NOT EXISTS (
                          SELECT 1 FROM cayu_work_attempt_admissions
                          WHERE task_id = ?
                      )
                    """,
                    (
                        str(status),
                        reason,
                        None if payload is None else sqlite_support.json_dumps(payload),
                        sqlite_support.format_datetime(now),
                        task_id,
                        str(TaskStatus.PENDING),
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.PAUSED),
                        str(TaskStatus.BLOCKED),
                        str(TaskStatus.NEEDS_ATTENTION),
                        str(TaskStatus.RUNNING),
                        _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
                        _TASK_CANCELLATION_REQUESTED_REASON,
                        task_id,
                    ),
                )
                if cursor.rowcount != 1:
                    if (
                        self._connection.execute(
                            "SELECT 1 FROM cayu_work_attempt_admissions WHERE task_id = ? LIMIT 1",
                            (task_id,),
                        ).fetchone()
                        is not None
                    ):
                        raise WorkAttemptExecutionClaimLost(
                            "Admitted work attempts cannot use ordinary task holds."
                        )
                    task = self._require_task_unlocked(task_id)
                    _ensure_can_hold_task(task, status)
                    raise ValueError(f"Task {task.id} cannot transition to {status}")
                updated = self._require_task_unlocked(task_id)
                return updated.model_copy(deep=True)

    def _task_filter_clauses(self, query: TaskQuery) -> tuple[list[str], list[object]]:
        clauses: list[str] = []
        params: list[object] = []
        if query.type is not None:
            clauses.append("type = ?")
            params.append(query.type)
        if query.session_id is not None:
            clauses.append("session_id = ?")
            params.append(query.session_id)
        if query.parent_task_id is not None:
            clauses.append("parent_task_id = ?")
            params.append(query.parent_task_id)
        if query.assigned_agent_name is not None:
            clauses.append("assigned_agent_name = ?")
            params.append(query.assigned_agent_name)
        return clauses, params

    def _raise_task_active_lease_error(
        self,
        task_id: str,
        worker_id: str,
        *,
        now: datetime,
    ) -> None:
        task = self._require_task_unlocked(task_id)
        _ensure_owned_active_task_lease(task, worker_id, now=now)
        raise RuntimeError(f"Task {task.id} active-lease mutation did not update a row.")

    def _raise_task_release_error(
        self,
        task_id: str,
        worker_id: str,
        *,
        now: datetime,
    ) -> None:
        task = self._require_task_unlocked(task_id)
        _ensure_owned_active_task_lease(task, worker_id, now=now)
        if task.session_id is not None:
            raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
        if task.status is not TaskStatus.CLAIMED:
            raise ValueError(f"Task {task.id} is not claimed.")
        if task.status_reason in {
            _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
            _TASK_CANCELLATION_REQUESTED_REASON,
        }:
            raise TaskTerminalizationConflict(
                "Task cancellation is still draining under its current owner."
            )
        raise RuntimeError(f"Task {task.id} active claim could not be released.")

    def _raise_attached_task_worker_release_error(
        self,
        task_id: str,
        worker_id: str,
        *,
        now: datetime,
    ) -> None:
        task = self._require_task_unlocked(task_id)
        _ensure_owned_active_task_lease(task, worker_id, now=now)
        if task.status is not TaskStatus.RUNNING:
            raise ValueError(f"Task {task.id} is not running.")
        if task.session_id is None:
            raise ValueError(f"Task {task.id} is not attached to a session.")
        if _task_cancellation_requested(task):
            raise TaskTerminalizationConflict(
                "Task cancellation is still draining under its current owner."
            )
        raise RuntimeError(f"Task {task.id} active attached claim could not be released.")

    def _raise_task_claim_attach_error(
        self,
        task_id: str,
        worker_id: str,
        *,
        now: datetime,
    ) -> None:
        task = self._require_task_unlocked(task_id)
        _raise_task_claim_attach_error(task, worker_id, now=now)


def _sqlite_task_terminalization_receipt(
    *,
    task_id: str,
    idempotency_key: str,
    row: sqlite3.Row,
) -> TaskTerminalizationReceipt:
    try:
        return TaskTerminalizationReceipt(
            task_id=task_id,
            idempotency_key=idempotency_key,
            worker_id=row["worker_id"],
            kind=row["terminal_kind"],
            request_sha256=row["request_sha256"],
            task=Task.model_validate(json.loads(row["task_json"])),
            committed_at=sqlite_support.parse_datetime(row["committed_at"]),
        )
    except Exception as exc:
        raise TaskTerminalizationConflict("Task terminalization receipt is malformed.") from exc


def _sqlite_interrupted_task_handoff_receipt(
    *,
    task_id: str,
    handoff_id: str,
    row: sqlite3.Row,
) -> TaskInterruptedHandoffReceipt:
    try:
        receipt = TaskInterruptedHandoffReceipt(
            request=TaskInterruptedHandoffRequest.model_validate(json.loads(row["request_json"])),
            request_sha256=row["request_sha256"],
            task=Task.model_validate(json.loads(row["task_json"])),
            committed_at=sqlite_support.parse_datetime(row["committed_at"]),
        )
        if receipt.request.task_id != task_id or receipt.request.handoff_id != handoff_id:
            raise ValueError("Interrupted-task handoff receipt conflicts with its storage key.")
        return receipt
    except Exception as exc:
        raise TaskInterruptedHandoffConflict(
            "Interrupted-task handoff receipt is malformed."
        ) from exc


def _validate_task_positive_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1.")
    return value
