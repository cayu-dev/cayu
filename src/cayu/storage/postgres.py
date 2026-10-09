from __future__ import annotations

import asyncio
import heapq
import hmac
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any, ClassVar, Literal, LiteralString, cast
from uuid import uuid4

from cayu._resource_store_surface import model_store_surface
from cayu.budgets.pricing import PriceBook
from cayu.collaboration.peer_content import (
    PeerAppendKey,
    PeerContentAppendRequest,
    PeerContentConflict,
    PeerContentExposureReceipt,
    PeerContentExposureRequest,
    PeerContentReceipt,
    PeerContentUnavailable,
)
from cayu.runtime import _session_message_queue as message_queue
from cayu.runtime._cost_accounting import CostAccountingSnapshot
from cayu.runtime._usage_accounting import UsageAccountingSnapshot
from cayu.sessions import creation_fence
from cayu.sessions import event_delivery as side_effect_health
from cayu.sessions.access import (
    require_resource_session,
    runtime_session_mutation,
    runtime_session_query,
)
from cayu.sessions.base import (
    _check_closure_lineage_owner,
    _closure_progress_targets,
    _validate_closure_progress_update,
    _validate_session_closure_detach_replay,
)
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectHealth,
    PersistedEventSideEffectPage,
    PersistedEventSideEffectQuery,
)
from cayu.sessions.messaging import (
    SESSION_MESSAGE_QUEUE_STORAGE_VALUE_MAX_BYTES,
    SessionMessageActionRequest,
    SessionMessageActionResult,
    SessionMessageConditions,
    SessionMessageConflict,
    SessionMessageDeliveryMode,
    SessionMessageInspection,
    SessionMessageQuery,
    SessionMessageSource,
    session_message_rejection,
)
from cayu.storage import _creation_fence
from cayu.storage import _postgres_base as postgres_base
from cayu.storage import _postgres_support as pg_support
from cayu.storage._context_selection_fence import PostgresContextSelectionFenceMixin
from cayu.storage._creation_fence import PostgresCreationFenceMixin
from cayu.storage._external_wait_postgres import PostgresExternalWaitMixin
from cayu.storage._session_execution import PostgresSessionExecutionMixin
from cayu.storage.budget_postgres import PostgresBudgetLedger as PostgresBudgetLedger
from cayu.storage.event_watchers_postgres import (
    PostgresEventWatcherStore as PostgresEventWatcherStore,
)
from cayu.storage.knowledge_embedding_postgres import (
    PostgresEmbeddingKnowledgeStore as PostgresEmbeddingKnowledgeStore,
)
from cayu.storage.knowledge_postgres import PostgresKnowledgeStore as PostgresKnowledgeStore
from cayu.storage.tasks_postgres import PostgresTaskStore as PostgresTaskStore
from cayu.storage.work_context_postgres import (
    PostgresAgentWorkContextStore as PostgresAgentWorkContextStore,
)

if TYPE_CHECKING:
    from cayu.runtime._zero_work_interruption import (
        ZeroWorkInterruptionPublication,
        ZeroWorkInterruptionRequest,
    )
    from cayu.sessions._temporary_continuation import TemporaryServiceAdmission
    from cayu.sessions.access import _SessionAccessBounds
    from cayu.sessions.exports import SessionExportLimits, SessionExportSnapshot

try:
    from psycopg.errors import (
        ForeignKeyViolation,
        UniqueViolation,
    )
    from psycopg.types.json import Jsonb
    from psycopg_pool import AsyncConnectionPool
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
    raise RuntimeError(
        "Cayu's Postgres stores require the optional psycopg packages. "
        'Install them with `pip install "cayu[postgres]"`.'
    ) from exc

from cayu._clock import utc_clock, utc_duration_cutoff
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    JsonUtf8SizeCounter,
    canonical_bounded_durable_json_bytes,
    copy_durable_json_object,
    copy_durable_json_value,
    copy_label_map,
    require_nonblank,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu.approvals.tools import (
    ResolutionActor,
    resolution_actor_payload,
)
from cayu.budgets.aggregates import EXACT_AGGREGATE, UsageRollupStoreResult
from cayu.events import (
    EVENT_ID_MAX_CHARS,
    Event,
    EventType,
    event_with_runtime_payload_authority,
)
from cayu.execution_profiles import (
    ExecutionProfileDecision,
    ExecutionProfileIdentity,
    ExecutionProfileRejectionResult,
)
from cayu.execution_units import (
    ToolRoundIdentity,
    copy_tool_round_identity,
)
from cayu.memory.evidence import (
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
from cayu.messages import Message, MessageRole
from cayu.runtime._child_session_notifications import (
    ChildSessionLifecycleOccurrence,
    ChildSessionLifecycleOccurrenceSource,
    ChildSessionLifecyclePage,
    ChildSessionLifecycleQuery,
    child_session_notification_stage_binding,
    child_session_notification_storage_key,
)
from cayu.runtime.evidence_spool import EvidenceSpool
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, parse_public_authority_alias
from cayu.runtime.service_manifest import RuntimeStoreDurability
from cayu.sessions._checkpoint_preservation import (
    _checkpoint_transform_result_preserving_completion_result_event_publications,
    _copy_checkpoint_for_transform,
    _replace_checkpoint_preserving_completion_result_event_publications,
)
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
)
from cayu.sessions._invocation_terminal_decision import InvocationTerminalDecision
from cayu.sessions._provider_operation_cancellation_claim import (
    active_provider_operation_cancellation_claim_from_checkpoint,
)
from cayu.sessions._terminal_evidence import _session_run_operation_from_checkpoint
from cayu.sessions.authority import CheckpointValueAuthority
from cayu.sessions.base import (
    _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES,
    _TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT,
    _TOOL_ROUND_LIFECYCLE_EVENT_TYPES,
    CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS,
    DELETE_BLOCKED_SESSION_STATUSES,
    FORK_TRANSCRIPT_VALIDATION_ERROR,
    INHERIT_INTERACTION,
    LATEST_TRANSCRIPT_TEXT_MAX_CHARS,
    LATEST_TRANSCRIPT_TEXT_MAX_PARTS,
    LATEST_TRANSCRIPT_TEXT_MAX_SOURCE_BYTES,
    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
    BudgetReservationIdentityConflict,
    CheckpointRootFieldGuard,
    CheckpointTransform,
    DeferredInteractionInput,
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
    ProfiledSessionForkResult,
    QueuedDispatchTerminalReceipt,
    QueuedDispatchTerminalReceiptQuery,
    QueuedInteractionProfileHandoff,
    RunRequest,
    RuntimePublicationMutation,
    RuntimePublicationReceipt,
    RuntimePublicationResult,
    SessionForkActiveModelStageConflict,
    SessionForkProfileRelationship,
    SessionIdentity,
    SessionInvocationSnapshot,
    SessionMessageQueueStatus,
    SessionModelCompletionDispatchAlreadyAuthorized,
    SessionModelCompletionStageConflict,
    SessionModelTransition,
    SessionOperationInitializer,
    SessionOperationPublication,
    SessionOperationTransform,
    SessionRunFenced,
    SessionRuntimeIdentity,
    SessionRuntimePublicationConflict,
    SessionStateSnapshot,
    SessionStatusConflict,
    SessionStore,
    StoreTimeCheckpointTransform,
    StoreTimeSessionOperationTransform,
    TranscriptSnapshot,
    TranscriptTextReadLimitExceeded,
    _activate_session_run_fence,
    _active_model_completion_stage_record,
    _active_unexpired_incomplete_recovery_claim_id,
    _active_unexpired_session_operation_id,
    _apply_queue_completion_checkpoint_mutation,
    _apply_runtime_publication_checkpoint_mutation,
    _apply_runtime_publication_operation_record_mutations,
    _assert_session_run_epoch,
    _assert_session_run_epoch_value,
    _authenticated_public_authority_alias_private_value,
    _build_runtime_publication_receipt,
    _checkpoint_after_exact_invocation_terminal_decision,
    _checkpoint_after_initial_transcript_publication,
    _checkpoint_after_queued_interaction_profile_handoff,
    _child_session_lifecycle_entry,
    _child_session_lifecycle_entry_sort_key,
    _child_session_lifecycle_occurrence,
    _child_session_notification_consumption_record,
    _child_session_notification_consumption_replays,
    _completion_result_event_publication_delete_block_reason,
    _copy_failed_first_delivery_retirement,
    _copy_historical_queued_interaction_profile_handoff,
    _copy_mcp_manifest_publication,
    _copy_optional_event_id,
    _copy_optional_execution_profile,
    _copy_optional_execution_profile_decision,
    _copy_optional_interaction_admission,
    _copy_optional_tool_capability_ceiling,
    _copy_pending_first_event_delivery,
    _copy_profiled_fork_authority,
    _copy_queued_interaction_profile_handoff,
    _copy_queued_interaction_started_event,
    _copy_session_event_batch,
    _copy_session_model_transition,
    _copy_transition_interaction_admission,
    _copy_workflow_step_reservation,
    _current_session_run_epoch,
    _deactivate_session_interaction,
    _deactivate_session_run_fence,
    _durable_subagent_parent_delete_block_reason,
    _event_file_attachment_attestations_are_runtime_owned,
    _event_input_contract_is_runtime_owned,
    _execution_profile_rejection_events_equivalent,
    _historical_queued_handoff_stage_from_records,
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
    _model_failover_admission_storage_keys,
    _model_failover_checkpoint_after_profile_admission,
    _model_failover_predecessor_storage_keys,
    _model_failover_preparation_checkpoint,
    _model_failover_selection_event,
    _ModelCompletionStagePromotionContext,
    _next_runtime_publication_timestamp,
    _prepare_execution_profile_rejection,
    _prepare_initial_session_operation_records,
    _prepare_interaction_transition,
    _prepare_interaction_transition_receipt_lookup,
    _prepare_model_completion_stage_promotion,
    _prepare_profiled_fork_checkpoint_result,
    _prepare_queue_completion_checkpoint_mutation,
    _prepare_session_fork_request,
    _PreparedModelCompletionStage,
    _PreparedModelCompletionStageAbandonment,
    _PreparedModelCompletionStageTerminal,
    _PreparedRuntimePublication,
    _profiled_fork_authority_validation_error,
    _project_interruption_cascade_marker_fields,
    _public_authority_alias_store_key,
    _queued_dispatch_terminal_receipts_from_checkpoint,
    _reconstruct_active_model_completion_stage,
    _reconstruct_active_model_completion_stage_record,
    _reconstruct_interaction_transition_receipt,
    _reconstruct_model_completion_stage,
    _reconstruct_model_completion_stage_abandonment,
    _reconstruct_model_completion_stage_dispatch,
    _reconstruct_runtime_publication_receipt,
    _reject_reserved_runtime_publication_key,
    _reject_settled_model_completion_stage,
    _replay_model_completion_stage_abandonment,
    _replay_promoted_model_completion_stage,
    _require_invocation_release_recovery_claim,
    _require_invocation_release_settlement_record,
    _require_invocation_release_terminal_session_event,
    _require_live_incomplete_recovery_claim_for_run_epoch_transfer,
    _require_session_export_target,
    _run_session_commit_guard_owned,
    _runtime_publication_json_equal,
    _runtime_publication_receipt_record,
    _runtime_publication_referenced_event_ids,
    _runtime_publication_storage_key,
    _session_metadata_after_model_transition,
    _session_metadata_after_runtime_identity_adoption,
    _session_metadata_after_tool_capability_ceiling_admission,
    _stored_mcp_manifest_baseline,
    _terminal_publication_delete_block_reason,
    _tool_lifecycle_publication_identity,
    _tool_round_lifecycle_event_limit,
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
    _validate_model_completion_stage_recovery_fence,
    _validate_model_completion_stage_release,
    _validate_model_completion_stage_repreparation,
    _validate_model_completion_stage_terminal_replay,
    _validate_model_failover_selection_replay,
    _validate_profiled_fork_authority,
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
    checkpoint_root_field_projection_from_storage,
    copy_run_request,
    copy_session_identity,
    copy_session_runtime_identity,
    copy_session_user_metadata,
    copy_transcript_messages,
    deferred_interaction_input_for_run_request,
    deferred_interaction_input_from_storage_payload,
    deferred_interaction_input_storage_payload,
    fork_transcript_is_accepted,
    replace_session_user_metadata,
    require_deferred_initial_transcript_replacement,
    resolve_interaction_attribution,
    restore_persisted_event_authority,
    session_instance_id_for_run_request,
    session_invocation_for_run_request,
    session_messages_input_contract_evidence,
    session_metadata_for_creation,
    transform_fork_checkpoint,
)
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectClaim,
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectDelivery,
    PersistedEventSideEffectStatus,
    validate_persisted_event_side_effect_error,
)
from cayu.sessions.event_queries import EventQuery, EventQueryResultTooLarge, copy_event_query
from cayu.sessions.inspection import SESSION_INSPECTION_LABEL_LIMIT, SessionInspectionIdentity
from cayu.sessions.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
    INTERACTION_TERMINAL_EVENT_TYPES,
)
from cayu.sessions.invocation import SessionInvocation
from cayu.sessions.lineage import (
    SESSION_LINEAGE_MAX_EVENT_ID_BYTES,
    SESSION_LINEAGE_MAX_IDENTIFIER_BYTES,
    SESSION_LINEAGE_MAX_ORIGIN_EVENTS,
    SessionLineageNode,
    SessionLineageOrigin,
    SessionLineageQuery,
    SessionLineageResult,
    copy_session_lineage_query,
    decode_session_lineage_cursor,
    encode_session_lineage_cursor,
)
from cayu.sessions.messaging import (
    SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
    EnqueueSessionMessageRequest,
    EnqueueSessionMessageResult,
    SessionMessageDeliveryBatch,
    SessionQueuedMessage,
    SessionQueuedMessagesPending,
    _queued_session_message_event_payload,
    _validate_equivalent_queued_session_message,
    copy_enqueue_session_message_request,
    enqueue_session_message_input,
    queued_session_message_input,
)
from cayu.sessions.pending_action_contracts import (
    MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL,
    MAX_PENDING_ACTION_TOOL_CALLS,
    PendingActionIssue,
    PendingActionListResult,
    PendingActionQuery,
    enforce_pending_action_result_size,
)
from cayu.sessions.queries import (
    SessionAggregateFilter,
    SessionListResult,
    SessionOrder,
    SessionQuery,
    SessionStatusCounts,
    copy_session_aggregate_filter,
    copy_session_query,
    decode_session_cursor,
    encode_session_cursor,
    session_next_cursor,
    session_query_from_aggregate_filter,
)
from cayu.sessions.records import (
    EventRecord,
    PendingActionKind,
    PendingActionSession,
    RunnerObservedEventIdentity,
    Session,
    SessionStatus,
    TranscriptRecord,
)
from cayu.sessions.summaries import (
    EventSummary,
    SessionOperationalSnapshot,
    SessionOutcome,
    session_outcome,
)
from cayu.sessions.terminal_evidence import (
    TerminalPublicationMarker,
    TerminalSessionEvidence,
    TerminalSessionEvidenceError,
    TerminalSessionEvidenceErrorCode,
    TerminalSessionEvidenceLimits,
    _assemble_terminal_session_evidence,
    _classify_terminal_session_evidence_records,
    _copy_runner_owned_interruption_proof,
    _copy_terminal_session_evidence_limits,
    _terminal_session_evidence_expected_event_type,
    _validate_runner_observed_event_identity_snapshot,
)
from cayu.sessions.topology import (
    SessionTopologyCycle,
    SessionTopologyDepthExceeded,
    SessionTopologyQuery,
    SessionTopologyStoreResult,
    build_session_topology_result,
    decode_session_topology_cursor,
)
from cayu.sessions.transcript_queries import (
    TranscriptPage,
    TranscriptQuery,
    TranscriptSearchHit,
    TranscriptSearchQuery,
    TranscriptSearchResult,
    copy_transcript_query,
    copy_transcript_search_query,
    decode_transcript_search_cursor,
    encode_transcript_search_cursor,
    filter_transcript_records,
    transcript_search_document,
    transcript_search_document_score,
    transcript_search_hit_from_message,
    transcript_search_position_after_cursor,
    transcript_search_query_document,
    transcript_search_session_token,
)
from cayu.sessions.usage import UsageRollupQuery, copy_usage_rollup_query
from cayu.storage import _postgres_aggregates as postgres_aggregates
from cayu.storage import _session_store_sql as session_store_sql
from cayu.storage import migration_authority
from cayu.storage import migrations as schema
from cayu.storage._participant_bindings_schema import (
    PARTICIPANT_BINDING_PROJECTION,
)
from cayu.tools.exposure import ToolCapabilityCeiling
from cayu.tools.grants import (
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
from cayu.workflows.base import WORKFLOW_ATTEMPT_EVENT_TYPE

_POSTGRES_SESSION_MIN_REQUIRED_REVISION = 113


async def _postgres_lock_memory_evidence_id(cur: Any, lock_id: str) -> None:
    await cur.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
        (lock_id,),
    )


def _postgres_json_document(value: Any, field_name: str) -> dict[str, Any]:
    if type(value) is str:
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Postgres {field_name} contains invalid JSON.") from exc
    if type(value) is not dict:
        raise RuntimeError(f"Postgres {field_name} must be a JSON object.")
    return value


def _postgres_recall_receipt(row: Sequence[Any]) -> RecallReceipt:
    try:
        receipt = RecallReceipt.model_validate(_postgres_json_document(row[5], "recall receipt"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Postgres recall receipt contains invalid durable material.") from exc
    document = memory_evidence_document_bytes(receipt, "stored recall receipt")
    if (
        receipt.receipt_id != row[0]
        or receipt.session_id != row[1]
        or receipt.interaction_id != row[2]
        or receipt.model_step_id != row[3]
        or receipt.created_at != pg_support.to_utc(row[4])
        or len(document) != row[6]
    ):
        raise RuntimeError("Postgres recall receipt index columns conflict with its document.")
    return receipt


def _postgres_context_exposure(row: Sequence[Any]) -> ContextExposure:
    try:
        exposure = ContextExposure.model_validate(
            _postgres_json_document(row[10], "context exposure")
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Postgres context exposure contains invalid durable material.") from exc
    document = memory_evidence_document_bytes(exposure, "stored context exposure")
    if (
        exposure.exposure_id != row[0]
        or exposure.session_id != row[1]
        or exposure.interaction_id != row[2]
        or exposure.model_step_id != row[3]
        or exposure.model_attempt_id != row[4]
        or exposure.provider_attempt_id != row[5]
        or str(exposure.state) != row[6]
        or exposure.state_revision != row[7]
        or exposure.created_at != pg_support.to_utc(row[8])
        or exposure.updated_at != pg_support.to_utc(row[9])
        or len(document) != row[11]
    ):
        raise RuntimeError("Postgres context exposure index columns conflict with its document.")
    return exposure


def _postgres_recall_item_exposures(
    rows: Sequence[Sequence[Any]],
) -> tuple[RecallItemExposure, ...]:
    items: list[RecallItemExposure] = []
    for row in rows:
        try:
            item = RecallItemExposure.model_validate(
                _postgres_json_document(row[4], "recall item exposure")
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Postgres recall item exposure contains invalid durable material."
            ) from exc
        document = memory_evidence_document_bytes(item, "stored recall item exposure")
        if (
            item.exposure_id != row[0]
            or item.ordinal != row[1]
            or item.receipt_id != row[2]
            or item.receipt_item_ordinal != row[3]
            or len(document) != row[5]
        ):
            raise RuntimeError(
                "Postgres recall item exposure index columns conflict with its document."
            )
        items.append(item)
    if tuple(item.ordinal for item in items) != tuple(range(len(items))):
        raise RuntimeError("Postgres recall item exposure ordinals are incomplete.")
    return tuple(items)


_EVENT_QUERY_SESSION_IDS_BATCH_SIZE = 500
_PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL = """
    event_type IN (
        'tool.call.approval_requested',
        'session.awaiting_user_input',
        'session.interrupted',
        'session.delegated_action.updated',
        'tool.call.started',
        'tool.call.completed',
        'tool.call.failed',
        'tool.call.blocked',
        'tool.call.approval_denied'
    )
    AND pending_action_lookup_key IS NOT NULL
"""
_SQL_DIALECT = session_store_sql.SessionStoreSqlDialect(
    placeholder="%s",
    contains_style="postgres_ilike",
    datetime_param=pg_support.to_utc,
)


def _targeted_tool_grant_from_postgres(value: object) -> TargetedToolGrantRecord:
    try:
        decoded = json.loads(value) if type(value) is str else value
        return copy_targeted_tool_grant_record(TargetedToolGrantRecord.model_validate(decoded))
    except (TypeError, ValueError):
        raise ValueError("Stored targeted tool grant is malformed.") from None


def _targeted_tool_use_from_postgres(value: object) -> TargetedToolUseBinding:
    try:
        decoded = json.loads(value) if type(value) is str else value
        return TargetedToolUseBinding.model_validate(decoded)
    except (TypeError, ValueError):
        raise ValueError("Stored targeted tool use is malformed.") from None


def _targeted_tool_grant_from_postgres_row(row: Sequence[object]) -> TargetedToolGrantRecord:
    if len(row) != 16:
        raise ValueError("Stored targeted tool grant row is malformed.")
    record = _targeted_tool_grant_from_postgres(row[15])
    indexed = (
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
        pg_support.to_utc(record.issued_at),
        pg_support.to_utc(record.expires_at),
        record.max_calls,
        record.used_calls,
        None if record.revoked_at is None else pg_support.to_utc(record.revoked_at),
    )
    if tuple(row[:15]) != indexed:
        raise ValueError("Stored targeted tool grant conflicts with indexed authority.")
    return record


def _targeted_tool_use_from_postgres_row(row: Sequence[object]) -> TargetedToolUseBinding:
    if len(row) != 10:
        raise ValueError("Stored targeted tool use row is malformed.")
    binding = _targeted_tool_use_from_postgres(row[9])
    indexed = (
        binding.use_id,
        binding.grant_id,
        binding.session_id,
        binding.interaction_id,
        binding.model_step_id,
        binding.outer_tool_call_id,
        binding.arguments_sha256,
        binding.invocation_id,
        pg_support.to_utc(binding.bound_at),
    )
    if tuple(row[:9]) != indexed:
        raise ValueError("Stored targeted tool use conflicts with indexed authority.")
    return binding


async def _validate_targeted_tool_use_counts(
    cur: Any,
    records: Iterable[TargetedToolGrantRecord],
) -> None:
    expected = {record.grant_id: record.used_calls for record in records}
    if not expected:
        return
    await cur.execute(
        "SELECT grant_id, COUNT(*) FROM cayu_targeted_tool_grant_uses "
        "WHERE grant_id = ANY(%s) GROUP BY grant_id",
        (list(expected),),
    )
    actual = dict.fromkeys(expected, 0)
    for row in await cur.fetchall():
        actual[str(row[0])] = int(row[1])
    if actual != expected:
        raise ValueError("Targeted grant call counter conflicts with durable uses.")


def _postgres_transcript_search_expression(query: TranscriptSearchQuery) -> str:
    session_terms = " | ".join(
        transcript_search_session_token(session_id) for session_id in query.session_ids
    )
    text_terms = " | ".join(transcript_search_query_document(query.text).split())
    return f"({session_terms}) & ({text_terms})"


def _postgres_transcript_index_document(session_id: str, message: Message) -> str:
    narrative_document = transcript_search_document(message)
    session_term = transcript_search_session_token(session_id)
    return session_term if not narrative_document else f"{session_term} {narrative_document}"


_SESSION_MESSAGE_QUEUE_COLUMNS = (
    "ordering_key, queue_id, session_id, idempotency_key, content, delivery_mode, status, "
    "requested_by, accepted_run_epoch, accepted_transcript_cursor, accepted_event_id, "
    "accepted_at, delivered_run_epoch, delivered_transcript_cursor, delivered_event_id, "
    "delivered_at, message_json, conditions_json, terminal_json"
)


def _session_message_raw_row(row: Any) -> dict[str, Any]:
    return dict(zip(_SESSION_MESSAGE_QUEUE_COLUMNS.split(", "), row, strict=True))


def _queued_session_message_from_row(row: Any) -> SessionQueuedMessage:
    requested_by = row[7]
    return SessionQueuedMessage(
        ordering_key=row[0],
        queue_id=row[1],
        session_id=row[2],
        idempotency_key=row[3],
        conditions=SessionMessageConditions.model_validate(
            {} if row[17] is None else pg_support._json_obj(row[17])
        ),
        content=row[4],
        message=(
            None if row[16] is None else Message.model_validate(pg_support._json_obj(row[16]))
        ),
        delivery_mode=row[5],
        status=row[6],
        requested_by=(
            None
            if requested_by is None
            else ResolutionActor.model_validate(pg_support._json_obj(requested_by))
        ),
        accepted_run_epoch=row[8],
        accepted_transcript_cursor=row[9],
        accepted_event_id=row[10],
        accepted_at=row[11],
        delivered_run_epoch=row[12],
        delivered_transcript_cursor=row[13],
        delivered_event_id=row[14],
        delivered_at=row[15],
    )


async def _raise_session_write_conflict(
    cur: Any,
    session_id: str,
    expected_run_epoch: int,
) -> None:
    await cur.execute("SELECT run_epoch FROM cayu_sessions WHERE id = %s", (session_id,))
    row = await cur.fetchone()
    if row is None:
        raise KeyError(f"Session not found: {session_id}")
    raise SessionRunFenced(
        f"Session run epoch no longer owns {session_id}: expected {expected_run_epoch}, "
        f"current {row[0]}."
    )


async def _touch_session_activity(cur: Any, session_id: str, activity_at: datetime) -> None:
    expected_run_epoch = _current_session_run_epoch(session_id)
    if expected_run_epoch is None:
        await cur.execute(
            "UPDATE cayu_sessions SET last_activity_at = %s WHERE id = %s",
            (activity_at, session_id),
        )
        if cur.rowcount != 1:
            raise KeyError(f"Session not found: {session_id}")
        return
    await cur.execute(
        "UPDATE cayu_sessions SET last_activity_at = %s WHERE id = %s AND run_epoch = %s",
        (activity_at, session_id, expected_run_epoch),
    )
    if cur.rowcount != 1:
        await _raise_session_write_conflict(cur, session_id, expected_run_epoch)


def _event_query_session_id_batches(
    session_ids: tuple[str, ...],
) -> list[tuple[str, ...]]:
    return [
        session_ids[index : index + _EVENT_QUERY_SESSION_IDS_BATCH_SIZE]
        for index in range(0, len(session_ids), _EVENT_QUERY_SESSION_IDS_BATCH_SIZE)
    ]


def _event_query_is_single_session(query: EventQuery) -> bool:
    return query.session_id is not None or len(query.session_ids) == 1


def _event_query_needs_snapshot_cutoff(query: EventQuery) -> bool:
    return (
        query.after_sequence is not None or query.before_sequence is not None
    ) and not _event_query_is_single_session(query)


async def _transcript_cursor(cur: Any, session_id: str) -> int:
    """Return the permanent next transcript position, independent of retention."""

    await cur.execute(
        "SELECT transcript_seq FROM cayu_sessions WHERE id = %s",
        (session_id,),
    )
    row = await cur.fetchone()
    if row is None:
        raise KeyError(f"Session not found: {session_id}")
    return int(row[0])


@model_store_surface("sessions")
class PostgresSessionStore(
    PostgresExternalWaitMixin,
    PostgresSessionExecutionMixin,
    PostgresContextSelectionFenceMixin,
    PostgresCreationFenceMixin,
    postgres_base._PostgresStoreBase,
    SessionStore,
):
    """Postgres-backed session store for shared durable runtime state."""

    session_access_version: ClassVar[int | None] = 1

    async def _access_create_session(self, bounds, request, identity):
        from cayu.sessions.access import _creation_bounds

        token = _creation_bounds.set(bounds)
        try:
            return await self.create(request, identity=identity)
        finally:
            _creation_bounds.reset(token)

    async def _access_update_metadata(self, bounds, session_id, metadata):
        return await self.update_metadata(session_id, metadata, _access_bounds=bounds)

    async def _access_delete_session(self, bounds, session_id):
        await self.delete_session(session_id, _access_bounds=bounds)

    async def _access_read_records(self, bounds, session_id, kind, offset, limit, max_bytes):
        from cayu.storage._session_access_records import postgres_read

        return await postgres_read(self, bounds, session_id, kind, offset, limit, max_bytes)

    async def _access_list_sessions(
        self, bounds: _SessionAccessBounds, query: SessionQuery
    ) -> SessionListResult:
        return await self._list_sessions(
            query, pending_interruption_cascade_only=False, access_bounds=bounds
        )

    async def _access_update_labels(
        self, bounds: _SessionAccessBounds, session_id: str, labels: dict[str, str]
    ) -> Session:
        return await self.update_labels(session_id, labels, _access_bounds=bounds)

    async def _access_load_session(self, bounds: _SessionAccessBounds, session_id: str) -> Session:
        from cayu.sessions.access import SessionAccessDenied

        session_id = require_clean_nonblank(session_id, "session_id")
        clause = session_store_sql.session_access_clause(bounds, dialect=_SQL_DIALECT)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            await cur.execute(
                f"SELECT {pg_support.SESSION_COLUMNS} FROM cayu_sessions WHERE id = %s AND ({clause.sql})",
                (session_id, *clause.params),
            )
            row = await cur.fetchone()
            if row is None:
                raise SessionAccessDenied()
            labels = await self._load_labels(cur, session_id)
            return bounds.require_read(pg_support.session_from_row(row, labels=labels))

    supports_usage_aggregates: ClassVar[bool] = True
    supports_private_argument_continuity: ClassVar[bool] = True
    supports_mcp_manifest_history: ClassVar[bool] = True
    supports_public_authority_aliases: ClassVar[bool] = True
    supports_targeted_tool_grants: ClassVar[bool] = True
    supports_session_topology: ClassVar[bool] = True
    supports_session_lineage: ClassVar[bool] = True
    child_session_notification_version: ClassVar[int | None] = 1
    supports_incremental_terminal_evidence: ClassVar[bool] = True
    supports_terminal_session_evidence: ClassVar[bool] = True
    supports_runner_owned_interrupted_evidence: ClassVar[bool] = True
    supports_execution_profile_admission: ClassVar[bool] = True
    model_failover_stage_version: ClassVar[int] = 1
    supports_active_invocation_execution_profiles: ClassVar[bool] = True
    invocation_lifecycle_command_version: ClassVar[int | None] = 1
    terminal_interaction_publication_version: ClassVar[int | None] = 1
    durable_model_terminalization_version: ClassVar[int | None] = 1
    queued_interaction_profile_handoff_version: ClassVar[int | None] = 1
    session_message_lifecycle_version: ClassVar[int | None] = 1
    supports_pending_session_initial_checkpoint: ClassVar[bool] = True
    supports_profiled_forks: ClassVar[bool] = True
    supports_atomic_session_operation_initialization: ClassVar[bool] = True
    supports_atomic_model_completion_stage_release: ClassVar[bool] = True
    model_completion_recovery_fence_version: ClassVar[int] = 1
    session_steering_version: ClassVar[int | None] = 1
    session_export_version: ClassVar[int] = 1
    session_continuation_version: ClassVar[int] = 1
    external_wait_version: ClassVar[int] = 1
    _producer_attachment_version: ClassVar[int] = 1
    supports_completion_result_event_publication_reservations: ClassVar[bool] = True
    supports_transcript_search: ClassVar[bool] = True
    supports_recall_evidence: ClassVar[bool] = True
    supports_owned_off_thread_session_commit_guards: ClassVar[bool] = True
    supports_session_closure_receipts: ClassVar[bool] = True
    supports_session_closure_detachment: ClassVar[bool] = True
    supports_session_closure_recursive_deletion: ClassVar[bool] = True
    supports_session_closure_progress: ClassVar[bool] = True
    participant_session_binding_version: ClassVar[int | None] = 1
    recipient_continuation_selection_version: ClassVar[int | None] = 1
    context_view_selection_fence_version: ClassVar[int | None] = 1
    context_view_version: ClassVar[int | None] = 1
    peer_content_version: ClassVar[int | None] = 1
    service_durability: RuntimeStoreDurability = RuntimeStoreDurability.DURABLE
    _min_required_revision = _POSTGRES_SESSION_MIN_REQUIRED_REVISION
    _supports_read_only = True

    def __init__(
        self,
        conninfo: str | None = None,
        *,
        pool: AsyncConnectionPool | None = None,
        min_size: int = 1,
        max_size: int = 8,
        schema_mode: schema.SchemaMode = schema.SchemaMode.VALIDATE,
        read_only: bool = False,
        public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
        migration_reset_empty_recall_state: bool = False,
        migration_expected_input_state: schema.SchemaState | None = None,
        migration_operation_sha256: str | None = None,
        migration_receipt_json: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        from cayu.runtime._cost_accounting_refresh import CostAccountingAuthority
        from cayu.runtime._usage_accounting import SessionUsageCache

        self._cost_accounting_authority = CostAccountingAuthority()
        self._session_usage_cache = SessionUsageCache()
        self._clock = utc_clock(clock)
        if public_authority_alias_codec is not None and not isinstance(
            public_authority_alias_codec,
            PublicAuthorityAliasCodec,
        ):
            raise TypeError("public_authority_alias_codec must be a PublicAuthorityAliasCodec.")
        super().__init__(
            conninfo,
            pool=pool,
            min_size=min_size,
            max_size=max_size,
            schema_mode=schema_mode,
            read_only=read_only,
            migration_reset_empty_recall_state=migration_reset_empty_recall_state,
            migration_expected_input_state=migration_expected_input_state,
            migration_operation_sha256=migration_operation_sha256,
            migration_receipt_json=migration_receipt_json,
        )
        self.service_durability = (
            RuntimeStoreDurability.READ_ONLY if read_only else RuntimeStoreDurability.DURABLE
        )
        self._public_authority_alias_codec = public_authority_alias_codec
        self._public_authority_alias_backfill_lock = asyncio.Lock()
        self._public_authority_aliases_reconciled = False

    @property
    def public_authority_alias_codec(self) -> PublicAuthorityAliasCodec | None:
        """Return the immutable codec configured for durable alias registration."""

        return self._public_authority_alias_codec

    async def _terminalize_zero_work_interruption(
        self,
        request: ZeroWorkInterruptionRequest,
    ) -> ZeroWorkInterruptionPublication | None:
        """Atomically prove and terminalize zero work; unsupported stores decline."""
        from cayu.storage._zero_work_interruption import postgres_terminalize

        return await postgres_terminalize(self, request)

    async def _preflight_migration_authority(self, cur: Any) -> None:
        await migration_authority.preflight_postgres_public_authority(
            cur,
            self.public_authority_alias_codec,
        )

    @staticmethod
    async def _session_store_now(cur: Any) -> datetime:
        await cur.execute("SELECT clock_timestamp()")
        row = await cur.fetchone()
        if row is None or type(row[0]) is not datetime:
            raise RuntimeError("Postgres did not return authoritative session-store time.")
        return pg_support.to_utc(row[0])

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
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await self._register_public_authority_alias_row(
                        cur,
                        field_name=field_name,
                        scope_key=scope_key,
                        public_alias=public_alias,
                        private_value=private_value,
                    )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

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
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT private_value
                FROM cayu_public_authority_aliases
                WHERE field_name = %s
                  AND scope_session_id = %s
                  AND public_alias = %s
                """,
                (field_name, scope_key, public_alias),
            )
            row = await cur.fetchone()
            return _authenticated_public_authority_alias_private_value(
                self.public_authority_alias_codec,
                public_alias,
                None if row is None else str(row[0]),
                field_name=field_name,
                scope_session_id=scope_session_id,
            )

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
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT EXISTS(
                    SELECT 1
                    FROM cayu_public_authority_aliases
                    WHERE field_name = %s
                      AND scope_session_id = %s
                      AND private_value = %s
                )
                """,
                (field_name, scope_key, private_value),
            )
            row = await cur.fetchone()
            return bool(row is not None and row[0])

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
        await self._ensure_ready()
        resolved: list[TargetedToolGrantRecord] = []
        outcomes: list[TargetedToolGrantIssueOutcome] = []
        resolved_events: list[Event] = []
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT agent_name, environment_name, status, run_epoch, invocation "
                        "FROM cayu_sessions WHERE id = %s FOR UPDATE",
                        (session_id,),
                    )
                    session_row = await cur.fetchone()
                    if session_row is None:
                        raise KeyError(f"Session not found: {session_id}")
                    if int(session_row[3]) != expected_run_epoch:
                        raise SessionRunFenced(
                            "Session source run epoch is stale: expected "
                            f"{expected_run_epoch}, current {session_row[3]}."
                        )
                    if str(session_row[2]) != str(SessionStatus.RUNNING):
                        raise SessionStatusConflict("Targeted grants require a running session.")
                    if interaction_ids:
                        await cur.execute(
                            "SELECT interaction_id, event_type FROM cayu_events "
                            "WHERE session_id = %s AND event_type = ANY(%s) "
                            "ORDER BY sequence DESC LIMIT 1",
                            (
                                session_id,
                                [str(value) for value in INTERACTION_LIFECYCLE_EVENT_TYPES],
                            ),
                        )
                        latest_interaction = await cur.fetchone()
                        if (
                            latest_interaction is None
                            or latest_interaction[0] != next(iter(interaction_ids))
                            or EventType(str(latest_interaction[1]))
                            in INTERACTION_TERMINAL_EVENT_TYPES
                        ):
                            raise ValueError(
                                "Targeted grants require the current open interaction."
                            )
                        await cur.execute(
                            "SELECT event FROM cayu_events WHERE session_id = %s "
                            "AND interaction_id = %s AND event_type = %s "
                            "ORDER BY sequence ASC LIMIT 1",
                            (
                                session_id,
                                next(iter(interaction_ids)),
                                str(EventType.INTERACTION_STARTED),
                            ),
                        )
                        interaction_started_row = await cur.fetchone()
                        if interaction_started_row is None:
                            raise RuntimeError(
                                "Targeted grant issuance lost interaction admission."
                            )
                        validate_targeted_tool_grant_batch_evidence(
                            copied_records,
                            Event(**pg_support._json_obj(interaction_started_row[0])),
                        )
                    invocation = SessionInvocation.model_validate(session_row[4])
                    new_events: list[Event] = []
                    for record, event in zip(copied_records, copied_events, strict=True):
                        if (
                            record.session_id != session_id
                            or record.agent_name != session_row[0]
                            or record.environment_name != session_row[1]
                            or record.principal != invocation.origin.subject
                            or record.tenant != invocation.origin.tenant
                        ):
                            raise ValueError("Targeted grant scope is inconsistent.")
                        await cur.execute(
                            "SELECT grant_id, session_id, interaction_id, request_id, tool_ref, "
                            "generation_id, tool_id, tool_name, catalogue_revision, "
                            "descriptor_version, issued_at, expires_at, max_calls, used_calls, "
                            "revoked_at, record FROM cayu_targeted_tool_grants "
                            "WHERE session_id = %s AND interaction_id = %s "
                            "AND (request_id = %s OR tool_id = %s) LIMIT 2 FOR UPDATE",
                            (
                                session_id,
                                record.interaction_id,
                                record.request_id,
                                record.tool_id,
                            ),
                        )
                        existing_row = await cur.fetchone()
                        if existing_row is not None:
                            existing = _targeted_tool_grant_from_postgres_row(existing_row)
                            await _validate_targeted_tool_use_counts(cur, (existing,))
                            if existing.request_id != record.request_id:
                                raise ValueError(
                                    "Targeted grant tool identity conflicts with durable authority."
                                )
                            if existing.grant_id != record.grant_id:
                                raise ValueError(
                                    "Targeted grant request identity conflicts with "
                                    "durable authority."
                                )
                            resolved.append(
                                targeted_tool_grant_with_active_reference(existing, codec)
                            )
                            outcomes.append(TargetedToolGrantIssueOutcome.REUSED)
                            await cur.execute(
                                "SELECT event FROM cayu_events "
                                "WHERE session_id = %s AND event_id = %s",
                                (session_id, event.id),
                            )
                            issued_row = await cur.fetchone()
                            if issued_row is None:
                                raise RuntimeError(
                                    "Targeted grant lost its durable issuance evidence."
                                )
                            validate_targeted_tool_grant_issuance_evidence(
                                existing,
                                Event(**pg_support._json_obj(issued_row[0])),
                            )
                            reused_event = targeted_tool_grant_event(
                                existing,
                                event_type=EventType.TARGETED_TOOL_GRANT_REUSED,
                                timestamp=event.timestamp,
                                outcome=TargetedToolGrantIssueOutcome.REUSED.value,
                                event_id_suffix="reused",
                            )
                            resolved_events.append(
                                await self._append_event_once_with_cursor(
                                    cur,
                                    reused_event,
                                    expected_run_epoch=expected_run_epoch,
                                )
                            )
                            continue
                        await cur.execute(
                            "SELECT 1 FROM cayu_targeted_tool_grants "
                            "WHERE grant_id = %s FOR UPDATE",
                            (record.grant_id,),
                        )
                        if await cur.fetchone() is not None:
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
                            await self._register_public_authority_alias_row(
                                cur,
                                field_name=field_name,
                                scope_key=scope_key,
                                public_alias=public_alias,
                                private_value=record.grant_id,
                            )
                        await cur.execute(
                            """
                            INSERT INTO cayu_targeted_tool_grants (
                                grant_id, session_id, interaction_id, request_id, tool_ref,
                                generation_id, tool_id, tool_name, catalogue_revision,
                                descriptor_version, issued_at, expires_at, max_calls,
                                used_calls, revoked_at, record
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s
                            )
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
                                pg_support.to_utc(record.issued_at),
                                pg_support.to_utc(record.expires_at),
                                record.max_calls,
                                record.used_calls,
                                None,
                                pg_support._dumps(record.model_dump(mode="json")),
                            ),
                        )
                        resolved.append(record)
                        outcomes.append(TargetedToolGrantIssueOutcome.ISSUED)
                        resolved_events.append(event)
                        new_events.append(event)
                    if interaction_ids:
                        await cur.execute(
                            "SELECT COUNT(*) FROM cayu_targeted_tool_grants "
                            "WHERE session_id = %s AND interaction_id = %s",
                            (session_id, next(iter(interaction_ids))),
                        )
                        interaction_count_row = await cur.fetchone()
                        if (
                            interaction_count_row is None
                            or int(interaction_count_row[0]) > TARGETED_TOOL_GRANT_MAX_REQUESTS
                        ):
                            raise ValueError(
                                "Targeted grant interaction exceeds its bounded count."
                            )
                    await self._append_events_with_cursor(
                        cur,
                        session_id,
                        new_events,
                        expected_run_epoch=expected_run_epoch,
                    )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
        return TargetedToolGrantIssueResult(
            records=tuple(resolved),
            outcomes=tuple(outcomes),
            events=tuple(resolved_events),
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
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT 1 FROM cayu_sessions WHERE id = %s FOR SHARE",
                (session_id,),
            )
            if await cur.fetchone() is None:
                raise KeyError(f"Session not found: {session_id}")
            if interaction_id is None:
                await cur.execute(
                    "SELECT grant_id, session_id, interaction_id, request_id, tool_ref, "
                    "generation_id, tool_id, tool_name, catalogue_revision, "
                    "descriptor_version, issued_at, expires_at, max_calls, used_calls, "
                    "revoked_at, record FROM cayu_targeted_tool_grants "
                    "WHERE session_id = %s ORDER BY issued_at, grant_id LIMIT %s",
                    (session_id, limit + 1),
                )
            else:
                await cur.execute(
                    "SELECT grant_id, session_id, interaction_id, request_id, tool_ref, "
                    "generation_id, tool_id, tool_name, catalogue_revision, "
                    "descriptor_version, issued_at, expires_at, max_calls, used_calls, "
                    "revoked_at, record FROM cayu_targeted_tool_grants "
                    "WHERE session_id = %s AND interaction_id = %s "
                    "ORDER BY issued_at, grant_id LIMIT %s",
                    (session_id, interaction_id, limit + 1),
                )
            rows = await cur.fetchall()
            if len(rows) > limit:
                raise ValueError("Targeted grant inspection exceeds its bounded result limit.")
            if not rows:
                return ()
            records = tuple(_targeted_tool_grant_from_postgres_row(row) for row in rows)
            await _validate_targeted_tool_use_counts(cur, records)
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

    async def load_targeted_tool_grant_state(
        self,
        session_id: str,
    ) -> TargetedToolGrantStateSnapshot:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT 1 FROM cayu_sessions WHERE id = %s FOR SHARE",
                (session_id,),
            )
            if await cur.fetchone() is None:
                raise KeyError(f"Session not found: {session_id}")
            await cur.execute(
                "SELECT grant_id, session_id, interaction_id, request_id, tool_ref, "
                "generation_id, tool_id, tool_name, catalogue_revision, descriptor_version, "
                "issued_at, expires_at, max_calls, used_calls, revoked_at, record "
                "FROM cayu_targeted_tool_grants "
                "WHERE session_id = %s ORDER BY issued_at, grant_id",
                (session_id,),
            )
            grant_rows = await cur.fetchall()
            await cur.execute(
                "SELECT use_id, grant_id, session_id, interaction_id, model_step_id, "
                "outer_tool_call_id, arguments_sha256, invocation_id, bound_at, record "
                "FROM cayu_targeted_tool_grant_uses "
                "WHERE session_id = %s ORDER BY bound_at, use_id",
                (session_id,),
            )
            use_rows = await cur.fetchall()
            if not grant_rows:
                if use_rows:
                    raise ValueError("Targeted grant uses exist without grant records.")
                return TargetedToolGrantStateSnapshot()
            codec = self.public_authority_alias_codec
            if codec is None:
                raise RuntimeError("Targeted grants require a public authority alias codec.")
            records: list[TargetedToolGrantRecord] = []
            for row in grant_rows:
                record = _targeted_tool_grant_from_postgres_row(row)
                records.append(targeted_tool_grant_with_active_reference(record, codec))
            return TargetedToolGrantStateSnapshot(
                records=tuple(records),
                uses=tuple(_targeted_tool_use_from_postgres_row(row) for row in use_rows),
            )

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
        await self._ensure_ready()
        result: TargetedToolUseResult
        new_event: Event | None = None
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT agent_name, environment_name, status, run_epoch "
                        "FROM cayu_sessions WHERE id = %s FOR UPDATE",
                        (request.session_id,),
                    )
                    session_row = await cur.fetchone()
                    if session_row is None:
                        raise KeyError(f"Session not found: {request.session_id}")
                    if int(session_row[3]) != request.expected_run_epoch:
                        raise SessionRunFenced(
                            "Session source run epoch is stale: expected "
                            f"{request.expected_run_epoch}, current {session_row[3]}."
                        )
                    if str(session_row[2]) != str(SessionStatus.RUNNING):
                        raise SessionStatusConflict("Targeted tool use requires a running session.")

                    async def unresolved(
                        reason: TargetedToolUseRejectionReason,
                    ) -> TargetedToolUseResult:
                        session_agent_name = str(session_row[0])
                        session_environment_name = (
                            None if session_row[1] is None else str(session_row[1])
                        )
                        event = targeted_tool_unresolved_rejection_event(
                            request,
                            reason=reason,
                            timestamp=observed_at,
                            agent_name=session_agent_name,
                            environment_name=session_environment_name,
                        )
                        persisted = await self._append_event_once_with_cursor(
                            cur,
                            event,
                            expected_run_epoch=request.expected_run_epoch,
                        )
                        validate_targeted_tool_unresolved_rejection_evidence(
                            request,
                            reason=reason,
                            event=persisted,
                            agent_name=session_agent_name,
                            environment_name=session_environment_name,
                        )
                        return TargetedToolUseResult(
                            disposition=TargetedToolUseDisposition.REJECTED,
                            reason=reason,
                            event=persisted,
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
                        result = await unresolved(TargetedToolUseRejectionReason.MALFORMED)
                        await conn.commit()
                        return result
                    await cur.execute(
                        "SELECT scope_session_id, private_value "
                        "FROM cayu_public_authority_aliases "
                        "WHERE field_name = %s AND public_alias = %s LIMIT 2",
                        (TARGETED_TOOL_REFERENCE_FIELD_NAME, request.tool_ref),
                    )
                    aliases = await cur.fetchall()
                    if not aliases:
                        result = await unresolved(TargetedToolUseRejectionReason.UNKNOWN)
                    elif len(aliases) != 1:
                        raise RuntimeError("Targeted tool reference registry is ambiguous.")
                    else:
                        scope_session_id = str(aliases[0][0])
                        grant_id = str(aliases[0][1])
                        if not codec.matches(
                            request.tool_ref,
                            grant_id,
                            field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                            session_id=scope_session_id,
                        ):
                            result = await unresolved(TargetedToolUseRejectionReason.UNKNOWN)
                        elif scope_session_id != request.session_id:
                            result = await unresolved(TargetedToolUseRejectionReason.CROSS_SESSION)
                        else:
                            await cur.execute(
                                "SELECT grant_id, session_id, interaction_id, request_id, "
                                "tool_ref, generation_id, tool_id, tool_name, "
                                "catalogue_revision, descriptor_version, issued_at, expires_at, "
                                "max_calls, used_calls, revoked_at, record "
                                "FROM cayu_targeted_tool_grants "
                                "WHERE grant_id = %s FOR UPDATE",
                                (grant_id,),
                            )
                            grant_row = await cur.fetchone()
                            if grant_row is None:
                                raise RuntimeError("Targeted tool reference lost its grant record.")
                            record = _targeted_tool_grant_from_postgres_row(grant_row)
                            await _validate_targeted_tool_use_counts(cur, (record,))

                            async def rejected(
                                reason: TargetedToolUseRejectionReason,
                            ) -> TargetedToolUseResult:
                                if reason is TargetedToolUseRejectionReason.EXPIRED:
                                    expiry_event = targeted_tool_grant_event(
                                        record,
                                        event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                                        timestamp=observed_at,
                                        outcome="expired",
                                        event_id_suffix="expired",
                                        rejection_reason=reason,
                                    )
                                    persisted_expiry = await self._append_event_once_with_cursor(
                                        cur,
                                        expiry_event,
                                        expected_run_epoch=request.expected_run_epoch,
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
                                persisted = await self._append_event_once_with_cursor(
                                    cur,
                                    rejection_event,
                                    expected_run_epoch=request.expected_run_epoch,
                                )
                                validate_targeted_tool_use_rejection_evidence(
                                    record,
                                    request,
                                    reason=reason,
                                    event=persisted,
                                )
                                return TargetedToolUseResult(
                                    disposition=TargetedToolUseDisposition.REJECTED,
                                    reason=reason,
                                    grant=record,
                                    event=persisted,
                                )

                            await cur.execute(
                                "SELECT 1 FROM cayu_events "
                                "WHERE session_id = %s AND interaction_id = %s "
                                "AND event_type = ANY(%s) LIMIT 1",
                                (
                                    request.session_id,
                                    record.interaction_id,
                                    [
                                        str(event_type)
                                        for event_type in INTERACTION_TERMINAL_EVENT_TYPES
                                    ],
                                ),
                            )
                            if await cur.fetchone() is not None:
                                result = await rejected(TargetedToolUseRejectionReason.EXPIRED)
                                await conn.commit()
                                return result
                            await cur.execute(
                                "SELECT use_id, grant_id, session_id, interaction_id, "
                                "model_step_id, outer_tool_call_id, arguments_sha256, "
                                "invocation_id, bound_at, record "
                                "FROM cayu_targeted_tool_grant_uses "
                                "WHERE session_id = %s AND interaction_id = %s "
                                "AND (invocation_id = %s OR outer_tool_call_id = %s) LIMIT 2",
                                (
                                    request.session_id,
                                    request.interaction_id,
                                    request.invocation_id,
                                    request.outer_tool_call_id,
                                ),
                            )
                            use_rows = await cur.fetchall()
                            if use_rows:
                                scope_rejection = targeted_tool_use_scope_rejection_reason(
                                    record,
                                    request,
                                )
                                if scope_rejection is not None:
                                    result = await rejected(scope_rejection)
                                    new_event = result.event
                                elif len(use_rows) != 1:
                                    result = await rejected(
                                        TargetedToolUseRejectionReason.ALTERED_REPLAY
                                    )
                                    new_event = result.event
                                else:
                                    binding = _targeted_tool_use_from_postgres_row(use_rows[0])
                                    candidate = targeted_tool_use_binding(
                                        grant_id,
                                        request,
                                        bound_at=binding.bound_at,
                                    )
                                    if binding != candidate:
                                        result = await rejected(
                                            TargetedToolUseRejectionReason.ALTERED_REPLAY
                                        )
                                        new_event = result.event
                                    else:
                                        expected_event = targeted_tool_grant_event(
                                            record,
                                            event_type=(EventType.TARGETED_TOOL_REFERENCE_CONSUMED),
                                            timestamp=binding.bound_at,
                                            outcome=TargetedToolUseDisposition.BOUND.value,
                                            event_id_suffix=f"use:{binding.use_id}",
                                            binding=binding,
                                        )
                                        await cur.execute(
                                            "SELECT event FROM cayu_events "
                                            "WHERE session_id = %s AND event_id = %s",
                                            (request.session_id, expected_event.id),
                                        )
                                        event_row = await cur.fetchone()
                                        if event_row is None:
                                            raise RuntimeError(
                                                "Targeted tool use lost its durable event evidence."
                                            )
                                        validate_targeted_tool_grant_lifecycle_event(
                                            record,
                                            Event(**pg_support._json_obj(event_row[0])),
                                            event_type=(EventType.TARGETED_TOOL_REFERENCE_CONSUMED),
                                            outcome=TargetedToolUseDisposition.BOUND.value,
                                            event_id_suffix=f"use:{binding.use_id}",
                                            binding=binding,
                                            require_current_call_count=False,
                                        )
                                        rejoined_event = targeted_tool_grant_event(
                                            record,
                                            event_type=(EventType.TARGETED_TOOL_REFERENCE_REJOINED),
                                            timestamp=observed_at,
                                            outcome=TargetedToolUseDisposition.REJOINED.value,
                                            event_id_suffix=f"rejoined:{binding.use_id}",
                                            binding=binding,
                                        )
                                        new_event = await self._append_event_once_with_cursor(
                                            cur,
                                            rejoined_event,
                                            expected_run_epoch=request.expected_run_epoch,
                                        )
                                        result = TargetedToolUseResult(
                                            disposition=TargetedToolUseDisposition.REJOINED,
                                            grant=record,
                                            binding=binding,
                                            event=new_event,
                                        )
                            else:
                                rejection = targeted_tool_use_rejection_reason(
                                    record,
                                    request,
                                    observed_at=observed_at,
                                )
                                if rejection is not None:
                                    result = await rejected(rejection)
                                    new_event = result.event
                                else:
                                    binding = targeted_tool_use_binding(
                                        grant_id,
                                        request,
                                        bound_at=observed_at,
                                    )
                                    updated = TargetedToolGrantRecord.model_validate(
                                        record.model_copy(
                                            update={"used_calls": record.used_calls + 1}
                                        ).model_dump(mode="python")
                                    )
                                    await cur.execute(
                                        """
                                        INSERT INTO cayu_targeted_tool_grant_uses (
                                            use_id, grant_id, session_id, interaction_id,
                                            model_step_id, outer_tool_call_id,
                                            arguments_sha256, invocation_id, bound_at, record
                                        ) VALUES (
                                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                                        )
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
                                            pg_support.to_utc(binding.bound_at),
                                            pg_support._dumps(binding.model_dump(mode="json")),
                                        ),
                                    )
                                    await cur.execute(
                                        "UPDATE cayu_targeted_tool_grants "
                                        "SET used_calls = %s, record = %s "
                                        "WHERE grant_id = %s AND used_calls = %s",
                                        (
                                            updated.used_calls,
                                            pg_support._dumps(updated.model_dump(mode="json")),
                                            grant_id,
                                            record.used_calls,
                                        ),
                                    )
                                    if cur.rowcount != 1:
                                        raise RuntimeError("Targeted grant use lost its row lock.")
                                    new_event = targeted_tool_grant_event(
                                        updated,
                                        event_type=(EventType.TARGETED_TOOL_REFERENCE_CONSUMED),
                                        timestamp=observed_at,
                                        outcome=TargetedToolUseDisposition.BOUND.value,
                                        event_id_suffix=f"use:{binding.use_id}",
                                        binding=binding,
                                    )
                                    result = TargetedToolUseResult(
                                        disposition=TargetedToolUseDisposition.BOUND,
                                        grant=updated,
                                        binding=binding,
                                        event=new_event,
                                    )
                                    await self._append_events_with_cursor(
                                        cur,
                                        request.session_id,
                                        [new_event],
                                        expected_run_epoch=request.expected_run_epoch,
                                    )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
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
        await self._ensure_ready()
        record: TargetedToolGrantRecord | None = None
        event: Event | None = None
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT run_epoch FROM cayu_sessions WHERE id = %s FOR UPDATE",
                        (session_id,),
                    )
                    session_row = await cur.fetchone()
                    if session_row is None:
                        raise KeyError(f"Session not found: {session_id}")
                    if int(session_row[0]) != expected_run_epoch:
                        raise SessionRunFenced(
                            "Session source run epoch is stale: expected "
                            f"{expected_run_epoch}, current {session_row[0]}."
                        )
                    try:
                        parsed = parse_public_authority_alias(tool_ref)
                    except (TypeError, ValueError):
                        parsed = None
                    if parsed is None or parsed.field_name != TARGETED_TOOL_REFERENCE_FIELD_NAME:
                        await conn.commit()
                        return None
                    await cur.execute(
                        "SELECT scope_session_id, private_value "
                        "FROM cayu_public_authority_aliases "
                        "WHERE field_name = %s AND public_alias = %s LIMIT 2",
                        (TARGETED_TOOL_REFERENCE_FIELD_NAME, tool_ref),
                    )
                    aliases = await cur.fetchall()
                    if aliases:
                        if len(aliases) != 1:
                            raise RuntimeError("Targeted tool reference registry is ambiguous.")
                        scope = str(aliases[0][0])
                        grant_id = str(aliases[0][1])
                        if scope == session_id and codec.matches(
                            tool_ref,
                            grant_id,
                            field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                            session_id=scope,
                        ):
                            await cur.execute(
                                "SELECT grant_id, session_id, interaction_id, request_id, "
                                "tool_ref, generation_id, tool_id, tool_name, "
                                "catalogue_revision, descriptor_version, issued_at, expires_at, "
                                "max_calls, used_calls, revoked_at, record "
                                "FROM cayu_targeted_tool_grants "
                                "WHERE grant_id = %s FOR UPDATE",
                                (grant_id,),
                            )
                            row = await cur.fetchone()
                            if row is None:
                                raise RuntimeError("Targeted tool reference lost its grant record.")
                            stored = _targeted_tool_grant_from_postgres_row(row)
                            await _validate_targeted_tool_use_counts(cur, (stored,))
                            if stored.revoked_at is not None:
                                if stored.revocation_reason != reason:
                                    raise ValueError(
                                        "Targeted grant was revoked with a different reason."
                                    )
                                record = stored
                                expected_event = targeted_tool_grant_event(
                                    stored,
                                    event_type=EventType.TARGETED_TOOL_GRANT_REVOKED,
                                    timestamp=stored.revoked_at,
                                    outcome="revoked",
                                    event_id_suffix="revoked",
                                )
                                await cur.execute(
                                    "SELECT event FROM cayu_events "
                                    "WHERE session_id = %s AND event_id = %s",
                                    (session_id, expected_event.id),
                                )
                                event_row = await cur.fetchone()
                                if event_row is None:
                                    raise RuntimeError(
                                        "Targeted grant revocation lost its durable event evidence."
                                    )
                                event = Event(**pg_support._json_obj(event_row[0]))
                                validate_targeted_tool_grant_revocation_evidence(
                                    stored,
                                    event,
                                )
                            else:
                                for owner in await self._closure_lineage_owners(cur, (session_id,)):
                                    _check_closure_lineage_owner(owner, (session_id,))
                                await cur.execute(
                                    "SELECT MAX(bound_at) "
                                    "FROM cayu_targeted_tool_grant_uses WHERE grant_id = %s",
                                    (grant_id,),
                                )
                                latest_use_row = await cur.fetchone()
                                latest_bound_at = latest_use_row[0]
                                if latest_bound_at is not None and (
                                    pg_support.to_utc(latest_bound_at) > revoked_at
                                ):
                                    raise ValueError(
                                        "revoked_at cannot precede a bound targeted tool use."
                                    )
                                record = TargetedToolGrantRecord.model_validate(
                                    stored.model_copy(
                                        update={
                                            "revoked_at": revoked_at,
                                            "revocation_reason": reason,
                                        }
                                    ).model_dump(mode="python")
                                )
                                await cur.execute(
                                    "UPDATE cayu_targeted_tool_grants "
                                    "SET revoked_at = %s, record = %s "
                                    "WHERE grant_id = %s AND revoked_at IS NULL",
                                    (
                                        pg_support.to_utc(revoked_at),
                                        pg_support._dumps(record.model_dump(mode="json")),
                                        grant_id,
                                    ),
                                )
                                if cur.rowcount != 1:
                                    raise RuntimeError(
                                        "Targeted grant revocation lost its row lock."
                                    )
                                event = targeted_tool_grant_event(
                                    record,
                                    event_type=EventType.TARGETED_TOOL_GRANT_REVOKED,
                                    timestamp=revoked_at,
                                    outcome="revoked",
                                    event_id_suffix="revoked",
                                )
                                await self._append_events_with_cursor(
                                    cur,
                                    session_id,
                                    [event],
                                    expected_run_epoch=expected_run_epoch,
                                )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
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
        await self._ensure_ready()
        result: TargetedToolGrantReconstructionResult
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT status, run_epoch FROM cayu_sessions WHERE id = %s FOR UPDATE",
                        (session_id,),
                    )
                    session_row = await cur.fetchone()
                    if session_row is None:
                        raise KeyError(f"Session not found: {session_id}")
                    if int(session_row[1]) != expected_run_epoch:
                        raise SessionRunFenced(
                            "Session source run epoch is stale: expected "
                            f"{expected_run_epoch}, current {session_row[1]}."
                        )
                    if str(session_row[0]) != str(SessionStatus.RUNNING):
                        raise SessionStatusConflict(
                            "Grant reconstruction requires a running session."
                        )
                    await cur.execute(
                        "SELECT grant_id, session_id, interaction_id, request_id, tool_ref, "
                        "generation_id, tool_id, tool_name, catalogue_revision, "
                        "descriptor_version, issued_at, expires_at, max_calls, used_calls, "
                        "revoked_at, record FROM cayu_targeted_tool_grants "
                        "WHERE session_id = %s AND interaction_id = %s "
                        "ORDER BY issued_at, grant_id LIMIT %s",
                        (
                            session_id,
                            interaction_id,
                            TARGETED_TOOL_GRANT_MAX_REQUESTS + 1,
                        ),
                    )
                    records = tuple(
                        _targeted_tool_grant_from_postgres_row(row) for row in await cur.fetchall()
                    )
                    if len(records) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
                        raise ValueError("Targeted grant interaction exceeds its bounded count.")
                    await _validate_targeted_tool_use_counts(cur, records)
                    await cur.execute(
                        "SELECT event FROM cayu_events WHERE session_id = %s "
                        "AND interaction_id = %s AND event_type = %s "
                        "ORDER BY sequence ASC LIMIT 1",
                        (session_id, interaction_id, str(EventType.INTERACTION_STARTED)),
                    )
                    interaction_started_row = await cur.fetchone()
                    if interaction_started_row is None:
                        raise RuntimeError(
                            "Targeted grant reconstruction lost interaction admission."
                        )
                    validate_targeted_tool_grant_batch_evidence(
                        records,
                        Event(**pg_support._json_obj(interaction_started_row[0])),
                    )
                    await cur.execute(
                        "SELECT 1 FROM cayu_events "
                        "WHERE session_id = %s AND interaction_id = %s "
                        "AND event_type = ANY(%s) LIMIT 1",
                        (
                            session_id,
                            interaction_id,
                            [str(value) for value in INTERACTION_TERMINAL_EVENT_TYPES],
                        ),
                    )
                    interaction_ended = await cur.fetchone() is not None
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
                                persisted_expiry = await self._append_event_once_with_cursor(
                                    cur,
                                    targeted_tool_grant_event(
                                        record,
                                        event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                                        timestamp=observed_at,
                                        outcome="expired",
                                        event_id_suffix="expired",
                                        rejection_reason=reason,
                                    ),
                                    expected_run_epoch=expected_run_epoch,
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
                        persisted = await self._append_event_once_with_cursor(
                            cur,
                            event,
                            expected_run_epoch=expected_run_epoch,
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
                    result = TargetedToolGrantReconstructionResult(
                        valid=tuple(valid),
                        rejected=tuple(rejected),
                        events=tuple(events),
                    )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
        return result

    async def _register_public_authority_alias_row(
        self,
        cur: Any,
        *,
        field_name: str,
        scope_key: str,
        public_alias: str,
        private_value: str,
    ) -> None:
        await cur.execute(
            """
            INSERT INTO cayu_public_authority_aliases (
                field_name,
                scope_session_id,
                public_alias,
                private_value
            ) VALUES (%s, %s, %s, %s)
            ON CONFLICT (field_name, scope_session_id, public_alias)
            DO UPDATE SET private_value = cayu_public_authority_aliases.private_value
            RETURNING private_value
            """,
            (field_name, scope_key, public_alias, private_value),
        )
        row = await cur.fetchone()
        if row is None:  # pragma: no cover - RETURNING is unconditional above
            raise RuntimeError("Public authority alias registration was not persisted.")
        stored = str(row[0])
        if not hmac.compare_digest(
            stored.encode("utf-8"),
            private_value.encode("utf-8"),
        ):
            raise ValueError("Public authority alias conflicts with existing private authority.")

    async def _ensure_ready(self) -> None:
        await super()._ensure_ready()
        if not self._public_authority_aliases_reconciled:
            async with self._public_authority_alias_backfill_lock:
                if not self._public_authority_aliases_reconciled:
                    await self._reconcile_public_authority_alias_keys()
                    self._public_authority_aliases_reconciled = True
        await self._assert_current_public_authority_configuration()

    async def _assert_current_public_authority_configuration(self) -> None:
        codec = self.public_authority_alias_codec
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT active_key_id, keyring_fingerprint "
                "FROM cayu_public_authority_alias_config "
                "WHERE singleton = TRUE"
            )
            row = await cur.fetchone()
        if codec is None:
            if row is not None:
                raise RuntimeError(
                    "Postgres public authority aliases require the deployment keyring."
                )
            return
        if (
            row is None
            or str(row[0]) != codec.keyring.active_key_id
            or str(row[1]) != codec.keyring_fingerprint()
        ):
            raise RuntimeError(
                "Postgres public authority alias key configuration is stale; reopen the store."
            )

    async def _reconcile_public_authority_alias_keys(self) -> None:
        codec = self.public_authority_alias_codec
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await postgres_base._acquire_schema_transaction_lock(
                        conn,
                        cur,
                        read_only=self._read_only,
                    )
                    await cur.execute(
                        "SELECT key_id, fingerprint, backfill_completed "
                        "FROM cayu_public_authority_alias_keys ORDER BY key_id"
                    )
                    durable = {
                        str(row[0]): (str(row[1]), bool(row[2])) for row in await cur.fetchall()
                    }
                    if codec is None:
                        await cur.execute(
                            "SELECT EXISTS(SELECT 1 FROM cayu_public_authority_alias_config)"
                        )
                        config_exists = await cur.fetchone()
                        if durable or (config_exists is not None and bool(config_exists[0])):
                            raise RuntimeError(
                                "Postgres public authority aliases are initialized; "
                                "configure the deployment's alias keyring before opening "
                                "this session store."
                            )
                        await conn.commit()
                        return

                    configured = {
                        key_id: codec.key_fingerprint(key_id) for key_id in codec.keyring.key_ids
                    }
                    unavailable_incomplete = [
                        key_id
                        for key_id, (_fingerprint, completed) in durable.items()
                        if key_id not in configured and not completed
                    ]
                    if unavailable_incomplete:
                        raise RuntimeError(
                            "Public authority alias backfill is incomplete for an "
                            "unavailable historical key; restore that key before startup."
                        )
                    for key_id, fingerprint in configured.items():
                        existing = durable.get(key_id)
                        if existing is not None and not hmac.compare_digest(
                            existing[0].encode("utf-8"),
                            fingerprint.encode("utf-8"),
                        ):
                            raise RuntimeError(
                                "Public authority alias key ID is already bound to "
                                "different key material."
                            )
                    missing = [key_id for key_id in configured if key_id not in durable]
                    incomplete = [
                        key_id
                        for key_id in configured
                        if key_id in durable and not durable[key_id][1]
                    ]
                    if self._read_only and (missing or incomplete):
                        raise RuntimeError(
                            "Read-only Postgres stores require a completed writable "
                            "public authority alias backfill for every configured key."
                        )
                    if missing or incomplete:
                        # Fence every identity producer while the new key's reverse
                        # index is backfilled. Writers that started first commit
                        # before this lock; writers that start later observe the key
                        # marker and must register aliases in their own transaction.
                        await cur.execute(
                            "LOCK TABLE cayu_sessions, cayu_events, "
                            "cayu_transcript_messages, cayu_targeted_tool_grants "
                            "IN SHARE ROW EXCLUSIVE MODE"
                        )
                    for key_id in missing:
                        await cur.execute(
                            "INSERT INTO cayu_public_authority_alias_keys "
                            "(key_id, fingerprint, backfill_completed) "
                            "VALUES (%s, %s, FALSE)",
                            (key_id, configured[key_id]),
                        )

                if missing or incomplete:
                    await self._backfill_public_authority_aliases(conn)
                    async with conn.cursor() as cur:
                        await cur.execute(
                            "UPDATE cayu_public_authority_alias_keys "
                            "SET backfill_completed = TRUE WHERE key_id = ANY(%s)",
                            (list(configured),),
                        )
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT active_key_id, keyring_fingerprint, generation, "
                        "retired_key_ids "
                        "FROM cayu_public_authority_alias_config WHERE singleton = TRUE"
                        + ("" if self._read_only else " FOR UPDATE")
                    )
                    config = await cur.fetchone()
                    desired_active = codec.keyring.active_key_id
                    desired_keyring_fingerprint = codec.keyring_fingerprint()
                    if config is None:
                        if self._read_only:
                            raise RuntimeError(
                                "Read-only Postgres store has no active alias-key state."
                            )
                        await cur.execute(
                            "INSERT INTO cayu_public_authority_alias_config "
                            "(singleton, active_key_id, keyring_fingerprint, generation, "
                            "retired_key_ids) VALUES (TRUE, %s, %s, 1, '[]'::jsonb)",
                            (desired_active, desired_keyring_fingerprint),
                        )
                    elif (
                        str(config[0]) != desired_active
                        or str(config[1]) != desired_keyring_fingerprint
                    ):
                        if self._read_only:
                            raise RuntimeError(
                                "Read-only Postgres public authority alias active key is stale."
                            )
                        retired_value = config[3]
                        retired = (
                            retired_value
                            if type(retired_value) is list
                            else json.loads(str(retired_value))
                        )
                        if type(retired) is not list or not all(
                            type(value) is str for value in retired
                        ):
                            raise RuntimeError(
                                "Postgres public authority alias rotation state is malformed."
                            )
                        if str(config[0]) != desired_active and desired_active in retired:
                            raise RuntimeError(
                                "A retired public authority alias key cannot become active again."
                            )
                        if str(config[0]) != desired_active:
                            retired.append(str(config[0]))
                        await cur.execute(
                            "UPDATE cayu_public_authority_alias_config "
                            "SET active_key_id = %s, keyring_fingerprint = %s, "
                            "generation = %s, retired_key_ids = %s::jsonb "
                            "WHERE singleton = TRUE",
                            (
                                desired_active,
                                desired_keyring_fingerprint,
                                int(config[2]) + 1,
                                json.dumps(list(dict.fromkeys(retired))),
                            ),
                        )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def _backfill_public_authority_aliases(self, conn: Any) -> None:
        codec = self.public_authority_alias_codec
        if codec is None:  # pragma: no cover - guarded by reconciliation
            raise AssertionError("Public authority alias backfill requires a codec.")
        cursor_name = f"cayu_public_authority_backfill_{uuid4().hex}"
        async with conn.cursor(name=cursor_name) as source, conn.cursor() as target:
            await source.execute("SELECT id FROM cayu_sessions ORDER BY id")
            while rows := await source.fetchmany(500):
                for (session_id,) in rows:
                    private_session_id = str(session_id)
                    for public_alias in codec.aliases(
                        private_session_id,
                        field_name="session_id",
                    ):
                        await self._register_public_authority_alias_row(
                            target,
                            field_name="session_id",
                            scope_key="",
                            public_alias=public_alias,
                            private_value=private_session_id,
                        )

        interaction_cursor_name = f"{cursor_name}_interactions"
        async with (
            conn.cursor(name=interaction_cursor_name) as source,
            conn.cursor() as target,
        ):
            await source.execute(
                """
                        SELECT DISTINCT authority.session_id, authority.interaction_id
                        FROM (
                            SELECT session_id, interaction_id
                            FROM cayu_events
                            WHERE interaction_id IS NOT NULL
                            UNION
                            SELECT session_id, interaction_id
                            FROM cayu_transcript_messages
                            WHERE interaction_id IS NOT NULL
                            UNION
                            SELECT event.session_id, nested.value #>> '{}' AS interaction_id
                            FROM cayu_events AS event
                            CROSS JOIN LATERAL jsonb_array_elements(
                                CASE
                                    WHEN jsonb_typeof(event.payload -> 'interaction_ids') = 'array'
                                    THEN event.payload -> 'interaction_ids'
                                    ELSE '[]'::jsonb
                                END
                            ) AS nested(value)
                            WHERE event.event_type = 'turn.completed'
                              AND jsonb_typeof(nested.value) = 'string'
                              AND btrim(nested.value #>> '{}') <> ''
                        ) AS authority
                        ORDER BY authority.session_id, authority.interaction_id
                        """
            )
            while rows := await source.fetchmany(500):
                for session_id, interaction_id in rows:
                    private_session_id = str(session_id)
                    private_interaction_id = str(interaction_id)
                    for public_alias in codec.aliases(
                        private_interaction_id,
                        field_name="interaction_id",
                        session_id=private_session_id,
                    ):
                        await self._register_public_authority_alias_row(
                            target,
                            field_name="interaction_id",
                            scope_key=private_session_id,
                            public_alias=public_alias,
                            private_value=private_interaction_id,
                        )

        grant_cursor_name = f"{cursor_name}_targeted_grants"
        async with (
            conn.cursor(name=grant_cursor_name) as source,
            conn.cursor() as target,
        ):
            await source.execute(
                "SELECT session_id, grant_id "
                "FROM cayu_targeted_tool_grants ORDER BY session_id, grant_id"
            )
            while rows := await source.fetchmany(500):
                for session_id, grant_id in rows:
                    private_session_id = str(session_id)
                    private_grant_id = str(grant_id)
                    for public_alias in codec.aliases(
                        private_grant_id,
                        field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                        session_id=private_session_id,
                    ):
                        await self._register_public_authority_alias_row(
                            target,
                            field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                            scope_key=private_session_id,
                            public_alias=public_alias,
                            private_value=private_grant_id,
                        )

    async def _register_public_authorities(
        self,
        cur: Any,
        session_id: str,
        *,
        interaction_ids: tuple[str, ...] = (),
    ) -> None:
        codec = self.public_authority_alias_codec
        if codec is None:
            await cur.execute("SELECT EXISTS(SELECT 1 FROM cayu_public_authority_alias_keys)")
            row = await cur.fetchone()
            if row is not None and row[0] is True:
                raise RuntimeError(
                    "Postgres public authority aliases are initialized; this writer "
                    "must configure the deployment's alias keyring."
                )
            return
        await cur.execute(
            "SELECT active_key_id, keyring_fingerprint "
            "FROM cayu_public_authority_alias_config "
            "WHERE singleton = TRUE"
        )
        active = await cur.fetchone()
        if (
            active is None
            or str(active[0]) != codec.keyring.active_key_id
            or str(active[1]) != codec.keyring_fingerprint()
        ):
            raise RuntimeError("Postgres public authority alias writer uses a stale active key.")
        for public_alias in codec.aliases(session_id, field_name="session_id"):
            await self._register_public_authority_alias_row(
                cur,
                field_name="session_id",
                scope_key="",
                public_alias=public_alias,
                private_value=session_id,
            )
        for interaction_id in dict.fromkeys(interaction_ids):
            for public_alias in codec.aliases(
                interaction_id,
                field_name="interaction_id",
                session_id=session_id,
            ):
                await self._register_public_authority_alias_row(
                    cur,
                    field_name="interaction_id",
                    scope_key=session_id,
                    public_alias=public_alias,
                    private_value=interaction_id,
                )

    async def _register_event_public_authorities(
        self,
        cur: Any,
        session_id: str,
        events: list[Event] | tuple[Event, ...],
    ) -> None:
        interaction_ids: list[str] = []
        for event in events:
            if event.interaction_id is not None:
                interaction_ids.append(event.interaction_id)
            if event.type == EventType.TURN_COMPLETED:
                nested = event.payload.get("interaction_ids")
                if type(nested) is list:
                    interaction_ids.extend(
                        value for value in nested if type(value) is str and value
                    )
        await self._register_public_authorities(
            cur,
            session_id,
            interaction_ids=tuple(interaction_ids),
        )

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
        participant_binding_factory: Callable[[Session], tuple[Any, Any]] | None = None,
        participant_request_commitment: str | None = None,
        participant_provenance=None,
        recipient_selection=None,
        creation_target=None,
        recipient_receipt_validator=None,
    ) -> Session:
        from cayu.sessions.pending_actions import pending_action_event_storage_values

        request = copy_run_request(request)
        identity = copy_session_identity(identity)
        if result_checkpoint_transform is not None and not callable(result_checkpoint_transform):
            raise TypeError("result_checkpoint_transform must be callable.")
        if participant_binding_factory is not None and not callable(participant_binding_factory):
            raise TypeError("participant_binding_factory must be callable.")
        if (
            participant_binding_factory is not None
            and type(participant_request_commitment) is not str
        ):
            raise TypeError("participant_request_commitment is required for participant creation.")
        await self._ensure_ready()
        session_id = request.session_id if request.session_id is not None else _new_id()
        if request.parent_session_id == session_id:
            raise ValueError("Session cannot be its own parent.")
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await self._lock_closure_lineage(cur)
                    await self._require_available_closure_identity(cur, session_id)
                    await self._require_external_creation(conn, request)
                    from cayu.sessions.base import _RECIPIENT_PROVENANCE_CAPABILITY

                    if (
                        participant_provenance is _RECIPIENT_PROVENANCE_CAPABILITY
                        and creation_target is None
                    ):
                        raise PermissionError(
                            "Recipient creation requires its exact durable target."
                        )
                    if creation_target is not None:
                        creation_target = creation_fence.snapshot_target(creation_target)
                        await _creation_fence.postgres_lock(cur, creation_target)
                        creation_fence.require_pending(
                            creation_target,
                            await _creation_fence.postgres_read(cur, creation_target),
                            request_commitment=participant_request_commitment,
                            requested_session_id=request.session_id,
                        )
                    parent_session = (
                        None
                        if request.parent_session_id is None
                        else await self._load_for_key_share(cur, request.parent_session_id)
                    )
                    if request.parent_session_id is not None and parent_session is None:
                        raise ValueError(f"Parent session not found: {request.parent_session_id}")
                    if parent_session is not None:
                        for owner in await self._closure_lineage_owners(cur, (parent_session.id,)):
                            _check_closure_lineage_owner(owner, (parent_session.id,))
                    now = await self._session_store_now(cur)
                    session = Session(
                        id=session_id,
                        instance_id=session_instance_id_for_run_request(
                            request,
                            session_id=session_id,
                        ),
                        agent_name=request.agent_name,
                        provider_name=identity.provider_name,
                        model=identity.model,
                        parent_session_id=request.parent_session_id,
                        causal_budget_id=request.causal_budget_id or request.task_id or session_id,
                        runtime_name=identity.runtime_name,
                        runtime_version=identity.runtime_version,
                        environment_name=request.environment_name,
                        status=SessionStatus.PENDING,
                        created_at=now,
                        updated_at=now,
                        last_activity_at=now,
                        invocation=session_invocation_for_run_request(
                            request,
                            session_id=session_id,
                            parent_session=parent_session,
                        ),
                        metadata=session_metadata_for_creation(
                            request.metadata,
                            identity=identity,
                            tool_capability_ceiling=request.tool_capability_ceiling,
                            execution_deadline=request.execution_deadline,
                            parent_session=parent_session,
                            prepared_request=request,
                        ),
                        labels=request.labels,
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
                    await cur.execute(
                        f"""
                        INSERT INTO cayu_sessions ({pg_support.SESSION_COLUMNS})
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        pg_support.session_insert_values(session),
                    )
                    if participant_binding_factory is not None:
                        binding, receipt = participant_binding_factory(
                            session.model_copy(deep=True)
                        )
                        if type(binding).__name__ != "ParticipantSessionBinding":
                            raise TypeError(
                                "Participant binding factory returned an invalid binding."
                            )
                        if type(receipt).__name__ != "ParticipantSessionCreationReceipt":
                            raise TypeError(
                                "Participant binding factory returned an invalid receipt."
                            )
                        if (
                            binding.session_id != session.id
                            or binding.session_instance_id != session.instance_id
                        ):
                            raise ValueError("Participant binding session identity conflicts.")
                        if receipt.binding != binding:
                            raise ValueError("Participant receipt binding conflicts.")
                        await cur.execute(
                            """
                            INSERT INTO cayu_participant_session_bindings (
                                creation_key, request_commitment, session_id, session_instance_id,
                                application_scope, participant_owner_id, participant_owner_incarnation,
                                participant_id, participant_incarnation, lifecycle_revision,
                                configuration_revision, admission_generation, creator_commitment,
                                authorization_commitment, initial_input_commitment,
                                execution_profile_commitment, binding_json, receipt_json
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                binding.creation_key,
                                participant_request_commitment,
                                session.id,
                                session.instance_id,
                                binding.application_scope,
                                binding.participant.owner.owner_id,
                                binding.participant.owner.incarnation,
                                binding.participant.participant_id,
                                binding.participant.incarnation,
                                binding.lifecycle_revision,
                                binding.configuration_revision,
                                binding.admission_generation,
                                binding.creator_commitment,
                                binding.authorization_commitment,
                                binding.initial_input_commitment,
                                binding.execution_profile_commitment,
                                Jsonb(binding.model_dump(mode="json")),
                                Jsonb(receipt.model_dump(mode="json")),
                            ),
                        )
                        if receipt.recipient_metadata_json is not None:
                            from cayu.sessions.base import _RECIPIENT_PROVENANCE_CAPABILITY

                            if participant_provenance is not _RECIPIENT_PROVENANCE_CAPABILITY:
                                raise PermissionError(
                                    "Recipient provenance requires the trusted application boundary."
                                )
                            await cur.executemany(
                                "INSERT INTO cayu_transcript_messages "
                                "(session_id, interaction_id, message, transcript_search_document) "
                                "VALUES (%s, %s, %s, %s)",
                                [
                                    (
                                        session.id,
                                        None,
                                        pg_support._dumps(message.model_dump(mode="json")),
                                        _postgres_transcript_index_document(session.id, message),
                                    )
                                    for message in request.messages
                                ],
                            )
                        if recipient_selection is not None:
                            from cayu.sessions.context_views import ContextViewSelectionReceipt

                            await cur.execute(
                                "SELECT receipt_json FROM cayu_context_view_selections "
                                "WHERE selection_key = %s FOR UPDATE",
                                (recipient_selection.selection_key,),
                            )
                            row = await cur.fetchone()
                            stored_selection = (
                                None
                                if row is None
                                else ContextViewSelectionReceipt.model_validate(row[0])
                            )
                            if (
                                stored_selection is None
                                or stored_selection != recipient_selection
                                or stored_selection.state not in {"adopted", "transferred"}
                            ):
                                raise PermissionError(
                                    "Recipient context-view ownership changed before child creation."
                                )
                        if recipient_receipt_validator is not None:
                            recipient_receipt_validator(session, receipt)
                        if creation_target is not None:
                            creation_fence.validate_binding(
                                creation_target,
                                binding,
                                requested_session_id=receipt.requested_session_id,
                            )
                            await _creation_fence.postgres_write(
                                cur, creation_fence.created(creation_target, session, receipt)
                            )
                    elif creation_target is not None:
                        raise PermissionError("Creation targets require participant ownership.")
                    if initial_operation_records:
                        await cur.executemany(
                            "INSERT INTO cayu_session_operations "
                            "(session_id, idempotency_key, record, updated_at) "
                            "VALUES (%s, %s, %s, %s)",
                            [
                                (
                                    session.id,
                                    key,
                                    pg_support._dumps(record),
                                    session.updated_at,
                                )
                                for key, record in initial_operation_records.items()
                            ],
                        )
                    await self._register_event_public_authorities(
                        cur,
                        session.id,
                        [] if admission is None else [admission[0]],
                    )
                    if session.labels:
                        await cur.executemany(
                            """
                            INSERT INTO cayu_session_labels (session_id, key, value)
                            VALUES (%s, %s, %s)
                            """,
                            pg_support.session_label_insert_values(session),
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
                        await cur.execute(
                            "UPDATE cayu_sessions SET event_seq = 1 WHERE id = %s",
                            (session.id,),
                        )
                        await cur.execute(
                            """
                            INSERT INTO cayu_events (
                                session_id, session_order, event_id, interaction_id,
                                event_type, timestamp, agent_name, environment_name,
                                workflow_name, tool_name, payload, event,
                                pending_action_lookup_key, pending_action_projection,
                                pending_action_projection_bytes
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s
                            )
                            """,
                            (
                                session.id,
                                1,
                                started_event.id,
                                interaction_id,
                                str(started_event.type),
                                pg_support.to_utc(started_event.timestamp),
                                started_event.agent_name,
                                started_event.environment_name,
                                started_event.workflow_name,
                                started_event.tool_name,
                                pg_support._dumps(started_event.payload),
                                pg_support._dumps(started_event.model_dump(mode="json")),
                                lookup_key,
                                projection,
                                projection_bytes,
                            ),
                        )
                        await self._enqueue_persisted_event_side_effects(
                            cur,
                            session.id,
                            [started_event],
                        )
                        await cur.execute(
                            "INSERT INTO cayu_deferred_interaction_inputs "
                            "(session_id, interaction_id, source_messages) "
                            "VALUES (%s, %s, %s)",
                            (
                                session.id,
                                interaction_id,
                                pg_support._dumps(
                                    deferred_interaction_input_storage_payload(deferred_input)
                                ),
                            ),
                        )
                        await self._upsert_checkpoint(
                            cur,
                            session.id,
                            _initial_transcript_pending_checkpoint(
                                session,
                                interaction_id,
                                checkpoint_transform=checkpoint_transform,
                            ),
                            session.updated_at,
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
                            await self._upsert_checkpoint(
                                cur,
                                session.id,
                                transformed,
                                session.updated_at,
                            )
                    if result_checkpoint_transform is not None:
                        current_checkpoint = await self._load_checkpoint(cur, session.id)
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
                        await self._upsert_checkpoint(
                            cur,
                            session.id,
                            _checkpoint_transform_result_preserving_completion_result_event_publications(
                                current_checkpoint,
                                transformed,
                                session_id=session.id,
                            ),
                            session.updated_at,
                        )
                await conn.commit()
            except UniqueViolation as exc:
                await conn.rollback()
                constraint_name = getattr(getattr(exc, "diag", None), "constraint_name", None)
                if participant_binding_factory is not None and constraint_name in {
                    "cayu_participant_session_bindings_pkey",
                    "cayu_participant_session_bindings_session_id_key",
                }:
                    raise
                raise ValueError(f"Session already exists: {session.id}") from exc
            except ForeignKeyViolation as exc:
                await conn.rollback()
                if session.parent_session_id is not None:
                    raise ValueError(
                        f"Parent session not found: {session.parent_session_id}"
                    ) from exc
                raise
        if admission is not None:
            _activate_session_run_fence(session)
        return session.model_copy(deep=True)

    async def create_participant_owned_session(
        self,
        creation_request,
        *,
        resolved_request,
        identity,
        binding_factory,
        recipient_provenance=None,
        recipient_selection=None,
        creation_target=None,
        recipient_receipt_validator=None,
    ):
        from cayu.storage._participant_session_records import validate_replay

        if creation_request.metadata_json is not None:
            from cayu.sessions.base import _RECIPIENT_PROVENANCE_CAPABILITY

            if recipient_provenance is not _RECIPIENT_PROVENANCE_CAPABILITY:
                raise PermissionError(
                    "Recipient provenance requires the trusted application boundary."
                )

        existing = await self.lookup_participant_session_creation(creation_request)
        if existing is not None:
            if creation_target is not None:
                await _creation_fence.validate_replay(self, creation_target, existing[0])
            return validate_replay(existing, identity, binding_factory)
        try:
            session = await self.create(
                resolved_request,
                identity=identity,
                participant_binding_factory=binding_factory,
                participant_provenance=recipient_provenance,
                recipient_selection=recipient_selection,
                creation_target=creation_target,
                recipient_receipt_validator=recipient_receipt_validator,
                participant_request_commitment=creation_request.request_commitment,
            )
        except (UniqueViolation, creation_fence.SessionCreationConflict):
            existing = await self.lookup_participant_session_creation(creation_request)
            if existing is None:
                raise
            if creation_target is not None:
                await _creation_fence.validate_replay(self, creation_target, existing[0])
            return validate_replay(existing, identity, binding_factory)
        receipt = await self.load_participant_session_creation_receipt(session.id)
        if receipt is None:
            raise RuntimeError("Participant creation did not persist its receipt.")
        return validate_replay((session, receipt), identity, binding_factory)

    async def lookup_participant_session_creation(self, creation_request):
        from cayu.sessions.context_views import ParticipantSessionCreationRequest
        from cayu.storage._participant_session_records import reconstruct, row_mapping

        if type(creation_request) is not ParticipantSessionCreationRequest:
            raise TypeError("Participant creation requires a typed creation request.")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                f"SELECT {PARTICIPANT_BINDING_PROJECTION} FROM cayu_participant_session_bindings WHERE creation_key = %s",
                (creation_request.creation_key,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            row = row_mapping(row)
            if row["request_commitment"] != creation_request.request_commitment:
                raise ValueError("Participant creation key conflicts with the request.")
            session = await self._load(cur, row["session_id"])
            receipt = reconstruct(row, session)
            assert session is not None
            return session.model_copy(deep=True), receipt

    async def load_participant_session_binding(self, session_id):
        receipt = await self.load_participant_session_creation_receipt(session_id)
        return None if receipt is None else receipt.binding

    async def _scan_participant_session_bindings(self, participant, *, after=None, limit=32):
        from cayu.sessions._participant_discovery import prepare_scan, reference, scan_parameters
        from cayu.storage._participant_session_records import reconstruct, row_mapping

        query = prepare_scan(participant, after, limit)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            await cur.execute(
                f"SELECT {PARTICIPANT_BINDING_PROJECTION} FROM cayu_participant_session_bindings WHERE "
                "application_scope=%s AND participant_owner_id=%s AND "
                "participant_owner_incarnation=%s AND participant_id=%s AND "
                'participant_incarnation=%s AND creation_key COLLATE "C">%s '
                'ORDER BY creation_key COLLATE "C" LIMIT %s',
                scan_parameters(query),
            )
            rows = await cur.fetchall()
            result = []
            for raw in rows:
                row = row_mapping(raw)
                receipt = reconstruct(row, await self._load(cur, row["session_id"]))
                result.append(reference(receipt, query))
            return tuple(result)

    async def load_participant_session_creation_receipt(self, session_id):
        from cayu.storage._participant_session_records import reconstruct, row_mapping

        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                f"SELECT {PARTICIPANT_BINDING_PROJECTION} FROM cayu_participant_session_bindings WHERE session_id = %s",
                (session_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            row = row_mapping(row)
            session = await self._load(cur, session_id)
            return reconstruct(row, session)

    async def capture_context_view_publication_source(self, session_id):
        return (await self._capture_completed_turn_snapshot(session_id)).publication

    async def _capture_completed_turn_snapshot(self, session_id):
        from cayu.sessions._context_view_source import (
            CompletedTurnSnapshot,
            capture_source,
            closed_round_publication_id,
            completed_boundary,
            publication_frontier,
        )
        from cayu.storage._participant_session_records import reconstruct, row_mapping

        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            session = await self._load(cur, session_id)
            await cur.execute(
                f"SELECT {PARTICIPANT_BINDING_PROJECTION} FROM cayu_participant_session_bindings WHERE session_id = %s",
                (session_id,),
            )
            row = await cur.fetchone()
            binding = None if row is None else reconstruct(row_mapping(row), session).binding
            checkpoint = await self._load_checkpoint(cur, session_id)
            pointer = completed_boundary(session, binding, checkpoint)
            assert session is not None and binding is not None
            tool_receipt = None
            if pointer.tool_round_id is not None:
                publication_id = f"tool-round:{pointer.tool_round_id}"
                key = _runtime_publication_storage_key(publication_id)
                await cur.execute(
                    "SELECT record FROM cayu_session_operations WHERE session_id = %s AND idempotency_key = %s",
                    (session_id, key),
                )
                row = await cur.fetchone()
                if row is None:
                    await cur.execute(
                        "SELECT event FROM cayu_events WHERE session_id = %s AND event_type = %s "
                        "AND event -> 'payload' ->> 'tool_round_id' = %s "
                        "AND (event -> 'payload' -> 'cleared' = 'true'::jsonb OR event -> 'payload' ->> 'transition' = 'answered') LIMIT 2",
                        (session_id, EventType.SESSION_CHECKPOINTED.value, pointer.tool_round_id),
                    )
                    closures = await cur.fetchall()
                    publication_id = closed_round_publication_id(
                        pointer,
                        tuple(
                            Event.model_validate(pg_support._json_obj(item[0])) for item in closures
                        ),
                    )
                    key = _runtime_publication_storage_key(publication_id)
                    await cur.execute(
                        "SELECT record FROM cayu_session_operations WHERE session_id = %s AND idempotency_key = %s",
                        (session_id, key),
                    )
                    row = await cur.fetchone()
                if row is not None:
                    tool_receipt = _reconstruct_runtime_publication_receipt(
                        _decode_runtime_publication_record(row[0]),
                        storage_key=key,
                        session_id=session_id,
                        publication_id=publication_id,
                    )
                publication_frontier(pointer, tool_receipt)
                assert tool_receipt is not None
                await self._validate_runtime_publication_material(
                    cur, tool_receipt, lock_events=False
                )
            end = publication_frontier(pointer, tool_receipt)
            await cur.execute(
                "SELECT event FROM cayu_events WHERE session_id = %s AND event_id = %s",
                (session_id, pointer.completion_event_id),
            )
            row = await cur.fetchone()
            completion = None if row is None else Event.model_validate(pg_support._json_obj(row[0]))
            await cur.execute(
                "SELECT session_order, interaction_id, message FROM cayu_transcript_messages "
                "WHERE session_id = %s AND session_order > %s AND session_order <= %s "
                "ORDER BY session_order",
                (session_id, pointer.source_transcript_cursor, end),
            )
            rows = await cur.fetchall()
            records = tuple(
                TranscriptRecord(
                    index=row[0] - 1,
                    interaction_id=row[1],
                    message=Message.model_validate(pg_support._json_obj(row[2])),
                )
                for row in rows
            )
            publication = capture_source(
                session, binding, checkpoint, pointer, completion, records, tool_receipt
            )
            await cur.execute(
                "SELECT 1 FROM cayu_session_message_queue "
                "WHERE session_id = %s AND status = 'queued' LIMIT 1",
                (session_id,),
            )
            queued = await cur.fetchone()
            await cur.execute(
                "SELECT 1 FROM cayu_session_closure_progress AS p "
                "WHERE root_session_id = %s OR EXISTS "
                "(SELECT 1 FROM jsonb_array_elements(p.progress_json->'descendants') AS child "
                "WHERE child->>'session_id' = %s) LIMIT 1",
                (session_id, session_id),
            )
            closure = await cur.fetchone()
            await cur.execute(
                "SELECT 1 FROM cayu_session_operations "
                "WHERE session_id = %s AND idempotency_key = %s LIMIT 1",
                (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
            )
            active = await cur.fetchone()
            return CompletedTurnSnapshot(
                publication=publication,
                current_session=session,
                checkpoint=checkpoint,
                has_queued_input=queued is not None,
                has_closure_owner=closure is not None,
                has_active_model_stage=active is not None,
                current_transcript_cursor=await _transcript_cursor(cur, session_id),
            )

    async def _lock_context_view_admission(
        self,
        cur,
        *,
        owner=None,
        session_id: str | None = None,
        lifecycle: bool = False,
    ) -> None:
        """Acquire context-view advisory locks in the canonical order.

        Selection and publication both cover the source session and owner, so
        they must never acquire those locks in opposite orders.  Keep the
        lifecycle lock between them for operations that publish lifecycle
        evidence.  Session closure/deletion only acquire the session lock and
        therefore cannot form a cycle with this order.
        """

        if owner is not None:
            await cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (
                    f"context-view-owner:{owner.application_scope}:"
                    f"{owner.owner_id}:{owner.incarnation}",
                ),
            )
        if lifecycle:
            await cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                ("context-view-lifecycle",),
            )
        if session_id is not None:
            await cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"context-view-session:{session_id}",),
            )

    async def publish_context_view(self, manifest, *, publication_key):
        from cayu.sessions.context_views import (
            CONTEXT_VIEW_MAX_PUBLICATIONS_PER_OWNER,
            ContextViewManifest,
            validate_context_view_manifest_storage,
        )

        if type(manifest) is not ContextViewManifest or type(publication_key) is not str:
            raise TypeError("Context-view publication requires typed manifest and key.")
        if len(publication_key.encode("utf-8")) > 512:
            raise ValueError("publication_key must be at most 512 UTF-8 bytes.")
        publication_key = require_clean_nonblank(publication_key, "publication_key")
        await self._ensure_ready()
        owner = manifest.source_owner
        async with self._connection() as conn, conn.cursor() as cur:
            await self._lock_context_view_admission(
                cur,
                owner=owner,
                session_id=manifest.source_session_id,
            )
            await cur.execute(
                "SELECT * FROM cayu_context_views WHERE publication_key = %s",
                (publication_key,),
            )
            existing = await cur.fetchone()
            if existing is not None:
                restored = validate_context_view_manifest_storage(
                    ContextViewManifest.model_validate(existing[-1]),
                    view_id=existing[0],
                    owner_scope=existing[2],
                    owner_id=existing[3],
                    owner_incarnation=existing[4],
                    source_session_id=existing[5],
                    source_session_instance_id=existing[6],
                    transcript_cursor=existing[7],
                    projection_schema=existing[8],
                    extension_set_commitment=existing[9],
                )
                if restored != manifest:
                    raise ValueError(
                        "Context-view publication key conflicts with its manifest."
                    ) from None
                return restored.model_copy(deep=True)
            await cur.execute(
                "SELECT 1 FROM cayu_context_views WHERE view_id = %s",
                (manifest.view_id,),
            )
            if await cur.fetchone() is not None:
                raise ValueError("Context-view ID is already bound to another manifest.")
            source = await self._load(cur, manifest.source_session_id)
            if source is None or source.instance_id != manifest.source_session_instance_id:
                raise LookupError("The source session incarnation is unavailable.")
            await cur.execute(
                "SELECT COUNT(*) FROM cayu_context_views "
                "WHERE source_owner_scope = %s AND source_owner_id = %s "
                "AND source_owner_incarnation = %s",
                (owner.application_scope, owner.owner_id, owner.incarnation),
            )
            if (await cur.fetchone())[0] >= CONTEXT_VIEW_MAX_PUBLICATIONS_PER_OWNER:
                raise OverflowError("Context-view publication quota exceeded for the owner.")
            try:
                await cur.execute(
                    """
                    INSERT INTO cayu_context_views (
                        view_id, publication_key, source_owner_scope, source_owner_id,
                        source_owner_incarnation, source_session_id, source_session_instance_id,
                        transcript_cursor, projection_schema, extension_set_commitment, manifest_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        manifest.view_id,
                        publication_key,
                        owner.application_scope,
                        owner.owner_id,
                        owner.incarnation,
                        manifest.source_session_id,
                        manifest.source_session_instance_id,
                        manifest.transcript_cursor,
                        manifest.projection_schema,
                        manifest.extension_set_commitment,
                        Jsonb(manifest.model_dump(mode="json")),
                    ),
                )
            except UniqueViolation:
                await conn.rollback()
                await cur.execute(
                    "SELECT * FROM cayu_context_views WHERE publication_key = %s",
                    (publication_key,),
                )
                existing = await cur.fetchone()
                if existing is None:
                    raise RuntimeError(
                        "Context-view publication conflict was not reconstructable."
                    ) from None
                restored = validate_context_view_manifest_storage(
                    ContextViewManifest.model_validate(existing[-1]),
                    view_id=existing[0],
                    owner_scope=existing[2],
                    owner_id=existing[3],
                    owner_incarnation=existing[4],
                    source_session_id=existing[5],
                    source_session_instance_id=existing[6],
                    transcript_cursor=existing[7],
                    projection_schema=existing[8],
                    extension_set_commitment=existing[9],
                )
                if restored != manifest:
                    raise ValueError(
                        "Context-view publication key conflicts with its manifest."
                    ) from None
                return restored.model_copy(deep=True)
            await conn.commit()
            return manifest.model_copy(deep=True)

    async def lookup_context_view_publication(self, publication_key):
        from cayu.sessions.context_views import (
            ContextViewManifest,
            validate_context_view_manifest_storage,
        )

        if type(publication_key) is not str:
            raise TypeError("Context-view publication keys must be strings.")
        if len(publication_key.encode("utf-8")) > 512:
            raise ValueError("publication_key must be at most 512 UTF-8 bytes.")
        publication_key = require_nonblank(publication_key, "publication_key")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT * FROM cayu_context_views WHERE publication_key = %s",
                (publication_key,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            return validate_context_view_manifest_storage(
                ContextViewManifest.model_validate(row[-1]),
                view_id=row[0],
                owner_scope=row[2],
                owner_id=row[3],
                owner_incarnation=row[4],
                source_session_id=row[5],
                source_session_instance_id=row[6],
                transcript_cursor=row[7],
                projection_schema=row[8],
                extension_set_commitment=row[9],
            ).model_copy(deep=True)

    async def _require_context_view_lifecycle_capacity(
        self, cur, view_id: str, *, additional_slots: int = 0
    ) -> None:
        from cayu.sessions.context_views import validate_context_view_lifecycle_capacity

        await cur.execute(
            "SELECT COUNT(*) FROM cayu_context_view_lifecycle_events WHERE view_id = %s",
            (view_id,),
        )
        events = (await cur.fetchone())[0]
        await cur.execute(
            "SELECT COUNT(*) FROM cayu_context_view_selections WHERE view_id = %s "
            "AND state IN ('selected', 'adopted', 'transferred')",
            (view_id,),
        )
        unsettled = (await cur.fetchone())[0]
        validate_context_view_lifecycle_capacity(
            events, unsettled, additional_slots=additional_slots
        )

    async def select_context_view(self, request):
        return await self._select_context_view(request)

    async def _select_context_view(self, request, *, target=None):
        from cayu.sessions._context_selection_fence import (
            require_not_excluded,
            require_selected_participant,
        )
        from cayu.sessions.context_views import (
            CONTEXT_VIEW_EXPIRY_BATCH_SIZE,
            CONTEXT_VIEW_MAX_PUBLICATIONS_PER_OWNER,
            CONTEXT_VIEW_MAX_SELECTIONS_PER_OWNER,
            ContextViewManifest,
            ContextViewSelectionReceipt,
            ContextViewSelectionRequest,
            context_view_manifest_bytes,
            json_commitment,
            validate_context_view_manifest_storage,
            validate_context_view_receipt_storage,
        )
        from cayu.storage._context_selection_fence import (
            lock_selection_key,
            postgres_exclusion,
            reconstruct_exclusion,
        )

        if type(request) is not ContextViewSelectionRequest:
            raise TypeError("Context-view selection requires a typed request.")
        request = ContextViewSelectionRequest.model_validate(request)
        await self._ensure_ready()
        request_commitment = json_commitment(
            canonical_bounded_durable_json_bytes(
                request.model_dump(mode="json"),
                "context view selection request",
                max_bytes=256 * 1024,
                max_nodes=8192,
                max_nesting=64,
            ).decode("utf-8"),
            "context view selection request",
        )
        owner = request.source_owner
        async with self._connection() as conn, conn.cursor() as cur:
            await lock_selection_key(cur, request.selection_key)
            await self._lock_context_view_admission(
                cur,
                owner=owner,
                lifecycle=True,
                session_id=request.source_session_id,
            )
            require_not_excluded(
                request,
                reconstruct_exclusion(await postgres_exclusion(cur, request.selection_key)),
                target=target,
            )
            now_ms = int(self._clock().timestamp() * 1000)
            await cur.execute(
                "SELECT selection_key, receipt_json FROM cayu_context_view_selections "
                "WHERE owner_scope = %s AND owner_id = %s AND owner_incarnation = %s "
                "AND expires_at_ms <= %s AND state = 'selected' "
                "ORDER BY (selection_key = %s) DESC, expires_at_ms, selection_key LIMIT %s FOR UPDATE",
                (
                    owner.application_scope,
                    owner.owner_id,
                    owner.incarnation,
                    now_ms,
                    request.selection_key,
                    CONTEXT_VIEW_EXPIRY_BATCH_SIZE,
                ),
            )
            expired_rows = await cur.fetchall()
            await cur.execute(
                """
                WITH expired AS (
                    SELECT selection_key
                    FROM cayu_context_view_selections
                    WHERE owner_scope = %s AND owner_id = %s AND owner_incarnation = %s
                      AND expires_at_ms <= %s
                      AND state = 'selected'
                    ORDER BY (selection_key = %s) DESC, expires_at_ms, selection_key
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE cayu_context_view_selections AS selection
                SET state = 'expired',
                    receipt_json = jsonb_set(selection.receipt_json, '{state}', '"expired"'::jsonb)
                FROM expired
                WHERE selection.selection_key = expired.selection_key
                """,
                (
                    owner.application_scope,
                    owner.owner_id,
                    owner.incarnation,
                    now_ms,
                    request.selection_key,
                    CONTEXT_VIEW_EXPIRY_BATCH_SIZE,
                ),
            )
            from cayu.sessions.context_views import ContextViewLifecycleEvent

            for expired_row in expired_rows:
                expired_receipt = ContextViewSelectionReceipt.model_validate(expired_row[1])
                await self._require_context_view_lifecycle_capacity(
                    cur, expired_receipt.view.view_id, additional_slots=1
                )
                operation_key = (
                    f"expiry:{expired_receipt.selection_key}:{expired_receipt.ownership_revision}"
                )
                event = ContextViewLifecycleEvent(
                    event_id="sha256:"
                    + sha256(f"context-view-event:{operation_key}".encode()).hexdigest(),
                    operation_key=operation_key,
                    selection_key=expired_receipt.selection_key,
                    view_id=expired_receipt.view.view_id,
                    state="expired",
                    owner=expired_receipt.owner,
                    owner_participant=expired_receipt.owner_participant,
                    pin_commitment=expired_receipt.pin_commitment,
                    ownership_revision=expired_receipt.ownership_revision,
                )
                await cur.execute(
                    "INSERT INTO cayu_context_view_lifecycle_events "
                    "(event_id, operation_key, selection_key, view_id, state, owner_scope, owner_id, "
                    "owner_incarnation, pin_commitment, ownership_revision, event_json) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (operation_key) DO NOTHING",
                    (
                        event.event_id,
                        event.operation_key,
                        event.selection_key,
                        event.view_id,
                        event.state,
                        event.owner.application_scope,
                        event.owner.owner_id,
                        event.owner.incarnation,
                        event.pin_commitment,
                        event.ownership_revision,
                        Jsonb(event.model_dump(mode="json")),
                    ),
                )

            await cur.execute(
                "SELECT * FROM cayu_context_view_selections WHERE selection_key = %s",
                (request.selection_key,),
            )
            existing = await cur.fetchone()
            if existing is not None:
                if existing[1] != request_commitment:
                    raise ValueError("Context-view selection key conflicts with its request.")
                receipt = validate_context_view_receipt_storage(
                    ContextViewSelectionReceipt.model_validate(existing[9]),
                    selection_key=existing[0],
                    view_id=existing[2],
                    owner_scope=existing[3],
                    owner_id=existing[4],
                    owner_incarnation=existing[5],
                    state=existing[6],
                    pin_commitment=existing[7],
                    ownership_revision=existing[10],
                )
                if receipt.state == "selected" and receipt.expires_at_ms <= now_ms:
                    raise ValueError("Expired context-view selection lacks cleanup evidence.")
                return receipt
            source = await self._load_for_update(cur, request.source_session_id)
            if source is None or source.instance_id != request.source_session_instance_id:
                raise LookupError("Context-view source session incarnation is unavailable.")
            if target is not None:
                from cayu.sessions._context_selection_fence import require_selection_source
                from cayu.storage._participant_session_records import reconstruct, row_mapping

                await cur.execute(
                    f"SELECT {PARTICIPANT_BINDING_PROJECTION} FROM cayu_participant_session_bindings WHERE session_id = %s",
                    (request.source_session_id,),
                )
                binding_row = await cur.fetchone()
                binding = (
                    None
                    if binding_row is None
                    else reconstruct(row_mapping(binding_row), source).binding
                )
                require_selection_source(target, binding, now_ms=now_ms)
            await cur.execute(
                """
                SELECT view_id FROM cayu_context_views
                WHERE source_owner_scope = %s AND source_owner_id = %s
                  AND source_owner_incarnation = %s AND source_session_id = %s
                  AND source_session_instance_id = %s AND projection_schema = %s
                  AND extension_set_commitment = %s
                  AND (%s::text <> 'exact' OR view_id = %s)
                  AND (%s::bigint IS NULL OR transcript_cursor >= %s)
                  ORDER BY transcript_cursor DESC, view_id COLLATE "C" DESC
                  LIMIT %s
                """,
                (
                    owner.application_scope,
                    owner.owner_id,
                    owner.incarnation,
                    request.source_session_id,
                    request.source_session_instance_id,
                    request.projection_schema,
                    request.extension_set_commitment,
                    request.selector,
                    request.exact_view_id,
                    request.minimum_transcript_cursor,
                    request.minimum_transcript_cursor,
                    min(request.limits.max_views, CONTEXT_VIEW_MAX_PUBLICATIONS_PER_OWNER) + 1,
                ),
            )
            rows = await cur.fetchall()
            if not rows:
                raise LookupError("No eligible context view is available.")
            if len(rows) > request.limits.max_views:
                raise OverflowError("Context-view count exceeds the requested limit.")
            await cur.execute("SELECT * FROM cayu_context_views WHERE view_id = %s", (rows[0][0],))
            row = await cur.fetchone()
            selected = validate_context_view_manifest_storage(
                ContextViewManifest.model_validate(row[-1]),
                view_id=row[0],
                owner_scope=row[2],
                owner_id=row[3],
                owner_incarnation=row[4],
                source_session_id=row[5],
                source_session_instance_id=row[6],
                transcript_cursor=row[7],
                projection_schema=row[8],
                extension_set_commitment=row[9],
            )
            await cur.execute(
                "SELECT COUNT(*) FROM cayu_context_view_selections "
                "WHERE owner_scope = %s AND owner_id = %s AND owner_incarnation = %s",
                (owner.application_scope, owner.owner_id, owner.incarnation),
            )
            if (await cur.fetchone())[0] >= CONTEXT_VIEW_MAX_SELECTIONS_PER_OWNER:
                raise OverflowError("Context-view selection quota exceeded for the owner.")
            await cur.execute(
                """
                SELECT COUNT(*) FROM cayu_context_view_selections
                WHERE owner_scope = %s AND owner_id = %s AND owner_incarnation = %s
                  AND state IN ('selected', 'adopted', 'transferred')
                  AND (state <> 'selected' OR expires_at_ms > %s)
                """,
                (owner.application_scope, owner.owner_id, owner.incarnation, now_ms),
            )
            if (await cur.fetchone())[0] >= request.limits.max_pins:
                raise OverflowError("Context-view pin count exceeds the requested limit.")
            await self._require_context_view_lifecycle_capacity(
                cur, selected.view_id, additional_slots=1
            )
            selected_bytes = context_view_manifest_bytes(selected)
            if selected_bytes > request.limits.max_view_bytes:
                raise OverflowError("Context-view manifest exceeds the requested byte limit.")
            await cur.execute(
                "SELECT * FROM cayu_context_view_selections "
                "WHERE owner_scope = %s AND owner_id = %s AND owner_incarnation = %s "
                "AND state IN ('selected', 'adopted', 'transferred') "
                "AND (state <> 'selected' OR expires_at_ms > %s)",
                (owner.application_scope, owner.owner_id, owner.incarnation, now_ms),
            )
            retained_rows = await cur.fetchall()
            retained_views = {
                receipt.view.view_id: receipt.view
                for row in retained_rows
                for receipt in [
                    validate_context_view_receipt_storage(
                        ContextViewSelectionReceipt.model_validate(row[9]),
                        selection_key=row[0],
                        view_id=row[2],
                        owner_scope=row[3],
                        owner_id=row[4],
                        owner_incarnation=row[5],
                        state=row[6],
                        pin_commitment=row[7],
                        ownership_revision=row[10],
                    )
                ]
            }
            retained_views[selected.view_id] = selected
            retained_bytes = sum(
                context_view_manifest_bytes(view) for view in retained_views.values()
            )
            if retained_bytes > request.limits.max_retained_bytes:
                raise OverflowError("Context-view retained bytes exceed the requested limit.")
            expires_at_ms = int(
                self._clock().timestamp() * 1000 + request.limits.max_lifetime_seconds * 1000
            )
            pin_commitment = (
                "sha256:"
                + sha256(
                    f"context-pin:{request.selection_key}:{selected.view_id}".encode()
                ).hexdigest()
            )
            receipt = ContextViewSelectionReceipt(
                selection_key=request.selection_key,
                view=selected,
                owner=owner,
                state="selected",
                pin_commitment=pin_commitment,
                owner_participant=selected.participant,
                expires_at_ms=expires_at_ms,
            )
            require_selected_participant(target, selected)
            await cur.execute(
                """
                INSERT INTO cayu_context_view_selections (
                    selection_key, request_commitment, view_id, owner_scope, owner_id,
                    owner_incarnation, state, pin_commitment, expires_at_ms,
                    ownership_revision, receipt_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    request.selection_key,
                    request_commitment,
                    selected.view_id,
                    owner.application_scope,
                    owner.owner_id,
                    owner.incarnation,
                    receipt.state,
                    receipt.pin_commitment,
                    receipt.expires_at_ms,
                    receipt.ownership_revision,
                    Jsonb(receipt.model_dump(mode="json")),
                ),
            )
            await conn.commit()
            return receipt.model_copy(deep=True)

    async def lookup_context_view_selection(self, selection_key):
        from cayu.sessions.context_views import ContextViewSelectionReceipt

        if type(selection_key) is not str:
            raise TypeError("Context-view selection key must be a string.")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT receipt_json FROM cayu_context_view_selections WHERE selection_key = %s",
                (selection_key,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        value = row[0]
        if isinstance(value, str):
            return ContextViewSelectionReceipt.model_validate_json(value)
        return ContextViewSelectionReceipt.model_validate(value)

    async def transition_context_view_ownership(self, request):
        return await self._transition_context_view_ownership(request)

    async def _transition_context_view_ownership(self, request, *, target=None):
        from cayu.sessions._context_selection_fence import require_selection_adoption
        from cayu.sessions.context_views import (
            ContextViewLifecycleEvent,
            ContextViewOwnershipRequest,
            ContextViewSelectionReceipt,
            canonical_bounded_durable_json_bytes,
            json_commitment,
            validate_context_view_receipt_storage,
        )
        from cayu.storage._context_selection_fence import (
            lock_selection_key,
            postgres_decision,
            postgres_exclusion,
            reconstruct_exclusion,
        )

        if type(request) is not ContextViewOwnershipRequest:
            raise TypeError("Context-view ownership requires a typed request.")
        request = ContextViewOwnershipRequest.model_validate(request)
        await self._ensure_ready()
        request_commitment = json_commitment(
            canonical_bounded_durable_json_bytes(
                request.model_dump(mode="json"),
                "context view ownership request",
                max_bytes=256 * 1024,
                max_nodes=8192,
                max_nesting=64,
            ).decode("utf-8"),
            "context view ownership request",
        )
        async with self._connection() as conn, conn.cursor() as cur:
            # Serialize before replay lookup, including concurrent identical
            # requests that would otherwise observe a stale post-transition row.
            await lock_selection_key(cur, request.selection_key)
            await cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                ("context-view-lifecycle",),
            )
            control = reconstruct_exclusion(await postgres_exclusion(cur, request.selection_key))
            require_selection_adoption(
                request,
                await postgres_decision(cur, control.request) if control is not None else None,
                target=target,
            )
            await cur.execute(
                "SELECT request_commitment, receipt_json "
                "FROM cayu_context_view_ownership_operations "
                "WHERE operation_key = %s",
                (request.operation_key,),
            )
            operation = await cur.fetchone()
            if operation is not None:
                if operation[0] != request_commitment:
                    raise ValueError("Ownership operation key conflicts with its request.")
                return ContextViewSelectionReceipt.model_validate(operation[1])
            await cur.execute(
                "SELECT * FROM cayu_context_view_selections WHERE selection_key = %s FOR UPDATE",
                (request.selection_key,),
            )
            row = await cur.fetchone()
            if row is None:
                raise LookupError("Context-view selection is unavailable.")
            receipt = validate_context_view_receipt_storage(
                ContextViewSelectionReceipt.model_validate(row[9]),
                selection_key=row[0],
                view_id=row[2],
                owner_scope=row[3],
                owner_id=row[4],
                owner_incarnation=row[5],
                state=row[6],
                pin_commitment=row[7],
                ownership_revision=row[10],
            )
            if (
                receipt.view.view_id != request.view_id
                or receipt.pin_commitment != request.pin_commitment
            ):
                raise ValueError("Context-view pin identity conflicts with the request.")
            if (
                receipt.state != request.expected_state
                or receipt.ownership_revision != request.expected_revision
            ):
                raise ValueError("Context-view ownership state or revision is stale.")
            if receipt.owner != request.current_owner:
                raise PermissionError("The current owner does not control this context-view pin.")
            if (
                request.current_participant is not None
                and receipt.owner_participant is not None
                and receipt.owner_participant != request.current_participant
            ):
                raise PermissionError(
                    "The current participant does not control this context-view pin."
                )
            now_ms = int(self._clock().timestamp() * 1000)
            from cayu.sessions._context_selection_fence import require_adoption_deadline

            require_adoption_deadline(target, now_ms)
            await self._require_context_view_lifecycle_capacity(cur, receipt.view.view_id)
            if receipt.state == "selected" and receipt.expires_at_ms <= now_ms:
                expired = receipt.model_copy(update={"state": "expired"}, deep=True)
                operation_key = f"expiry:{request.selection_key}:{receipt.ownership_revision}"
                event = ContextViewLifecycleEvent(
                    event_id="sha256:"
                    + sha256(f"context-view-event:{operation_key}".encode()).hexdigest(),
                    operation_key=operation_key,
                    selection_key=request.selection_key,
                    view_id=receipt.view.view_id,
                    state="expired",
                    owner=expired.owner,
                    owner_participant=expired.owner_participant,
                    pin_commitment=expired.pin_commitment,
                    ownership_revision=expired.ownership_revision,
                )
                await cur.execute(
                    "UPDATE cayu_context_view_selections SET state = 'expired', receipt_json = %s "
                    "WHERE selection_key = %s AND ownership_revision = %s",
                    (
                        Jsonb(expired.model_dump(mode="json")),
                        request.selection_key,
                        request.expected_revision,
                    ),
                )
                await cur.execute(
                    "INSERT INTO cayu_context_view_lifecycle_events "
                    "(event_id, operation_key, selection_key, view_id, state, owner_scope, owner_id, "
                    "owner_incarnation, pin_commitment, ownership_revision, event_json) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (operation_key) DO NOTHING",
                    (
                        event.event_id,
                        event.operation_key,
                        event.selection_key,
                        event.view_id,
                        event.state,
                        event.owner.application_scope,
                        event.owner.owner_id,
                        event.owner.incarnation,
                        event.pin_commitment,
                        event.ownership_revision,
                        Jsonb(event.model_dump(mode="json")),
                    ),
                )
                await conn.commit()
                raise LookupError("Context-view selection has expired.")
            if request.operation == "release":
                next_state, next_owner = "released", receipt.owner
            elif request.operation == "adopt":
                assert request.destination_owner is not None
                next_state, next_owner = "adopted", request.destination_owner
            else:
                assert request.destination_owner is not None
                next_state, next_owner = "transferred", request.destination_owner
            next_participant = receipt.owner_participant
            if request.operation != "release":
                if request.destination_participant is None:
                    if request.destination_owner != receipt.owner:
                        raise PermissionError(
                            "A transfer to another owner requires destination participant evidence."
                        )
                else:
                    next_participant = request.destination_participant
            updated = receipt.model_copy(
                update={
                    "state": next_state,
                    "owner": next_owner,
                    "ownership_revision": receipt.ownership_revision + 1,
                    "owner_participant": next_participant,
                },
                deep=True,
            )
            updated_json = Jsonb(updated.model_dump(mode="json"))
            event = ContextViewLifecycleEvent(
                event_id="sha256:"
                + sha256(f"context-view-event:{request.operation_key}".encode()).hexdigest(),
                operation_key=request.operation_key,
                selection_key=request.selection_key,
                view_id=request.view_id,
                state=next_state,
                owner=updated.owner,
                owner_participant=updated.owner_participant,
                pin_commitment=updated.pin_commitment,
                ownership_revision=updated.ownership_revision,
            )
            event_json = Jsonb(event.model_dump(mode="json"))
            await self._require_context_view_lifecycle_capacity(
                cur, updated.view.view_id, additional_slots=int(next_state != "released")
            )
            await cur.execute(
                "UPDATE cayu_context_view_selections SET owner_scope = %s, owner_id = %s, "
                "owner_incarnation = %s, state = %s, ownership_revision = %s, receipt_json = %s "
                "WHERE selection_key = %s AND ownership_revision = %s AND state = %s",
                (
                    updated.owner.application_scope,
                    updated.owner.owner_id,
                    updated.owner.incarnation,
                    updated.state,
                    updated.ownership_revision,
                    updated_json,
                    request.selection_key,
                    request.expected_revision,
                    request.expected_state,
                ),
            )
            if cur.rowcount != 1:
                raise ValueError("Context-view ownership transition lost its compare-and-set race.")
            try:
                await cur.execute(
                    "INSERT INTO cayu_context_view_ownership_operations "
                    "(operation_key, selection_key, request_commitment, receipt_json) VALUES (%s, %s, %s, %s)",
                    (
                        request.operation_key,
                        request.selection_key,
                        request_commitment,
                        updated_json,
                    ),
                )
            except UniqueViolation:
                await conn.rollback()
                async with self._connection() as replay_conn, replay_conn.cursor() as replay_cur:
                    await replay_cur.execute(
                        "SELECT request_commitment, receipt_json "
                        "FROM cayu_context_view_ownership_operations "
                        "WHERE operation_key = %s",
                        (request.operation_key,),
                    )
                    replay = await replay_cur.fetchone()
                    if replay is None or replay[0] != request_commitment:
                        raise ValueError(
                            "Ownership operation key conflicts with its request."
                        ) from None
                    return ContextViewSelectionReceipt.model_validate(replay[1])
            await cur.execute(
                "INSERT INTO cayu_context_view_lifecycle_events "
                "(event_id, operation_key, selection_key, view_id, state, owner_scope, owner_id, "
                "owner_incarnation, pin_commitment, ownership_revision, event_json) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    event.event_id,
                    event.operation_key,
                    event.selection_key,
                    event.view_id,
                    event.state,
                    event.owner.application_scope,
                    event.owner.owner_id,
                    event.owner.incarnation,
                    event.pin_commitment,
                    event.ownership_revision,
                    event_json,
                ),
            )
            await conn.commit()
            return updated.model_copy(deep=True)

    async def validate_context_view_source_closure(self, session_id: str) -> None:
        await self._ensure_ready()
        now_ms = int(self._clock().timestamp() * 1000)
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT 1 FROM cayu_context_view_selections s "
                "LEFT JOIN cayu_context_views v ON v.view_id = s.view_id "
                "WHERE (v.source_session_id = %s OR v.view_id IS NULL) "
                "AND s.state IN ('selected', 'adopted', 'transferred') "
                "AND (s.state <> 'selected' OR s.expires_at_ms > %s) "
                "LIMIT 1",
                (session_id, now_ms),
            )
            if await cur.fetchone() is not None:
                raise ValueError("Session has an active context-view retention pin.")

    async def validate_context_view_compaction(
        self, session_id: str, expected_transcript_cursor: int
    ) -> None:
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await self._validate_context_view_compaction(
                cur, session_id, expected_transcript_cursor
            )

    async def _validate_context_view_compaction(
        self, cur, session_id: str, expected_transcript_cursor: int
    ) -> None:
        from cayu.sessions.context_views import (
            ContextViewManifest,
            require_independent_context_view_material,
            validate_context_view_manifest_storage,
        )

        now_ms = int(self._clock().timestamp() * 1000)
        await cur.execute(
            "SELECT DISTINCT v.view_id FROM cayu_context_view_selections s "
            "LEFT JOIN cayu_context_views v ON v.view_id = s.view_id "
            "WHERE (v.source_session_id = %s OR v.view_id IS NULL) "
            "AND s.state IN ('selected', 'adopted', 'transferred') "
            "AND (s.state <> 'selected' OR s.expires_at_ms > %s) "
            "AND (v.view_id IS NULL OR v.transcript_cursor <= %s)",
            (session_id, now_ms, expected_transcript_cursor),
        )
        # Buffer only identities. Each bounded manifest is reconstructed separately.
        identities = await cur.fetchall()
        for (view_id,) in identities:
            if view_id is None:
                raise ValueError("Pinned context-view material is unavailable for compaction.")
            await cur.execute("SELECT * FROM cayu_context_views WHERE view_id = %s", (view_id,))
            row = await cur.fetchone()
            if row is None:
                raise ValueError("Pinned context-view material is unavailable for compaction.")
            manifest = validate_context_view_manifest_storage(
                ContextViewManifest.model_validate(row[-1]),
                view_id=row[0],
                owner_scope=row[2],
                owner_id=row[3],
                owner_incarnation=row[4],
                source_session_id=row[5],
                source_session_instance_id=row[6],
                transcript_cursor=row[7],
                projection_schema=row[8],
                extension_set_commitment=row[9],
            )
            require_independent_context_view_material(manifest)

    async def read_context_view_lifecycle_events(
        self, view_id: str, *, limit: int = 256, owner_participant=None
    ):
        from cayu.sessions.context_views import (
            ContextViewLifecycleEvent,
            validate_context_view_lifecycle_storage,
        )

        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("Context-view lifecycle event limits must be between 1 and 256.")
        participant_filter = ""
        parameters: list[Any] = [view_id]
        if owner_participant is not None:
            from cayu.collaboration.participants import ParticipantRef

            participant = ParticipantRef.model_validate(owner_participant)
            participant_filter = " AND event_json -> 'owner_participant' = %s"
            parameters.append(Jsonb(participant.model_dump(mode="json")))
        parameters.append(limit)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT event_id, operation_key, selection_key, view_id, state, owner_scope, "
                "owner_id, owner_incarnation, pin_commitment, ownership_revision, event_json "
                "FROM cayu_context_view_lifecycle_events "
                "WHERE view_id = %s"
                + participant_filter
                + " ORDER BY ownership_revision, event_id LIMIT %s",
                parameters,
            )
            rows = await cur.fetchall()
            columns = (
                "event_id",
                "operation_key",
                "selection_key",
                "view_id",
                "state",
                "owner_scope",
                "owner_id",
                "owner_incarnation",
                "pin_commitment",
                "ownership_revision",
            )
            return tuple(
                validate_context_view_lifecycle_storage(
                    ContextViewLifecycleEvent.model_validate(row[10]),
                    dict(zip(columns, row[:10], strict=True)),
                )
                for row in rows
            )

    async def read_context_view(self, view_id: str, *, source_session_id: str):
        from cayu.sessions.context_views import (
            ContextViewManifest,
            ContextViewReadback,
            validate_context_view_manifest_storage,
        )

        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT * FROM cayu_context_views WHERE view_id = %s AND source_session_id = %s",
                (view_id, source_session_id),
            )
            row = await cur.fetchone()
            if row is None:
                raise LookupError("Context view is unavailable.")
            return ContextViewReadback(
                view=validate_context_view_manifest_storage(
                    ContextViewManifest.model_validate(row[-1]),
                    view_id=row[0],
                    owner_scope=row[2],
                    owner_id=row[3],
                    owner_incarnation=row[4],
                    source_session_id=row[5],
                    source_session_instance_id=row[6],
                    transcript_cursor=row[7],
                    projection_schema=row[8],
                    extension_set_commitment=row[9],
                )
            )

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

        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await self._lock_closure_lineage(cur)
                    for owner in await self._closure_lineage_owners(cur, (source_session_id,)):
                        _check_closure_lineage_owner(owner, (source_session_id,))
                    await self._require_available_closure_identity(cur, fork.id)
                    source_session = _validate_session_fork_source(
                        source_session=await self._load_for_update(cur, source_session_id),
                        source_session_id=source_session_id,
                        fork=fork,
                        allowed_statuses=allowed_statuses,
                        expected_source_run_epoch=expected_source_run_epoch,
                        profile_relationship=profile_relationship,
                    )
                    await cur.execute(
                        "SELECT 1 FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s",
                        (
                            source_session_id,
                            MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                        ),
                    )
                    if await cur.fetchone() is not None:
                        raise SessionForkActiveModelStageConflict(
                            "Cannot fork a session while a model-completion stage is active."
                        )

                    source_checkpoint = await self._load_checkpoint(cur, source_session_id)
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

                    source_transcript_cursor = await _transcript_cursor(cur, source_session_id)
                    if (
                        transcript_cursor is not None
                        and transcript_cursor > source_transcript_cursor
                    ):
                        raise ValueError(
                            "transcript_cursor is greater than source transcript length."
                        )
                    await cur.execute(
                        """
                        SELECT session_order, message, interaction_id
                        FROM cayu_transcript_messages
                        WHERE session_id = %s
                          AND session_order <= %s
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
                    )
                    selected_transcript_rows = await cur.fetchall()
                    copied_messages = [
                        Message(**pg_support._json_obj(row[1])) for row in selected_transcript_rows
                    ]
                    copied_interaction_ids = [row[2] for row in selected_transcript_rows]
                    source_transcript_snapshot = (
                        None
                        if transcript_validator is None
                        else TranscriptSnapshot(
                            records=[
                                TranscriptRecord(
                                    index=int(row[0]) - 1,
                                    interaction_id=row[2],
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
                        copied_checkpoint = _prepare_profiled_fork_checkpoint_result(
                            supports_model_failover=self._supports_model_failover_stage_protocol(),
                            fork=fork,
                            transcript_cursor=len(copied_messages),
                            relationship=profile_relationship,
                            source_checkpoint_present=source_checkpoint_present,
                            copied_checkpoint=copied_checkpoint,
                        )
                    initial_operation_records = _prepare_initial_session_operation_records(
                        fork,
                        operation_initializer,
                    )

                    await cur.execute(
                        f"""
                        INSERT INTO cayu_sessions ({pg_support.SESSION_COLUMNS})
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        pg_support.session_insert_values(fork),
                    )
                    if initial_operation_records:
                        await cur.executemany(
                            "INSERT INTO cayu_session_operations "
                            "(session_id, idempotency_key, record, updated_at) "
                            "VALUES (%s, %s, %s, %s)",
                            [
                                (
                                    fork.id,
                                    key,
                                    pg_support._dumps(record),
                                    fork.updated_at,
                                )
                                for key, record in initial_operation_records.items()
                            ],
                        )
                    await self._register_public_authorities(
                        cur,
                        fork.id,
                        interaction_ids=tuple(
                            value for value in copied_interaction_ids if value is not None
                        ),
                    )
                    if fork.labels:
                        await cur.executemany(
                            """
                            INSERT INTO cayu_session_labels (session_id, key, value)
                            VALUES (%s, %s, %s)
                            """,
                            pg_support.session_label_insert_values(fork),
                        )
                    if copied_messages:
                        await cur.executemany(
                            """
                            INSERT INTO cayu_transcript_messages
                                (session_id, interaction_id, message,
                                 transcript_search_document)
                            VALUES (%s, %s, %s, %s)
                            """,
                            [
                                (
                                    fork.id,
                                    copied_interaction_ids[index],
                                    pg_support._dumps(message.model_dump(mode="json")),
                                    _postgres_transcript_index_document(fork.id, message),
                                )
                                for index, message in enumerate(copied_messages)
                            ],
                        )
                    if copied_checkpoint is not None:
                        await cur.execute(
                            """
                            INSERT INTO cayu_checkpoints (
                                session_id, state, updated_at,
                                pending_action_source_bytes,
                                pending_action_tool_call_count,
                                pending_action_flags,
                                pending_action_metrics_ready
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """,
                            _checkpoint_row_values(fork.id, copied_checkpoint, fork.updated_at),
                        )
                    if events:
                        from cayu.sessions.pending_actions import (
                            pending_action_event_storage_values,
                        )

                        activity_at = await self._session_store_now(cur)
                        await cur.execute(
                            "UPDATE cayu_sessions SET event_seq = %s, last_activity_at = %s "
                            "WHERE id = %s",
                            (len(events), activity_at, fork.id),
                        )
                        await self._register_event_public_authorities(cur, fork.id, events)
                        rows = []
                        for session_order, event in enumerate(events, start=1):
                            lookup_key, projection, projection_bytes = (
                                pending_action_event_storage_values(event)
                            )
                            rows.append(
                                (
                                    fork.id,
                                    session_order,
                                    event.id,
                                    event.interaction_id,
                                    str(event.type),
                                    pg_support.to_utc(event.timestamp),
                                    event.agent_name,
                                    event.environment_name,
                                    event.workflow_name,
                                    event.tool_name,
                                    pg_support._dumps(event.payload),
                                    pg_support._dumps(event.model_dump(mode="json")),
                                    lookup_key,
                                    projection,
                                    projection_bytes,
                                )
                            )
                        await cur.executemany(
                            """
                            INSERT INTO cayu_events (
                                session_id, session_order, event_id, interaction_id,
                                event_type, timestamp, agent_name, environment_name,
                                workflow_name, tool_name, payload, event,
                                pending_action_lookup_key, pending_action_projection,
                                pending_action_projection_bytes
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s
                            )
                            """,
                            rows,
                        )
                        await self._enqueue_persisted_event_side_effects(
                            cur,
                            fork.id,
                            events,
                        )
                    loaded = await self._load(cur, fork.id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {fork.id}")
                await conn.commit()
            except UniqueViolation as exc:
                await conn.rollback()
                raise ValueError(f"Session already exists: {fork.id}") from exc
            except Exception:
                await conn.rollback()
                raise

            return loaded

    async def load(self, session_id: str) -> Session | None:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        if access_bounds is not None:
            from cayu._resource_access_errors import ResourceAccessDenied

            try:
                return await self._access_load_session(access_bounds, session_id)
            except ResourceAccessDenied:
                return None
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            return await self._load(cur, session_id)

    async def load_state(self, session_id: str) -> SessionStateSnapshot | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, status, updated_at, last_activity_at
                FROM cayu_sessions
                WHERE id = %s
                """,
                (session_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            return SessionStateSnapshot(
                id=row[0],
                status=SessionStatus(row[1]),
                updated_at=pg_support.to_utc(row[2]),
                last_activity_at=pg_support.to_utc(row[3]),
            )

    async def load_invocation_snapshot(
        self,
        session_id: str,
    ) -> SessionInvocationSnapshot | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT id, instance_id, status, invocation FROM cayu_sessions WHERE id = %s",
                (session_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            invocation_value = row[3]
            if isinstance(invocation_value, str):
                invocation_value = json.loads(invocation_value)
            return SessionInvocationSnapshot(
                id=row[0],
                session_instance_id=row[1],
                status=SessionStatus(row[2]),
                invocation=SessionInvocation.model_validate(invocation_value),
            )

    async def create_recall_receipt(self, receipt: RecallReceipt) -> RecallReceipt:
        copied = copy_recall_receipt(receipt)
        document = memory_evidence_document_bytes(copied, "recall receipt")
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await self._lock_closure_lineage(cur)
                    await _postgres_lock_memory_evidence_id(
                        cur,
                        f"recall-receipt:{copied.receipt_id}",
                    )
                    await cur.execute(
                        """
                        SELECT receipt_id, session_id, interaction_id, model_step_id,
                               created_at, receipt_json, document_bytes
                        FROM cayu_recall_receipts
                        WHERE receipt_id = %s
                        """,
                        (copied.receipt_id,),
                    )
                    row = await cur.fetchone()
                    if row is not None:
                        current = _postgres_recall_receipt(row)
                        if (
                            memory_evidence_document_bytes(current, "stored recall receipt")
                            != document
                        ):
                            raise RecallEvidenceConflict("Recall receipt", copied.receipt_id)
                        await conn.commit()
                        return current
                    for owner in await self._closure_lineage_owners(cur, (copied.session_id,)):
                        _check_closure_lineage_owner(owner, (copied.session_id,))
                    await cur.execute(
                        "SELECT 1 FROM cayu_sessions WHERE id = %s FOR KEY SHARE",
                        (copied.session_id,),
                    )
                    if await cur.fetchone() is None:
                        raise KeyError(f"Session not found: {copied.session_id}")
                    await cur.execute(
                        """
                        INSERT INTO cayu_recall_receipts (
                            receipt_id, session_id, interaction_id, model_step_id,
                            created_at, receipt_json, document_bytes
                        ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                        """,
                        (
                            copied.receipt_id,
                            copied.session_id,
                            copied.interaction_id,
                            copied.model_step_id,
                            copied.created_at,
                            document.decode("utf-8"),
                            len(document),
                        ),
                    )
                await conn.commit()
                return copied
            except BaseException:
                await conn.rollback()
                raise

    async def load_recall_receipt(
        self,
        session_id: str,
        receipt_id: str,
    ) -> RecallReceipt | None:
        session_id = require_memory_evidence_session_id(session_id)
        receipt_id = require_memory_evidence_id(receipt_id, "receipt_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT receipt_id, session_id, interaction_id, model_step_id,
                       created_at, receipt_json, document_bytes
                FROM cayu_recall_receipts
                WHERE session_id = %s AND receipt_id = %s
                """,
                (session_id, receipt_id),
            )
            row = await cur.fetchone()
            return None if row is None else _postgres_recall_receipt(row)

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
        clauses = ["session_id = %s"]
        parameters: list[object] = [copied_query.session_id]
        if copied_query.interaction_id is not None:
            clauses.append("interaction_id = %s")
            parameters.append(copied_query.interaction_id)
        if copied_query.model_step_id is not None:
            clauses.append("model_step_id = %s")
            parameters.append(copied_query.model_step_id)
        if after is not None:
            clauses.append('(created_at, receipt_id COLLATE "C") > (%s, %s)')
            parameters.extend(after)
        where = " AND ".join(clauses)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT receipt_id, session_id, interaction_id, model_step_id,
                       created_at, receipt_json, document_bytes
                FROM cayu_recall_receipts
                WHERE {where}
                ORDER BY created_at, receipt_id COLLATE "C"
                LIMIT %s
                """,
                (*parameters, copied_query.limit + 1),
            )
            rows = await cur.fetchall()
        retained: list[RecallReceipt] = []
        retained_bytes = 2
        for row in rows:
            if len(retained) >= copied_query.limit:
                break
            receipt = _postgres_recall_receipt(row)
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
        lock_ids = sorted(
            {
                f"context-exposure:{copied.exposure_id}",
                f"model-attempt:{copied.session_id}:{copied.model_attempt_id}",
                f"provider-attempt:{copied.session_id}:{copied.provider_attempt_id}",
            }
        )
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await self._lock_closure_lineage(cur)
                    for lock_id in lock_ids:
                        await _postgres_lock_memory_evidence_id(cur, lock_id)
                    await cur.execute(
                        """
                        SELECT exposure_id, session_id, interaction_id, model_step_id,
                               model_attempt_id, provider_attempt_id, state, state_revision,
                               created_at, updated_at, exposure_json, document_bytes
                        FROM cayu_context_exposures
                        WHERE exposure_id = %s
                        FOR UPDATE
                        """,
                        (copied.exposure_id,),
                    )
                    row = await cur.fetchone()
                    if row is not None:
                        current = _postgres_context_exposure(row)
                        await cur.execute(
                            """
                            SELECT exposure_id, ordinal, receipt_id, receipt_item_ordinal,
                                   item_json, document_bytes
                            FROM cayu_recall_item_exposures
                            WHERE exposure_id = %s
                            ORDER BY ordinal
                            LIMIT %s
                            """,
                            (copied.exposure_id, MAX_RECALL_RECEIPT_ITEMS + 1),
                        )
                        current_item_rows = await cur.fetchall()
                        if len(current_item_rows) > MAX_RECALL_RECEIPT_ITEMS:
                            raise ValueError(
                                "Stored recall item exposures exceed their count bound."
                            )
                        current_items = _postgres_recall_item_exposures(current_item_rows)
                        if (
                            not context_exposure_creation_matches(current, copied)
                            or tuple(
                                memory_evidence_document_bytes(item, "stored recall item exposure")
                                for item in current_items
                            )
                            != item_documents
                        ):
                            raise RecallEvidenceConflict(
                                "Context exposure",
                                copied.exposure_id,
                            )
                        await conn.commit()
                        return current
                    for owner in await self._closure_lineage_owners(cur, (copied.session_id,)):
                        _check_closure_lineage_owner(owner, (copied.session_id,))
                    await cur.execute(
                        "SELECT 1 FROM cayu_sessions WHERE id = %s FOR KEY SHARE",
                        (copied.session_id,),
                    )
                    if await cur.fetchone() is None:
                        raise KeyError(f"Session not found: {copied.session_id}")
                    await cur.execute(
                        """
                        SELECT exposure_id
                        FROM cayu_context_exposures
                        WHERE session_id = %s AND model_attempt_id = %s
                        """,
                        (copied.session_id, copied.model_attempt_id),
                    )
                    model_attempt_collision = await cur.fetchone()
                    if model_attempt_collision is not None:
                        raise RecallEvidenceConflict(
                            "Model-attempt exposure",
                            str(model_attempt_collision[0]),
                        )
                    await cur.execute(
                        """
                        SELECT exposure_id
                        FROM cayu_context_exposures
                        WHERE session_id = %s AND provider_attempt_id = %s
                        """,
                        (copied.session_id, copied.provider_attempt_id),
                    )
                    provider_attempt_collision = await cur.fetchone()
                    if provider_attempt_collision is not None:
                        raise RecallEvidenceConflict(
                            "Provider-attempt exposure",
                            str(provider_attempt_collision[0]),
                        )
                    receipts: dict[str, RecallReceipt] = {}
                    for receipt_id in copied.receipt_ids:
                        await cur.execute(
                            """
                            SELECT receipt_id, session_id, interaction_id, model_step_id,
                                   created_at, receipt_json, document_bytes
                            FROM cayu_recall_receipts
                            WHERE receipt_id = %s
                            FOR SHARE
                            """,
                            (receipt_id,),
                        )
                        receipt_row = await cur.fetchone()
                        if receipt_row is None:
                            raise KeyError(f"Recall receipt not found: {receipt_id}")
                        receipt = _postgres_recall_receipt(receipt_row)
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
                    await cur.execute(
                        """
                        INSERT INTO cayu_context_exposures (
                            exposure_id, session_id, interaction_id, model_step_id,
                            model_attempt_id, provider_attempt_id, state, state_revision,
                            created_at, updated_at, exposure_json, document_bytes
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s::jsonb, %s
                        )
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
                            copied.created_at,
                            copied.updated_at,
                            document.decode("utf-8"),
                            len(document),
                        ),
                    )
                    await cur.executemany(
                        """
                        INSERT INTO cayu_recall_item_exposures (
                            exposure_id, ordinal, receipt_id, receipt_item_ordinal,
                            item_json, document_bytes
                        ) VALUES (%s, %s, %s, %s, %s::jsonb, %s)
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
                await conn.commit()
                return copied
            except BaseException:
                await conn.rollback()
                raise

    async def load_context_exposure(
        self,
        session_id: str,
        exposure_id: str,
    ) -> ContextExposure | None:
        session_id = require_memory_evidence_session_id(session_id)
        exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT exposure_id, session_id, interaction_id, model_step_id,
                       model_attempt_id, provider_attempt_id, state, state_revision,
                       created_at, updated_at, exposure_json, document_bytes
                FROM cayu_context_exposures
                WHERE session_id = %s AND exposure_id = %s
                """,
                (session_id, exposure_id),
            )
            row = await cur.fetchone()
            return None if row is None else _postgres_context_exposure(row)

    async def load_recall_item_exposures(
        self,
        session_id: str,
        exposure_id: str,
    ) -> tuple[RecallItemExposure, ...]:
        session_id = require_memory_evidence_session_id(session_id)
        exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT item.exposure_id, item.ordinal, item.receipt_id,
                       item.receipt_item_ordinal, item.item_json, item.document_bytes
                FROM cayu_recall_item_exposures AS item
                JOIN cayu_context_exposures AS exposure
                  ON exposure.exposure_id = item.exposure_id
                WHERE exposure.session_id = %s AND exposure.exposure_id = %s
                ORDER BY item.ordinal
                LIMIT %s
                """,
                (session_id, exposure_id, MAX_RECALL_RECEIPT_ITEMS + 1),
            )
            rows = await cur.fetchall()
            if len(rows) > MAX_RECALL_RECEIPT_ITEMS:
                raise ValueError("Stored recall item exposures exceed their count bound.")
            return _postgres_recall_item_exposures(rows)

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
        clauses = ["session_id = %s"]
        parameters: list[object] = [copied_query.session_id]
        if copied_query.interaction_id is not None:
            clauses.append("interaction_id = %s")
            parameters.append(copied_query.interaction_id)
        if copied_query.model_step_id is not None:
            clauses.append("model_step_id = %s")
            parameters.append(copied_query.model_step_id)
        if after is not None:
            clauses.append('(created_at, exposure_id COLLATE "C") > (%s, %s)')
            parameters.extend(after)
        where = " AND ".join(clauses)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT exposure_id, session_id, interaction_id, model_step_id,
                       model_attempt_id, provider_attempt_id, state, state_revision,
                       created_at, updated_at, exposure_json, document_bytes
                FROM cayu_context_exposures
                WHERE {where}
                ORDER BY created_at, exposure_id COLLATE "C"
                LIMIT %s
                """,
                (*parameters, copied_query.limit + 1),
            )
            rows = await cur.fetchall()
        retained: list[ContextExposure] = []
        retained_bytes = 2
        for row in rows:
            if len(retained) >= copied_query.limit:
                break
            exposure = _postgres_context_exposure(row)
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
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await self._lock_closure_lineage(cur)
                    await cur.execute(
                        """
                        SELECT exposure_id, session_id, interaction_id, model_step_id,
                               model_attempt_id, provider_attempt_id, state, state_revision,
                               created_at, updated_at, exposure_json, document_bytes
                        FROM cayu_context_exposures
                        WHERE session_id = %s AND exposure_id = %s
                        FOR UPDATE
                        """,
                        (session_id, exposure_id),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        raise KeyError(f"Context exposure not found: {exposure_id}")
                    current = _postgres_context_exposure(row)
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
                        await conn.commit()
                        return current
                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
                    updated = append_context_exposure_transition(current, copied_request)
                    updated_document = memory_evidence_document_bytes(
                        updated,
                        "context exposure",
                    )
                    await cur.execute(
                        """
                        UPDATE cayu_context_exposures
                        SET state = %s, state_revision = %s, updated_at = %s,
                            exposure_json = %s::jsonb, document_bytes = %s
                        WHERE session_id = %s AND exposure_id = %s
                          AND state = %s AND state_revision = %s
                        """,
                        (
                            str(updated.state),
                            updated.state_revision,
                            updated.updated_at,
                            updated_document.decode("utf-8"),
                            len(updated_document),
                            session_id,
                            exposure_id,
                            str(copied_request.expected_state),
                            copied_request.expected_revision,
                        ),
                    )
                    if cur.rowcount != 1:
                        raise ContextExposureTransitionConflict(
                            exposure_id,
                            expected_state=copied_request.expected_state,
                            expected_revision=copied_request.expected_revision,
                            actual_state=current.state,
                            actual_revision=current.state_revision,
                        )
                await conn.commit()
                return updated
            except BaseException:
                await conn.rollback()
                raise

    async def inspect_identity(self, session_id: str) -> SessionInspectionIdentity:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, agent_name, provider_name, model, parent_session_id,
                       causal_budget_id, runtime_name, runtime_version, environment_name,
                       status, created_at, updated_at, last_activity_at, run_epoch,
                       metadata -> 'cayu:runtime_build_provenance'
                           AS runtime_build_provenance
                FROM cayu_sessions
                WHERE id = %s
                """,
                (session_id,),
            )
            row = await cur.fetchone()
            if row is None:
                raise KeyError(session_id)
            await cur.execute(
                """
                SELECT key, value,
                       (SELECT COUNT(*)
                        FROM cayu_session_labels
                        WHERE session_id = %s) AS label_count
                FROM cayu_session_labels
                WHERE session_id = %s
                ORDER BY key COLLATE "C" ASC
                LIMIT %s
                """,
                (session_id, session_id, SESSION_INSPECTION_LABEL_LIMIT),
            )
            label_rows = await cur.fetchall()
            label_count = 0 if not label_rows else label_rows[0][2]
            return SessionInspectionIdentity(
                id=row[0],
                agent_name=row[1],
                provider_name=row[2],
                model=row[3],
                parent_session_id=row[4],
                causal_budget_id=row[5],
                runtime_name=row[6],
                runtime_version=row[7],
                runtime_build_provenance=(
                    pg_support.runtime_build_provenance_from_session_metadata(
                        {} if row[14] is None else {"cayu:runtime_build_provenance": row[14]}
                    )
                ),
                environment_name=row[8],
                status=SessionStatus(row[9]),
                created_at=pg_support.to_utc(row[10]),
                updated_at=pg_support.to_utc(row[11]),
                last_activity_at=pg_support.to_utc(row[12]),
                run_epoch=row[13],
                labels={label_row[0]: label_row[1] for label_row in label_rows},
                label_count=label_count,
                labels_truncated=label_count > len(label_rows),
            )

    async def update_status(self, session_id: str, status: SessionStatus) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(status, SessionStatus):
            raise ValueError("Session status must be a SessionStatus.")
        return await self.transition_status(
            session_id,
            from_statuses=set(SessionStatus),
            to_status=status,
        )

    async def load_session_closure_receipt(
        self, session_id: str, plan_id: str
    ) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        plan_id = require_clean_nonblank(plan_id, "plan_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT receipt_json FROM cayu_session_closure_receipts "
                "WHERE session_id = %s AND plan_id = %s",
                (session_id, plan_id),
            )
            row = await cur.fetchone()
        return None if row is None else dict(row[0])

    async def load_session_closure_progress(self, session_id: str, plan_id: str):
        session_id = require_clean_nonblank(session_id, "session_id")
        plan_id = require_clean_nonblank(plan_id, "plan_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT progress_json FROM cayu_session_closure_progress "
                "WHERE root_session_id = %s AND plan_id = %s",
                (session_id, plan_id),
            )
            row = await cur.fetchone()
        return None if row is None else dict(row[0])

    async def save_session_closure_progress(self, progress: dict[str, Any]) -> None:
        root_id = progress.get("root_session_id")
        plan_id = progress.get("plan_id")
        await self._ensure_ready()
        async with self._connection() as conn:
            await self._lock_closure_lineage(conn)
            cursor = await conn.execute(
                "SELECT progress_json FROM cayu_session_closure_progress "
                "WHERE root_session_id = %s AND plan_id = %s",
                (root_id, plan_id),
            )
            row = await cursor.fetchone()
            if row is not None:
                _validate_closure_progress_update(dict(row[0]), progress)
            await conn.execute(
                "INSERT INTO cayu_session_closure_progress "
                "(root_session_id, plan_id, progress_json) VALUES (%s, %s, %s) "
                "ON CONFLICT (root_session_id, plan_id) DO UPDATE "
                "SET progress_json = EXCLUDED.progress_json",
                (root_id, plan_id, Jsonb(progress)),
            )

    @staticmethod
    async def _lock_closure_lineage(executor: Any) -> None:
        # Short admission/publication transactions only: never held across
        # dependent-store cleanup. The progress row retains durable ownership.
        await executor.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            ("cayu-session-closure-lineage",),
        )

    async def _closure_lineage_owners(
        self, cur: Any, targets: Iterable[str]
    ) -> tuple[dict[str, Any], ...]:
        targets = list(targets)
        await cur.execute(
            "SELECT progress_json FROM cayu_session_closure_progress AS p "
            "WHERE root_session_id = ANY(%s) OR EXISTS "
            "(SELECT 1 FROM jsonb_array_elements(p.progress_json->'descendants') AS child "
            "WHERE child->>'session_id' = ANY(%s))",
            (targets, targets),
        )
        return tuple(dict(row[0]) for row in await cur.fetchall())

    async def claim_session_closure_progress(self, progress: dict[str, Any]) -> None:
        progress = deepcopy(progress)
        targets = _closure_progress_targets(progress)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await self._lock_closure_lineage(cur)
            for owner in await self._closure_lineage_owners(cur, targets):
                if (owner["root_session_id"], owner["plan_id"]) == (
                    progress["root_session_id"],
                    progress["plan_id"],
                ):
                    _validate_closure_progress_update(owner, progress)
                    return
                _check_closure_lineage_owner(owner, targets)
            await cur.execute(
                "SELECT id, parent_session_id FROM cayu_sessions WHERE id = ANY(%s) FOR UPDATE",
                (list(targets),),
            )
            parents = dict(await cur.fetchall())
            if progress["root_session_id"] not in parents:
                raise ValueError("Closure root disappeared before lineage admission.")
            for item in progress["descendants"]:
                if parents.get(item["session_id"]) != item["parent_session_id"]:
                    raise ValueError("Child lineage changed before closure admission.")
            if progress["phase"] in {"recursive", "reject"}:
                expected = {
                    (item["session_id"], item["parent_session_id"])
                    for item in progress["descendants"]
                }
                await cur.execute(
                    "SELECT id, parent_session_id FROM cayu_sessions "
                    "WHERE parent_session_id = ANY(%s) LIMIT %s",
                    (list(targets), len(expected) + 1),
                )
                if set(await cur.fetchall()) != expected:
                    raise ValueError("Child lineage changed before closure admission.")
            for target_id in targets:
                target = await self._load_for_update(cur, target_id)
                if target is None:
                    raise ValueError("Closure target disappeared before admission.")
                await self._require_session_erasure_quiescence(cur, target)
                await self._load_session_closure_records(
                    cur,
                    target_id,
                    max_records=progress["max_records"],
                    max_bytes=progress["max_bytes"],
                )
            await cur.execute(
                "INSERT INTO cayu_session_closure_progress "
                "(root_session_id, plan_id, progress_json) VALUES (%s, %s, %s)",
                (progress["root_session_id"], progress["plan_id"], Jsonb(progress)),
            )

    async def _require_available_closure_identity(self, cur: Any, session_id: str) -> None:
        for owner in await self._closure_lineage_owners(cur, (session_id,)):
            _check_closure_lineage_owner(owner, (session_id,))
        await cur.execute(
            "SELECT 1 FROM cayu_session_closure_receipts WHERE session_id = %s LIMIT 1",
            (session_id,),
        )
        if await cur.fetchone() is not None:
            raise ValueError("Session identity was retired by closure.")

    async def load_session_closure_tombstones(
        self, root_session_id: str, plan_id: str
    ) -> tuple[dict[str, Any], ...]:
        root_session_id = require_clean_nonblank(root_session_id, "root_session_id")
        plan_id = require_clean_nonblank(plan_id, "plan_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT tombstone_json FROM cayu_session_closure_tombstones "
                "WHERE root_session_id = %s AND plan_id = %s "
                'ORDER BY child_session_id COLLATE "C"',
                (root_session_id, plan_id),
            )
            rows = await cur.fetchall()
        return tuple(dict(row[0]) for row in rows)

    async def detach_session_children(
        self,
        parent_session_id: str,
        child_session_ids: tuple[str, ...],
        *,
        closure_receipt: dict[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        parent_session_id = require_clean_nonblank(parent_session_id, "parent_session_id")
        root_id = closure_receipt.get("root_session_id")
        plan_id = closure_receipt.get("plan_id")
        if type(root_id) is not str or type(plan_id) is not str:
            raise ValueError("Closure detachment receipt is missing identity.")
        if len(set(child_session_ids)) != len(child_session_ids):
            raise ValueError("Detached child session IDs must be unique.")
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await self._lock_closure_lineage(cur)
                    await cur.execute(
                        "SELECT tombstone_json FROM cayu_session_closure_tombstones "
                        "WHERE root_session_id = %s AND plan_id = %s "
                        'ORDER BY child_session_id COLLATE "C"',
                        (root_id, plan_id),
                    )
                    existing = await cur.fetchall()
                    if existing:
                        tombstones = tuple(dict(row[0]) for row in existing)
                        _validate_session_closure_detach_replay(
                            tombstones,
                            root_id=root_id,
                            plan_id=plan_id,
                            parent_session_id=parent_session_id,
                            child_session_ids=child_session_ids,
                        )
                        return tombstones
                    for owner in await self._closure_lineage_owners(cur, child_session_ids):
                        _check_closure_lineage_owner(owner, child_session_ids)
                    if not child_session_ids:
                        return ()
                    await cur.execute(
                        "SELECT id, parent_session_id FROM cayu_sessions "
                        "WHERE id = ANY(%s) FOR UPDATE",
                        (list(child_session_ids),),
                    )
                    rows = await cur.fetchall()
                    if {row[0] for row in rows} != set(child_session_ids) or any(
                        row[1] != parent_session_id for row in rows
                    ):
                        raise ValueError("Child lineage changed before detachment.")
                    detached_at = await self._session_store_now(cur)
                    tombstones = tuple(
                        {
                            "root_session_id": root_id,
                            "plan_id": plan_id,
                            "child_session_id": child_id,
                            "original_parent_session_id": parent_session_id,
                            "detached_at": detached_at.isoformat(),
                        }
                        for child_id in sorted(child_session_ids)
                    )
                    await cur.execute(
                        "UPDATE cayu_sessions SET parent_session_id = NULL, updated_at = %s "
                        "WHERE id = ANY(%s)",
                        (detached_at, list(child_session_ids)),
                    )
                    await cur.executemany(
                        "INSERT INTO cayu_session_closure_tombstones "
                        "(root_session_id, plan_id, child_session_id, original_parent_session_id, "
                        "detached_at, tombstone_json) VALUES (%s, %s, %s, %s, %s, %s)",
                        [
                            (
                                item["root_session_id"],
                                item["plan_id"],
                                item["child_session_id"],
                                item["original_parent_session_id"],
                                detached_at,
                                Jsonb(item),
                            )
                            for item in tombstones
                        ],
                    )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
            return tombstones

    async def _require_session_erasure_quiescence(self, cur, session: Session) -> None:
        """Shared admission for closure and final deletion; no mutations."""
        await cur.execute(
            "SELECT 1 FROM cayu_external_waits WHERE session_id=%s "
            "AND session_instance_id=%s AND pending_handoff=1 LIMIT 1",
            (session.id, session.instance_id),
        )
        if await cur.fetchone() is not None:
            raise ValueError("Session has a pending external-wait handoff.")
        from cayu._validation import DURABLE_DOCUMENT_LIMITS
        from cayu.collaboration import _session_export_store as session_exports
        from cayu.runtime._session_closure_records import require_terminal_protected_effect
        from cayu.sessions import _session_continuation_store as continuations

        session_id = session.id
        export_records: dict[str, dict[str, Any]] = {}
        continuation_records: dict[str, dict[str, Any]] = {}
        producer_records = {}
        after_key = ""
        while True:
            # Read one bounded document at a time, allowing JSON text overhead.
            # The shared validator applies the durable document limit.
            await cur.execute(
                "SELECT idempotency_key, CASE WHEN octet_length(record::text) <= %s "
                "THEN record END FROM cayu_session_operations "
                "WHERE session_id = %s AND (idempotency_key LIKE 'tool-effect:%%' "
                "OR idempotency_key LIKE 'session-export:%%' "
                "OR idempotency_key LIKE 'producer-output:%%' "
                "OR idempotency_key LIKE 'session-continuation:%%') "
                "AND idempotency_key > %s ORDER BY idempotency_key LIMIT 1",
                (8 * DURABLE_DOCUMENT_LIMITS.max_bytes, session_id, after_key),
            )
            effects = await cur.fetchall()
            for key, raw in effects:
                if key.startswith(continuations.CONTINUATION_OPERATION_PREFIX):
                    continuations.collect_retained_record(continuation_records, key, raw)
                elif key.startswith("producer-output:"):
                    producer_records[key] = raw
                elif key.startswith(session_exports.OPERATION_PREFIX):
                    if type(raw) is not dict:
                        raise ValueError("Session export retention evidence is malformed.")
                    export_records[key] = raw
                else:
                    require_terminal_protected_effect(session_id, session.instance_id, key, raw)
            if not effects:
                break
            after_key = effects[-1][0]
        if session.status in DELETE_BLOCKED_SESSION_STATUSES:
            raise ValueError("Session closure requires a non-running target.")
        await cur.execute(
            "SELECT 1 FROM cayu_persisted_event_side_effects "
            "WHERE session_id = %s AND status = 'leased' LIMIT 1",
            (session_id,),
        )
        if await cur.fetchone() is not None:
            raise ValueError("Session closure requires settled event side-effect deliveries.")
        checkpoint = await self._load_checkpoint(cur, session_id)
        deletion_now = await self._session_store_now(cur)
        from cayu.runtime._producer_output_store import require_erasure_quiescence

        require_erasure_quiescence(session=session, checkpoint=checkpoint, records=producer_records)
        continuations.require_erasure_quiescence(
            session=session, checkpoint=checkpoint, records=continuation_records
        )
        session_exports.require_erasure_quiescence(
            session=session, checkpoint=checkpoint, export_records=export_records
        )
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
        await cur.execute(
            "SELECT event FROM cayu_events "
            "WHERE session_id = %s AND event_type = ANY(%s) "
            "ORDER BY session_order DESC LIMIT %s",
            (
                session_id,
                [str(event_type) for event_type in _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES],
                _TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT,
            ),
        )
        terminal_publication_block = _terminal_publication_delete_block_reason(
            session=session,
            checkpoint=checkpoint,
            evidence_events=[Event(**pg_support._json_obj(row[0])) for row in await cur.fetchall()],
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
                f"Cannot delete a session while {completion_result_publication_block}: {session_id}"
            )
        await cur.execute(
            "SELECT 1 FROM cayu_session_operations WHERE session_id = %s AND idempotency_key = %s",
            (
                session_id,
                MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
            ),
        )
        if await cur.fetchone() is not None:
            raise ValueError(
                f"Cannot delete a session while a model-completion stage is active: {session_id}"
            )
        await cur.execute(
            """
            SELECT identity.reservation_id
            FROM cayu_budget_reservation_identities AS identity
            LEFT JOIN cayu_events AS event
              ON event.session_id = identity.publication_session_id
             AND event.event_type IN (
                 'budget.reconciled',
                 'budget.reservation_released'
             )
             AND event.payload ->> 'reservation_id'
                 = identity.reservation_id
            LEFT JOIN cayu_persisted_event_side_effects AS delivery
              ON delivery.session_id = event.session_id
             AND delivery.event_id = event.event_id
            WHERE identity.publication_session_id = %s
            GROUP BY identity.reservation_id
            HAVING COUNT(event.event_id) <> 1
                OR COUNT(*) FILTER (
                    WHERE delivery.status = 'delivered'
                ) <> 1
            LIMIT 1
            """,
            (session_id,),
        )
        if await cur.fetchone() is not None:
            raise ValueError(
                "Cannot delete a session while a budget settlement audit "
                f"event is pending: {session_id}"
            )

    async def validate_session_closure_admission(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            session = await self._load_for_update(cur, session_id)
            if session is None:
                raise ValueError("Closure target is unavailable.")
            await self._require_session_erasure_quiescence(cur, session)

    @runtime_session_mutation
    async def delete_session(
        self,
        session_id: str,
        *,
        closure_receipt: dict[str, Any] | None = None,
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await self._lock_closure_lineage(cur)
                    await cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (f"context-view-session:{session_id}",),
                    )
                    session = await self._load_for_update(cur, session_id)
                    if _access_bounds is not None:
                        _access_bounds.require_action(session, "delete")
                    if session is None:
                        await conn.rollback()
                        return
                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,), closure_receipt)
                    await cur.execute(
                        "SELECT 1 FROM cayu_session_closure_progress AS p "
                        "JOIN cayu_sessions AS child ON child.id = p.root_session_id "
                        "WHERE child.parent_session_id = %s LIMIT 1",
                        (session_id,),
                    )
                    if await cur.fetchone() is not None:
                        raise ValueError(
                            "Session lineage is owned by an unfinished recursive closure."
                        )
                    if (
                        closure_receipt is not None
                        and closure_receipt.get("operation") == "recursive"
                    ):
                        expected_parent = closure_receipt.get("original_parent_session_id")
                        if (
                            type(expected_parent) is not str
                            or session.parent_session_id != expected_parent
                        ):
                            raise ValueError("Recursive closure child parent identity conflict.")
                    if session.status in DELETE_BLOCKED_SESSION_STATUSES:
                        raise ValueError(
                            f"Cannot delete a session while it is {session.status}; "
                            f"interrupt it first: {session_id}"
                        )
                    if self.context_view_version is not None:
                        await cur.execute(
                            "SELECT 1 FROM cayu_context_view_selections s "
                            "LEFT JOIN cayu_context_views v ON v.view_id = s.view_id "
                            "WHERE (v.source_session_id = %s OR v.view_id IS NULL) "
                            "AND s.state IN ('selected', 'adopted', 'transferred') "
                            "AND (s.state <> 'selected' OR s.expires_at_ms > %s) LIMIT 1",
                            (session_id, int(self._clock().timestamp() * 1000)),
                        )
                        if await cur.fetchone() is not None:
                            raise ValueError("Session has an active context-view retention pin.")
                    await cur.execute(
                        "SELECT id FROM cayu_sessions "
                        "WHERE parent_session_id = %s "
                        "AND metadata #>> '{subagent,mode}' = %s "
                        'ORDER BY id COLLATE "C" LIMIT 1',
                        (session_id, "durable"),
                    )
                    durable_child = await cur.fetchone()
                    if closure_receipt is not None or _access_bounds is not None:
                        await cur.execute(
                            "SELECT 1 FROM cayu_sessions WHERE parent_session_id = %s LIMIT 1",
                            (session_id,),
                        )
                        if await cur.fetchone() is not None:
                            raise ValueError("Closure deletion requires no remaining child edges.")
                    if durable_child is not None:
                        raise ValueError(
                            _durable_subagent_parent_delete_block_reason(durable_child[0])
                        )
                    await self._require_session_erasure_quiescence(cur, session)
                    await cur.execute(
                        "UPDATE cayu_peer_content_receipts SET target_deleted = TRUE "
                        "WHERE receipt_json->>'status' = 'appended' "
                        "AND receipt_json->>'target_session_id' = %s "
                        "AND receipt_json->>'target_session_instance_id' = %s",
                        (session.id, session.instance_id),
                    )
                    # ON DELETE CASCADE removes events/labels/checkpoint/transcript;
                    # the self-FK is ON DELETE SET NULL so children keep loading.
                    await cur.execute(
                        "DELETE FROM cayu_sessions WHERE id = %s",
                        (session_id,),
                    )
                    if closure_receipt is not None:
                        receipt_plan_id = closure_receipt.get("plan_id")
                        if type(receipt_plan_id) is not str:
                            raise ValueError("Session closure receipt is missing plan identity.")
                        await cur.execute(
                            "INSERT INTO cayu_session_closure_receipts "
                            "(session_id, plan_id, committed_at, receipt_json) "
                            "VALUES (%s, %s, clock_timestamp(), %s) "
                            "ON CONFLICT (session_id, plan_id) DO UPDATE SET receipt_json = EXCLUDED.receipt_json",
                            (session_id, receipt_plan_id, Jsonb(closure_receipt)),
                        )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    @runtime_session_mutation
    async def update_labels(
        self,
        session_id: str,
        labels: dict[str, str],
        *,
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        new_labels = copy_label_map(labels, "labels", allow_reserved=False)
        await self._ensure_ready()
        expected_run_epoch = _current_session_run_epoch(session_id)
        async with self._connection() as conn:
            async with conn.cursor() as cur:
                access_session = await self._load_for_update(cur, session_id)
                if _access_bounds is not None:
                    _access_bounds.require_label_update(access_session, new_labels)
                if access_session is None:
                    raise KeyError(f"Session not found: {session_id}")
                for owner in await self._closure_lineage_owners(cur, (session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
                updated_at = await self._session_store_now(cur)
                if expected_run_epoch is None:
                    await cur.execute(
                        "UPDATE cayu_sessions SET updated_at = %s WHERE id = %s",
                        (updated_at, session_id),
                    )
                else:
                    await cur.execute(
                        "UPDATE cayu_sessions SET updated_at = %s WHERE id = %s AND run_epoch = %s",
                        (updated_at, session_id, expected_run_epoch),
                    )
                if cur.rowcount != 1:
                    if expected_run_epoch is not None:
                        await _raise_session_write_conflict(cur, session_id, expected_run_epoch)
                    raise KeyError(f"Session not found: {session_id}")
                await cur.execute(
                    "DELETE FROM cayu_session_labels WHERE session_id = %s",
                    (session_id,),
                )
                if new_labels:
                    await cur.executemany(
                        """
                        INSERT INTO cayu_session_labels (session_id, key, value)
                        VALUES (%s, %s, %s)
                        """,
                        [(session_id, key, value) for key, value in new_labels.items()],
                    )
                if _access_bounds is not None:
                    audit = _access_bounds.label_audit(access_session, new_labels, updated_at)
                    if audit is not None:
                        key, record = audit
                        await cur.execute(
                            "INSERT INTO cayu_session_operations (session_id, idempotency_key, record, updated_at) VALUES (%s, %s, %s, %s)",
                            (session_id, key, Jsonb(record), updated_at),
                        )
                loaded = await self._load(cur, session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                from cayu.sessions._invocation_lifecycle import (
                    require_invocation_lifecycle_release_capacity,
                )

                require_invocation_lifecycle_release_capacity(
                    await self._load_checkpoint(cur, session_id),
                    loaded,
                )
            await conn.commit()
            return loaded

    @runtime_session_mutation
    async def update_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any],
        *,
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        user_metadata = copy_session_user_metadata(metadata)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    if _access_bounds is not None:
                        _access_bounds.require_action(
                            await self._load_for_update(cur, session_id), "modify"
                        )
                    await cur.execute(
                        "SELECT run_epoch, metadata FROM cayu_sessions WHERE id = %s FOR UPDATE",
                        (session_id,),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        raise KeyError(f"Session not found: {session_id}")
                    _assert_session_run_epoch_value(session_id, row[0])
                    updated_at = await self._session_store_now(cur)
                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
                    new_metadata = replace_session_user_metadata(
                        pg_support._json_obj(row[1]), user_metadata
                    )
                    await cur.execute(
                        "UPDATE cayu_sessions SET metadata = %s, updated_at = %s WHERE id = %s",
                        (pg_support._dumps(new_metadata), updated_at, session_id),
                    )
                    loaded = await self._load(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    from cayu.sessions._invocation_lifecycle import (
                        require_invocation_lifecycle_release_capacity,
                    )

                    require_invocation_lifecycle_release_capacity(
                        await self._load_checkpoint(cur, session_id),
                        loaded,
                    )
                await conn.commit()
            except Exception:
                await conn.rollback()
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
        await self._ensure_ready()
        expected_run_epoch = _current_session_run_epoch(session_id)
        async with self._connection() as conn:
            async with conn.cursor() as cur:
                admission_source = await self._load_for_update(cur, session_id)
                if admission_source is None:
                    raise KeyError(f"Session not found: {session_id}")
                for owner in await self._closure_lineage_owners(cur, (session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
                updated_at = await self._session_store_now(cur)
                params: list[object] = [
                    str(to_status),
                    updated_at,
                    updated_at,
                    1 if to_status == SessionStatus.RUNNING else 0,
                    session_id,
                    [str(status) for status in allowed_statuses],
                ]
                epoch_clause = ""
                if expected_run_epoch is not None:
                    epoch_clause = " AND run_epoch = %s"
                    params.append(expected_run_epoch)
                await cur.execute(
                    f"""
                    UPDATE cayu_sessions
                    SET status = %s, updated_at = %s, last_activity_at = %s,
                        run_epoch = run_epoch + %s
                    WHERE id = %s AND status = ANY(%s){epoch_clause}
                    """,
                    params,
                )
                if cur.rowcount != 1:
                    loaded = await self._load(cur, session_id)
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
                    await self._require_external_wait_admission(cur, admission_source)
                    _require_live_incomplete_recovery_claim_for_run_epoch_transfer(
                        await self._load_checkpoint(cur, session_id),
                        now=updated_at,
                    )
                loaded = await self._load(cur, session_id)
            await conn.commit()
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
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
        expected_latest_interaction_event_id: str | None = None,
        require_no_active_model_completion_dispatch: bool = False,
        temporary_service_admission: TemporaryServiceAdmission | None = None,
    ) -> Session:
        from cayu.sessions._temporary_continuation_scope import prepare_temporary_transition
        from cayu.sessions.pending_actions import pending_action_event_storage_values

        temporary_service_admission = prepare_temporary_transition(temporary_service_admission)

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
        expected_latest_interaction_event_id = _copy_optional_event_id(
            expected_latest_interaction_event_id,
            "expected_latest_interaction_event_id",
        )
        if type(require_no_active_model_completion_dispatch) is not bool:
            raise TypeError("require_no_active_model_completion_dispatch must be a boolean.")
        if prepared_execution_profile_decision is not None and admission is None:
            raise ValueError("An execution-profile decision requires atomic interaction admission.")
        if admission is not None and to_status is not SessionStatus.RUNNING:
            raise ValueError("Interaction admission requires a transition to running.")
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded = await self._load_for_update(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    updated_at = await self._session_store_now(cur)
                    _assert_session_run_epoch(session_id, loaded)
                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
                    if loaded.status not in allowed_statuses:
                        raise SessionStatusConflict(
                            f"Session status transition not allowed: {loaded.status} -> {to_status}"
                        )
                    if to_status is SessionStatus.RUNNING:
                        await self._require_external_wait_admission(cur, loaded)
                    if expected_latest_interaction_event_id is not None:
                        await cur.execute(
                            "SELECT retained.event_id "
                            "FROM cayu_interaction_latest_events AS event "
                            "JOIN cayu_events AS retained "
                            "ON retained.sequence = event.latest_event_sequence "
                            "WHERE event.session_id = %s "
                            "ORDER BY event.latest_event_sequence DESC LIMIT 1 FOR UPDATE",
                            (session_id,),
                        )
                        latest_interaction_row = await cur.fetchone()
                        if (
                            latest_interaction_row is None
                            or latest_interaction_row[0] != expected_latest_interaction_event_id
                        ):
                            raise SessionRunFenced(
                                "Session latest interaction changed before the status transition."
                            )
                    if require_no_active_model_completion_dispatch:
                        await cur.execute(
                            "SELECT record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = %s",
                            (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                        )
                        active_row = await cur.fetchone()
                        if active_row is not None:
                            active_marker = _reconstruct_active_model_completion_stage_record(
                                _decode_model_completion_stage_record(active_row[0]),
                                session_id=session_id,
                            )
                            await cur.execute(
                                "SELECT 1 FROM cayu_session_operations "
                                "WHERE session_id = %s AND idempotency_key = %s",
                                (
                                    session_id,
                                    _model_completion_stage_dispatch_storage_key(
                                        active_marker.stage_id
                                    ),
                                ),
                            )
                            if await cur.fetchone() is not None:
                                raise SessionModelCompletionDispatchAlreadyAuthorized(
                                    "Provider dispatch was authorized before terminal decision election."
                                )
                    transition_profile_metadata = _validate_execution_profile_admission(
                        loaded,
                        candidate_profile=prepared_execution_profile,
                        model_transition=prepared_model_transition,
                        decision=prepared_execution_profile_decision,
                    )
                    if (
                        loaded.status == SessionStatus.PENDING
                        and admission is not None
                        and admission[3]
                    ):
                        await cur.execute(
                            f"SELECT {PARTICIPANT_BINDING_PROJECTION} FROM cayu_participant_session_bindings WHERE session_id = %s",
                            (session_id,),
                        )
                        binding_row = await cur.fetchone()
                        if binding_row is not None:
                            from cayu.sessions._participant_execution_identity import (
                                require_initial_execution_input,
                            )
                            from cayu.storage._participant_session_records import reconstruct

                            await cur.execute(
                                "SELECT message FROM cayu_transcript_messages WHERE session_id = %s ORDER BY session_order",
                                (session_id,),
                            )
                            transcript_rows = await cur.fetchall()
                            require_initial_execution_input(
                                loaded,
                                reconstruct(binding_row, loaded),
                                [
                                    Message.model_validate(pg_support._json_obj(row[0]))
                                    for row in transcript_rows
                                ],
                                admission[2],
                            )
                    transition_metadata = transition_profile_metadata
                    if prepared_model_transition is not None:
                        await cur.execute(
                            "SELECT message FROM cayu_transcript_messages "
                            "WHERE session_id = %s ORDER BY session_order ASC FOR UPDATE",
                            (session_id,),
                        )
                        transcript_rows = await cur.fetchall()
                        _validate_session_model_transition(
                            loaded,
                            [Message.model_validate(row[0]) for row in transcript_rows],
                            await _transcript_cursor(cur, session_id),
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

                    current_checkpoint = await self._load_checkpoint(cur, session_id)
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
                        # Reads after acquiring the row lock may themselves wait.
                        # Deadline checks consume fresh receiving-owner time at
                        # the final synchronous admission callback.
                        updated_at = await self._session_store_now(cur)
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

                    admission_events = []
                    if prepared_execution_profile_decision is not None:
                        admission_events.append(prepared_execution_profile_decision.event)
                    if prepared_model_transition is not None:
                        admission_events.append(prepared_model_transition.event)
                    if admission is not None and admission[0] is not None:
                        admission_events.append(admission[0])

                    transition_values = (
                        str(to_status),
                        updated_at,
                        updated_at,
                        1 if to_status == SessionStatus.RUNNING else 0,
                        len(admission_events),
                    )
                    if prepared_model_transition is None and transition_metadata is None:
                        await cur.execute(
                            """
                            UPDATE cayu_sessions
                            SET status = %s, updated_at = %s, last_activity_at = %s,
                                run_epoch = run_epoch + %s,
                                event_seq = event_seq + %s
                            WHERE id = %s
                            RETURNING event_seq
                            """,
                            (*transition_values, session_id),
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
                        await cur.execute(
                            """
                            UPDATE cayu_sessions
                            SET status = %s, updated_at = %s, last_activity_at = %s,
                                run_epoch = run_epoch + %s,
                                event_seq = event_seq + %s,
                                provider_name = %s, model = %s,
                                runtime_name = %s, runtime_version = %s, metadata = %s
                            WHERE id = %s
                            RETURNING event_seq
                            """,
                            (
                                *transition_values,
                                target_provider_name,
                                target_model,
                                prepared_adopted_runtime_identity.runtime_name,
                                prepared_adopted_runtime_identity.runtime_version,
                                pg_support._dumps(transition_metadata),
                                session_id,
                            ),
                        )
                    elif prepared_model_transition is not None:
                        await cur.execute(
                            """
                            UPDATE cayu_sessions
                            SET status = %s, updated_at = %s, last_activity_at = %s,
                                run_epoch = run_epoch + %s,
                                event_seq = event_seq + %s,
                                provider_name = %s, model = %s, metadata = %s
                            WHERE id = %s
                            RETURNING event_seq
                            """,
                            (
                                *transition_values,
                                prepared_model_transition.target.provider_name,
                                prepared_model_transition.target.model,
                                pg_support._dumps(transition_metadata),
                                session_id,
                            ),
                        )
                    else:
                        await cur.execute(
                            """
                            UPDATE cayu_sessions
                            SET status = %s, updated_at = %s, last_activity_at = %s,
                                run_epoch = run_epoch + %s,
                                event_seq = event_seq + %s, metadata = %s
                            WHERE id = %s
                            RETURNING event_seq
                            """,
                            (
                                *transition_values,
                                pg_support._dumps(transition_metadata),
                                session_id,
                            ),
                        )
                    order_row = await cur.fetchone()
                    if order_row is None:
                        raise KeyError(f"Session not found: {session_id}")
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
                    failover_keys = _model_failover_admission_storage_keys(
                        current_checkpoint, prepared_execution_profile
                    )
                    if failover_keys:
                        await cur.execute(
                            "SELECT idempotency_key, record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                            (session_id, list(failover_keys)),
                        )
                        route_records = {
                            row[0]: _decode_model_completion_stage_record(row[1])
                            for row in await cur.fetchall()
                        }
                        transformed_checkpoint = _model_failover_checkpoint_after_profile_admission(
                            source_session=loaded,
                            admitted_session=transitioned,
                            source_checkpoint=current_checkpoint,
                            admitted_checkpoint=transformed_checkpoint,
                            candidate_profile=prepared_execution_profile,
                            records=route_records,
                            transcript_cursor=await _transcript_cursor(cur, session_id),
                            supports_model_failover=self._supports_model_failover_stage_protocol(),
                        )
                    if result_checkpoint_transform is not None:
                        result_checkpoint = result_checkpoint_transform(
                            transitioned,
                            _copy_checkpoint_for_transform(
                                transformed_checkpoint,
                                session_id=session_id,
                            ),
                        )
                        if result_checkpoint is None:
                            raise ValueError(
                                "Result checkpoint transform must return a checkpoint."
                            )
                        transformed_checkpoint = _checkpoint_transform_result_preserving_completion_result_event_publications(
                            transformed_checkpoint,
                            result_checkpoint,
                            session_id=session_id,
                        )
                    if temporary_service_admission is not None:
                        from cayu.sessions._session_continuation import continuation_operation_key
                        from cayu.sessions._temporary_continuation import temporary_service_key
                        from cayu.sessions._temporary_continuation_store import (
                            compose_temporary_service_admission,
                        )

                        intent = temporary_service_admission.dispatch.intent
                        parent_key = continuation_operation_key(intent.ticket)
                        from cayu.sessions._temporary_service_target import target_service_key

                        child_key = (
                            temporary_service_key(intent.operation)
                            if intent.mode == "same_session"
                            else target_service_key(intent.operation)
                        )
                        await cur.execute(
                            "SELECT idempotency_key, record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key IN (%s, %s) FOR UPDATE",
                            (session_id, parent_key, child_key),
                        )
                        service_records = dict(await cur.fetchall())
                        publication = compose_temporary_service_admission(
                            source_session=loaded,
                            source_checkpoint=current_checkpoint,
                            parent_record=service_records.get(parent_key),
                            child_record=service_records.get(child_key),
                            admitted_session=transitioned,
                            admitted_checkpoint=transformed_checkpoint,
                            admission=temporary_service_admission,
                            now=updated_at,
                        )
                        transformed_checkpoint = publication.checkpoint
                        await cur.executemany(
                            "INSERT INTO cayu_session_operations "
                            "(session_id, idempotency_key, record, updated_at) VALUES (%s, %s, %s, %s) "
                            "ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                            "record = excluded.record, updated_at = excluded.updated_at",
                            [
                                (session_id, key, pg_support._dumps(record), updated_at)
                                for key, record in publication.operation_records.items()
                            ],
                        )
                    if transformed_checkpoint is not None:
                        await self._upsert_checkpoint(
                            cur, session_id, transformed_checkpoint, updated_at
                        )
                    if admission is not None:
                        _started_event, interaction_id, source_messages, defer_source = admission
                        await self._register_public_authorities(
                            cur,
                            session_id,
                            interaction_ids=(interaction_id,),
                        )
                        await cur.execute(
                            "SELECT interaction_id FROM cayu_deferred_interaction_inputs "
                            "WHERE session_id = %s FOR UPDATE",
                            (session_id,),
                        )
                        existing_deferred = await cur.fetchone()
                        if existing_deferred is not None and (
                            not defer_source or existing_deferred[0] != interaction_id
                        ):
                            raise RuntimeError("Session already has deferred interaction input.")
                        for event_offset, admission_event in enumerate(admission_events):
                            lookup_key, projection, projection_bytes = (
                                pending_action_event_storage_values(admission_event)
                            )
                            await cur.execute(
                                """
                                INSERT INTO cayu_events (
                                    session_id, session_order, event_id, interaction_id,
                                    event_type, timestamp, agent_name, environment_name,
                                    workflow_name, tool_name, payload, event,
                                    pending_action_lookup_key, pending_action_projection,
                                    pending_action_projection_bytes
                                ) VALUES (
                                    %s, %s, %s, %s, %s, %s, %s, %s,
                                    %s, %s, %s, %s, %s, %s, %s
                                )
                                """,
                                (
                                    session_id,
                                    order_row[0] - len(admission_events) + event_offset + 1,
                                    admission_event.id,
                                    admission_event.interaction_id,
                                    str(admission_event.type),
                                    pg_support.to_utc(admission_event.timestamp),
                                    admission_event.agent_name,
                                    admission_event.environment_name,
                                    admission_event.workflow_name,
                                    admission_event.tool_name,
                                    pg_support._dumps(admission_event.payload),
                                    pg_support._dumps(admission_event.model_dump(mode="json")),
                                    lookup_key,
                                    projection,
                                    projection_bytes,
                                ),
                            )
                        if admission_events:
                            await self._enqueue_persisted_event_side_effects(
                                cur, session_id, admission_events
                            )
                        if defer_source:
                            deferred_input = DeferredInteractionInput(
                                interaction_id=interaction_id,
                                source_messages=source_messages,
                            )
                            await cur.execute(
                                "INSERT INTO cayu_deferred_interaction_inputs "
                                "(session_id, interaction_id, source_messages) "
                                "VALUES (%s, %s, %s) "
                                "ON CONFLICT(session_id) DO UPDATE SET "
                                "interaction_id = EXCLUDED.interaction_id, "
                                "source_messages = EXCLUDED.source_messages",
                                (
                                    session_id,
                                    interaction_id,
                                    pg_support._dumps(
                                        deferred_interaction_input_storage_payload(deferred_input)
                                    ),
                                ),
                            )
                        else:
                            await cur.executemany(
                                "INSERT INTO cayu_transcript_messages "
                                "(session_id, interaction_id, message, "
                                "transcript_search_document) VALUES (%s, %s, %s, %s)",
                                [
                                    (
                                        session_id,
                                        interaction_id,
                                        pg_support._dumps(message.model_dump(mode="json")),
                                        _postgres_transcript_index_document(session_id, message),
                                    )
                                    for message in source_messages
                                ],
                            )
                await conn.commit()
            except UniqueViolation as exc:
                await conn.rollback()
                existing_event_id = None
                if admission is not None:
                    existing_event_id = await self._first_existing_event_id(
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
                if existing_event_id is not None:
                    raise ValueError(
                        f"Event already exists for session {session_id}: {existing_event_id}"
                    ) from exc
                raise
            except Exception:
                await conn.rollback()
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
        from cayu.sessions.pending_actions import pending_action_event_storage_values

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
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await self._lock_closure_lineage(cur)
                    session = await self._load_for_update(cur, session_id)
                    if session is None:
                        raise KeyError(f"Session not found: {session_id}")
                    _validate_execution_profile_rejection_session(
                        session,
                        checkpoint=await self._load_checkpoint(cur, session_id),
                        expected_session_instance_id=expected_session_instance_id,
                        expected_statuses=statuses,
                        expected_run_epoch=expected_run_epoch,
                        expected_profile=expected_profile,
                        event=copied_event,
                        expected_active_invocation_profile_authority=(
                            expected_active_invocation_profile_authority
                        ),
                    )
                    await cur.execute(
                        "SELECT event FROM cayu_events WHERE session_id = %s AND event_id = %s",
                        (session_id, copied_event.id),
                    )
                    existing_row = await cur.fetchone()
                    if existing_row is not None:
                        existing = restore_persisted_event_authority(
                            Event.model_validate(existing_row[0])
                        )
                        if not _execution_profile_rejection_events_equivalent(
                            existing,
                            copied_event,
                        ):
                            raise ValueError(
                                f"Execution-profile rejection id was reused: {copied_event.id}"
                            )
                        await conn.commit()
                        return ExecutionProfileRejectionResult(event=existing, replayed=True)

                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
                    activity_at = await self._session_store_now(cur)
                    await cur.execute(
                        "UPDATE cayu_sessions "
                        "SET event_seq = event_seq + 1, last_activity_at = %s "
                        "WHERE id = %s RETURNING event_seq",
                        (activity_at, session_id),
                    )
                    order_row = await cur.fetchone()
                    if order_row is None:
                        raise KeyError(f"Session not found: {session_id}")
                    await self._register_event_public_authorities(
                        cur,
                        session_id,
                        [copied_event],
                    )
                    lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                        copied_event
                    )
                    await cur.execute(
                        """
                        INSERT INTO cayu_events (
                            session_id, session_order, event_id, interaction_id,
                            event_type, timestamp, agent_name, environment_name,
                            workflow_name, tool_name, payload, event,
                            pending_action_lookup_key, pending_action_projection,
                            pending_action_projection_bytes
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s
                        )
                        """,
                        (
                            session_id,
                            order_row[0],
                            copied_event.id,
                            copied_event.interaction_id,
                            str(copied_event.type),
                            pg_support.to_utc(copied_event.timestamp),
                            copied_event.agent_name,
                            copied_event.environment_name,
                            copied_event.workflow_name,
                            copied_event.tool_name,
                            pg_support._dumps(copied_event.payload),
                            pg_support._dumps(copied_event.model_dump(mode="json")),
                            lookup_key,
                            projection,
                            projection_bytes,
                        ),
                    )
                    await self._enqueue_persisted_event_side_effects(
                        cur,
                        session_id,
                        [copied_event],
                    )
                await conn.commit()
                return ExecutionProfileRejectionResult(event=copied_event, replayed=False)
            except Exception:
                await conn.rollback()
                raise

    async def transition_status_if_no_queued_messages(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_mutation: dict[str, Any] | None = None,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        mutation = _prepare_queue_completion_checkpoint_mutation(checkpoint_mutation, to_status)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded = await self._load_for_update(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    _assert_session_run_epoch(session_id, loaded)
                    updated_at = await self._session_store_now(cur)
                    if loaded.status not in allowed_statuses:
                        raise SessionStatusConflict(
                            f"Session status transition not allowed: {loaded.status} -> {to_status}"
                        )
                    await cur.execute(
                        "SELECT 1 FROM cayu_session_message_queue "
                        "WHERE session_id = %s AND status = 'queued' LIMIT 1",
                        (session_id,),
                    )
                    if await cur.fetchone() is not None:
                        raise SessionQueuedMessagesPending(
                            f"Session has durable queued messages: {session_id}"
                        )
                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
                    if mutation is not None:
                        checkpoint = _apply_queue_completion_checkpoint_mutation(
                            loaded, mutation, await self._load_checkpoint(cur, session_id)
                        )
                        if checkpoint is None:
                            raise ValueError(
                                "Queue completion mutation cannot delete its checkpoint."
                            )
                        await self._upsert_checkpoint(cur, session_id, checkpoint, updated_at)
                    await cur.execute(
                        "UPDATE cayu_sessions SET status = %s, updated_at = %s, "
                        "last_activity_at = %s, run_epoch = run_epoch + %s WHERE id = %s",
                        (
                            str(to_status),
                            updated_at,
                            updated_at,
                            1 if to_status == SessionStatus.RUNNING and mutation is None else 0,
                            session_id,
                        ),
                    )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
            transitioned = loaded.model_copy(
                update={
                    "status": to_status,
                    "updated_at": updated_at,
                    "last_activity_at": updated_at,
                    "run_epoch": loaded.run_epoch
                    + (to_status == SessionStatus.RUNNING and mutation is None),
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
        terminal_event: Event | None = None,
        terminal_decision: InvocationTerminalDecision | None = None,
        expected_session_instance_id: str | None = None,
        expected_active_invocation_profile: ActiveInvocationExecutionProfile | None = None,
        expected_invocation_authority_state: Literal["active", "released"] = "active",
        expected_recovery_claim_id: str | None = None,
        terminalization_only: bool = False,
        terminalization_plan_ownership: Any = None,
    ) -> InteractionTransitionResult:
        from cayu.sessions.pending_actions import pending_action_event_storage_values

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
            terminal_event=terminal_event,
            terminal_decision=terminal_decision,
        )
        copied_event = transition.event
        allowed_statuses = set(transition.from_statuses)
        target_status = transition.to_status
        conditional = transition.only_if_no_queued_messages
        settlement_request = transition.model_completion_stage_settlement
        checkpoint_mutation_request = transition.checkpoint_mutation
        copied_terminal_event = transition.terminal_event
        terminal_decision = transition.terminal_decision
        receipt_storage_key = _interaction_transition_storage_key(copied_event.id)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded = await self._load_for_update(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    if expected_active_invocation_profile is None:
                        _assert_session_run_epoch(session_id, loaded)
                    await cur.execute(
                        "SELECT record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s",
                        (session_id, receipt_storage_key),
                    )
                    receipt_row = await cur.fetchone()
                    await cur.execute(
                        "SELECT event FROM cayu_events WHERE session_id = %s AND event_id = %s",
                        (session_id, copied_event.id),
                    )
                    existing_row = await cur.fetchone()
                    existing_terminal_row = None
                    if copied_terminal_event is not None:
                        await cur.execute(
                            "SELECT event FROM cayu_events WHERE session_id = %s AND event_id = %s",
                            (session_id, copied_terminal_event.id),
                        )
                        existing_terminal_row = await cur.fetchone()
                    if receipt_row is not None:
                        receipt = _reconstruct_interaction_transition_receipt(
                            pg_support._json_obj(receipt_row[0]),
                            transition=transition,
                        )
                        _validate_interaction_transition_receipt_authority(
                            receipt,
                            current_session=loaded,
                            current_checkpoint=await self._load_checkpoint(cur, session_id),
                            expected_session_instance_id=expected_session_instance_id,
                            expected_active_invocation_profile=expected_active_invocation_profile,
                            expected_invocation_authority_state=(
                                expected_invocation_authority_state
                            ),
                            expected_recovery_claim_id=expected_recovery_claim_id,
                        )
                        if (
                            existing_row is not None
                            and Event(**pg_support._json_obj(existing_row[0])) != receipt.event
                        ):
                            raise RuntimeError(
                                "Interaction transition receipt conflicts with retained event history."
                            )
                        if copied_terminal_event is not None and (
                            receipt.terminal_event != copied_terminal_event
                            or (
                                existing_terminal_row is not None
                                and Event(**pg_support._json_obj(existing_terminal_row[0]))
                                != receipt.terminal_event
                            )
                        ):
                            raise RuntimeError(
                                "Interaction transition receipt conflicts with its terminal session event."
                            )
                        await conn.commit()
                        return InteractionTransitionResult(
                            session=receipt.session,
                            event=receipt.event,
                            terminal_event=receipt.terminal_event,
                            status_changed=receipt.status_changed,
                            replayed=True,
                        )
                    if existing_row is not None:
                        raise RuntimeError(
                            "Interaction transition event exists without its immutable receipt."
                        )
                    if existing_terminal_row is not None:
                        raise RuntimeError(
                            "Terminal session event exists without its interaction receipt."
                        )
                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
                    checkpoint = await self._load_checkpoint(cur, session_id)
                    if terminalization_only:
                        from cayu.runtime._durable_model_terminalization import (
                            require_terminalization_checkpoint,
                            require_terminalization_plan_owner,
                        )

                        require_terminalization_checkpoint(loaded, checkpoint)

                        require_terminalization_plan_owner(
                            checkpoint,
                            terminalization_plan_ownership,
                            await self._session_store_now(cur),
                        )
                        for table, column in (
                            ("cayu_sessions", "parent_session_id"),
                            ("cayu_deferred_interaction_inputs", "session_id"),
                            ("cayu_session_message_queue", "session_id"),
                        ):
                            await cur.execute(
                                f"SELECT 1 FROM {table} WHERE {column} = %s "
                                + (
                                    "AND status = 'queued' "
                                    if table == "cayu_session_message_queue"
                                    else ""
                                )
                                + "LIMIT 1",
                                (session_id,),
                            )
                            if await cur.fetchone() is not None:
                                raise SessionRunFenced("Model terminalization has dependent work.")
                    settled_checkpoint = _checkpoint_after_exact_invocation_terminal_decision(
                        checkpoint,
                        session=loaded,
                        expected=terminal_decision,
                    )
                    active_recovery_claim_id = _active_unexpired_incomplete_recovery_claim_id(
                        checkpoint,
                        now=await self._session_store_now(cur),
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
                        from cayu.sessions._invocation_lifecycle import (
                            require_invocation_command_authority,
                            require_released_invocation_command_authority,
                        )

                        assert expected_session_instance_id is not None
                        if expected_invocation_authority_state == "released":
                            require_released_invocation_command_authority(
                                loaded,
                                checkpoint,
                                session_id=session_id,
                                session_instance_id=expected_session_instance_id,
                                active_profile=expected_active_invocation_profile,
                                events=tuple(
                                    event
                                    for event in (copied_event, copied_terminal_event)
                                    if event is not None
                                ),
                            )
                        else:
                            require_invocation_command_authority(
                                loaded,
                                checkpoint,
                                session_id=session_id,
                                session_instance_id=expected_session_instance_id,
                                run_epochs=frozenset(
                                    {expected_active_invocation_profile.run_epoch}
                                ),
                                active_profile=expected_active_invocation_profile,
                                events=tuple(
                                    event
                                    for event in (copied_event, copied_terminal_event)
                                    if event is not None
                                ),
                            )
                    if loaded.status not in allowed_statuses:
                        raise SessionStatusConflict(
                            "Session status transition not allowed: "
                            f"{loaded.status} -> {target_status}"
                        )
                    queued = False
                    if conditional:
                        await cur.execute(
                            "SELECT 1 FROM cayu_session_message_queue "
                            "WHERE session_id = %s AND status = 'queued' LIMIT 1",
                            (session_id,),
                        )
                        queued = await cur.fetchone() is not None
                    from cayu.runtime._session_steering import (
                        interaction_completion_steering_key,
                        prepare_interaction_completion_steering_record,
                    )

                    steering_key = interaction_completion_steering_key(
                        loaded, checkpoint, copied_event
                    )
                    completion_record = None
                    if steering_key is not None:
                        await cur.execute(
                            "SELECT record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = %s",
                            (session_id, steering_key),
                        )
                        steering_row = await cur.fetchone()
                        completion_record = prepare_interaction_completion_steering_record(
                            loaded,
                            checkpoint,
                            copied_event,
                            None if steering_row is None else pg_support._json_obj(steering_row[0]),
                            keeps_running=queued or target_status is SessionStatus.RUNNING,
                        )
                    updated_at = await self._session_store_now(cur)
                    settlement_record = None
                    settlement_storage_key = None
                    if settlement_request is not None:
                        await cur.execute(
                            "SELECT record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = %s",
                            (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                        )
                        active_row = await cur.fetchone()
                        if active_row is None:
                            raise SessionModelCompletionStageConflict(
                                "The interaction transition has no active model-completion "
                                "stage to settle."
                            )
                        active_record = _decode_model_completion_stage_record(active_row[0])
                        marker = _reconstruct_active_model_completion_stage_record(
                            active_record,
                            session_id=session_id,
                        )
                        _, _, preparation_key, terminal_key = (
                            _model_completion_stage_storage_identity(
                                session_id,
                                marker.stage_id,
                            )
                        )
                        await cur.execute(
                            "SELECT idempotency_key, record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                            (session_id, [preparation_key, terminal_key]),
                        )
                        stage_records = {
                            row[0]: _decode_model_completion_stage_record(row[1])
                            for row in await cur.fetchall()
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
                        related_keys = [
                            settlement_storage_key,
                            _model_completion_stage_winner_storage_key(stage.logical_step_id),
                            _runtime_publication_storage_key(stage.logical_step_id),
                        ]
                        await cur.execute(
                            "SELECT idempotency_key, record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                            (session_id, related_keys),
                        )
                        related_records = {
                            row[0]: _decode_model_completion_stage_record(row[1])
                            for row in await cur.fetchall()
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
                    committed_events = [
                        event
                        for event in (copied_event, copied_terminal_event)
                        if event is not None
                    ]
                    await cur.execute(
                        """
                        UPDATE cayu_sessions
                        SET status = CASE WHEN %s THEN status ELSE %s END,
                            updated_at = CASE WHEN %s THEN updated_at ELSE %s END,
                            last_activity_at = %s,
                            event_seq = event_seq + %s
                        WHERE id = %s
                        RETURNING event_seq
                        """,
                        (
                            queued,
                            str(target_status),
                            queued,
                            updated_at,
                            updated_at,
                            len(committed_events),
                            session_id,
                        ),
                    )
                    order_row = await cur.fetchone()
                    if order_row is None:
                        raise KeyError(f"Session not found: {session_id}")
                    if not queued and checkpoint_mutation_request is not None:
                        transformed_checkpoint = _apply_runtime_publication_checkpoint_mutation(
                            RuntimePublicationMutation.model_validate(checkpoint_mutation_request),
                            await self._load_checkpoint(cur, session_id),
                        )
                        if transformed_checkpoint is None:
                            raise AssertionError(
                                "Interaction checkpoint mutation deleted its checkpoint."
                            )
                        await self._upsert_checkpoint(
                            cur,
                            session_id,
                            transformed_checkpoint,
                            updated_at,
                        )
                    if settlement_record is not None and settlement_storage_key is not None:
                        await cur.execute(
                            "INSERT INTO cayu_session_operations "
                            "(session_id, idempotency_key, record, updated_at) "
                            "VALUES (%s, %s, %s, %s)",
                            (
                                session_id,
                                settlement_storage_key,
                                pg_support._dumps(settlement_record),
                                updated_at,
                            ),
                        )
                        await cur.execute(
                            "DELETE FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = %s",
                            (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                        )
                        if cur.rowcount != 1:
                            raise SessionModelCompletionStageConflict(
                                "The active model-completion stage changed during settlement."
                            )
                    await self._register_event_public_authorities(
                        cur,
                        session_id,
                        committed_events,
                    )
                    first_session_order = order_row[0] - len(committed_events) + 1
                    for event_offset, committed_event in enumerate(committed_events):
                        lookup_key, projection, projection_bytes = (
                            pending_action_event_storage_values(committed_event)
                        )
                        await cur.execute(
                            """
                        INSERT INTO cayu_events (
                            session_id, session_order, event_id, interaction_id,
                            event_type, timestamp, agent_name, environment_name,
                            workflow_name, tool_name, payload, event,
                            pending_action_lookup_key, pending_action_projection,
                            pending_action_projection_bytes
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s
                        )
                            """,
                            (
                                session_id,
                                first_session_order + event_offset,
                                committed_event.id,
                                committed_event.interaction_id,
                                str(committed_event.type),
                                pg_support.to_utc(committed_event.timestamp),
                                committed_event.agent_name,
                                committed_event.environment_name,
                                committed_event.workflow_name,
                                committed_event.tool_name,
                                pg_support._dumps(committed_event.payload),
                                pg_support._dumps(committed_event.model_dump(mode="json")),
                                lookup_key,
                                projection,
                                projection_bytes,
                            ),
                        )
                    await self._record_invocation_terminal_event_receipts(
                        cur, session_id, committed_events, activity_at=updated_at
                    )
                    await self._enqueue_persisted_event_side_effects(
                        cur,
                        session_id,
                        committed_events,
                    )
                    if terminal_decision is not None:
                        assert settled_checkpoint is not None
                        await self._upsert_checkpoint(
                            cur,
                            session_id,
                            settled_checkpoint,
                            updated_at,
                        )
                    transitioned = await self._load(cur, session_id)
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
                        terminal_event=copied_terminal_event,
                        terminal_decision=terminal_decision,
                        status_changed=not queued,
                        invocation_session_instance_id=expected_session_instance_id,
                        invocation_active_profile=expected_active_invocation_profile,
                        invocation_authority_state=expected_invocation_authority_state,
                        recovery_claim_id=expected_recovery_claim_id,
                    )
                    await cur.execute(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record, updated_at) "
                        "VALUES (%s, %s, %s, %s)",
                        (
                            session_id,
                            receipt_storage_key,
                            pg_support._dumps(receipt_record),
                            updated_at,
                        ),
                    )
                    if completion_record is not None:
                        assert steering_key is not None
                        await cur.execute(
                            "INSERT INTO cayu_session_operations "
                            "(session_id, idempotency_key, record, updated_at) "
                            "VALUES (%s, %s, %s, %s)",
                            (
                                session_id,
                                steering_key,
                                pg_support._dumps(completion_record),
                                updated_at,
                            ),
                        )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return InteractionTransitionResult(
            session=transitioned,
            event=copied_event,
            terminal_event=copied_terminal_event,
            status_changed=not queued,
        )

    async def settle_session_invocation(self, command: Any) -> InteractionTransitionResult:
        from cayu.sessions._invocation_lifecycle import (
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
        if transition.terminal_event is not None:
            kwargs["terminal_event"] = transition.terminal_event
            kwargs["terminal_decision"] = transition.terminal_decision
        if copied.recovery_claim_id is not None:
            kwargs["expected_recovery_claim_id"] = copied.recovery_claim_id
        if copied.terminalization_only:
            kwargs["terminalization_only"] = True
            kwargs["terminalization_plan_ownership"] = copied.terminalization_plan_ownership
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
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT operation.record, retained.event "
                "FROM cayu_sessions AS session "
                "LEFT JOIN cayu_session_operations AS operation "
                "ON operation.session_id = session.id AND operation.idempotency_key = %s "
                "LEFT JOIN cayu_events AS retained "
                "ON retained.session_id = session.id AND retained.event_id = %s "
                "WHERE session.id = %s",
                (receipt_storage_key, copied_event.id, session_id),
            )
            row = await cur.fetchone()
            if row is None:
                raise KeyError(f"Session not found: {session_id}")
            receipt_record, existing_record = row
            if receipt_record is None:
                if existing_record is not None:
                    raise RuntimeError(
                        "Interaction transition event exists without its immutable receipt."
                    )
                return None
            receipt = _reconstruct_interaction_transition_receipt(
                pg_support._json_obj(receipt_record),
                transition=copied_transition,
            )
            _validate_interaction_transition_receipt_recovery_authority(
                receipt,
                current_checkpoint=await self._load_checkpoint(cur, session_id),
                expected_recovery_claim_id=expected_recovery_claim_id,
            )
            if (
                existing_record is not None
                and Event(**pg_support._json_obj(existing_record)) != receipt.event
            ):
                raise RuntimeError(
                    "Interaction transition receipt conflicts with retained event history."
                )
            return InteractionTransitionReceiptResult(
                session=receipt.session,
                transition=_interaction_transition_spec_from_receipt(receipt),
                status_changed=receipt.status_changed,
            )

    async def _load_historical_interaction_settlement_record(
        self, session_id: str, event_id: str
    ) -> dict[str, Any] | None:
        key = _interaction_transition_storage_key(event_id)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT record FROM cayu_session_operations "
                "WHERE session_id = %s AND idempotency_key = %s",
                (session_id, key),
            )
            row = await cur.fetchone()
            return None if row is None else pg_support._json_obj(row[0])

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
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT operation.record, retained.event "
                "FROM cayu_sessions AS session "
                "LEFT JOIN cayu_session_operations AS operation "
                "ON operation.session_id = session.id AND operation.idempotency_key = %s "
                "LEFT JOIN cayu_events AS retained "
                "ON retained.session_id = session.id AND retained.event_id = %s "
                "WHERE session.id = %s",
                (receipt_storage_key, event_id, session_id),
            )
            row = await cur.fetchone()
            if row is None:
                raise KeyError(f"Session not found: {session_id}")
            receipt_record, existing_record = row
            if receipt_record is None:
                if existing_record is not None:
                    raise RuntimeError(
                        "Interaction transition event exists without its immutable receipt."
                    )
                return None
            receipt = _load_interaction_transition_receipt(pg_support._json_obj(receipt_record))
            current_session = await self._load(cur, session_id)
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
            if (
                existing_record is not None
                and Event(**pg_support._json_obj(existing_record)) != receipt.event
            ):
                raise RuntimeError(
                    "Interaction transition receipt conflicts with retained event history."
                )
            return InteractionTransitionReceiptResult(
                session=receipt.session,
                transition=_interaction_transition_spec_from_receipt(receipt),
                status_changed=receipt.status_changed,
            )

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
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded = await self._load_for_update(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    checkpoint = await self._load_checkpoint(cur, session_id)
                    now = await self._session_store_now(cur)
                    if await self._has_live_execution_owner(cur, loaded, now):
                        await conn.commit()
                        return None
                    if (
                        active_provider_operation_cancellation_claim_from_checkpoint(
                            checkpoint,
                            now=now,
                        )
                        is not None
                        or _incomplete_recovery_claim_from_checkpoint(checkpoint) is not None
                    ):
                        await conn.commit()
                        return None
                    inactive_before = utc_duration_cutoff(
                        now,
                        validated_inactive_for_seconds,
                    )
                    if (
                        inactive_before is None
                        or loaded.status not in allowed_statuses
                        or loaded.last_activity_at > inactive_before
                    ):
                        await conn.commit()
                        return None
                    await cur.execute(
                        "UPDATE cayu_sessions SET run_epoch = run_epoch + 1, "
                        "last_activity_at = %s WHERE id = %s",
                        (now, session_id),
                    )
                    loaded = await self._load(cur, session_id)
                    if loaded is None:  # pragma: no cover - row remains locked
                        raise KeyError(f"Session not found: {session_id}")
                    await conn.commit()
            except Exception:
                await conn.rollback()
                raise
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
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded = await self._load_for_update(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    current = await self._load_checkpoint(cur, session_id)
                    now = await self._session_store_now(cur)
                    if inactive_for_seconds is not None and await self._has_live_execution_owner(
                        cur, loaded, now
                    ):
                        await conn.commit()
                        return None
                    inactive_before = (
                        None
                        if inactive_for_seconds is None
                        else utc_duration_cutoff(now, inactive_for_seconds)
                    )
                    if (
                        loaded.status not in allowed_statuses
                        or (
                            inactive_for_seconds is not None
                            and (
                                inactive_before is None or loaded.last_activity_at > inactive_before
                            )
                        )
                        or active_provider_operation_cancellation_claim_from_checkpoint(
                            current,
                            now=now,
                        )
                        is not None
                    ):
                        await conn.commit()
                        return None
                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
                    transformed = checkpoint_transform(
                        loaded,
                        _copy_checkpoint_for_transform(
                            current,
                            session_id=session_id,
                        ),
                        now,
                    )
                    if transformed is None:
                        await conn.commit()
                        return None
                    transformed = _checkpoint_transform_result_preserving_completion_result_event_publications(
                        current,
                        transformed,
                        session_id=session_id,
                    )
                    await self._upsert_checkpoint(cur, session_id, transformed, now)
                await conn.commit()
                return loaded
            except BaseException:
                await conn.rollback()
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
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded = await self._load_for_update(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    updated_at = await self._session_store_now(cur)
                    if loaded.status not in allowed_statuses:
                        raise SessionStatusConflict(
                            f"Session status cannot be fenced: {loaded.status}"
                        )
                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
                    current_checkpoint = await self._load_checkpoint(cur, session_id)
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
                    transformed = _checkpoint_transform_result_preserving_completion_result_event_publications(
                        current_checkpoint,
                        transformed,
                        session_id=session_id,
                    )
                    await cur.execute(
                        "UPDATE cayu_sessions SET run_epoch = run_epoch + 1, "
                        "last_activity_at = %s WHERE id = %s",
                        (updated_at, session_id),
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
                            raise ValueError(
                                "Result checkpoint transform must return a checkpoint."
                            )
                        transformed = _checkpoint_transform_result_preserving_completion_result_event_publications(
                            transformed,
                            result_checkpoint,
                            session_id=session_id,
                        )
                    await self._upsert_checkpoint(cur, session_id, transformed, updated_at)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
            _activate_session_run_fence(fenced)
            return fenced

    async def release_run_fence(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        expected_run_epoch = _current_session_run_epoch(session_id)
        if expected_run_epoch is None:
            _deactivate_session_interaction(session_id)
            return
        await self._ensure_ready()
        try:
            async with self._connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "UPDATE cayu_sessions SET run_epoch = run_epoch + 1 "
                        "WHERE id = %s AND run_epoch = %s",
                        (session_id, expected_run_epoch),
                    )
                await conn.commit()
        finally:
            _deactivate_session_run_fence(session_id)
            _deactivate_session_interaction(session_id)

    async def release_session_invocation(self, command: Any) -> Any:
        from cayu.runtime._invocation_lifecycle import (
            checkpoint_with_invocation_lifecycle_receipt,
        )
        from cayu.sessions._invocation_lifecycle import (
            InvocationReleaseResult,
            ReleaseInvocationCommand,
            _invocation_lifecycle_receipt_ledger_from_checkpoint,
            copy_invocation_lifecycle_command,
            invocation_release_replay_from_state,
            require_invocation_command_authority,
            require_invocation_release_store_authority,
        )

        copied = copy_invocation_lifecycle_command(command)
        if type(copied) is not ReleaseInvocationCommand:
            raise TypeError("command must be a ReleaseInvocationCommand.")
        require_invocation_release_store_authority(copied)
        await self._ensure_ready()
        released = False
        try:
            async with self._connection() as conn:
                try:
                    async with conn.cursor() as cur:
                        session = await self._load_for_update(cur, copied.session_id)
                        if session is None:
                            raise KeyError(f"Session not found: {copied.session_id}")
                        checkpoint = await self._load_checkpoint(cur, copied.session_id)
                        ledger = _invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
                        replay = invocation_release_replay_from_state(
                            session,
                            checkpoint,
                            copied,
                            _ledger=ledger,
                        )
                        if replay is not None:
                            await conn.commit()
                            released = True
                            return replay
                        if copied.terminal_session_event is not None:
                            await cur.execute(
                                "SELECT event.event, operation.record "
                                "FROM cayu_events AS event "
                                "LEFT JOIN cayu_session_operations AS operation "
                                "ON operation.session_id = event.session_id "
                                "AND operation.idempotency_key = %s "
                                "WHERE event.session_id = %s AND event.event_id = %s",
                                (
                                    _invocation_terminal_event_storage_key(
                                        copied.terminal_session_event.id
                                    ),
                                    copied.session_id,
                                    copied.terminal_session_event.id,
                                ),
                            )
                            terminal_event_row = await cur.fetchone()
                            _require_invocation_release_terminal_session_event(
                                (
                                    None
                                    if terminal_event_row is None or terminal_event_row[1] is None
                                    else pg_support._json_obj(terminal_event_row[1])
                                ),
                                (
                                    None
                                    if terminal_event_row is None
                                    else restore_persisted_event_authority(
                                        Event.model_validate(terminal_event_row[0])
                                    )
                                ),
                                current_session=session,
                                expected_event=copied.terminal_session_event,
                                expected_session_instance_id=(copied.expected_session_instance_id),
                                expected_active_invocation_profile=(copied.expected_active_profile),
                            )
                        elif copied.settlement_transition is None:
                            assert copied.recovery_claim_id is not None
                            _require_invocation_release_recovery_claim(
                                checkpoint,
                                current_session=session,
                                recovery_claim_id=copied.recovery_claim_id,
                            )
                        else:
                            await cur.execute(
                                "SELECT operation.record "
                                "FROM cayu_session_operations AS operation "
                                "WHERE operation.session_id = %s "
                                "AND operation.idempotency_key = %s",
                                (
                                    copied.session_id,
                                    _interaction_transition_storage_key(
                                        copied.settlement_transition.event.id
                                    ),
                                ),
                            )
                            settlement_row = await cur.fetchone()
                            if settlement_row is None:
                                raise SessionRunFenced(
                                    "Invocation release lacks exact durable terminal settlement."
                                )
                            _require_invocation_release_settlement_record(
                                settlement_row[0],
                                current_session=session,
                                transition=copied.settlement_transition,
                                expected_session_instance_id=(copied.expected_session_instance_id),
                                expected_active_invocation_profile=(copied.expected_active_profile),
                            )
                        require_invocation_command_authority(
                            session,
                            checkpoint,
                            session_id=copied.session_id,
                            session_instance_id=copied.expected_session_instance_id,
                            run_epochs=frozenset({copied.expected_run_epoch}),
                            active_profile=copied.expected_active_profile,
                        )
                        await cur.execute(
                            "UPDATE cayu_sessions SET run_epoch = run_epoch + 1 "
                            "WHERE id = %s AND run_epoch = %s",
                            (copied.session_id, copied.expected_run_epoch),
                        )
                        if cur.rowcount != 1:
                            raise SessionRunFenced("Invocation release lost its exact run epoch.")
                        session = session.model_copy(
                            update={"run_epoch": copied.expected_run_epoch + 1}
                        )
                        updated_checkpoint = checkpoint_with_invocation_lifecycle_receipt(
                            checkpoint,
                            copied,
                            active_profile=copied.expected_active_profile,
                            result_session=session,
                            _ledger=ledger,
                        )
                        await self._upsert_checkpoint(
                            cur,
                            copied.session_id,
                            updated_checkpoint,
                            session.updated_at,
                        )
                    await conn.commit()
                except BaseException:
                    await conn.rollback()
                    raise
            result = InvocationReleaseResult(
                session=session,
                active_profile=copied.expected_active_profile,
                replayed=False,
            )
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
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    # Serialize the claim with run-epoch takeover and session
                    # deletion until this transaction commits.
                    await cur.execute(
                        "SELECT run_epoch FROM cayu_sessions WHERE id = %s FOR SHARE",
                        (publication_session_id,),
                    )
                    session_row = await cur.fetchone()
                    if session_row is None:
                        raise KeyError(f"Session not found: {publication_session_id}")
                    if expected_run_epoch is not None and session_row[0] != expected_run_epoch:
                        await _raise_session_write_conflict(
                            cur,
                            publication_session_id,
                            expected_run_epoch,
                        )
                    await cur.execute(
                        "SELECT 1 FROM cayu_budget_reservation_identities WHERE reservation_id = %s",
                        (reservation_id,),
                    )
                    if await cur.fetchone() is None:
                        for owner in await self._closure_lineage_owners(
                            cur, (publication_session_id,)
                        ):
                            _check_closure_lineage_owner(owner, (publication_session_id,))
                    await cur.execute(
                        """
                        INSERT INTO cayu_budget_reservation_identities (
                            reservation_id,
                            publication_session_id,
                            publication_id,
                            published
                        )
                        VALUES (%s, %s, %s, FALSE)
                        ON CONFLICT (reservation_id) DO NOTHING
                        """,
                        (reservation_id, publication_session_id, publication_id),
                    )
                    if cur.rowcount == 0:
                        await cur.execute(
                            """
                            SELECT publication_session_id, publication_id
                            FROM cayu_budget_reservation_identities
                            WHERE reservation_id = %s
                            """,
                            (reservation_id,),
                        )
                        existing = await cur.fetchone()
                        if existing is None or existing != (
                            publication_session_id,
                            publication_id,
                        ):
                            raise BudgetReservationIdentityConflict(
                                "Budget ledger reused a reservation identity."
                            )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    @staticmethod
    async def _publish_budget_reservation_identities(
        cur: Any,
        events: list[Event],
    ) -> None:
        for event in events:
            if event.type != EventType.BUDGET_RESERVED:
                continue
            raw_reservation_id = event.payload.get("reservation_id")
            if type(raw_reservation_id) is not str:
                continue
            await cur.execute(
                """
                UPDATE cayu_budget_reservation_identities
                SET published = TRUE
                WHERE reservation_id = %s
                  AND publication_session_id = %s
                  AND publication_id = %s
                  AND NOT published
                """,
                (raw_reservation_id, event.session_id, event.id),
            )
            if cur.rowcount == 1:
                continue
            await cur.execute(
                """
                INSERT INTO cayu_budget_reservation_identities (
                    reservation_id,
                    publication_session_id,
                    publication_id,
                    published
                )
                VALUES (%s, %s, %s, TRUE)
                ON CONFLICT (reservation_id) DO NOTHING
                """,
                (raw_reservation_id, event.session_id, event.id),
            )
            if cur.rowcount == 0:
                await cur.execute(
                    """
                    SELECT publication_session_id, publication_id, published
                    FROM cayu_budget_reservation_identities
                    WHERE reservation_id = %s
                    """,
                    (raw_reservation_id,),
                )
                existing = await cur.fetchone()
                if existing == (event.session_id, event.id, True):
                    await cur.execute(
                        """
                        SELECT 1
                        FROM cayu_events
                        WHERE session_id = %s AND event_id = %s
                        """,
                        (event.session_id, event.id),
                    )
                    if await cur.fetchone() is not None:
                        # The reservation belongs to this exact persisted event.
                        # Let the event insert below classify the replay as a
                        # duplicate event.
                        continue
                raise BudgetReservationIdentityConflict(
                    "Budget ledger reused a reservation identity."
                )

    async def _append_events_with_cursor(
        self,
        cur: Any,
        session_id: str,
        events: Sequence[Event],
        *,
        expected_run_epoch: int | None,
    ) -> None:
        """Append events and their delivery outbox rows in the caller's transaction."""

        # Serialize with every competing session writer before sampling the
        # liveness timestamp. Sampling database time before this row lock can
        # backdate activity by the duration of a blocked write and let recovery
        # steal a run earlier than the configured inactivity duration.
        await cur.execute(
            "SELECT 1 FROM cayu_sessions WHERE id = %s FOR UPDATE",
            (session_id,),
        )
        if await cur.fetchone() is None:
            raise KeyError(f"Session not found: {session_id}")
        activity_at = await self._session_store_now(cur)
        if expected_run_epoch is None:
            await cur.execute(
                "UPDATE cayu_sessions SET event_seq = event_seq + %s, "
                "last_activity_at = %s WHERE id = %s RETURNING event_seq",
                (len(events), activity_at, session_id),
            )
        else:
            await cur.execute(
                "UPDATE cayu_sessions SET event_seq = event_seq + %s, "
                "last_activity_at = %s WHERE id = %s AND run_epoch = %s "
                "RETURNING event_seq",
                (len(events), activity_at, session_id, expected_run_epoch),
            )
        order_row = await cur.fetchone()
        if order_row is None:
            if expected_run_epoch is not None:
                await _raise_session_write_conflict(cur, session_id, expected_run_epoch)
            raise KeyError(f"Session not found: {session_id}")
        if not events:
            return

        await self._insert_event_rows_with_cursor(
            cur, session_id, events, next_order=order_row[0] - len(events), activity_at=activity_at
        )

    async def _insert_event_rows_with_cursor(
        self,
        cur: Any,
        session_id: str,
        events: Sequence[Event],
        *,
        next_order: int,
        activity_at: datetime,
    ) -> None:
        """Insert prepared events after their transaction owner assigns order."""
        from cayu.sessions.pending_actions import pending_action_event_storage_values

        copied_events = list(events)
        await self._register_event_public_authorities(cur, session_id, copied_events)
        await self._publish_budget_reservation_identities(cur, copied_events)
        rows = []
        for event in copied_events:
            next_order += 1
            lookup_key, projection, projection_bytes = pending_action_event_storage_values(event)
            rows.append(
                (
                    session_id,
                    next_order,
                    event.id,
                    event.interaction_id,
                    str(event.type),
                    pg_support.to_utc(event.timestamp),
                    event.agent_name,
                    event.environment_name,
                    event.workflow_name,
                    event.tool_name,
                    pg_support._dumps(event.payload),
                    pg_support._dumps(event.model_dump(mode="json")),
                    lookup_key,
                    projection,
                    projection_bytes,
                )
            )
        await cur.executemany(
            """
            INSERT INTO cayu_events (
                session_id, session_order, event_id, interaction_id,
                event_type, timestamp,
                agent_name, environment_name, workflow_name, tool_name,
                payload, event, pending_action_lookup_key,
                pending_action_projection, pending_action_projection_bytes
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
            """,
            rows,
        )
        await self._record_invocation_terminal_event_receipts(
            cur, session_id, copied_events, activity_at=activity_at
        )
        await self._enqueue_persisted_event_side_effects(cur, session_id, copied_events)

    async def _record_invocation_terminal_event_receipts(
        self,
        cur: Any,
        session_id: str,
        events: Sequence[Event],
        *,
        activity_at: datetime,
    ) -> None:
        """Persist release evidence for terminal events in the owning transaction."""

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
            session = await self._load_for_update(cur, session_id)
            if session is None:  # pragma: no cover - session update already authenticated it
                raise KeyError(f"Session not found: {session_id}")
            checkpoint = await self._load_checkpoint(cur, session_id)
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
            await cur.executemany(
                "INSERT INTO cayu_session_operations "
                "(session_id, idempotency_key, record, updated_at) "
                "VALUES (%s, %s, %s, %s)",
                [
                    (session_id, receipt_key, pg_support._dumps(receipt_record), activity_at)
                    for receipt_key, receipt_record in terminal_receipts
                ],
            )

    async def _append_event_once_with_cursor(
        self,
        cur: Any,
        event: Event,
        *,
        expected_run_epoch: int,
    ) -> Event:
        """Return existing exact evidence or append it in the caller's transaction."""

        await cur.execute(
            "SELECT event FROM cayu_events WHERE session_id = %s AND event_id = %s",
            (event.session_id, event.id),
        )
        row = await cur.fetchone()
        if row is not None:
            return Event(**pg_support._json_obj(row[0]))
        await self._append_events_with_cursor(
            cur,
            event.session_id,
            [event],
            expected_run_epoch=expected_run_epoch,
        )
        return event

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        session_id, copied_events = _copy_session_event_batch(session_id, events)

        await self._ensure_ready()
        expected_run_epoch = _current_session_run_epoch(session_id)
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded = await self._load_for_update(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    if not copied_events:
                        _assert_session_run_epoch(session_id, loaded)
                        return
                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
                    await self._append_events_with_cursor(
                        cur,
                        session_id,
                        copied_events,
                        expected_run_epoch=expected_run_epoch,
                    )
                await conn.commit()
            except UniqueViolation as exc:
                await conn.rollback()
                existing = await self._first_existing_event_id(
                    session_id, [event.id for event in copied_events]
                )
                if existing is not None:
                    raise ValueError(
                        f"Event already exists for session {session_id}: {existing}"
                    ) from exc
                if (
                    getattr(exc.diag, "constraint_name", None)
                    == "idx_cayu_events_budget_reservation_identity"
                ):
                    raise BudgetReservationIdentityConflict(
                        "Budget ledger reused a reservation identity."
                    ) from exc
                raise
            except Exception:
                await conn.rollback()
                raise

    async def append_tool_effect_conflict(self, request: object) -> Event:
        from cayu.runtime._tool_effect_conflicts import (
            copy_tool_effect_conflict_audit,
            reconcile_tool_effect_conflict_event,
        )

        audit = copy_tool_effect_conflict_audit(request)
        session_id = audit.executing.intent.session_id
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await self._lock_closure_lineage(cur)
                    session = await self._load_for_update(cur, session_id)
                    if session is None:
                        raise KeyError("Tool effect audit session is unavailable.")
                    await cur.execute(
                        "SELECT record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s",
                        (session_id, audit.storage_key),
                    )
                    row = await cur.fetchone()
                    current = None if row is None else pg_support._json_obj(row[0])
                    event = audit.prepare_event(
                        session, current, now=await self._session_store_now(cur)
                    )
                    await cur.execute(
                        "SELECT event FROM cayu_events WHERE session_id = %s AND event_id = %s",
                        (session_id, event.id),
                    )
                    existing = await cur.fetchone()
                    if existing is not None:
                        event = reconcile_tool_effect_conflict_event(
                            event, Event(**pg_support._json_obj(existing[0]))
                        )
                    else:
                        for owner in await self._closure_lineage_owners(cur, (session_id,)):
                            _check_closure_lineage_owner(owner, (session_id,))
                        await cur.execute(
                            "UPDATE cayu_sessions SET event_seq = event_seq + 1 "
                            "WHERE id = %s RETURNING event_seq",
                            (session_id,),
                        )
                        order_row = await cur.fetchone()
                        if order_row is None:
                            raise KeyError("Tool effect audit session is unavailable.")
                        await self._insert_event_rows_with_cursor(
                            cur,
                            session_id,
                            [event],
                            next_order=order_row[0] - 1,
                            activity_at=event.timestamp,
                        )
                await conn.commit()
                return event
            except BaseException:
                await conn.rollback()
                raise

    async def append_workflow_step_started(
        self,
        session_id: str,
        event: Event,
        *,
        workflow_name: str,
        attempt_id: str,
    ) -> bool:
        from cayu.sessions.pending_actions import pending_action_event_storage_values

        session_id, copied_event, workflow_name, attempt_id = _copy_workflow_step_reservation(
            session_id,
            event,
            workflow_name=workflow_name,
            attempt_id=attempt_id,
        )
        await self._ensure_ready()
        expected_run_epoch = _current_session_run_epoch(session_id)
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await self._lock_closure_lineage(cur)
                    if await self._load_for_update(cur, session_id) is None:
                        raise KeyError(f"Session not found: {session_id}")
                    activity_at = await self._session_store_now(cur)
                    if expected_run_epoch is None:
                        await cur.execute(
                            """
                            UPDATE cayu_sessions
                            SET event_seq = event_seq + 1, last_activity_at = %s
                            WHERE id = %s
                            RETURNING event_seq
                            """,
                            (activity_at, session_id),
                        )
                    else:
                        await cur.execute(
                            """
                            UPDATE cayu_sessions
                            SET event_seq = event_seq + 1, last_activity_at = %s
                            WHERE id = %s AND run_epoch = %s
                            RETURNING event_seq
                            """,
                            (activity_at, session_id, expected_run_epoch),
                        )
                    order_row = await cur.fetchone()
                    if order_row is None:
                        if expected_run_epoch is not None:
                            await _raise_session_write_conflict(
                                cur,
                                session_id,
                                expected_run_epoch,
                            )
                        raise KeyError(f"Session not found: {session_id}")

                    await cur.execute(
                        """
                        SELECT event -> 'payload' ->> 'attempt_id'
                        FROM cayu_events
                        WHERE session_id = %s
                          AND workflow_name = %s
                          AND event_type = %s
                        ORDER BY sequence DESC
                        LIMIT 1
                        """,
                        (session_id, workflow_name, WORKFLOW_ATTEMPT_EVENT_TYPE),
                    )
                    latest_attempt = await cur.fetchone()
                    if latest_attempt is None or latest_attempt[0] != attempt_id:
                        await conn.rollback()
                        return False

                    await cur.execute(
                        "SELECT 1 FROM cayu_events WHERE session_id = %s AND event_id = %s",
                        (session_id, copied_event.id),
                    )
                    if await cur.fetchone() is not None:
                        await conn.rollback()
                        return False

                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
                    await self._register_event_public_authorities(
                        cur,
                        session_id,
                        [copied_event],
                    )
                    lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                        copied_event
                    )
                    await cur.execute(
                        """
                        INSERT INTO cayu_events (
                            session_id, session_order, event_id, interaction_id,
                            event_type, timestamp,
                            agent_name, environment_name, workflow_name, tool_name,
                            payload, event, pending_action_lookup_key,
                            pending_action_projection, pending_action_projection_bytes
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s
                        )
                        """,
                        (
                            session_id,
                            order_row[0],
                            copied_event.id,
                            copied_event.interaction_id,
                            str(copied_event.type),
                            pg_support.to_utc(copied_event.timestamp),
                            copied_event.agent_name,
                            copied_event.environment_name,
                            copied_event.workflow_name,
                            copied_event.tool_name,
                            pg_support._dumps(copied_event.payload),
                            pg_support._dumps(copied_event.model_dump(mode="json")),
                            lookup_key,
                            projection,
                            projection_bytes,
                        ),
                    )
                    await self._enqueue_persisted_event_side_effects(
                        cur,
                        session_id,
                        [copied_event],
                    )
                await conn.commit()
                return True
            except UniqueViolation as exc:
                await conn.rollback()
                existing = await self._first_existing_event_id(session_id, [copied_event.id])
                if existing is not None:
                    return False
                raise exc
            except Exception:
                await conn.rollback()
                raise

    async def load_mcp_manifest_baselines(
        self,
        history_keys: tuple[str, ...],
    ) -> McpManifestBaselineLoadResult:
        keys = _validate_mcp_manifest_history_keys(history_keys)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            rows = []
            if keys:
                await cur.execute(
                    "SELECT history_key, generation, baseline "
                    "FROM cayu_mcp_manifest_baselines "
                    "WHERE history_key = ANY(%s)",
                    (list(keys),),
                )
                rows = await cur.fetchall()
        return McpManifestBaselineLoadResult(
            baselines={
                row[0]: _stored_mcp_manifest_baseline(row[0], row[1], row[2]) for row in rows
            },
        )

    async def compare_and_publish_mcp_manifest_checks(
        self,
        session_id: str,
        *,
        expected_generations: dict[str, int | None],
        baseline_updates: dict[str, McpManifestBaseline],
        events: list[Event],
    ) -> McpManifestPublicationResult:
        from cayu.sessions.pending_actions import pending_action_event_storage_values

        session_id, expected, updates, copied_events = _copy_mcp_manifest_publication(
            session_id,
            expected_generations=expected_generations,
            baseline_updates=baseline_updates,
            events=events,
        )
        await self._ensure_ready()
        expected_run_epoch = _current_session_run_epoch(session_id)
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await self._lock_closure_lineage(cur)
                    # Missing rows need the same fence as existing rows. Locking
                    # the stable keys first also gives multi-toolset batches one
                    # deterministic lock order.
                    for key in sorted(expected):
                        await cur.execute(
                            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                            (key,),
                        )
                    await cur.execute(
                        "SELECT history_key, generation, baseline "
                        "FROM cayu_mcp_manifest_baselines "
                        "WHERE history_key = ANY(%s) FOR UPDATE",
                        (list(expected),),
                    )
                    current = {
                        row[0]: _stored_mcp_manifest_baseline(row[0], row[1], row[2])
                        for row in await cur.fetchall()
                    }
                    if any(
                        expected_generation
                        != (None if (baseline := current.get(key)) is None else baseline.generation)
                        for key, expected_generation in expected.items()
                    ):
                        await conn.rollback()
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
                    if await self._load_for_update(cur, session_id) is None:
                        raise KeyError(f"Session not found: {session_id}")
                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
                    activity_at = await self._session_store_now(cur)
                    if expected_run_epoch is None:
                        await cur.execute(
                            """
                            UPDATE cayu_sessions
                            SET event_seq = event_seq + %s, last_activity_at = %s
                            WHERE id = %s
                            RETURNING event_seq
                            """,
                            (len(copied_events), activity_at, session_id),
                        )
                    else:
                        await cur.execute(
                            """
                            UPDATE cayu_sessions
                            SET event_seq = event_seq + %s, last_activity_at = %s
                            WHERE id = %s AND run_epoch = %s
                            RETURNING event_seq
                            """,
                            (
                                len(copied_events),
                                activity_at,
                                session_id,
                                expected_run_epoch,
                            ),
                        )
                    order_row = await cur.fetchone()
                    if order_row is None:
                        if expected_run_epoch is not None:
                            await _raise_session_write_conflict(
                                cur,
                                session_id,
                                expected_run_epoch,
                            )
                        raise KeyError(f"Session not found: {session_id}")

                    next_order = order_row[0] - len(copied_events)
                    await self._register_event_public_authorities(
                        cur,
                        session_id,
                        copied_events,
                    )
                    event_rows = []
                    for event in copied_events:
                        next_order += 1
                        lookup_key, projection, projection_bytes = (
                            pending_action_event_storage_values(event)
                        )
                        event_rows.append(
                            (
                                session_id,
                                next_order,
                                event.id,
                                event.interaction_id,
                                str(event.type),
                                pg_support.to_utc(event.timestamp),
                                event.agent_name,
                                event.environment_name,
                                event.workflow_name,
                                event.tool_name,
                                pg_support._dumps(event.payload),
                                pg_support._dumps(event.model_dump(mode="json")),
                                lookup_key,
                                projection,
                                projection_bytes,
                            )
                        )
                    await cur.executemany(
                        """
                        INSERT INTO cayu_events (
                            session_id, session_order, event_id, interaction_id,
                            event_type, timestamp,
                            agent_name, environment_name, workflow_name, tool_name,
                            payload, event, pending_action_lookup_key,
                            pending_action_projection, pending_action_projection_bytes
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s
                        )
                        """,
                        event_rows,
                    )
                    await self._enqueue_persisted_event_side_effects(
                        cur,
                        session_id,
                        copied_events,
                    )
                    for key, baseline in updates.items():
                        await cur.execute(
                            """
                            INSERT INTO cayu_mcp_manifest_baselines (
                                history_key, generation, baseline, updated_at
                            )
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (history_key) DO UPDATE SET
                                generation = EXCLUDED.generation,
                                baseline = EXCLUDED.baseline,
                                updated_at = EXCLUDED.updated_at
                            """,
                            (
                                key,
                                baseline.generation,
                                pg_support._dumps(baseline.model_dump(mode="json")),
                                activity_at,
                            ),
                        )
                        current[key] = baseline.model_copy(deep=True)
                await conn.commit()
                return McpManifestPublicationResult(
                    published=True,
                    baselines=current,
                )
            except UniqueViolation as exc:
                await conn.rollback()
                existing = await self._first_existing_event_id(
                    session_id,
                    [event.id for event in copied_events],
                )
                if existing is not None:
                    raise ValueError(
                        f"Event already exists for session {session_id}: {existing}"
                    ) from exc
                raise
            except Exception:
                await conn.rollback()
                raise

    @staticmethod
    async def _enqueue_persisted_event_side_effects(
        cur: Any,
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
        # Presence alone is not authority: rows predating revision 31 may contain
        # caller-authored payload text but cannot carry this proof bit.
        if runtime_owned_input_contract_event_ids:
            await cur.execute(
                """
                UPDATE cayu_events
                SET input_contract_runtime_owned = TRUE
                WHERE session_id = %s
                  AND event_id = ANY(%s)
                  AND event_type IN (
                      'session.started',
                      'session.resumed',
                      'session.message.queued',
                      'session.message.delivered'
                  )
                  AND jsonb_typeof(payload -> 'input_contract') = 'string'
                """,
                (session_id, runtime_owned_input_contract_event_ids),
            )
        if runtime_owned_file_attestation_event_ids:
            await cur.execute(
                """
                UPDATE cayu_events
                SET file_attachment_attestations_runtime_owned = TRUE
                WHERE session_id = %s
                  AND event_id = ANY(%s)
                  AND event_type = 'model.started'
                  AND jsonb_typeof(payload -> 'file_attachment_attestations') = 'string'
                """,
                (session_id, runtime_owned_file_attestation_event_ids),
            )
        await cur.execute(
            """
            INSERT INTO cayu_persisted_event_side_effects (
                session_id, event_id, event_sequence, status, attempts, updated_at
            )
            SELECT session_id, event_id, sequence, 'pending', 0, timestamp
            FROM cayu_events
            WHERE session_id = %s
              AND event_id = ANY(%s)
              AND event_type <> 'runtime.sink.failed'
            """,
            (session_id, event_ids),
        )

    async def claim_first_persisted_event_side_effect(
        self, expected: PersistedEventSideEffectDelivery
    ) -> PersistedEventSideEffectClaim | None:
        expected = _copy_pending_first_event_delivery(expected)
        return await self._claim_persisted_event_side_effect(
            session_id=expected.session_id, event_id=expected.event_id, expected=expected
        )

    async def claim_persisted_event_side_effect(
        self,
        *,
        session_id: str | None = None,
        event_id: str | None = None,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectClaim | None:
        return await self._claim_persisted_event_side_effect(
            session_id=session_id, event_id=event_id, lease_seconds=lease_seconds
        )

    async def _claim_persisted_event_side_effect(
        self,
        *,
        session_id: str | None = None,
        event_id: str | None = None,
        lease_seconds: float = 300.0,
        expected: PersistedEventSideEffectDelivery | None = None,
    ) -> PersistedEventSideEffectClaim | None:
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        if event_id is not None:
            event_id = require_clean_nonblank(event_id, "event_id")
        if (session_id is None) != (event_id is None):
            raise ValueError("session_id and event_id must be supplied together.")
        if type(lease_seconds) not in {int, float} or lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than 0.")
        claim_id = str(uuid4())
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    exact_filter = ""
                    params: list[Any] = []
                    # The same transaction-level lock protects closure admission.
                    # Exclude closed targets in selection, so one retained target
                    # cannot starve unrelated event deliveries.
                    await self._lock_closure_lineage(cur)
                    if expected is not None:
                        await cur.execute(
                            "SELECT session_id, event_id, event_sequence, status, attempts, claim_id, "
                            "lease_expires_at, next_attempt_at, last_error, updated_at "
                            "FROM cayu_persisted_event_side_effects "
                            "WHERE session_id = %s AND event_id = %s FOR UPDATE",
                            (expected.session_id, expected.event_id),
                        )
                        expected_row = await cur.fetchone()
                        if (
                            expected_row is None
                            or _persisted_event_side_effect_delivery_from_row(expected_row)
                            != expected
                        ):
                            await conn.rollback()
                            return None
                    if session_id is not None and event_id is not None:
                        exact_filter = (
                            "AND candidate_delivery.session_id = %s "
                            "AND candidate_delivery.event_id = %s"
                        )
                        params.extend([session_id, event_id])
                    params.extend([claim_id, float(lease_seconds)])
                    await cur.execute(
                        f"""
                        WITH timing AS MATERIALIZED (
                            SELECT clock_timestamp() AS now
                        ), candidate AS (
                            SELECT candidate_delivery.session_id,
                                   candidate_delivery.event_id
                            FROM cayu_persisted_event_side_effects AS candidate_delivery,
                                 timing
                            WHERE (
                                candidate_delivery.status = 'pending'
                                OR (candidate_delivery.status = 'failed' AND (
                                    candidate_delivery.next_attempt_at IS NULL
                                    OR candidate_delivery.next_attempt_at <= timing.now
                                ))
                                OR (candidate_delivery.status = 'leased'
                                    AND candidate_delivery.lease_expires_at <= timing.now)
                            )
                            {exact_filter}
                            AND NOT EXISTS (
                                SELECT 1 FROM cayu_session_closure_progress AS p
                                WHERE p.root_session_id = candidate_delivery.session_id
                                   OR EXISTS (
                                       SELECT 1 FROM jsonb_array_elements(
                                           p.progress_json->'descendants'
                                       ) AS child
                                       WHERE child->>'session_id' = candidate_delivery.session_id
                                   )
                            )
                            ORDER BY candidate_delivery.event_sequence ASC
                            FOR UPDATE OF candidate_delivery SKIP LOCKED
                            LIMIT 1
                        )
                        UPDATE cayu_persisted_event_side_effects AS delivery
                        SET status = 'leased', attempts = delivery.attempts + 1,
                            claim_id = %s,
                            lease_expires_at = timing.now + (%s * INTERVAL '1 second'),
                            next_attempt_at = NULL, last_error = NULL,
                            updated_at = timing.now
                        FROM candidate, timing
                        WHERE delivery.session_id = candidate.session_id
                          AND delivery.event_id = candidate.event_id
                        RETURNING delivery.session_id, delivery.event_id,
                                  delivery.event_sequence, delivery.attempts,
                                  delivery.lease_expires_at
                        """,
                        params,
                    )
                    row = await cur.fetchone()
                    if row is None:
                        await conn.commit()
                        return None
                    await cur.execute(
                        "SELECT event FROM cayu_events WHERE session_id = %s AND event_id = %s",
                        (row[0], row[1]),
                    )
                    event_row = await cur.fetchone()
                    if event_row is None:
                        raise RuntimeError("Persisted side-effect delivery lost its source event.")
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return PersistedEventSideEffectClaim(
            session_id=row[0],
            event_id=row[1],
            event_sequence=row[2],
            event=Event(**pg_support._json_obj(event_row[0])),
            attempt=row[3],
            claim_id=claim_id,
            lease_expires_at=pg_support.to_utc(row[4]),
        )

    async def get_persisted_event_side_effect_delivery(
        self,
        *,
        session_id: str,
        event_id: str,
    ) -> PersistedEventSideEffectDelivery | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        event_id = require_clean_nonblank(event_id, "event_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT session_id, event_id, event_sequence, status, attempts,
                       claim_id, lease_expires_at, next_attempt_at, last_error, updated_at
                FROM cayu_persisted_event_side_effects
                WHERE session_id = %s AND event_id = %s
                """,
                (session_id, event_id),
            )
            row = await cur.fetchone()
        return None if row is None else _persisted_event_side_effect_delivery_from_row(row)

    async def retire_failed_first_event_delivery(
        self, expected: PersistedEventSideEffectDelivery
    ) -> PersistedEventSideEffectDelivery | None:
        expected = _copy_failed_first_delivery_retirement(expected)
        await self._ensure_ready()
        async with self._connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.execute(
                "SELECT session_id, event_id, event_sequence, status, attempts, claim_id, "
                "lease_expires_at, next_attempt_at, last_error, updated_at "
                "FROM cayu_persisted_event_side_effects WHERE session_id = %s AND event_id = %s FOR UPDATE",
                (expected.session_id, expected.event_id),
            )
            row = await cur.fetchone()
            if row is None or _persisted_event_side_effect_delivery_from_row(row) != expected:
                return None
            retired = expected.model_copy(
                update={
                    "status": PersistedEventSideEffectStatus.DEAD_LETTERED,
                    "next_attempt_at": None,
                    "updated_at": await self._session_store_now(cur),
                }
            )
            await cur.execute(
                "UPDATE cayu_persisted_event_side_effects SET status = 'dead_lettered', "
                "next_attempt_at = NULL, updated_at = %s WHERE session_id = %s AND event_id = %s",
                (retired.updated_at, expected.session_id, expected.event_id),
            )
            return retired

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

    async def defer_persisted_event_side_effect(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        claim = PersistedEventSideEffectClaim.model_validate(claim)
        return await self._finish_persisted_event_side_effect_claim(
            claim,
            status=PersistedEventSideEffectStatus.PENDING,
            error=None,
            retry_delay_seconds=None,
            deferred=True,
        )

    async def renew_persisted_event_side_effect(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectDelivery:
        claim = PersistedEventSideEffectClaim.model_validate(claim)
        if type(lease_seconds) not in {int, float} or not 0 < lease_seconds <= 86_400:
            raise ValueError("lease_seconds must be positive and at most 86400.")
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    # Sample ownership time only after acquiring the row lock:
                    # lock contention must not turn a pre-wait timestamp into
                    # permission to revive a lease that expired while waiting.
                    await cur.execute(
                        "SELECT 1 FROM cayu_persisted_event_side_effects "
                        "WHERE session_id = %s AND event_id = %s FOR UPDATE",
                        (claim.session_id, claim.event_id),
                    )
                    if await cur.fetchone() is None:
                        raise PersistedEventSideEffectClaimLost(
                            "Persisted event side-effect claim is no longer active."
                        )
                    await cur.execute(
                        """
                        WITH timing AS MATERIALIZED (
                            SELECT clock_timestamp() AS now
                        )
                        UPDATE cayu_persisted_event_side_effects
                        SET lease_expires_at = GREATEST(
                                lease_expires_at, timing.now + (%s * INTERVAL '1 second')
                            ), updated_at = timing.now
                        FROM timing
                        WHERE session_id = %s AND event_id = %s AND status = 'leased'
                          AND claim_id = %s AND attempts = %s
                          AND lease_expires_at > timing.now
                        RETURNING session_id, event_id, event_sequence, status,
                                  attempts, claim_id, lease_expires_at, next_attempt_at,
                                  last_error, updated_at
                        """,
                        (
                            float(lease_seconds),
                            claim.session_id,
                            claim.event_id,
                            claim.claim_id,
                            claim.attempt,
                        ),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        raise PersistedEventSideEffectClaimLost(
                            "Persisted event side-effect claim is no longer active."
                        )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return _persisted_event_side_effect_delivery_from_row(row)

    async def _finish_persisted_event_side_effect_claim(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        status: PersistedEventSideEffectStatus,
        error: str | None,
        retry_delay_seconds: float | None,
        deferred: bool = False,
    ) -> PersistedEventSideEffectDelivery:
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        WITH timing AS MATERIALIZED (
                            SELECT clock_timestamp() AS now
                        )
                        UPDATE cayu_persisted_event_side_effects
                        SET status = %s, claim_id = NULL, lease_expires_at = NULL,
                            next_attempt_at = CASE
                                WHEN %s::double precision IS NULL THEN NULL
                                ELSE timing.now + (%s * INTERVAL '1 second')
                            END,
                            last_error = %s, updated_at = timing.now,
                            attempts = attempts - %s
                        FROM timing
                        WHERE session_id = %s AND event_id = %s AND status = 'leased'
                          AND claim_id = %s AND attempts = %s
                        RETURNING session_id, event_id, event_sequence, status,
                                  attempts, claim_id, lease_expires_at, next_attempt_at,
                                  last_error, updated_at
                        """,
                        (
                            str(status),
                            retry_delay_seconds,
                            retry_delay_seconds,
                            error,
                            int(deferred),
                            claim.session_id,
                            claim.event_id,
                            claim.claim_id,
                            claim.attempt,
                        ),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        await cur.execute(
                            "SELECT 1 FROM cayu_persisted_event_side_effects "
                            "WHERE session_id = %s AND event_id = %s",
                            (claim.session_id, claim.event_id),
                        )
                        if await cur.fetchone() is None:
                            raise ValueError("Persisted event side-effect delivery was not found.")
                        raise PersistedEventSideEffectClaimLost(
                            "Persisted event side-effect claim is no longer active."
                        )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return _persisted_event_side_effect_delivery_from_row(row)

    async def get_persisted_event_side_effect_health(self) -> PersistedEventSideEffectHealth:
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute("SELECT clock_timestamp()")
            now = (await cur.fetchone())[0]
            await cur.execute(
                cast("LiteralString", side_effect_health.health_sql("%s::timestamptz")), (now,)
            )
            row = await cur.fetchone()
        return side_effect_health.finish_health(
            dict(zip(side_effect_health.AGGREGATES, row, strict=True)), now
        )

    async def query_persisted_event_side_effect_deliveries(
        self,
        query: PersistedEventSideEffectQuery,
    ) -> PersistedEventSideEffectPage:
        query = PersistedEventSideEffectQuery.model_validate(query)
        side_effect_health.cursor_key(query)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute("SELECT clock_timestamp()")
            now = (await cur.fetchone())[0]
            sql, params = side_effect_health.page_sql(query, now, "%s")
            sql = sql.replace("SELECT %s AS observed_at", "SELECT %s::timestamptz AS observed_at")
            await cur.execute(cast("LiteralString", sql), params)
            rows = await cur.fetchall()
        return side_effect_health.page(
            [_persisted_event_side_effect_delivery_from_row(row) for row in rows],
            query,
            now,
        )

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
        if selected_statuses == []:
            return []
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            clauses: list[str] = []
            params: list[Any] = []
            if after_sequence is not None:
                clauses.append("event_sequence > %s")
                params.append(after_sequence)
            if selected_statuses is not None:
                clauses.append("status = ANY(%s)")
                params.append(selected_statuses)
            if claimable_only:
                clauses.append(
                    "(status = 'pending' "
                    "OR (status = 'failed' AND "
                    "(next_attempt_at IS NULL OR next_attempt_at <= clock_timestamp())) "
                    "OR (status = 'leased' AND lease_expires_at <= clock_timestamp()))"
                )
            where = "" if not clauses else "WHERE " + " AND ".join(clauses)
            params.append(limit)
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    SELECT session_id, event_id, event_sequence, status, attempts,
                           claim_id, lease_expires_at, next_attempt_at, last_error, updated_at
                    FROM cayu_persisted_event_side_effects
                    {where}
                    ORDER BY event_sequence ASC
                    LIMIT %s
                    """,
                ),
                params,
            )
            rows = await cur.fetchall()
        return [_persisted_event_side_effect_delivery_from_row(row) for row in rows]

    async def _session_message_source_locked(
        self,
        cur: Any,
        session: Session,
        *,
        include_transcript_digest: bool,
        include_checkpoint_digest: bool,
    ) -> SessionMessageSource:
        cursor = await _transcript_cursor(cur, session.id)
        transcript_digest = None
        if include_transcript_digest:
            hasher = message_queue.SourceTranscriptHasher(cursor)
            after = 0
            while True:
                await cur.execute(
                    "SELECT session_order, message FROM cayu_transcript_messages "
                    "WHERE session_id = %s AND session_order > %s ORDER BY session_order LIMIT 100",
                    (session.id, after),
                )
                rows = await cur.fetchall()
                if not rows:
                    break
                for row in rows:
                    hasher.add(row[0] - 1, Message.model_validate(pg_support._json_obj(row[1])))
                after = rows[-1][0]
            transcript_digest = hasher.hexdigest()
        checkpoint = None
        if include_checkpoint_digest:
            checkpoint = await self._load_checkpoint(cur, session.id)
        return message_queue.source_snapshot(
            session,
            cursor,
            transcript_sha256=transcript_digest,
            checkpoint=checkpoint,
            include_checkpoint_digest=include_checkpoint_digest,
        )

    async def _session_message_read_session(self, cur: Any, session_id: str) -> Session:
        # Inspection must also work on read-only connections. One MVCC snapshot
        # binds session identity, queue pages, transcript and checkpoint reads.
        await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        await cur.execute(
            f"SELECT {pg_support.SESSION_COLUMNS} FROM cayu_sessions WHERE id = %s",
            (session_id,),
        )
        row = await cur.fetchone()
        if row is None:
            require_resource_session(None)
            raise KeyError("Session not found.")
        return pg_support.session_from_row(row, labels=await self._load_labels(cur, session_id))

    @runtime_session_query
    async def snapshot_session_message_source(
        self,
        session_id: str,
        *,
        include_transcript_digest: bool = False,
        include_checkpoint_digest: bool = False,
        expected_authorized_session_instance_id: str | None = None,
    ) -> SessionMessageSource:
        session_id = require_clean_nonblank(session_id, "session_id")
        if (
            type(include_transcript_digest) is not bool
            or type(include_checkpoint_digest) is not bool
        ):
            raise TypeError("Snapshot digest flags must be bool.")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            session = await self._session_message_read_session(cur, session_id)
            require_resource_session(session, "read")
            message_queue.require_authorized_session_instance(
                session, expected_authorized_session_instance_id
            )
            return await self._session_message_source_locked(
                cur,
                session,
                include_transcript_digest=include_transcript_digest,
                include_checkpoint_digest=include_checkpoint_digest,
            )

    async def _session_message_raw_bounded(
        self,
        cur: Any,
        session_id: str,
        queue_id: str,
    ) -> dict[str, Any]:
        columns = _SESSION_MESSAGE_QUEUE_COLUMNS.split(", ")
        projection = ", ".join(
            f"CASE WHEN octet_length({name}::text) <= {SESSION_MESSAGE_QUEUE_STORAGE_VALUE_MAX_BYTES} "
            f"THEN {name} END AS {name}"
            for name in columns
        )
        hashes = ", ".join(
            f"CASE WHEN octet_length({name}::text) > {SESSION_MESSAGE_QUEUE_STORAGE_VALUE_MAX_BYTES} "
            f"THEN encode(sha256(convert_to({name}::text, 'UTF8')), 'hex') END"
            for name in columns
        )
        await cur.execute(
            f"SELECT {projection}, {hashes} FROM cayu_session_message_queue "
            "WHERE session_id = %s AND queue_id = %s",
            (session_id, queue_id),
        )
        row = await cur.fetchone()
        if row is None:
            raise SessionMessageConflict()
        raw: dict[str, Any] = dict(zip(columns, row[: len(columns)], strict=True))
        for name, digest in zip(columns, row[len(columns) :], strict=True):
            if digest is not None:
                raw[name] = message_queue.OversizedStorageValue(digest)
        return raw

    async def _session_message_acceptance_events(
        self,
        cur: Any,
        session_id: str,
        rows: list[dict[str, Any]],
        *,
        quarantine_queue_id: str | None = None,
    ) -> dict[str, Event]:
        """Batch-read bounded audit projections of canonical events under the session lock."""
        ids = [row["accepted_event_id"] for row in rows]
        if quarantine_queue_id is None and any(
            type(event_id) is not str or len(event_id) > 512 for event_id in ids
        ):
            raise SessionMessageConflict()
        if not ids:
            return {}
        projection = (
            "jsonb_build_object('queue_id', event->'payload'->'queue_id', "
            "'source', event->'payload'->'source')"
        )
        predicate = (
            "event_id = ANY(%s)"
            if quarantine_queue_id is None
            else "event #>> '{payload,queue_id}' = %s LIMIT 2"
        )
        await cur.execute(
            f"SELECT event_id, CASE WHEN octet_length(({projection})::text) <= 32768 "
            f"THEN {projection} END FROM cayu_events WHERE session_id = %s "
            f"AND event_type = 'session.message.queued' AND {predicate}",
            (session_id, ids if quarantine_queue_id is None else quarantine_queue_id),
        )
        events = await cur.fetchall()
        if quarantine_queue_id is not None and len(events) != 1:
            raise SessionMessageConflict()
        if any(row[1] is None for row in events):
            raise SessionMessageConflict()
        return {
            row[0]: Event(
                id=row[0],
                type=EventType.SESSION_MESSAGE_QUEUED,
                session_id=session_id,
                payload=pg_support._json_obj(row[1]),
            )
            for row in events
        }

    @runtime_session_query
    async def inspect_session_messages(
        self,
        query: SessionMessageQuery,
        *,
        expected_authorized_session_instance_id: str | None = None,
    ) -> SessionMessageInspection:
        query = message_queue.copy_inspection_query(query)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            session = await self._session_message_read_session(cur, query.session_id)
            require_resource_session(session, "read")
            message_queue.require_authorized_session_instance(
                session, expected_authorized_session_instance_id
            )
            maximum = 0
            if query.cursor is None:
                await cur.execute(
                    "SELECT COALESCE(MAX(ordering_key), 0) FROM cayu_session_message_queue "
                    "WHERE session_id = %s",
                    (session.id,),
                )
                maximum = (await cur.fetchone())[0]
            boundary = message_queue.inspection_boundary(session, query.cursor, maximum)
            await cur.execute(
                "WITH ordered AS (SELECT queue_id, ordering_key, "
                "CASE delivery_mode WHEN 'next_turn' THEN 0 WHEN 'on_idle' THEN 1 ELSE 2 END "
                "AS priority FROM cayu_session_message_queue "
                "WHERE session_id = %s AND ordering_key <= %s) "
                "SELECT queue_id, ordering_key, priority FROM ordered "
                "WHERE (priority, ordering_key) > (%s, %s) "
                "ORDER BY priority, ordering_key LIMIT %s",
                (
                    session.id,
                    boundary.through_ordering_key,
                    boundary.after_priority,
                    boundary.after_ordering_key,
                    query.limit + 1,
                ),
            )
            rows = await cur.fetchall()
            raw_rows = [
                await self._session_message_raw_bounded(cur, session.id, row[0])
                for row in rows[: query.limit]
            ]
            records = tuple(
                message_queue.inspect_record(
                    raw, lambda raw=raw: _queued_session_message_from_row(tuple(raw.values()))
                )
                for raw in raw_rows
            )
            return SessionMessageInspection(
                session_id=session.id,
                session_instance_id=session.instance_id,
                records=records,
                next_cursor=(
                    message_queue.inspection_next_cursor(
                        boundary,
                        rows[query.limit - 1][2],
                        records[-1].ordering_key,
                    )
                    if len(rows) > query.limit
                    else None
                ),
            )

    @runtime_session_query
    async def apply_session_message_action(
        self,
        request: SessionMessageActionRequest,
    ) -> SessionMessageActionResult:
        request = SessionMessageActionRequest(**message_queue.action_material(request))
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    session = await self._load_for_update(cur, request.session_id)
                    require_resource_session(session, "modify")
                    if session is None or session.instance_id != request.session_instance_id:
                        raise SessionMessageConflict()
                    await cur.execute(
                        "SELECT queue_id FROM cayu_session_message_queue "
                        "WHERE session_id = %s AND queue_id = %s FOR UPDATE",
                        (session.id, request.queue_id),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        raise SessionMessageConflict()
                    raw = await self._session_message_raw_bounded(cur, session.id, request.queue_id)
                    accepted_events = await self._session_message_acceptance_events(
                        cur,
                        session.id,
                        [raw],
                        quarantine_queue_id=request.queue_id
                        if request.action == "quarantine"
                        else None,
                    )
                    accepted_event = (
                        next(iter(accepted_events.values()))
                        if request.action == "quarantine"
                        else accepted_events.get(raw["accepted_event_id"])
                    )
                    replay = message_queue.replay_action(
                        raw, request, accepted_event=accepted_event
                    )
                    record = message_queue.inspect_record(
                        raw, lambda: _queued_session_message_from_row(tuple(raw.values()))
                    )
                    if replay is not None:
                        await conn.commit()
                        return SessionMessageActionResult(
                            record=record, event=replay, replayed=True
                        )
                    for owner in await self._closure_lineage_owners(cur, (session.id,)):
                        _check_closure_lineage_owner(owner, (session.id,))
                    if (
                        record.revision != request.expected_revision
                        or raw["status"] != "queued"
                        or any(
                            raw[key] is not None
                            for key in (
                                "delivered_event_id",
                                "delivered_at",
                                "delivered_run_epoch",
                                "delivered_transcript_cursor",
                            )
                        )
                    ):
                        raise SessionMessageConflict()
                    await cur.execute(
                        "SELECT 1 FROM cayu_session_message_deliveries "
                        "WHERE session_id = %s AND queue_ids @> %s::jsonb LIMIT 1",
                        (session.id, pg_support._dumps([request.queue_id])),
                    )
                    if await cur.fetchone() is not None or (
                        request.action == "withdraw" and record.validity != "valid"
                    ):
                        raise SessionMessageConflict()
                    status = SessionMessageQueueStatus(
                        "withdrawn" if request.action == "withdraw" else "quarantined"
                    )
                    event = message_queue.terminal_event(
                        session,
                        raw,
                        status,
                        await self._session_store_now(cur),
                        actor=request.requested_by,
                        accepted_event=accepted_event,
                    )
                    await cur.execute(
                        "UPDATE cayu_session_message_queue SET status = %s, terminal_json = %s "
                        "WHERE session_id = %s AND queue_id = %s AND status = 'queued'",
                        (
                            str(status),
                            pg_support._dumps(
                                message_queue.terminal_receipt(status, event, request)
                            ),
                            session.id,
                            request.queue_id,
                        ),
                    )
                    await self._append_events_with_cursor(
                        cur, session.id, [event], expected_run_epoch=None
                    )
                    updated = await self._session_message_raw_bounded(
                        cur, session.id, request.queue_id
                    )
                    result = SessionMessageActionResult(
                        record=message_queue.inspect_record(
                            updated,
                            lambda: _queued_session_message_from_row(tuple(updated.values())),
                        ),
                        event=event,
                    )
                await conn.commit()
                return result
            except BaseException:
                await conn.rollback()
                raise

    @runtime_session_query
    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
        *,
        expected_authorized_target_instance_id: str | None = None,
    ) -> EnqueueSessionMessageResult:
        from cayu.sessions.pending_actions import pending_action_event_storage_values

        request = copy_enqueue_session_message_request(request)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    # A->B and B->A provenance admissions acquire the same lock order.
                    source_id = (
                        None
                        if request.conditions.source is None
                        else request.conditions.source.session_id
                    )
                    locked = {
                        sid: await self._load_for_update(cur, sid)
                        for sid in sorted(
                            {request.session_id} | ({source_id} if source_id else set())
                        )
                    }
                    loaded = locked[request.session_id]
                    require_resource_session(loaded, "modify")
                    if loaded is None:
                        raise KeyError(f"Session not found: {request.session_id}")
                    if expected_authorized_target_instance_id is not None and (
                        type(expected_authorized_target_instance_id) is not str
                        or loaded.instance_id != expected_authorized_target_instance_id
                    ):
                        raise SessionMessageConflict()
                    if request.conditions.source is not None:
                        source = locked[request.conditions.source.session_id]
                        require_resource_session(source, "read")
                        if (
                            source is None
                            or source.instance_id != request.conditions.source.session_instance_id
                        ):
                            raise SessionMessageConflict()
                    await cur.execute(
                        f"SELECT {_SESSION_MESSAGE_QUEUE_COLUMNS} "
                        "FROM cayu_session_message_queue "
                        "WHERE session_id = %s AND idempotency_key = %s",
                        (request.session_id, request.idempotency_key),
                    )
                    existing_row = await cur.fetchone()
                    if existing_row is not None:
                        existing = _queued_session_message_from_row(existing_row)
                        _validate_equivalent_queued_session_message(existing, request)
                        await cur.execute(
                            "SELECT event FROM cayu_events WHERE session_id = %s AND event_id = %s",
                            (request.session_id, existing.accepted_event_id),
                        )
                        event_row = await cur.fetchone()
                        if event_row is None:
                            raise RuntimeError(
                                "Queued session message is missing its durable acceptance event."
                            )
                        await conn.commit()
                        return EnqueueSessionMessageResult(
                            message=existing,
                            event=Event(**pg_support._json_obj(event_row[0])),
                            replayed=True,
                        )
                    for owner in await self._closure_lineage_owners(cur, (request.session_id,)):
                        _check_closure_lineage_owner(owner, (request.session_id,))
                    checkpoint = await self._load_checkpoint(cur, request.session_id)
                    message_queue.require_open_admission(loaded.status, checkpoint)
                    if request.conditions.source is not None:
                        expected_source = request.conditions.source
                        source = locked[expected_source.session_id]
                        require_resource_session(source, "read")
                        if (
                            source is None
                            or await self._session_message_source_locked(
                                cur,
                                source,
                                include_transcript_digest=expected_source.transcript_sha256
                                is not None,
                                include_checkpoint_digest=expected_source.checkpoint_sha256
                                is not None,
                            )
                            != expected_source
                        ):
                            raise SessionMessageConflict()
                    transcript_cursor = await _transcript_cursor(cur, request.session_id)
                    accepted_at = await self._session_store_now(cur)
                    queue_id = str(uuid4())
                    accepted_event_id = str(uuid4())
                    await cur.execute(
                        """
                        INSERT INTO cayu_session_message_queue (
                            queue_id, session_id, idempotency_key, content, message_json,
                            delivery_mode, status, requested_by,
                            accepted_run_epoch, accepted_transcript_cursor,
                            accepted_event_id, accepted_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, 'queued', %s, %s, %s, %s, %s)
                        RETURNING ordering_key
                        """,
                        (
                            queue_id,
                            request.session_id,
                            request.idempotency_key,
                            request.content,
                            (
                                None
                                if request.message is None
                                else pg_support._dumps(request.message.model_dump(mode="json"))
                            ),
                            str(request.delivery_mode),
                            (
                                None
                                if request.requested_by is None
                                else pg_support._dumps(
                                    resolution_actor_payload(request.requested_by)
                                )
                            ),
                            loaded.run_epoch,
                            transcript_cursor,
                            accepted_event_id,
                            accepted_at,
                        ),
                    )
                    ordering_row = await cur.fetchone()
                    if ordering_row is None:
                        raise RuntimeError("Postgres queue insert did not return an ordering key.")
                    ordering_key = ordering_row[0]
                    await cur.execute(
                        "UPDATE cayu_session_message_queue SET conditions_json = %s WHERE queue_id = %s",
                        (
                            pg_support._dumps(request.conditions.model_dump(mode="json")),
                            queue_id,
                        ),
                    )
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
                                **message_queue.source_event_payload(request.conditions.source),
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
                    await cur.execute(
                        "UPDATE cayu_sessions SET event_seq = event_seq + 1, "
                        "last_activity_at = %s WHERE id = %s RETURNING event_seq",
                        (accepted_at, request.session_id),
                    )
                    event_order_row = await cur.fetchone()
                    if event_order_row is None:
                        raise KeyError(f"Session not found: {request.session_id}")
                    lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                        accepted_event
                    )
                    await self._register_event_public_authorities(
                        cur,
                        request.session_id,
                        [accepted_event],
                    )
                    await cur.execute(
                        """
                        INSERT INTO cayu_events (
                            session_id, session_order, event_id, interaction_id,
                            event_type, timestamp,
                            agent_name, environment_name, workflow_name, tool_name,
                            payload, event, pending_action_lookup_key,
                            pending_action_projection, pending_action_projection_bytes
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            request.session_id,
                            event_order_row[0],
                            accepted_event.id,
                            accepted_event.interaction_id,
                            str(accepted_event.type),
                            accepted_event.timestamp,
                            accepted_event.agent_name,
                            accepted_event.environment_name,
                            accepted_event.workflow_name,
                            accepted_event.tool_name,
                            pg_support._dumps(accepted_event.payload),
                            pg_support._dumps(accepted_event.model_dump(mode="json")),
                            lookup_key,
                            projection,
                            projection_bytes,
                        ),
                    )
                    await self._enqueue_persisted_event_side_effects(
                        cur,
                        request.session_id,
                        [accepted_event],
                    )
                    await cur.execute(
                        f"SELECT {_SESSION_MESSAGE_QUEUE_COLUMNS} "
                        "FROM cayu_session_message_queue WHERE queue_id = %s",
                        (queue_id,),
                    )
                    stored_row = await cur.fetchone()
                    if stored_row is None:
                        raise RuntimeError("Queued session message disappeared after acceptance.")
                await conn.commit()
                return EnqueueSessionMessageResult(
                    message=_queued_session_message_from_row(stored_row),
                    event=accepted_event,
                )
            except Exception:
                await conn.rollback()
                raise

    async def deliver_queued_session_messages(
        self,
        session_id: str,
        *,
        include_on_idle: bool,
        reject_only: bool = False,
        delivery_id: str | None = None,
        eligible_through: int | None = None,
        limit: int = SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
        interaction_id: str | None = None,
        interaction_started_event: Event | None = None,
        profile_handoff: QueuedInteractionProfileHandoff | None = None,
    ) -> SessionMessageDeliveryBatch:
        from cayu.sessions.pending_actions import pending_action_event_storage_values

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
        profile_handoff = _copy_queued_interaction_profile_handoff(
            session_id,
            delivery_id,
            interaction_id,
            interaction_started_event,
            profile_handoff,
        )
        if type(include_on_idle) is not bool:
            raise TypeError("include_on_idle must be a bool.")
        if type(reject_only) is not bool:
            raise TypeError("reject_only must be a bool.")
        eligible_through = _validate_message_delivery_eligible_through(eligible_through)
        if type(limit) is not int or not 1 <= limit <= SESSION_MESSAGE_DELIVERY_BATCH_LIMIT:
            raise ValueError(f"limit must be between 1 and {SESSION_MESSAGE_DELIVERY_BATCH_LIMIT}.")
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded = await self._load_for_update(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    _assert_session_run_epoch(session_id, loaded)
                    await cur.execute(
                        """
                        SELECT session_id, interaction_id, include_on_idle,
                               requested_eligible_through, eligible_through,
                               batch_limit, has_more, interaction_started_event,
                               queue_ids, events, reject_only
                        FROM cayu_session_message_deliveries
                        WHERE delivery_id = %s
                        """,
                        (delivery_id,),
                    )
                    delivery_row = await cur.fetchone()
                    if delivery_row is not None:
                        stored_started_event = (
                            None
                            if delivery_row[7] is None
                            else Event(**pg_support._json_obj(delivery_row[7]))
                        )
                        if (
                            delivery_row[0] != session_id
                            or delivery_row[10] != reject_only
                            or delivery_row[1] != interaction_id
                            or delivery_row[2] != include_on_idle
                            or delivery_row[3] != eligible_through
                            or delivery_row[5] != limit
                            or stored_started_event != interaction_started_event
                        ):
                            raise ValueError(
                                "delivery_id was already used for a different queue delivery."
                            )
                        queue_ids = list(delivery_row[8])
                        queued_by_id: dict[str, SessionQueuedMessage] = {}
                        replayed_events = tuple(
                            Event(**pg_support._json_obj(event)) for event in delivery_row[9]
                        )
                        if queue_ids:
                            await cur.execute(
                                f"SELECT {_SESSION_MESSAGE_QUEUE_COLUMNS} "
                                "FROM cayu_session_message_queue "
                                "WHERE queue_id = ANY(%s)",
                                (queue_ids,),
                            )
                            queued_by_id = {
                                message.queue_id: message
                                for message in (
                                    _queued_session_message_from_row(row)
                                    for row in await cur.fetchall()
                                )
                            }
                        if len(queued_by_id) != len(queue_ids):
                            raise RuntimeError("Queue delivery replay lost a delivered message.")
                        if queue_ids and profile_handoff is not None:
                            await cur.execute(
                                "SELECT record FROM cayu_session_operations "
                                "WHERE session_id = %s AND idempotency_key = %s",
                                (
                                    session_id,
                                    _interaction_transition_storage_key(
                                        profile_handoff.predecessor_settlement_event_id
                                    ),
                                ),
                            )
                            receipt_row = await cur.fetchone()
                            if receipt_row is None:
                                raise SessionRunFenced(
                                    "Queued interaction handoff lost its predecessor "
                                    "settlement receipt."
                                )
                            active_model_stage = None
                            stage_dispatch = None
                            await cur.execute(
                                "SELECT record FROM cayu_session_operations "
                                "WHERE session_id = %s AND idempotency_key = %s",
                                (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                            )
                            active_row = await cur.fetchone()
                            if active_row is not None:
                                active_record = _decode_model_completion_stage_record(active_row[0])
                                marker = _reconstruct_active_model_completion_stage_record(
                                    active_record,
                                    session_id=session_id,
                                )
                                _, _, preparation_key, terminal_key = (
                                    _model_completion_stage_storage_identity(
                                        session_id,
                                        marker.stage_id,
                                    )
                                )
                                dispatch_key = _model_completion_stage_dispatch_storage_key(
                                    marker.stage_id
                                )
                                await cur.execute(
                                    "SELECT idempotency_key, record "
                                    "FROM cayu_session_operations WHERE session_id = %s "
                                    "AND idempotency_key = ANY(%s)",
                                    (
                                        session_id,
                                        [preparation_key, terminal_key, dispatch_key],
                                    ),
                                )
                                stage_records = {
                                    row[0]: _decode_model_completion_stage_record(row[1])
                                    for row in await cur.fetchall()
                                }
                                active_model_stage = _reconstruct_active_model_completion_stage(
                                    active_record,
                                    stage_records.get(preparation_key),
                                    stage_records.get(terminal_key),
                                    session_id=session_id,
                                )
                                dispatch_record = stage_records.get(dispatch_key)
                                if dispatch_record is not None:
                                    stage_dispatch = _reconstruct_model_completion_stage_dispatch(
                                        dispatch_record,
                                        session_id=session_id,
                                        stage_id=marker.stage_id,
                                        storage_key=dispatch_key,
                                    )
                            repaired_checkpoint = (
                                _checkpoint_after_queued_interaction_profile_handoff(
                                    loaded,
                                    await self._load_checkpoint(cur, session_id),
                                    profile_handoff,
                                    settlement_record=pg_support._json_obj(receipt_row[0]),
                                    replayed_delivery=True,
                                    active_model_stage=active_model_stage,
                                    stage_dispatch=stage_dispatch,
                                )
                            )
                            await self._upsert_checkpoint(
                                cur,
                                session_id,
                                repaired_checkpoint,
                                await self._session_store_now(cur),
                            )
                        await conn.commit()
                        return SessionMessageDeliveryBatch(
                            messages=tuple(queued_by_id[queue_id] for queue_id in queue_ids),
                            events=replayed_events,
                            delivery_id=delivery_id,
                            interaction_id=interaction_id,
                            eligible_through=delivery_row[4],
                            has_more=delivery_row[6],
                            replayed=True,
                            active_invocation_profile=(
                                None
                                if not queue_ids or profile_handoff is None
                                else profile_handoff.target_active_profile
                            ),
                        )
                    if loaded.status != SessionStatus.RUNNING:
                        raise SessionStatusConflict(
                            "Queued session messages may be delivered only while running."
                        )
                    boundary = eligible_through
                    if boundary is None:
                        # ``ordering_key`` is a global identity primary key. Its
                        # global maximum is an end-of-index lookup and still
                        # fences every message this session can currently
                        # contain; the locked session row serializes enqueues for
                        # this session until the transaction completes.
                        await cur.execute(
                            "SELECT COALESCE(MAX(ordering_key), 0) FROM cayu_session_message_queue"
                        )
                        boundary_row = await cur.fetchone()
                        boundary = boundary_row[0] if boundary_row is not None else 0
                    await cur.execute(
                        f"SELECT {_SESSION_MESSAGE_QUEUE_COLUMNS} "
                        "FROM cayu_session_message_queue WHERE session_id = %s "
                        "AND status = 'queued' AND delivery_mode = 'next_turn' "
                        "AND ordering_key <= %s ORDER BY ordering_key ASC LIMIT %s FOR UPDATE",
                        (session_id, boundary, limit),
                    )
                    rows = await cur.fetchall()
                    if not rows and include_on_idle:
                        await cur.execute(
                            f"SELECT {_SESSION_MESSAGE_QUEUE_COLUMNS} "
                            "FROM cayu_session_message_queue WHERE session_id = %s "
                            "AND status = 'queued' AND delivery_mode = 'on_idle' "
                            "AND ordering_key <= %s ORDER BY ordering_key ASC LIMIT %s FOR UPDATE",
                            (session_id, boundary, limit),
                        )
                        rows = await cur.fetchall()
                    reject_only_more = False
                    if reject_only:
                        # Eligible rows remain pending. Scan in bounded pages so they cannot hide
                        # an expired record behind the first delivery-sized prefix.
                        rows = []
                        scan_now = await self._session_store_now(cur)
                        scan_cursor = await _transcript_cursor(cur, session_id)
                        for mode in ("next_turn", "on_idle") if include_on_idle else ("next_turn",):
                            after = 0
                            while len(rows) < limit + 1:
                                await cur.execute(
                                    f"SELECT {_SESSION_MESSAGE_QUEUE_COLUMNS} FROM cayu_session_message_queue "
                                    "WHERE session_id = %s AND status = 'queued' AND delivery_mode = %s "
                                    "AND ordering_key > %s AND ordering_key <= %s "
                                    "ORDER BY ordering_key LIMIT 100 FOR UPDATE",
                                    (session_id, mode, after, boundary),
                                )
                                page = await cur.fetchall()
                                if not page:
                                    break
                                for candidate in page:
                                    queued = _queued_session_message_from_row(candidate)
                                    if (
                                        session_message_rejection(
                                            queued.conditions,
                                            session_instance_id=loaded.instance_id,
                                            run_epoch=loaded.run_epoch,
                                            transcript_cursor=scan_cursor,
                                            now=scan_now,
                                        )
                                        is not None
                                    ):
                                        rows.append(candidate)
                                        if len(rows) == limit + 1:
                                            reject_only_more = True
                                            break
                                after = page[-1][0]
                            if len(rows) == limit + 1:
                                break
                        rows = rows[:limit]
                    if not rows:
                        await cur.execute(
                            """
                            INSERT INTO cayu_session_message_deliveries (
                                delivery_id, session_id, interaction_id,
                                include_on_idle, requested_eligible_through,
                                eligible_through, batch_limit, has_more,
                                interaction_started_event, queue_ids, events,
                                created_at
                            )
                            VALUES (
                                %s, %s, %s, %s, %s, %s, %s, FALSE,
                                %s, '[]'::jsonb, '[]'::jsonb, %s
                            )
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
                                    else pg_support._dumps(
                                        interaction_started_event.model_dump(mode="json")
                                    )
                                ),
                                await self._session_store_now(cur),
                            ),
                        )
                        await cur.execute(
                            "UPDATE cayu_session_message_deliveries SET reject_only = %s WHERE delivery_id = %s",
                            (reject_only, delivery_id),
                        )
                        await conn.commit()
                        return SessionMessageDeliveryBatch(
                            delivery_id=delivery_id,
                            interaction_id=interaction_id,
                            eligible_through=boundary,
                            has_more=False,
                        )
                    transcript_cursor = await _transcript_cursor(cur, session_id)
                    delivered_at = scan_now if reject_only else await self._session_store_now(cur)
                    accepted_events = await self._session_message_acceptance_events(
                        cur,
                        session_id,
                        [_session_message_raw_row(row) for row in rows],
                    )
                    rejection_events: list[Event] = []
                    deliverable_rows = []
                    for row in rows:
                        queued = _queued_session_message_from_row(row)
                        rejection = session_message_rejection(
                            queued.conditions,
                            session_instance_id=loaded.instance_id,
                            run_epoch=loaded.run_epoch,
                            transcript_cursor=transcript_cursor + len(deliverable_rows),
                            now=delivered_at,
                        )
                        if rejection is None:
                            if not reject_only:
                                deliverable_rows.append(row)
                            continue
                        event = message_queue.terminal_event(
                            loaded,
                            _session_message_raw_row(row),
                            rejection,
                            delivered_at,
                            accepted_event=accepted_events.get(row[10]),
                            actor=queued.requested_by,
                            interaction_id=interaction_id,
                        )
                        rejection_events.append(event)
                        await cur.execute(
                            "UPDATE cayu_session_message_queue SET status = %s, terminal_json = %s "
                            "WHERE queue_id = %s AND status = 'queued'",
                            (
                                str(rejection),
                                pg_support._dumps(message_queue.terminal_receipt(rejection, event)),
                                queued.queue_id,
                            ),
                        )
                    rows = deliverable_rows
                    rebound_checkpoint: dict[str, Any] | None = None
                    if rows and profile_handoff is not None:
                        await cur.execute(
                            "SELECT record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = %s",
                            (
                                session_id,
                                _interaction_transition_storage_key(
                                    profile_handoff.predecessor_settlement_event_id
                                ),
                            ),
                        )
                        receipt_row = await cur.fetchone()
                        if receipt_row is None:
                            raise SessionRunFenced(
                                "Queued interaction handoff lost its predecessor "
                                "settlement receipt."
                            )
                        await self._reject_new_work_after_steering(
                            cur, loaded, allow_completed_interaction=True
                        )
                        rebound_checkpoint = _checkpoint_after_queued_interaction_profile_handoff(
                            loaded,
                            await self._load_checkpoint(cur, session_id),
                            profile_handoff,
                            settlement_record=pg_support._json_obj(receipt_row[0]),
                            replayed_delivery=False,
                        )
                    updated_messages: list[SessionQueuedMessage] = []
                    delivery_events: list[Event] = list(rejection_events)
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
                                    **message_queue.source_audit_payload(
                                        _session_message_raw_row(row), accepted_events.get(row[10])
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
                        updated_messages.append(
                            queued_message.model_copy(
                                update={
                                    "status": SessionMessageQueueStatus.DELIVERED,
                                    "delivered_run_epoch": loaded.run_epoch,
                                    "delivered_transcript_cursor": delivered_cursor,
                                    "delivered_event_id": delivery_event.id,
                                    "delivered_at": delivered_at,
                                },
                                deep=True,
                            )
                        )
                        delivery_events.append(delivery_event)
                        transcript_messages.append(delivered_message)
                    await cur.executemany(
                        "INSERT INTO cayu_transcript_messages "
                        "(session_id, interaction_id, message, "
                        "transcript_search_document) VALUES (%s, %s, %s, %s)",
                        [
                            (
                                session_id,
                                interaction_id,
                                pg_support._dumps(message.model_dump(mode="json")),
                                _postgres_transcript_index_document(session_id, message),
                            )
                            for message in transcript_messages
                        ],
                    )
                    await self._register_event_public_authorities(
                        cur,
                        session_id,
                        delivery_events,
                    )
                    await self._register_public_authorities(
                        cur,
                        session_id,
                        interaction_ids=(() if interaction_id is None else (interaction_id,)),
                    )
                    for updated in updated_messages:
                        await cur.execute(
                            "UPDATE cayu_session_message_queue SET status = 'delivered', "
                            "delivered_run_epoch = %s, delivered_transcript_cursor = %s, "
                            "delivered_event_id = %s, delivered_at = %s "
                            "WHERE queue_id = %s AND status = 'queued'",
                            (
                                updated.delivered_run_epoch,
                                updated.delivered_transcript_cursor,
                                updated.delivered_event_id,
                                delivered_at,
                                updated.queue_id,
                            ),
                        )
                    delivery_events.sort(key=lambda event: event.payload["ordering_key"])
                    persisted_events = [
                        *(
                            [interaction_started_event]
                            if updated_messages and interaction_started_event is not None
                            else []
                        ),
                        *delivery_events,
                    ]
                    await cur.execute(
                        "UPDATE cayu_sessions SET event_seq = event_seq + %s, "
                        "last_activity_at = %s WHERE id = %s RETURNING event_seq",
                        (len(persisted_events), delivered_at, session_id),
                    )
                    event_order_row = await cur.fetchone()
                    if event_order_row is None:
                        raise KeyError(f"Session not found: {session_id}")
                    next_order = event_order_row[0] - len(persisted_events)
                    event_rows = []
                    for event in persisted_events:
                        next_order += 1
                        lookup_key, projection, projection_bytes = (
                            pending_action_event_storage_values(event)
                        )
                        event_rows.append(
                            (
                                session_id,
                                next_order,
                                event.id,
                                event.interaction_id,
                                str(event.type),
                                event.timestamp,
                                event.agent_name,
                                event.environment_name,
                                event.workflow_name,
                                event.tool_name,
                                pg_support._dumps(event.payload),
                                pg_support._dumps(event.model_dump(mode="json")),
                                lookup_key,
                                projection,
                                projection_bytes,
                            )
                        )
                    await cur.executemany(
                        "INSERT INTO cayu_events (session_id, session_order, event_id, "
                        "interaction_id, event_type, timestamp, agent_name, "
                        "environment_name, workflow_name, "
                        "tool_name, payload, event, pending_action_lookup_key, "
                        "pending_action_projection, pending_action_projection_bytes) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        event_rows,
                    )
                    await self._enqueue_persisted_event_side_effects(
                        cur,
                        session_id,
                        persisted_events,
                    )
                    mode_clause = (
                        "delivery_mode IN ('next_turn', 'on_idle')"
                        if include_on_idle
                        else "delivery_mode = 'next_turn'"
                    )
                    await cur.execute(
                        "SELECT 1 FROM cayu_session_message_queue WHERE session_id = %s "
                        "AND status = 'queued' AND ordering_key <= %s "
                        f"AND {mode_clause} LIMIT 1",
                        (session_id, boundary),
                    )
                    remaining = await cur.fetchone()
                    has_more = reject_only_more if reject_only else remaining is not None
                    await cur.execute(
                        """
                        INSERT INTO cayu_session_message_deliveries (
                            delivery_id, session_id, interaction_id,
                            include_on_idle, requested_eligible_through,
                            eligible_through, batch_limit, has_more,
                            interaction_started_event, queue_ids, events,
                            created_at
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s
                        )
                        """,
                        (
                            delivery_id,
                            session_id,
                            interaction_id,
                            include_on_idle,
                            eligible_through,
                            boundary,
                            limit,
                            has_more,
                            (
                                None
                                if interaction_started_event is None
                                else pg_support._dumps(
                                    interaction_started_event.model_dump(mode="json")
                                )
                            ),
                            pg_support._dumps([message.queue_id for message in updated_messages]),
                            pg_support._dumps(
                                [event.model_dump(mode="json") for event in persisted_events]
                            ),
                            delivered_at,
                        ),
                    )
                    await cur.execute(
                        "UPDATE cayu_session_message_deliveries SET reject_only = %s WHERE delivery_id = %s",
                        (reject_only, delivery_id),
                    )
                    if rebound_checkpoint is not None:
                        await self._upsert_checkpoint(
                            cur,
                            session_id,
                            rebound_checkpoint,
                            delivered_at,
                        )
                await conn.commit()
                return SessionMessageDeliveryBatch(
                    messages=tuple(updated_messages),
                    events=tuple(persisted_events),
                    delivery_id=delivery_id,
                    interaction_id=interaction_id,
                    eligible_through=boundary,
                    has_more=has_more,
                    active_invocation_profile=(
                        None
                        if not updated_messages or profile_handoff is None
                        else profile_handoff.target_active_profile
                    ),
                )
            except Exception:
                await conn.rollback()
                raise

    async def repair_queued_interaction_profile_handoff(
        self,
        session_id: str,
        *,
        interaction_started_event: Event,
        profile_handoff: QueuedInteractionProfileHandoff,
    ) -> ActiveInvocationExecutionProfile:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_started_event, profile_handoff = (
            _copy_historical_queued_interaction_profile_handoff(
                session_id,
                interaction_started_event,
                profile_handoff,
            )
        )
        target = profile_handoff.target_active_profile
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded = await self._load_for_update(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    _assert_session_run_epoch(session_id, loaded)
                    await cur.execute(
                        "SELECT session_id, interaction_id, interaction_started_event, "
                        "queue_ids FROM cayu_session_message_deliveries "
                        "WHERE delivery_id = %s",
                        (target.interaction_id,),
                    )
                    delivery_row = await cur.fetchone()
                    stored_started_event = (
                        None
                        if delivery_row is None or delivery_row[2] is None
                        else Event(**pg_support._json_obj(delivery_row[2]))
                    )
                    if (
                        delivery_row is None
                        or delivery_row[0] != session_id
                        or delivery_row[1] != target.interaction_id
                        or stored_started_event != interaction_started_event
                        or not list(delivery_row[3])
                    ):
                        raise SessionRunFenced(
                            "Historical queued interaction handoff lacks its exact delivery receipt."
                        )
                    await cur.execute(
                        "SELECT record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s",
                        (
                            session_id,
                            _interaction_transition_storage_key(
                                profile_handoff.predecessor_settlement_event_id
                            ),
                        ),
                    )
                    receipt_row = await cur.fetchone()
                    if receipt_row is None:
                        raise SessionRunFenced(
                            "Historical queued interaction handoff lost its predecessor settlement."
                        )
                    await cur.execute(
                        "SELECT record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s",
                        (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                    )
                    active_row = await cur.fetchone()
                    stage_records: dict[str, Any] = {}
                    if active_row is not None:
                        active_record = _decode_model_completion_stage_record(active_row[0])
                        marker = _reconstruct_active_model_completion_stage_record(
                            active_record,
                            session_id=session_id,
                        )
                        _, _, preparation_key, terminal_key = (
                            _model_completion_stage_storage_identity(
                                session_id,
                                marker.stage_id,
                            )
                        )
                        dispatch_key = _model_completion_stage_dispatch_storage_key(marker.stage_id)
                        await cur.execute(
                            "SELECT idempotency_key, record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                            (
                                session_id,
                                [preparation_key, terminal_key, dispatch_key],
                            ),
                        )
                        stage_records = {
                            row[0]: _decode_model_completion_stage_record(row[1])
                            for row in await cur.fetchall()
                        }
                        stage_records[MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY] = active_record
                    active_model_stage, stage_dispatch = (
                        _historical_queued_handoff_stage_from_records(
                            session_id,
                            stage_records,
                        )
                    )
                    repaired_checkpoint = _checkpoint_after_queued_interaction_profile_handoff(
                        loaded,
                        await self._load_checkpoint(cur, session_id),
                        profile_handoff,
                        settlement_record=pg_support._json_obj(receipt_row[0]),
                        replayed_delivery=True,
                        active_model_stage=active_model_stage,
                        stage_dispatch=stage_dispatch,
                    )
                    await self._upsert_checkpoint(
                        cur,
                        session_id,
                        repaired_checkpoint,
                        await self._session_store_now(cur),
                    )
                await conn.commit()
                return target.model_copy(deep=True)
            except Exception:
                await conn.rollback()
                raise

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
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        idempotency_key = _reject_reserved_runtime_publication_key(
            idempotency_key,
            "idempotency_key",
            browser_control_read=True,
        )
        _require_session_export_target(session_id, idempotency_key)
        await self._ensure_ready()
        checkpoint_root_key = (
            "__cayu_no_checkpoint_root_guard__"
            if checkpoint_root_guard is None
            else checkpoint_root_guard.key
        )
        async with self._connection() as conn, conn.cursor() as cur:
            if access_bounds is not None:
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                access_bounds.require_read(await self._load(cur, session_id))
            await cur.execute(
                f"""
                SELECT
                    cayu_session_operations.record,
                    jsonb_typeof(cayu_checkpoints.state -> '{checkpoint_root_key}'),
                    left(
                        cayu_checkpoints.state ->> '{checkpoint_root_key}',
                        {CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS + 1}
                    )
                FROM cayu_sessions
                LEFT JOIN cayu_session_operations
                    ON cayu_session_operations.session_id = cayu_sessions.id
                    AND cayu_session_operations.idempotency_key = %s
                LEFT JOIN cayu_checkpoints
                    ON cayu_checkpoints.session_id = cayu_sessions.id
                WHERE cayu_sessions.id = %s
                """,
                (idempotency_key, session_id),
            )
            row = await cur.fetchone()
            if row is None:
                raise KeyError(f"Session not found: {session_id}")
            scalar_text = row[2]
            if checkpoint_root_guard is not None:
                checkpoint_root_guard.validate(
                    session_id,
                    checkpoint_root_field_projection_from_storage(
                        json_type=row[1],
                        scalar_text=scalar_text,
                    ),
                )
            return None if row[0] is None else pg_support._json_obj(row[0])

    async def _load_runtime_publication_receipt_record(
        self,
        session_id: str,
        storage_key: str,
        publication_id: str,
    ) -> dict[str, Any] | None:
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT record FROM cayu_session_operations "
                "WHERE session_id = %s AND idempotency_key = %s FOR SHARE",
                (session_id, storage_key),
            )
            row = await cur.fetchone()
            if row is not None:
                record = _decode_runtime_publication_record(row[0])
                receipt = _reconstruct_runtime_publication_receipt(
                    record,
                    storage_key=storage_key,
                    session_id=session_id,
                    publication_id=publication_id,
                )
                await self._validate_runtime_publication_material(cur, receipt)
                return record
            await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (session_id,))
            if await cur.fetchone() is None:
                raise KeyError(f"Session not found: {session_id}")
            return None

    async def _validate_runtime_publication_material(
        self,
        cur,
        receipt: RuntimePublicationReceipt,
        *,
        lock_events: bool = True,
    ) -> None:
        try:
            await cur.execute(
                "SELECT interaction_id, message FROM cayu_transcript_messages "
                "WHERE session_id = %s AND session_order > %s AND session_order <= %s "
                "ORDER BY session_order ASC",
                (
                    receipt.session_id,
                    receipt.transcript_start_cursor,
                    receipt.transcript_end_cursor,
                ),
            )
            transcript_rows = await cur.fetchall()
            transcript = [Message(**pg_support._json_obj(row[1])) for row in transcript_rows]
            transcript_interaction_ids = [row[0] for row in transcript_rows]

            referenced_event_ids = _runtime_publication_referenced_event_ids(
                receipt.referenced_events
            )
            requested_event_ids = tuple(
                dict.fromkeys((*receipt.appended_event_ids, *referenced_event_ids))
            )
            events_by_id: dict[str, Event] = {}
            if requested_event_ids:
                await cur.execute(
                    "SELECT event_id, event FROM cayu_events "
                    "WHERE session_id = %s AND event_id = ANY(%s)"
                    + (" FOR SHARE" if lock_events else ""),
                    (receipt.session_id, list(requested_event_ids)),
                )
                events_by_id = {
                    row[0]: Event(**pg_support._json_obj(row[1])) for row in await cur.fetchall()
                }
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
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT idempotency_key, record FROM cayu_session_operations "
                "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                (session_id, [preparation_storage_key, terminal_storage_key]),
            )
            records = {
                row[0]: _decode_model_completion_stage_record(row[1])
                for row in await cur.fetchall()
            }
            if records:
                return records.get(preparation_storage_key), records.get(terminal_storage_key)
            await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (session_id,))
            if await cur.fetchone() is None:
                raise KeyError(f"Session not found: {session_id}")
            return None, None

    async def _load_model_completion_stage_settlement_record(
        self,
        session_id: str,
        settlement_storage_key: str,
    ) -> dict[str, Any] | None:
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT record FROM cayu_session_operations "
                "WHERE session_id = %s AND idempotency_key = %s",
                (session_id, settlement_storage_key),
            )
            row = await cur.fetchone()
            if row is not None:
                return _decode_model_completion_stage_record(row[0])
            await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (session_id,))
            if await cur.fetchone() is None:
                raise KeyError(f"Session not found: {session_id}")
            return None

    async def _load_model_completion_stage_dispatch_record(
        self,
        session_id: str,
        dispatch_storage_key: str,
    ) -> dict[str, Any] | None:
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT record FROM cayu_session_operations "
                "WHERE session_id = %s AND idempotency_key = %s",
                (session_id, dispatch_storage_key),
            )
            row = await cur.fetchone()
            if row is not None:
                return _decode_model_completion_stage_record(row[0])
            await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (session_id,))
            if await cur.fetchone() is None:
                raise KeyError(f"Session not found: {session_id}")
            return None

    async def _load_active_model_completion_stage_records(
        self,
        session_id: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            await cur.execute(
                "SELECT 1 FROM cayu_sessions WHERE id = %s",
                (session_id,),
            )
            if await cur.fetchone() is None:
                raise KeyError(f"Session not found: {session_id}")
            await cur.execute(
                "SELECT record FROM cayu_session_operations "
                "WHERE session_id = %s AND idempotency_key = %s",
                (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
            )
            row = await cur.fetchone()
            if row is None:
                return None, None, None
            active_record = _decode_model_completion_stage_record(row[0])
            marker = _reconstruct_active_model_completion_stage_record(
                active_record,
                session_id=session_id,
            )
            _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
                session_id,
                marker.stage_id,
            )
            await cur.execute(
                "SELECT idempotency_key, record FROM cayu_session_operations "
                "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                (session_id, [preparation_key, terminal_key]),
            )
            records = {
                record_row[0]: _decode_model_completion_stage_record(record_row[1])
                for record_row in await cur.fetchall()
            }
            return (
                active_record,
                records.get(preparation_key),
                records.get(terminal_key),
            )

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
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded = await self._load_for_update(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    await cur.execute(
                        "SELECT idempotency_key, record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                        (
                            session_id,
                            [
                                MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                                preparation_key,
                                terminal_key,
                                settlement_key,
                                dispatch_key,
                            ],
                        ),
                    )
                    records = {
                        row[0]: _decode_model_completion_stage_record(row[1])
                        for row in await cur.fetchall()
                    }
                    _validate_model_completion_stage_for_dispatch(
                        session=loaded,
                        checkpoint=await self._load_checkpoint(cur, session_id),
                        current_transcript_cursor=await _transcript_cursor(cur, session_id),
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
                        await cur.execute(
                            "INSERT INTO cayu_session_operations "
                            "(session_id, idempotency_key, record, updated_at) "
                            "VALUES (%s, %s, %s, %s)",
                            (
                                session_id,
                                dispatch_key,
                                pg_support._dumps(dispatch_record),
                                published_at,
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
                            # Serialize canonical occurrence selection with child
                            # status changes and event appends, both of which
                            # update the child session row.
                            await cur.execute(
                                "SELECT 1 FROM cayu_sessions WHERE id = %s FOR SHARE",
                                (claim.child_session_id,),
                            )
                            child_exists = await cur.fetchone()
                            child = await self._load(cur, claim.child_session_id)
                            event_row = None
                            if child is not None:
                                await cur.execute(
                                    "SELECT sequence, event FROM cayu_events "
                                    "WHERE session_id = %s AND event_type = ANY(%s) "
                                    "ORDER BY sequence DESC LIMIT 1 FOR SHARE",
                                    (
                                        claim.child_session_id,
                                        [
                                            str(EventType.SESSION_STARTED),
                                            str(EventType.SESSION_RESUMED),
                                            str(EventType.SESSION_FORKED),
                                            str(EventType.SESSION_COMPLETED),
                                            str(EventType.SESSION_FAILED),
                                            str(EventType.SESSION_INTERRUPTED),
                                        ],
                                    ),
                                )
                                event_row = await cur.fetchone()
                            if child_exists is None or child is None or event_row is None:
                                raise SessionModelCompletionStageConflict(
                                    "Child-session notification occurrence is no longer canonical."
                                )
                            event = Event(**pg_support._json_obj(event_row[1]))
                            occurrence = ChildSessionLifecycleOccurrence(
                                source=ChildSessionLifecycleOccurrenceSource.EVENT,
                                source_id=event.id,
                                source_sequence=event_row[0],
                                source_type=str(event.type),
                                occurred_at=event.timestamp,
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
                            await cur.execute(
                                "SELECT record FROM cayu_session_operations "
                                "WHERE session_id = %s AND idempotency_key = %s FOR UPDATE",
                                (session_id, consumption_key),
                            )
                            consumption_row = await cur.fetchone()
                            material = consumption.model_dump(mode="json")
                            if consumption_row is not None:
                                if not _child_session_notification_consumption_replays(
                                    pg_support._json_obj(consumption_row[0]),
                                    consumption,
                                ):
                                    raise SessionModelCompletionStageConflict(
                                        "Child-session terminal notification was consumed by "
                                        "another stage."
                                    )
                            elif consume_child_session_notifications:
                                await cur.execute(
                                    "INSERT INTO cayu_session_operations "
                                    "(session_id, idempotency_key, record, updated_at) "
                                    "VALUES (%s, %s, %s, %s)",
                                    (
                                        session_id,
                                        consumption_key,
                                        pg_support._dumps(material),
                                        published_at,
                                    ),
                                )
                    await cur.execute(
                        "UPDATE cayu_sessions SET updated_at = %s, last_activity_at = %s "
                        "WHERE id = %s",
                        (published_at, published_at, session_id),
                    )
                    if cur.rowcount != 1:
                        raise KeyError(f"Session not found: {session_id}")
                await conn.commit()
                return dispatch
            except BaseException:
                await conn.rollback()
                raise

    async def _prepare_model_completion_stage_atomic(
        self,
        prepared: _PreparedModelCompletionStage,
    ) -> ModelCompletionStageResult:
        session_id = prepared.session_id
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded = await self._load_for_update(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    await cur.execute(
                        "SELECT idempotency_key, record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                        (
                            session_id,
                            [
                                prepared.preparation_storage_key,
                                prepared.terminal_storage_key,
                                prepared.abandonment_storage_key,
                                *_model_failover_predecessor_storage_keys(prepared),
                            ],
                        ),
                    )
                    records = {
                        row[0]: _decode_model_completion_stage_record(row[1])
                        for row in await cur.fetchall()
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

                    await cur.execute(
                        "SELECT record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s",
                        (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                    )
                    active_row = await cur.fetchone()
                    active = None
                    if active_row is not None:
                        active_record = _decode_model_completion_stage_record(active_row[0])
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
                        await cur.execute(
                            "SELECT idempotency_key, record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                            (
                                session_id,
                                [active_preparation_key, active_terminal_key],
                            ),
                        )
                        active_records = {
                            row[0]: _decode_model_completion_stage_record(row[1])
                            for row in await cur.fetchall()
                        }
                        active = _reconstruct_active_model_completion_stage(
                            active_record,
                            active_records.get(active_preparation_key),
                            active_records.get(active_terminal_key),
                            session_id=session_id,
                        )
                    await cur.execute(
                        "SELECT idempotency_key FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                        (
                            session_id,
                            [
                                prepared.winner_storage_key,
                                prepared.publication_storage_key,
                            ],
                        ),
                    )
                    publication_keys = {row[0] for row in await cur.fetchall()}
                    winner_exists = prepared.winner_storage_key in publication_keys
                    receipt_exists = prepared.publication_storage_key in publication_keys
                    if stage is not None:
                        _validate_model_completion_preparation_replay_state(
                            stage,
                            active=active,
                            winner_exists=winner_exists,
                            receipt_exists=receipt_exists,
                        )
                        _model_failover_preparation_checkpoint(
                            prepared,
                            session=loaded,
                            checkpoint=await self._load_checkpoint(cur, session_id),
                            current_transcript_cursor=await _transcript_cursor(cur, session_id),
                            active=active,
                            records=records,
                            replayed=True,
                        )
                        expected_selection = _model_failover_selection_event(
                            prepared, session=loaded, prepared_at=stage.prepared_at
                        )
                        if expected_selection is not None:
                            await cur.execute(
                                "SELECT event FROM cayu_events WHERE session_id = %s AND event_id = %s",
                                (session_id, expected_selection.id),
                            )
                            event_row = await cur.fetchone()
                            _validate_model_failover_selection_replay(
                                expected_selection,
                                None
                                if event_row is None
                                else Event(**pg_support._json_obj(event_row[0])),
                            )
                        await conn.rollback()
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
                    current_cursor = await _transcript_cursor(cur, session_id)
                    if current_cursor != prepared.expected_transcript_cursor:
                        raise ValueError(
                            "Session source transcript cursor is stale: expected "
                            f"{prepared.expected_transcript_cursor}, current {current_cursor}."
                        )

                    if active is None:
                        await self._reject_new_work_after_steering(cur, loaded)
                    route_checkpoint = _model_failover_preparation_checkpoint(
                        prepared,
                        session=loaded,
                        checkpoint=await self._load_checkpoint(cur, session_id),
                        current_transcript_cursor=current_cursor,
                        active=active,
                        records=records,
                        replayed=False,
                    )
                    prepared_at = _next_runtime_publication_timestamp(loaded)
                    if route_checkpoint is not None:
                        await self._upsert_checkpoint(
                            cur, session_id, route_checkpoint, prepared_at
                        )
                    record = _model_completion_stage_preparation_record(
                        prepared,
                        source_session=loaded,
                        prepared_at=prepared_at,
                    )
                    selection_event = _model_failover_selection_event(
                        prepared, session=loaded, prepared_at=prepared_at
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
                    await cur.execute(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record, updated_at) "
                        "VALUES (%s, %s, %s, %s)",
                        (
                            session_id,
                            prepared.preparation_storage_key,
                            pg_support._dumps(record),
                            prepared_at,
                        ),
                    )
                    active_record = _active_model_completion_stage_record(
                        stage,
                        activated_at=prepared_at,
                    )
                    if retry_settlement_request is not None:
                        assert active is not None
                        retry_settlement_storage_key = (
                            _model_completion_stage_settlement_storage_key(active.stage.stage_id)
                        )
                        await cur.execute(
                            "SELECT record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = %s",
                            (session_id, retry_settlement_storage_key),
                        )
                        settlement_row = await cur.fetchone()
                        _validate_model_completion_stage_for_settlement(
                            session=loaded,
                            stage=active.stage,
                            active=active,
                            request=retry_settlement_request,
                            settlement_record=(
                                None
                                if settlement_row is None
                                else _decode_model_completion_stage_record(settlement_row[0])
                            ),
                            winner_exists=winner_exists,
                            receipt_exists=receipt_exists,
                        )
                        retry_settlement_record = _model_completion_stage_settlement_record(
                            active.stage,
                            request=retry_settlement_request,
                            settled_at=prepared_at,
                        )
                        await cur.execute(
                            "INSERT INTO cayu_session_operations "
                            "(session_id, idempotency_key, record, updated_at) "
                            "VALUES (%s, %s, %s, %s)",
                            (
                                session_id,
                                retry_settlement_storage_key,
                                pg_support._dumps(retry_settlement_record),
                                prepared_at,
                            ),
                        )
                    await cur.execute(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record, updated_at) "
                        "VALUES (%s, %s, %s, %s) "
                        "ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                        "record = EXCLUDED.record, updated_at = EXCLUDED.updated_at",
                        (
                            session_id,
                            MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                            pg_support._dumps(active_record),
                            prepared_at,
                        ),
                    )
                    await cur.execute(
                        "UPDATE cayu_sessions SET updated_at = %s, last_activity_at = %s "
                        "WHERE id = %s",
                        (prepared_at, prepared_at, session_id),
                    )
                    if selection_event is not None:
                        await self._append_events_with_cursor(
                            cur,
                            session_id,
                            (selection_event,),
                            expected_run_epoch=prepared.expected_run_epoch,
                        )
                await conn.commit()
                return ModelCompletionStageResult(
                    stage=stage,
                    replayed=False,
                    dispatch_authorized=True,
                    prepared_events=() if selection_event is None else (selection_event,),
                )
            except BaseException:
                await conn.rollback()
                raise

    async def _complete_model_completion_stage_atomic(
        self,
        prepared: _PreparedModelCompletionStageTerminal,
    ) -> ModelCompletionStageResult:
        session_id = prepared.session_id
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded = await self._load_for_update(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    await cur.execute(
                        "SELECT idempotency_key, record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                        (
                            session_id,
                            [
                                prepared.preparation_storage_key,
                                prepared.terminal_storage_key,
                                prepared.settlement_storage_key,
                            ],
                        ),
                    )
                    records = {
                        row[0]: _decode_model_completion_stage_record(row[1])
                        for row in await cur.fetchall()
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
                        await conn.rollback()
                        return ModelCompletionStageResult(
                            stage=stage,
                            replayed=True,
                            dispatch_authorized=False,
                        )
                    if prepared.recovery_fence is not None:
                        await cur.execute(
                            "SELECT record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = %s",
                            (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                        )
                        active_row = await cur.fetchone()
                        await cur.execute(
                            "SELECT record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = %s",
                            (
                                session_id,
                                _model_completion_stage_dispatch_storage_key(stage.stage_id),
                            ),
                        )
                        dispatch_row = await cur.fetchone()
                        _validate_model_completion_stage_recovery_fence(
                            prepared.recovery_fence,
                            session=loaded,
                            checkpoint=await self._load_checkpoint(cur, session_id),
                            stage=stage,
                            active_record=(
                                None
                                if active_row is None
                                else _decode_model_completion_stage_record(active_row[0])
                            ),
                            dispatch_record=(
                                None
                                if dispatch_row is None
                                else _decode_model_completion_stage_record(dispatch_row[0])
                            ),
                            now=await self._session_store_now(cur),
                        )
                    _validate_model_completion_stage_publication(
                        prepared.publication,
                        session_id=session_id,
                        stage=stage,
                    )
                    if not _runtime_publication_json_equal(
                        prepared.publication.intent,
                        stage.intent,
                    ):
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
                    await cur.execute(
                        "SELECT record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s",
                        (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                    )
                    active_row = await cur.fetchone()
                    active_record = (
                        None
                        if active_row is None
                        else _decode_model_completion_stage_record(active_row[0])
                    )
                    advances_last_activity = _model_completion_terminal_advances_last_activity(
                        active_record,
                        stage=stage,
                        current_run_epoch=loaded.run_epoch,
                    )
                    await cur.execute(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record, updated_at) "
                        "VALUES (%s, %s, %s, %s)",
                        (
                            session_id,
                            prepared.terminal_storage_key,
                            pg_support._dumps(terminal_record),
                            completed_at,
                        ),
                    )
                    await cur.execute(
                        "UPDATE cayu_sessions SET updated_at = %s, "
                        "last_activity_at = CASE WHEN %s THEN %s ELSE last_activity_at END "
                        "WHERE id = %s",
                        (
                            completed_at,
                            advances_last_activity,
                            completed_at,
                            session_id,
                        ),
                    )
                await conn.commit()
                return ModelCompletionStageResult(
                    stage=completed_stage,
                    replayed=False,
                    dispatch_authorized=False,
                )
            except BaseException:
                await conn.rollback()
                raise

    async def _abandon_model_completion_stage_atomic(
        self,
        prepared: _PreparedModelCompletionStageAbandonment,
    ) -> ModelCompletionStageAbandonmentResult:
        session_id = prepared.session_id
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded = await self._load_for_update(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    await cur.execute(
                        "SELECT idempotency_key, record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                        (
                            session_id,
                            [
                                prepared.preparation_storage_key,
                                prepared.terminal_storage_key,
                                prepared.abandonment_storage_key,
                            ],
                        ),
                    )
                    records = {
                        row[0]: _decode_model_completion_stage_record(row[1])
                        for row in await cur.fetchall()
                    }
                    stage = _reconstruct_model_completion_stage(
                        records.get(prepared.preparation_storage_key),
                        records.get(prepared.terminal_storage_key),
                        session_id=session_id,
                        stage_id=prepared.stage_id,
                        preparation_storage_key=prepared.preparation_storage_key,
                        terminal_storage_key=prepared.terminal_storage_key,
                    )
                    await cur.execute(
                        "SELECT record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s",
                        (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                    )
                    active_row = await cur.fetchone()
                    active_record = (
                        None
                        if active_row is None
                        else _decode_model_completion_stage_record(active_row[0])
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
                        await cur.execute(
                            "SELECT idempotency_key FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                            (
                                session_id,
                                [
                                    _model_completion_stage_winner_storage_key(
                                        replayed.abandonment.logical_step_id
                                    ),
                                    _runtime_publication_storage_key(
                                        replayed.abandonment.logical_step_id
                                    ),
                                ],
                            ),
                        )
                        if await cur.fetchone() is not None:
                            raise SessionModelCompletionStageConflict(
                                "An abandoned model-completion stage has durable publication state."
                            )
                        await conn.rollback()
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
                    publication_storage_key = _runtime_publication_storage_key(
                        stage.logical_step_id
                    )
                    dispatch_storage_key = _model_completion_stage_dispatch_storage_key(
                        stage.stage_id
                    )
                    await cur.execute(
                        "SELECT idempotency_key FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                        (
                            session_id,
                            [winner_storage_key, publication_storage_key, dispatch_storage_key],
                        ),
                    )
                    publication_keys = {row[0] for row in await cur.fetchall()}
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
                    await cur.execute(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record, updated_at) "
                        "VALUES (%s, %s, %s, %s) "
                        "ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                        "record = EXCLUDED.record, updated_at = EXCLUDED.updated_at",
                        (
                            session_id,
                            prepared.abandonment_storage_key,
                            pg_support._dumps(abandonment_record),
                            abandoned_at,
                        ),
                    )
                    await cur.execute(
                        "DELETE FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s "
                        "AND record->>'record_digest' = %s",
                        (
                            session_id,
                            prepared.preparation_storage_key,
                            prepared.preparation_digest,
                        ),
                    )
                    if cur.rowcount != 1:
                        raise SessionModelCompletionStageConflict(
                            "The model-completion preparation changed during abandonment."
                        )
                    await cur.execute(
                        "DELETE FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s "
                        "AND record->>'record_digest' = %s "
                        "AND record->>'preparation_digest' = %s",
                        (
                            session_id,
                            MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                            active.marker_digest,
                            prepared.preparation_digest,
                        ),
                    )
                    if cur.rowcount != 1:
                        raise SessionModelCompletionStageConflict(
                            "The active model-completion marker changed during abandonment."
                        )
                    await cur.execute(
                        "UPDATE cayu_sessions SET updated_at = %s, last_activity_at = %s "
                        "WHERE id = %s",
                        (abandoned_at, abandoned_at, session_id),
                    )
                    if cur.rowcount != 1:
                        raise KeyError(f"Session not found: {session_id}")
                await conn.commit()
                return ModelCompletionStageAbandonmentResult(
                    abandonment=abandonment,
                    replayed=False,
                )
            except BaseException:
                await conn.rollback()
                raise

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
        from cayu.sessions.pending_actions import (
            pending_action_event_storage_values,
            pending_action_lookup_key,
        )

        session_id = prepared.session_id
        request = prepared.request
        checkpoint_codec = prepared.checkpoint_codec
        checkpoint_decode = None if checkpoint_codec is None else checkpoint_codec.decode
        published_at: datetime
        receipt: RuntimePublicationReceipt
        loaded: Session

        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded_result = await self._load_for_update(cur, session_id)
                    if loaded_result is None:
                        raise KeyError(f"Session not found: {session_id}")
                    loaded = loaded_result

                    locked_stage = None
                    active_record = None
                    winner_record = None
                    if _model_completion_stage is not None:
                        await cur.execute(
                            "SELECT idempotency_key, record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                            (
                                session_id,
                                [
                                    _model_completion_stage.preparation_storage_key,
                                    _model_completion_stage.terminal_storage_key,
                                    _model_completion_stage.active_storage_key,
                                    _model_completion_stage.winner_storage_key,
                                ],
                            ),
                        )
                        stage_records = {
                            row[0]: _decode_model_completion_stage_record(row[1])
                            for row in await cur.fetchall()
                        }
                        locked_stage = _reconstruct_model_completion_stage(
                            stage_records.get(_model_completion_stage.preparation_storage_key),
                            stage_records.get(_model_completion_stage.terminal_storage_key),
                            session_id=session_id,
                            stage_id=_model_completion_stage.stage_id,
                            preparation_storage_key=(
                                _model_completion_stage.preparation_storage_key
                            ),
                            terminal_storage_key=_model_completion_stage.terminal_storage_key,
                        )
                        if locked_stage is None:
                            raise KeyError(
                                "Model-completion stage not found: "
                                f"{_model_completion_stage.stage_id}"
                            )
                        if (
                            locked_stage.completion_digest
                            != _model_completion_stage.completion_digest
                            or locked_stage.publication is None
                            or not _runtime_publication_json_equal(
                                locked_stage.publication.model_dump(mode="json"),
                                prepared.request.model_dump(mode="json"),
                            )
                        ):
                            raise SessionModelCompletionStageConflict(
                                "Model-completion stage changed before atomic promotion."
                            )
                        active_record = stage_records.get(
                            _model_completion_stage.active_storage_key
                        )
                        winner_record = stage_records.get(
                            _model_completion_stage.winner_storage_key
                        )

                    await cur.execute(
                        "SELECT record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s",
                        (session_id, prepared.storage_key),
                    )
                    receipt_row = await cur.fetchone()
                    if receipt_row is not None:
                        receipt_record = _decode_runtime_publication_record(receipt_row[0])
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
                                await cur.execute(
                                    "SELECT idempotency_key, record "
                                    "FROM cayu_session_operations "
                                    "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                                    (
                                        session_id,
                                        [active_preparation_key, active_terminal_key],
                                    ),
                                )
                                active_records = {
                                    row[0]: _decode_model_completion_stage_record(row[1])
                                    for row in await cur.fetchall()
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
                            await self._validate_runtime_publication_material(cur, receipt)
                            result = _replay_promoted_model_completion_stage(
                                session=loaded,
                                stage=locked_stage,
                                receipt_record=receipt_record,
                                winner_record=winner_record,
                            )
                            await conn.rollback()
                            return result
                        receipt = _reconstruct_runtime_publication_receipt(
                            receipt_record,
                            storage_key=prepared.storage_key,
                            session_id=session_id,
                            publication_id=request.publication_id,
                            request_digest=prepared.request_digest,
                        )
                        _validate_runtime_publication_replay_receipt(receipt, prepared)
                        await self._validate_runtime_publication_material(cur, receipt)
                        await conn.rollback()
                        return RuntimePublicationResult(
                            session=loaded.model_copy(deep=True),
                            receipt=receipt,
                            replayed=True,
                        )

                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))

                    operation_mutation_records: dict[str, dict[str, Any]] = {}
                    if request.operation_record_mutations:
                        mutation_keys = [
                            mutation.key for mutation in request.operation_record_mutations
                        ]
                        await cur.execute(
                            "SELECT idempotency_key, record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = ANY(%s) FOR UPDATE",
                            (session_id, mutation_keys),
                        )
                        current_mutation_records = {
                            row[0]: pg_support._json_obj(row[1]) for row in await cur.fetchall()
                        }
                        operation_mutation_records = (
                            _apply_runtime_publication_operation_record_mutations(
                                request.operation_record_mutations,
                                current_mutation_records,
                            )
                        )

                    if request.argument_continuity is not None:
                        from cayu.sessions._argument_continuity import STORAGE_KEY, append_record

                        await cur.execute(
                            "SELECT record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = %s FOR UPDATE",
                            (session_id, STORAGE_KEY),
                        )
                        private_row = await cur.fetchone()
                        operation_mutation_records[STORAGE_KEY] = append_record(
                            None if private_row is None else pg_support._json_obj(private_row[0]),
                            continuity=request.argument_continuity,
                            request_digest=prepared.request_digest,
                            session=loaded,
                            messages=request.transcript_messages,
                        )

                    if locked_stage is not None:
                        assert _model_completion_stage is not None
                        if winner_record is not None:
                            raise SessionModelCompletionStageConflict(
                                "A model-completion winner exists without its runtime "
                                "publication receipt."
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
                            "Session status is not eligible for runtime publication: "
                            f"{loaded.status}"
                        )
                    if (
                        prepared.expected_run_epoch is not None
                        and loaded.run_epoch != prepared.expected_run_epoch
                    ):
                        raise SessionRunFenced(
                            "Session source run epoch is stale: expected "
                            f"{prepared.expected_run_epoch}, current {loaded.run_epoch}."
                        )
                    transcript_start_cursor = await _transcript_cursor(cur, session_id)
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
                        raise ValueError(
                            "Appended and referenced runtime publication events overlap."
                        )
                    durable_references: dict[str, Event] = {}
                    if referenced_event_ids:
                        await cur.execute(
                            "SELECT event_id, event FROM cayu_events "
                            "WHERE session_id = %s AND event_id = ANY(%s) FOR SHARE",
                            (session_id, list(referenced_event_ids)),
                        )
                        durable_references = {
                            row[0]: Event(**pg_support._json_obj(row[1]))
                            for row in await cur.fetchall()
                        }
                    _validate_runtime_publication_event_references(
                        request.referenced_events,
                        durable_references,
                        interaction_id=request.interaction_id,
                    )
                    stored_checkpoint = await self._load_checkpoint(cur, session_id)
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
                        durable_events_by_id=durable_references,
                    )
                    _validate_tool_round_checkpoint_mutation(
                        request,
                        current_checkpoint,
                    )
                    durable_tool_events: list[Event] = []
                    tool_round_identity = _tool_lifecycle_publication_identity(request)
                    if tool_round_identity is not None:
                        execution_identity, tool_call_ids = tool_round_identity
                        lookup_keys = [
                            pending_action_lookup_key(tool_call_id)
                            for tool_call_id in tool_call_ids
                        ]
                        lifecycle_event_types = [
                            str(event_type)
                            for event_type in sorted(
                                _TOOL_ROUND_LIFECYCLE_EVENT_TYPES,
                                key=str,
                            )
                        ]
                        await cur.execute(
                            "SELECT event_id, event FROM cayu_events "
                            "WHERE session_id = %s "
                            "AND pending_action_lookup_key = ANY(%s) "
                            "AND event_type = ANY(%s) "
                            f"AND ({_PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
                            "AND (event -> 'payload' ->> 'tool_round_id' = %s "
                            "OR (event -> 'payload' ->> 'model_step_id' = %s "
                            "AND event -> 'payload' ->> 'model_attempt_id' = %s) "
                            "OR ((event -> 'payload' ->> 'tool_round_id') "
                            "~ '^tround_[0-9a-f]{32}$') IS NOT TRUE "
                            "OR ((event -> 'payload' ->> 'model_step_id') "
                            "~ '^mstep_[0-9a-f]{32}$') IS NOT TRUE "
                            "OR ((event -> 'payload' ->> 'model_attempt_id') "
                            "~ '^matt_[0-9a-f]{32}$') IS NOT TRUE) "
                            "ORDER BY session_order ASC LIMIT %s FOR SHARE",
                            (
                                session_id,
                                lookup_keys,
                                lifecycle_event_types,
                                execution_identity.tool_round_id,
                                execution_identity.model_step_id,
                                execution_identity.model_attempt_id,
                                _tool_round_lifecycle_event_limit(tool_call_ids) + 1,
                            ),
                        )
                        rows = await cur.fetchall()
                        if len(rows) > _tool_round_lifecycle_event_limit(tool_call_ids):
                            raise ValueError(
                                "Tool-round lifecycle evidence exceeds the publication limit."
                            )
                        durable_tool_events = [
                            Event(**pg_support._json_obj(row[1])) for row in rows
                        ]
                    _validate_tool_round_publication(
                        request,
                        durable_references,
                        durable_tool_events=durable_tool_events,
                    )
                    if request.events:
                        await cur.execute(
                            "SELECT event_id FROM cayu_events "
                            "WHERE session_id = %s AND event_id = ANY(%s)",
                            (session_id, [event.id for event in request.events]),
                        )
                        existing_event_row = await cur.fetchone()
                        if existing_event_row is not None:
                            raise ValueError(
                                f"Event already exists for session {session_id}: "
                                f"{existing_event_row[0]}"
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
                            request.interaction_id,
                            pg_support._dumps(message_payload),
                            _postgres_transcript_index_document(session_id, message),
                        )
                        for message, message_payload in zip(
                            request.transcript_messages,
                            prepared.transcript_payloads,
                            strict=True,
                        )
                    ]
                    prepared_event_rows = []
                    for event, event_payload in zip(
                        request.events,
                        prepared.event_payloads,
                        strict=True,
                    ):
                        lookup_key, projection, projection_bytes = (
                            pending_action_event_storage_values(event)
                        )
                        prepared_event_rows.append(
                            (
                                session_id,
                                event.id,
                                event.interaction_id,
                                str(event.type),
                                pg_support.to_utc(event.timestamp),
                                event.agent_name,
                                event.environment_name,
                                event.workflow_name,
                                event.tool_name,
                                pg_support._dumps(event_payload["payload"]),
                                pg_support._dumps(event_payload),
                                lookup_key,
                                projection,
                                projection_bytes,
                            )
                        )

                    published_at = _next_runtime_publication_timestamp(loaded)
                    checkpoint_values = (
                        None
                        if stored_target_checkpoint is None or not request.mutation.operations
                        else _checkpoint_row_values(
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
                    receipt_record = _runtime_publication_receipt_record(receipt)
                    receipt_json = pg_support._dumps(receipt_record)

                    await self._register_event_public_authorities(
                        cur,
                        session_id,
                        request.events,
                    )
                    await self._register_public_authorities(
                        cur,
                        session_id,
                        interaction_ids=(
                            () if request.interaction_id is None else (request.interaction_id,)
                        ),
                    )
                    await cur.execute(
                        """
                        UPDATE cayu_sessions
                        SET event_seq = event_seq + %s,
                            updated_at = %s,
                            last_activity_at = %s
                        WHERE id = %s
                        RETURNING event_seq
                        """,
                        (len(request.events), published_at, published_at, session_id),
                    )
                    order_row = await cur.fetchone()
                    if order_row is None:
                        raise KeyError(f"Session not found: {session_id}")
                    if transcript_rows:
                        await cur.executemany(
                            """
                            INSERT INTO cayu_transcript_messages (
                                session_id, interaction_id, message,
                                transcript_search_document
                            )
                            VALUES (%s, %s, %s, %s)
                            """,
                            transcript_rows,
                        )
                    if checkpoint_values is not None:
                        await cur.execute(
                            """
                            INSERT INTO cayu_checkpoints (
                                session_id, state, updated_at,
                                pending_action_source_bytes,
                                pending_action_tool_call_count,
                                pending_action_flags,
                                pending_action_metrics_ready
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (session_id) DO UPDATE SET
                                state = EXCLUDED.state,
                                updated_at = EXCLUDED.updated_at,
                                pending_action_source_bytes = EXCLUDED.pending_action_source_bytes,
                                pending_action_tool_call_count = EXCLUDED.pending_action_tool_call_count,
                                pending_action_flags = EXCLUDED.pending_action_flags,
                                pending_action_metrics_ready = EXCLUDED.pending_action_metrics_ready
                            """,
                            checkpoint_values,
                        )

                    next_order = order_row[0] - len(prepared_event_rows)
                    event_rows = []
                    for row in prepared_event_rows:
                        next_order += 1
                        event_rows.append((row[0], next_order, *row[1:]))
                    if event_rows:
                        await cur.executemany(
                            """
                            INSERT INTO cayu_events (
                                session_id, session_order, event_id, interaction_id,
                                event_type, timestamp,
                                agent_name, environment_name, workflow_name, tool_name,
                                payload, event, pending_action_lookup_key,
                                pending_action_projection, pending_action_projection_bytes
                            )
                            VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s
                            )
                            """,
                            event_rows,
                        )
                        await self._enqueue_persisted_event_side_effects(
                            cur,
                            session_id,
                            request.events,
                        )
                    if operation_mutation_records:
                        await cur.executemany(
                            "INSERT INTO cayu_session_operations "
                            "(session_id, idempotency_key, record, updated_at) "
                            "VALUES (%s, %s, %s, %s) "
                            "ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                            "record = excluded.record, updated_at = excluded.updated_at",
                            [
                                (session_id, key, pg_support._dumps(record), published_at)
                                for key, record in operation_mutation_records.items()
                            ],
                        )
                    await cur.execute(
                        """
                        INSERT INTO cayu_session_operations (
                            session_id, idempotency_key, record, updated_at
                        )
                        VALUES (%s, %s, %s, %s)
                        """,
                        (
                            session_id,
                            prepared.storage_key,
                            receipt_json,
                            published_at,
                        ),
                    )
                    if locked_stage is not None and _model_completion_stage is not None:
                        winner = _model_completion_stage_winner_record(
                            locked_stage,
                            receipt=receipt,
                        )
                        await cur.execute(
                            "INSERT INTO cayu_session_operations "
                            "(session_id, idempotency_key, record, updated_at) "
                            "VALUES (%s, %s, %s, %s)",
                            (
                                session_id,
                                _model_completion_stage.winner_storage_key,
                                pg_support._dumps(winner),
                                published_at,
                            ),
                        )
                        await cur.execute(
                            "DELETE FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = %s",
                            (session_id, _model_completion_stage.active_storage_key),
                        )
                        if cur.rowcount != 1:
                            raise SessionModelCompletionStageConflict(
                                "The active model-completion marker changed before commit."
                            )
                await conn.commit()
            except UniqueViolation as exc:
                await conn.rollback()
                existing = None
                async with conn.cursor() as cur:
                    for event in request.events:
                        await cur.execute(
                            "SELECT 1 FROM cayu_events WHERE session_id = %s AND event_id = %s",
                            (session_id, event.id),
                        )
                        if await cur.fetchone() is not None:
                            existing = event.id
                            break
                    await cur.execute(
                        "SELECT 1 FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s",
                        (session_id, prepared.storage_key),
                    )
                    receipt_exists = await cur.fetchone() is not None
                if existing is not None:
                    raise ValueError(
                        f"Event already exists for session {session_id}: {existing}"
                    ) from exc
                if receipt_exists:
                    raise SessionRuntimePublicationConflict(
                        "Runtime publication receipt was inserted concurrently."
                    ) from exc
                raise
            except BaseException:
                await conn.rollback()
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

    async def _prepare_temporary_side_service(self, preparation):
        from cayu.sessions._side_service_preparation import (
            SidePreparationSnapshot,
            plan_preparation,
            preparation_record_keys,
            prepare_selection,
        )

        prepared = prepare_selection(preparation)
        keys = preparation_record_keys(prepared)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    snapshots = {}
                    # Every pair takes the same stable row-lock order. Single-
                    # session admission/deletion uses these same session rows.
                    for session_id in sorted(keys):
                        session = await self._load_for_update(cur, session_id)
                        if session is None:
                            raise KeyError("Side-service session is unavailable.")
                        _assert_session_run_epoch(session_id, session)
                        for owner in await self._closure_lineage_owners(cur, (session_id,)):
                            _check_closure_lineage_owner(owner, (session_id,))
                        await cur.execute(
                            "SELECT idempotency_key, record FROM cayu_session_operations WHERE session_id = %s AND idempotency_key = ANY(%s)",
                            (session_id, list(keys[session_id])),
                        )
                        records = {
                            row[0]: pg_support._json_obj(row[1]) for row in await cur.fetchall()
                        }
                        snapshots[session_id] = SidePreparationSnapshot(
                            session, await self._load_checkpoint(cur, session_id), records
                        )
                    now = await self._session_store_now(cur)
                    plans = plan_preparation(prepared, snapshots, now)
                    for session_id, plan in plans.items():
                        await self._upsert_checkpoint(cur, session_id, plan.checkpoint, now)
                        await cur.executemany(
                            "INSERT INTO cayu_session_operations (session_id, idempotency_key, record, updated_at) "
                            "VALUES (%s, %s, %s, %s) ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                            "record=excluded.record, updated_at=excluded.updated_at",
                            [
                                (session_id, key, pg_support._dumps(record), now)
                                for key, record in plan.operation_records.items()
                            ],
                        )
                        await cur.execute(
                            "UPDATE cayu_sessions SET updated_at=%s, last_activity_at=%s WHERE id=%s",
                            (now, now, session_id),
                        )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

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
        context_view_compaction_cursor: int | None = None,
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
            context_view_compaction_cursor=context_view_compaction_cursor,
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
        context_view_compaction_cursor: int | None = None,
        preserve_completion_result_publications: bool = True,
    ) -> Session:
        from cayu.sessions.pending_actions import pending_action_event_storage_values

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
        from cayu.collaboration import _session_export_store as session_exports

        if operation_idempotency_key is not None:
            session_exports.require_operation_key_access(operation_idempotency_key, read=False)
            _require_session_export_target(session_id, operation_idempotency_key)
        allowed_statuses = (
            None
            if expected_statuses is None
            else _validate_status_set(expected_statuses, "expected_statuses")
        )
        await self._ensure_ready()
        async with self._connection() as conn:
            commit_guard_signal: BaseException | None = None
            try:
                async with conn.cursor() as cur:
                    if context_view_compaction_cursor is not None:
                        await cur.execute(
                            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                            (f"context-view-session:{session_id}",),
                        )
                    loaded = await self._load_for_update(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    updated_at = await self._session_store_now(cur)
                    _assert_session_run_epoch(session_id, loaded)
                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
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
                    current_cursor = await _transcript_cursor(cur, session_id)
                    if (
                        expected_transcript_cursor is not None
                        and current_cursor != expected_transcript_cursor
                    ):
                        raise ValueError(
                            "Session source transcript cursor is stale: expected "
                            f"{expected_transcript_cursor}, current {current_cursor}."
                        )
                    if context_view_compaction_cursor is not None:
                        await self._validate_context_view_compaction(
                            cur, session_id, context_view_compaction_cursor
                        )
                    current_checkpoint = await self._load_checkpoint(cur, session_id)
                    callback_checkpoint = _copy_checkpoint_for_transform(
                        current_checkpoint,
                        session_id=session_id,
                    )
                    operation_records: dict[str, dict[str, Any]] = {}
                    model_completion_stage_release = None
                    if (
                        operation_transform is not None
                        or store_time_operation_transform is not None
                    ):
                        await cur.execute(
                            "SELECT record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = %s",
                            (session_id, operation_idempotency_key),
                        )
                        operation_row = await cur.fetchone()
                        current_operation = (
                            None
                            if operation_row is None
                            else pg_support._json_obj(operation_row[0])
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
                                "Session operation transform must return a "
                                "SessionOperationPublication."
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
                        indices = session_exports.selected_transcript_indices(session_id=session_id)
                        if (
                            indices
                            or session_exports.checkpoint_visible(session_id=session_id)
                            or any(
                                key.startswith(session_exports.OPERATION_PREFIX)
                                for key in operation_records
                            )
                        ):
                            selected_rows: tuple[TranscriptRecord, ...] = ()
                            if indices:
                                await cur.execute(
                                    "SELECT session_order - 1, interaction_id, message "
                                    "FROM cayu_transcript_messages "
                                    "WHERE session_id = %s AND session_order = ANY(%s) "
                                    "ORDER BY session_order",
                                    (session_id, [index + 1 for index in indices]),
                                )
                                selected_rows = tuple(
                                    TranscriptRecord(
                                        index=row[0],
                                        interaction_id=row[1],
                                        message=Message(**pg_support._json_obj(row[2])),
                                    )
                                    for row in await cur.fetchall()
                                )
                            session_exports.validate_publication(
                                session=loaded,
                                current_checkpoint=current_checkpoint,
                                proposed_checkpoint=transformed,
                                operation_records=operation_records,
                                selected_transcript_rows=selected_rows,
                                events=copied_events,
                            )
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

                    await self._register_event_public_authorities(
                        cur,
                        session_id,
                        copied_events,
                    )
                    await self._publish_budget_reservation_identities(cur, copied_events)
                    await cur.execute(
                        """
                        UPDATE cayu_sessions
                        SET event_seq = event_seq + %s,
                            updated_at = %s,
                            last_activity_at = %s
                        WHERE id = %s
                        RETURNING event_seq
                        """,
                        (len(copied_events), updated_at, updated_at, session_id),
                    )
                    order_row = await cur.fetchone()
                    if order_row is None:
                        raise KeyError(f"Session not found: {session_id}")
                    await self._upsert_checkpoint(cur, session_id, transformed, updated_at)
                    if operation_records:
                        await cur.executemany(
                            """
                            INSERT INTO cayu_session_operations (
                                session_id, idempotency_key, record, updated_at
                            )
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT(session_id, idempotency_key) DO UPDATE SET
                                record = excluded.record,
                                updated_at = excluded.updated_at
                            """,
                            [
                                (session_id, key, pg_support._dumps(record), updated_at)
                                for key, record in operation_records.items()
                            ],
                        )
                    if model_completion_stage_release is not None:
                        _, _, preparation_key, terminal_key = (
                            _model_completion_stage_storage_identity(
                                session_id,
                                model_completion_stage_release.stage_id,
                            )
                        )
                        await cur.execute(
                            "SELECT idempotency_key, record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                            (
                                session_id,
                                [
                                    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                                    preparation_key,
                                    terminal_key,
                                ],
                            ),
                        )
                        stage_records = {
                            row[0]: _decode_model_completion_stage_record(row[1])
                            for row in await cur.fetchall()
                        }
                        marker = _validate_model_completion_stage_release(
                            session=loaded,
                            active_record=stage_records.get(
                                MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY
                            ),
                            preparation_record=stage_records.get(preparation_key),
                            terminal_record=stage_records.get(terminal_key),
                            release=model_completion_stage_release,
                        )
                        await cur.execute(
                            "DELETE FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = %s "
                            "AND record->>'record_digest' = %s",
                            (
                                session_id,
                                MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                                marker.record_digest,
                            ),
                        )
                        if cur.rowcount != 1:
                            raise SessionModelCompletionStageConflict(
                                "The active model-completion stage changed during disposition."
                            )

                    next_order = order_row[0] - len(copied_events)
                    rows = []
                    for event in copied_events:
                        next_order += 1
                        lookup_key, projection, projection_bytes = (
                            pending_action_event_storage_values(event)
                        )
                        rows.append(
                            (
                                session_id,
                                next_order,
                                event.id,
                                event.interaction_id,
                                str(event.type),
                                pg_support.to_utc(event.timestamp),
                                event.agent_name,
                                event.environment_name,
                                event.workflow_name,
                                event.tool_name,
                                pg_support._dumps(event.payload),
                                pg_support._dumps(event.model_dump(mode="json")),
                                lookup_key,
                                projection,
                                projection_bytes,
                            )
                        )
                    if rows:
                        await cur.executemany(
                            """
                            INSERT INTO cayu_events (
                                session_id, session_order, event_id, interaction_id,
                                event_type, timestamp,
                                agent_name, environment_name, workflow_name, tool_name,
                                payload, event, pending_action_lookup_key,
                                pending_action_projection, pending_action_projection_bytes
                            )
                            VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s
                            )
                            """,
                            rows,
                        )
                        await self._record_invocation_terminal_event_receipts(
                            cur, session_id, copied_events, activity_at=updated_at
                        )
                        await self._enqueue_persisted_event_side_effects(
                            cur,
                            session_id,
                            copied_events,
                        )
                    if operation_commit_guard is not None:
                        commit_guard_signal = await _run_session_commit_guard_owned(
                            operation_commit_guard
                        )
                    activity_at = (
                        await self._session_store_now(cur)
                        if operation_commit_guard is not None
                        else updated_at
                    )
                    if operation_commit_time_guard is not None:
                        activity_at = await self._session_store_now(cur)
                        operation_commit_time_guard(activity_at)
                    if activity_at != updated_at:
                        await cur.execute(
                            "UPDATE cayu_sessions SET last_activity_at = %s WHERE id = %s",
                            (activity_at, session_id),
                        )
                await conn.commit()
                if commit_guard_signal is not None:
                    raise commit_guard_signal
            except UniqueViolation as exc:
                await conn.rollback()
                if commit_guard_signal is not None:
                    raise commit_guard_signal from exc
                existing = await self._first_existing_event_id(
                    session_id,
                    [event.id for event in copied_events],
                )
                if existing is not None:
                    raise ValueError(
                        f"Event already exists for session {session_id}: {existing}"
                    ) from exc
                raise
            except Exception as error:
                await conn.rollback()
                if commit_guard_signal is not None:
                    raise commit_guard_signal from error
                raise
            return loaded.model_copy(
                update={"updated_at": updated_at, "last_activity_at": activity_at}
            )

    async def load_session_closure_records(
        self, session_id: str, *, max_records: int, max_bytes: int
    ) -> dict[str, Any]:
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            return await self._load_session_closure_records(
                cur, session_id, max_records=max_records, max_bytes=max_bytes
            )

    async def _load_session_closure_records(
        self, cur: Any, session_id: str, *, max_records: int, max_bytes: int
    ) -> dict[str, Any]:
        from cayu.runtime._session_closure_records import (
            ClosureRecordsBuilder,
            ClosureRecordsTooLarge,
        )
        from cayu.storage._session_closure_sql import closure_size_statement

        session_id = require_clean_nonblank(session_id, "session_id")
        builder = ClosureRecordsBuilder(max_records=max_records, max_bytes=max_bytes)
        statement, source_count = closure_size_statement(postgres=True)
        await cur.execute(statement, (session_id, max_records + 1) * source_count)
        count, size = await cur.fetchone()
        if count > max_records or size > max_bytes:
            raise ClosureRecordsTooLarge()
        session = await self._load(cur, session_id)
        builder.add_class(
            "session",
            ()
            if session is None
            else (
                {
                    name: getattr(session, name)
                    for name in type(session).model_fields
                    if name not in {"labels", "metadata"}
                },
            ),
        )
        builder.add_class(
            "labels",
            ()
            if session is None
            else ({"key": key, "value": value} for key, value in session.labels.items()),
        )
        builder.add_class("metadata", () if session is None else (session.metadata,))

        def project_grant(row):
            codec = self.public_authority_alias_codec
            if codec is None:
                raise RuntimeError("Closure grant export requires an authority alias codec.")
            return targeted_tool_grant_with_active_reference(
                _targeted_tool_grant_from_postgres_row(row), codec
            )

        queries: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
            (
                "recall_receipts",
                "SELECT receipt_id, session_id, interaction_id, model_step_id, created_at, "
                "receipt_json, document_bytes FROM cayu_recall_receipts WHERE session_id = %s "
                "ORDER BY created_at, receipt_id LIMIT %s",
                _postgres_recall_receipt,
            ),
            (
                "context_exposures",
                "SELECT exposure_id, session_id, interaction_id, model_step_id, model_attempt_id, "
                "provider_attempt_id, state, state_revision, created_at, updated_at, exposure_json, "
                "document_bytes FROM cayu_context_exposures WHERE session_id = %s "
                "ORDER BY created_at, exposure_id LIMIT %s",
                _postgres_context_exposure,
            ),
            (
                "recall_item_exposures",
                "SELECT item.item_json FROM cayu_recall_item_exposures AS item "
                "JOIN cayu_context_exposures AS exposure ON exposure.exposure_id = item.exposure_id "
                "WHERE exposure.session_id = %s "
                "ORDER BY exposure.created_at, exposure.exposure_id, item.ordinal LIMIT %s",
                lambda row: pg_support._json_obj(row[0]),
            ),
            (
                "events",
                "SELECT sequence, event FROM cayu_events WHERE session_id = %s ORDER BY sequence LIMIT %s",
                lambda row: EventRecord(
                    sequence=row[0], event=Event(**pg_support._json_obj(row[1]))
                ),
            ),
            (
                "transcript",
                "SELECT session_order, interaction_id, message FROM cayu_transcript_messages WHERE session_id = %s ORDER BY session_order LIMIT %s",
                lambda row: {
                    "transcript_index": row[0] - 1,
                    "interaction_id": row[1],
                    "message": pg_support._json_obj(row[2]),
                },
            ),
            (
                "checkpoint",
                "SELECT state FROM cayu_checkpoints WHERE session_id = %s LIMIT %s",
                lambda row: pg_support._json_obj(row[0]),
            ),
            (
                "queued_messages",
                f"SELECT {_SESSION_MESSAGE_QUEUE_COLUMNS} FROM cayu_session_message_queue WHERE session_id = %s ORDER BY ordering_key LIMIT %s",
                lambda row: {
                    "message": _queued_session_message_from_row(row),
                    "terminal": None if row[18] is None else pg_support._json_obj(row[18]),
                },
            ),
            (
                "session_operations",
                "SELECT idempotency_key, record FROM cayu_session_operations WHERE session_id = %s ORDER BY idempotency_key LIMIT %s",
                lambda row: {"idempotency_key": row[0], "record": pg_support._json_obj(row[1])},
            ),
            (
                "event_side_effect_deliveries",
                "SELECT session_id, event_id, event_sequence, status, attempts, claim_id, lease_expires_at, next_attempt_at, last_error, updated_at FROM cayu_persisted_event_side_effects WHERE session_id = %s ORDER BY event_sequence LIMIT %s",
                _persisted_event_side_effect_delivery_from_row,
            ),
            (
                "queue_deliveries",
                "SELECT to_jsonb(d) - 'created_at' FROM cayu_session_message_deliveries AS d WHERE session_id = %s ORDER BY created_at, delivery_id LIMIT %s",
                lambda row: pg_support._json_obj(row[0]),
            ),
            (
                "deferred_interaction_inputs",
                "SELECT interaction_id, source_messages FROM cayu_deferred_interaction_inputs WHERE session_id = %s LIMIT %s",
                lambda row: deferred_interaction_input_from_storage_payload(
                    row[0], pg_support._json_obj(row[1])
                ),
            ),
            (
                "targeted_tool_grants",
                "SELECT grant_id, session_id, interaction_id, request_id, tool_ref, generation_id, tool_id, tool_name, catalogue_revision, descriptor_version, issued_at, expires_at, max_calls, used_calls, revoked_at, record FROM cayu_targeted_tool_grants WHERE session_id = %s ORDER BY issued_at, grant_id LIMIT %s",
                project_grant,
            ),
            (
                "targeted_tool_grant_uses",
                "SELECT use_id, grant_id, session_id, interaction_id, model_step_id, outer_tool_call_id, arguments_sha256, invocation_id, bound_at, record FROM cayu_targeted_tool_grant_uses WHERE session_id = %s ORDER BY bound_at, use_id LIMIT %s",
                _targeted_tool_use_from_postgres_row,
            ),
        )
        for name, statement, convert in queries:
            builder.add_class(name, ())
            await cur.execute(statement, (session_id, max_records + 1))
            while rows := await cur.fetchmany(100):
                for row in rows:
                    builder.add_record(name, convert(row))
        return builder.finish()

    async def _complete_native_producer_cleanup(self, registration, *, authority):
        from cayu.storage._producer_cleanup import postgres_cleanup

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer cleanup is not qualified.")
        return await postgres_cleanup(self, registration, authority=authority, commit=True)

    async def _retire_native_producer_cleanup(self, retirement, *, authority, limit):
        from cayu.storage._producer_retirement import postgres_retirement

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer retirement is not qualified.")
        return await postgres_retirement(self, retirement, authority=authority, limit=limit)

    async def _read_completed_native_producer_cleanup(self, registration):
        from cayu.storage._producer_cleanup import postgres_cleanup

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer cleanup readback is not qualified.")
        return await postgres_cleanup(self, registration)

    async def _read_native_producer_release(self, command):
        from cayu.storage._producer_observation import postgres_observation

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer release readback is not qualified.")
        return await postgres_observation(self, command)

    async def _read_native_producer_progress(self, command, *, kind):
        from cayu.storage._producer_observation import postgres_observation

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer progress readback is not qualified.")
        return await postgres_observation(self, command, kind=kind)

    async def _read_native_producer_attachment(self, command):
        from cayu.storage._producer_observation import postgres_observation

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer attachment readback is not qualified.")
        return await postgres_observation(self, command, attachment_only=True)

    async def load_session_export_snapshot(
        self,
        session_id: str,
        *,
        limits: SessionExportLimits | None = None,
    ) -> SessionExportSnapshot | None:
        from cayu.sessions.exports import SESSION_EXPORT_PAGE_SIZE, SessionExportBuilder

        session_id = require_clean_nonblank(session_id, "session_id")
        from cayu.storage._session_export_sql import export_size_statement

        builder = SessionExportBuilder(limits)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            statement, parameter_count = export_size_statement(postgres=True)
            await cur.execute(statement, (session_id,) * parameter_count)
            sizes = await cur.fetchone()
            builder.preflight_bytes(int(sizes[0]), int(sizes[1]))
            session = await self._load(cur, session_id)
            if session is None:
                return None
            cursor = await _transcript_cursor(cur, session_id)
            checkpoint = await self._load_checkpoint(cur, session_id)
            async with conn.cursor(name=f"session_export_events_{uuid4().hex}") as rows:
                await rows.execute(
                    "SELECT sequence, event FROM cayu_events WHERE session_id = %s ORDER BY sequence",
                    (session_id,),
                )
                while page := await rows.fetchmany(SESSION_EXPORT_PAGE_SIZE):
                    for row in page:
                        builder.event(
                            EventRecord(
                                sequence=row[0], event=Event(**pg_support._json_obj(row[1]))
                            )
                        )
            async with conn.cursor(name=f"session_export_transcript_{uuid4().hex}") as rows:
                await rows.execute(
                    "SELECT session_order, interaction_id, message FROM cayu_transcript_messages "
                    "WHERE session_id = %s ORDER BY session_order",
                    (session_id,),
                )
                while page := await rows.fetchmany(SESSION_EXPORT_PAGE_SIZE):
                    for row in page:
                        builder.message(
                            TranscriptRecord(
                                index=row[0] - 1,
                                interaction_id=row[1],
                                message=Message(**pg_support._json_obj(row[2])),
                            )
                        )
            await cur.execute(
                "SELECT interaction_id, source_messages FROM cayu_deferred_interaction_inputs "
                "WHERE session_id = %s",
                (session_id,),
            )
            row = await cur.fetchone()
            deferred = (
                None
                if row is None
                else deferred_interaction_input_from_storage_payload(
                    row[0], pg_support._json_obj(row[1])
                )
            )
            codec = self.public_authority_alias_codec
            grants = []
            uses = []
            async with conn.cursor(name=f"session_export_grants_{uuid4().hex}") as rows:
                await rows.execute(
                    "SELECT grant_id, session_id, interaction_id, request_id, tool_ref, "
                    "generation_id, tool_id, tool_name, catalogue_revision, descriptor_version, "
                    "issued_at, expires_at, max_calls, used_calls, revoked_at, record "
                    "FROM cayu_targeted_tool_grants WHERE session_id = %s ORDER BY issued_at, grant_id",
                    (session_id,),
                )
                while page := await rows.fetchmany(SESSION_EXPORT_PAGE_SIZE):
                    for row in page:
                        if codec is None:
                            raise RuntimeError(
                                "Exporting targeted grants requires an authority alias codec."
                            )
                        grant = targeted_tool_grant_with_active_reference(
                            _targeted_tool_grant_from_postgres_row(row), codec
                        )
                        builder.charge(grant.model_dump(mode="json"))
                        grants.append(grant)
            async with conn.cursor(name=f"session_export_uses_{uuid4().hex}") as rows:
                await rows.execute(
                    "SELECT use_id, grant_id, session_id, interaction_id, model_step_id, "
                    "outer_tool_call_id, arguments_sha256, invocation_id, bound_at, record "
                    "FROM cayu_targeted_tool_grant_uses WHERE session_id = %s ORDER BY bound_at, use_id",
                    (session_id,),
                )
                while page := await rows.fetchmany(SESSION_EXPORT_PAGE_SIZE):
                    for row in page:
                        use = _targeted_tool_use_from_postgres_row(row)
                        builder.charge(use.model_dump(mode="json"))
                        uses.append(use)
            return builder.finish(
                session=session,
                transcript_cursor=cursor,
                checkpoint=checkpoint,
                deferred=deferred,
                grants_charged=True,
                grants=TargetedToolGrantStateSnapshot(records=tuple(grants), uses=tuple(uses)),
            )

    async def load_events(self, session_id: str) -> list[Event]:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            if access_bounds is not None:
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                access_bounds.require_read(await self._load(cur, session_id))
            await cur.execute(
                "SELECT 1 FROM cayu_sessions WHERE id = %s",
                (session_id,),
            )
            if await cur.fetchone() is None:
                raise KeyError(f"Session not found: {session_id}")
            await cur.execute(
                """
                SELECT event
                FROM cayu_events
                WHERE session_id = %s
                ORDER BY session_order ASC
                """,
                (session_id,),
            )
            rows = await cur.fetchall()
            return [Event(**pg_support._json_obj(row[0])) for row in rows]

    async def load_user_input_supersession_events(
        self,
        session_id: str,
        input_id: str,
    ) -> list[Event]:
        from cayu.sessions.pending_actions import pending_action_lookup_key

        session_id = require_clean_nonblank(session_id, "session_id")
        input_id = require_clean_nonblank(input_id, "input_id")
        lookup_key = pending_action_lookup_key(input_id)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT 1 FROM cayu_sessions WHERE id = %s",
                (session_id,),
            )
            if await cur.fetchone() is None:
                raise KeyError(f"Session not found: {session_id}")
            await cur.execute(
                "SELECT event FROM cayu_events "
                "WHERE session_id = %s AND pending_action_lookup_key = %s "
                "AND event_type = %s "
                f"AND ({_PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
                "AND event -> 'payload' -> 'user_input_supersession_intent' "
                "->> 'input_id' = %s "
                "ORDER BY session_order ASC LIMIT 2",
                (
                    session_id,
                    lookup_key,
                    str(EventType.SESSION_INTERRUPTED),
                    input_id,
                ),
            )
            rows = await cur.fetchall()
            return [Event(**pg_support._json_obj(row[0])) for row in rows]

    async def load_tool_round_lifecycle_events(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
    ) -> list[Event]:
        from cayu.sessions.pending_actions import pending_action_lookup_key

        session_id = require_clean_nonblank(session_id, "session_id")
        copied_ids = _validate_tool_round_call_ids(tool_call_ids, "tool_call_ids")
        lookup_keys = [pending_action_lookup_key(call_id) for call_id in copied_ids]
        lifecycle_event_types = [
            str(event_type) for event_type in sorted(_TOOL_ROUND_LIFECYCLE_EVENT_TYPES, key=str)
        ]
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT 1 FROM cayu_sessions WHERE id = %s",
                (session_id,),
            )
            if await cur.fetchone() is None:
                raise KeyError(f"Session not found: {session_id}")
            await cur.execute(
                "SELECT event FROM cayu_events "
                "WHERE session_id = %s "
                "AND pending_action_lookup_key = ANY(%s) "
                "AND event_type = ANY(%s) "
                f"AND ({_PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
                "ORDER BY session_order ASC LIMIT %s",
                (
                    session_id,
                    lookup_keys,
                    lifecycle_event_types,
                    _tool_round_lifecycle_event_limit(copied_ids) + 1,
                ),
            )
            rows = await cur.fetchall()
            if len(rows) > _tool_round_lifecycle_event_limit(copied_ids):
                raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
            return [Event(**pg_support._json_obj(row[0])) for row in rows]

    async def load_tool_round_lifecycle_events_for_round(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
        *,
        tool_round_identity: ToolRoundIdentity,
    ) -> list[Event]:
        from cayu.sessions.pending_actions import pending_action_lookup_key

        session_id = require_clean_nonblank(session_id, "session_id")
        copied_ids = _validate_tool_round_call_ids(tool_call_ids, "tool_call_ids")
        tool_round_identity = copy_tool_round_identity(tool_round_identity)
        lookup_keys = [pending_action_lookup_key(call_id) for call_id in copied_ids]
        lifecycle_event_types = [
            str(event_type) for event_type in sorted(_TOOL_ROUND_LIFECYCLE_EVENT_TYPES, key=str)
        ]
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT 1 FROM cayu_sessions WHERE id = %s",
                (session_id,),
            )
            if await cur.fetchone() is None:
                raise KeyError(f"Session not found: {session_id}")
            await cur.execute(
                "SELECT event FROM cayu_events "
                "WHERE session_id = %s "
                "AND pending_action_lookup_key = ANY(%s) "
                "AND event_type = ANY(%s) "
                f"AND ({_PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
                "AND (event -> 'payload' ->> 'tool_round_id' = %s "
                "OR (event -> 'payload' ->> 'model_step_id' = %s "
                "AND event -> 'payload' ->> 'model_attempt_id' = %s) "
                "OR ((event -> 'payload' ->> 'tool_round_id') "
                "~ '^tround_[0-9a-f]{32}$') IS NOT TRUE "
                "OR ((event -> 'payload' ->> 'model_step_id') "
                "~ '^mstep_[0-9a-f]{32}$') IS NOT TRUE "
                "OR ((event -> 'payload' ->> 'model_attempt_id') "
                "~ '^matt_[0-9a-f]{32}$') IS NOT TRUE) "
                "ORDER BY session_order ASC LIMIT %s",
                (
                    session_id,
                    lookup_keys,
                    lifecycle_event_types,
                    tool_round_identity.tool_round_id,
                    tool_round_identity.model_step_id,
                    tool_round_identity.model_attempt_id,
                    _tool_round_lifecycle_event_limit(copied_ids) + 1,
                ),
            )
            rows = await cur.fetchall()
            if len(rows) > _tool_round_lifecycle_event_limit(copied_ids):
                raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
            return [Event(**pg_support._json_obj(row[0])) for row in rows]

    @runtime_session_query
    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        query = copy_event_query(query)
        if len(query.session_ids) > _EVENT_QUERY_SESSION_IDS_BATCH_SIZE:
            return await self._query_events_by_session_id_batches(query)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            return await self._query_events(cur, query, safe_insert_xid=None)

    @runtime_session_query
    async def event_exists(self, query: EventQuery) -> bool:
        plan = session_store_sql.build_accounting_event_query_sql(query, dialect=_SQL_DIALECT)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                cast(
                    "LiteralString",
                    "SELECT EXISTS(SELECT 1 FROM cayu_events "
                    "JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id "
                    f"{plan.where_sql})",
                ),
                plan.params,
            )
            row = await cur.fetchone()
            if row is None:
                raise RuntimeError("Failed to read event existence result.")
            return bool(row[0])

    @runtime_session_query
    async def read_usage_accounting(
        self, query: EventQuery, *, by_session: bool = False, by_identity: bool = False
    ) -> UsageAccountingSnapshot:
        from cayu.runtime._usage_accounting import (
            USAGE_ACCOUNTING_PAGE_SIZE,
            SessionUsageCache,
            UsageAccountingReducer,
            usage_accounting_query,
        )

        query = usage_accounting_query(query)
        cached_session_id = SessionUsageCache.session_scope(
            query, by_session=by_session, by_identity=by_identity
        )
        prior = None
        boundary = 0
        await self._ensure_ready()
        async with self._connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                await cur.execute(
                    "SELECT generation FROM cayu_accounting_state WHERE singleton = 1"
                )
                generation_row = await cur.fetchone()
                if generation_row is None:
                    raise RuntimeError("Accounting deletion revision is missing.")
                generation = generation_row[0]
                if cached_session_id is not None:
                    prior = self._session_usage_cache.resume_after(cached_session_id, generation)
                    if prior is not None:
                        query = copy_event_query(
                            query, update={"after_sequence": prior.scanned_through}
                        )
                    # The session's event counter row lock serializes its appends,
                    # so a later commit for this session cannot land below this.
                    await cur.execute(
                        "SELECT MAX(sequence) FROM cayu_events WHERE session_id = %s",
                        (cached_session_id,),
                    )
                    boundary_row = await cur.fetchone()
                    boundary = 0 if boundary_row is None else boundary_row[0] or 0

            reducer = UsageAccountingReducer(query, by_session=by_session, by_identity=by_identity)
            plan = session_store_sql.build_accounting_event_query_sql(query, dialect=_SQL_DIALECT)
            # A named server cursor prevents libpq from buffering all rows in the
            # client, while REPEATABLE READ pins one snapshot across fetches.
            async with conn.cursor(name=f"usage_{uuid4().hex}") as cur:
                await cur.execute(
                    cast(
                        "LiteralString",
                        "SELECT cayu_events.sequence, cayu_events.event FROM cayu_events "
                        "JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id "
                        f"{plan.where_sql} ORDER BY cayu_events.sequence ASC",
                    ),
                    plan.params,
                )
                while rows := await cur.fetchmany(USAGE_ACCOUNTING_PAGE_SIZE):
                    reducer.add_page(
                        [
                            EventRecord(
                                sequence=row[0], event=Event(**pg_support._json_obj(row[1]))
                            )
                            for row in rows
                        ]
                    )
            if cached_session_id is not None:
                return self._session_usage_cache.settle(
                    cached_session_id, generation, prior, reducer.snapshot(), boundary=boundary
                )
            return reducer.snapshot().model_copy(update={"generation": generation})

    @runtime_session_query
    async def read_cost_accounting(
        self,
        query: EventQuery,
        pricing: PriceBook,
        *,
        currency: str = "USD",
        details: bool = False,
        by_session: bool = False,
        additional_events: tuple[Event, ...] = (),
        max_detail_bytes: int | None = None,
        previous: CostAccountingSnapshot | None = None,
    ) -> CostAccountingSnapshot:
        from cayu.runtime._cost_accounting import (
            COST_ACCOUNTING_PAGE_SIZE,
            cost_accounting_query,
            cost_pending_events,
        )
        from cayu.runtime._cost_accounting_refresh import CostAccountingRead
        from cayu.storage._cost_accounting_sql import (
            changed_cost_groups_statement,
            cost_boundary_statement,
            cost_group_lookup_statement,
            cost_group_statement,
        )

        if previous is not None and type(previous) is not CostAccountingSnapshot:
            raise TypeError("previous must be a CostAccountingSnapshot.")
        query = cost_accounting_query(query)
        pending = cost_pending_events(query, additional_events)
        await self._ensure_ready()
        async with self._connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                await cur.execute(
                    "SELECT generation FROM cayu_accounting_state WHERE singleton = 1"
                )
                generation_row = await cur.fetchone()
                if generation_row is None:
                    raise RuntimeError("Accounting deletion revision is missing.")
                generation = generation_row[0]
                if query.causal_budget_id is not None and pending:
                    await cur.execute(
                        "SELECT id FROM cayu_sessions WHERE causal_budget_id = %s AND id = ANY(%s)",
                        (query.causal_budget_id, [event.session_id for event in pending]),
                    )
                    allowed_sessions = {row[0] for row in await cur.fetchall()}
                    pending = tuple(
                        event for event in pending if event.session_id in allowed_sessions
                    )

            plan = session_store_sql.build_accounting_event_query_sql(query, dialect=_SQL_DIALECT)
            unique_pending: list[Event] = []
            async with conn.cursor() as cur:
                for event in pending:
                    await cur.execute(
                        cast(
                            "LiteralString",
                            "SELECT 1 FROM cayu_events "
                            "JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id "
                            f"{plan.where_sql} AND cayu_events.session_id = %s AND cayu_events.event_id = %s LIMIT 1",
                        ),
                        (*plan.params, event.session_id, event.id),
                    )
                    if await cur.fetchone() is None:
                        unique_pending.append(event)
            boundary_sql, boundary_params = cost_boundary_statement(query, dialect=_SQL_DIALECT)
            async with conn.cursor() as cur:
                await cur.execute(cast("LiteralString", boundary_sql), boundary_params)
                boundary_row = await cur.fetchone()
                if boundary_row is None:
                    raise RuntimeError("Accounting event boundary is missing.")
                boundary = boundary_row[0] or 0
            reducer = CostAccountingRead(
                query,
                pricing,
                currency=currency,
                details=details,
                max_detail_bytes=max_detail_bytes,
                previous=previous if _event_query_is_single_session(query) else None,
                generation=generation,
                through_sequence=boundary,
                authority=self._cost_accounting_authority,
                by_session=by_session,
                additional_events=tuple(unique_pending),
            )
            source_plan = session_store_sql.build_accounting_event_query_sql(
                reducer.source_query,
                dialect=_SQL_DIALECT,
            )
            columns = "cayu_events.sequence, cayu_events.event, cayu_events.session_id, cayu_events.event_id, cayu_events.event_type"

            async def add_group(key: tuple[str, bool, str]) -> None:
                statement, params = cost_group_lookup_statement(
                    columns=columns, plan=source_plan, key=key, postgres=True
                )
                async with conn.cursor(name=f"cost_group_{uuid4().hex}") as group_cursor:
                    await group_cursor.execute(cast("LiteralString", statement), params)
                    while rows := await group_cursor.fetchmany(COST_ACCOUNTING_PAGE_SIZE):
                        for row in rows:
                            reducer.add(row[0], Event(**pg_support._json_obj(row[1])))

            if reducer.incremental:
                statement, params = changed_cost_groups_statement(reducer, dialect=_SQL_DIALECT)
                async with conn.cursor(name=f"changed_cost_{uuid4().hex}") as cur:
                    await cur.execute(cast("LiteralString", statement), params)
                    while groups := await cur.fetchmany(COST_ACCOUNTING_PAGE_SIZE):
                        for group in groups:
                            await add_group(
                                (
                                    group[0],
                                    group[1] is not None,
                                    group[1] if group[1] is not None else group[2],
                                )
                            )
                for key in reducer.remaining_pending_keys:
                    await add_group(key)
            else:
                statement, group_params = cost_group_statement(
                    columns=columns, where_sql=source_plan.where_sql, postgres=True
                )
                async with conn.cursor(name=f"cost_{uuid4().hex}") as cur:
                    await cur.execute(
                        cast("LiteralString", statement), (*group_params, *source_plan.params)
                    )
                    while rows := await cur.fetchmany(COST_ACCOUNTING_PAGE_SIZE):
                        for row in rows:
                            reducer.add(row[0], Event(**pg_support._json_obj(row[1])))
            result = reducer.snapshot()
            # Global sequence allocation is not commit order. A later commit can
            # become visible below this snapshot's maximum sequence, so it is not
            # a safe incremental baseline across independently written sessions.
            if not _event_query_is_single_session(query):
                result = result.model_copy(update={"cursor": None})
            return result

    @runtime_session_query
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
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            needs_snapshot_cutoff = _event_query_needs_snapshot_cutoff(query)
            safe_insert_xid = None
            extra_clauses: tuple[session_store_sql.SqlClause, ...] = ()
            if needs_snapshot_cutoff:
                await cur.execute("SELECT pg_snapshot_xmin(pg_current_snapshot())")
                snapshot_row = await cur.fetchone()
                if snapshot_row is None:
                    raise RuntimeError("Failed to read Postgres event visibility snapshot.")
                safe_insert_xid = snapshot_row[0]
                extra_clauses = (
                    session_store_sql.SqlClause(
                        "cayu_events.insert_xid < %s",
                        (safe_insert_xid,),
                    ),
                )
            plan = session_store_sql.build_event_query_sql(
                query,
                dialect=_SQL_DIALECT,
                extra_after_sequence_clauses=extra_clauses,
            )
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    WITH bounded_candidates AS (
                        SELECT octet_length(cayu_events.event::text) + 256
                                   AS serialized_bytes
                        FROM cayu_events
                        JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
                        {plan.where_sql}
                        ORDER BY cayu_events.sequence {plan.order_direction}
                        LIMIT %s
                    )
                    SELECT COALESCE(SUM(serialized_bytes), 0)
                    FROM bounded_candidates
                    """,
                ),
                (*plan.params, query.limit),
            )
            size_row = await cur.fetchone()
            if size_row is None or int(size_row[0]) > max_bytes:
                raise EventQueryResultTooLarge(max_bytes)
            return await self._query_events(
                cur,
                query,
                safe_insert_xid=safe_insert_xid,
                force_snapshot_cutoff=needs_snapshot_cutoff,
            )

    async def load_terminal_session_evidence(
        self,
        session_id: str,
        *,
        limits: TerminalSessionEvidenceLimits | None = None,
    ) -> TerminalSessionEvidence:
        result = await self._load_terminal_session_evidence(
            session_id,
            limits=limits,
            observed_interrupted_events=None,
            expected_interrupted_parent_session_id=None,
            require_interrupted_proof=False,
        )
        assert isinstance(result, TerminalSessionEvidence)
        return result

    async def load_runner_owned_interrupted_evidence(
        self,
        session_id: str,
        *,
        observed_events: tuple[RunnerObservedEventIdentity, ...] | None = None,
        expected_parent_session_id: str | None = None,
        limits: TerminalSessionEvidenceLimits | None = None,
    ) -> TerminalSessionEvidence:
        result = await self._load_terminal_session_evidence(
            session_id,
            limits=limits,
            observed_interrupted_events=observed_events,
            expected_interrupted_parent_session_id=expected_parent_session_id,
            require_interrupted_proof=True,
        )
        assert isinstance(result, TerminalSessionEvidence)
        return result

    async def load_bounded(self, session_id: str, *, max_bytes: int) -> Session | None:
        from cayu._validation import compact_json_utf8_size

        session_id = require_clean_nonblank(session_id, "session_id")
        if type(max_bytes) is not int or not 1 <= max_bytes <= 8_388_608:
            raise ValueError("max_bytes must be an integer in 1..8388608.")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            await cur.execute(
                "SELECT octet_length(to_jsonb(s)::text) + 1 + COALESCE((SELECT "
                "SUM(octet_length(key) + octet_length(value)) FROM cayu_session_labels "
                "WHERE session_id = s.id), 0) FROM cayu_sessions s WHERE id = %s",
                (session_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            if row[0] > max_bytes + (max_bytes + 1) // 2:
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED, limit=max_bytes
                )
            session = await self._load(cur, session_id)
            if (
                session is not None
                and compact_json_utf8_size(session.model_dump(mode="json")) > max_bytes
            ):
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED, limit=max_bytes
                )
            return session

    async def export_terminal_session_evidence(
        self,
        session_id: str,
        *,
        spool: EvidenceSpool,
    ) -> None:
        """Copy one stable terminal snapshot into caller-owned bounded backing."""
        async with asyncio.timeout(spool.limits.max_seconds):
            await self._load_terminal_session_evidence(
                session_id,
                limits=None,
                observed_interrupted_events=None,
                expected_interrupted_parent_session_id=None,
                require_interrupted_proof=False,
                spool=spool,
            )
            await spool.seal_async()

    async def _load_terminal_session_evidence(
        self,
        session_id: str,
        *,
        limits: TerminalSessionEvidenceLimits | None,
        observed_interrupted_events: tuple[RunnerObservedEventIdentity, ...] | None,
        expected_interrupted_parent_session_id: str | None,
        require_interrupted_proof: bool,
        spool: EvidenceSpool | None = None,
    ) -> TerminalSessionEvidence | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        eager_limits = _copy_terminal_session_evidence_limits(limits)
        selected_limits = eager_limits if spool is None else spool.limits
        observed, expected_parent_session_id = _copy_runner_owned_interruption_proof(
            session_id,
            observed_events=observed_interrupted_events,
            expected_parent_session_id=expected_interrupted_parent_session_id,
            limits=eager_limits,
            required=require_interrupted_proof,
        )
        allow_interrupted = observed is not None or expected_parent_session_id is not None
        evidence_event_types = [
            str(event_type) for event_type in _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES
        ]

        def jsonb_transport_bytes(expression: str) -> str:
            # Bound the complete JSONB text representation that PostgreSQL must
            # transfer and the driver must decode. This intentionally retains
            # serializer whitespace as well as every byte inside string values.
            # The extra byte accounts for JSONB's binary-protocol version prefix
            # when that transfer format is selected.
            return f"octet_length(({expression})::text) + 1"

        def transport_limit(canonical_limit: int) -> int:
            # This is an independent PostgreSQL working-set policy, not an
            # upper bound derived from Cayu's portable JSON representation.
            # Besides separator spaces, JSONB can expand scientific-notation
            # floats into fixed-point decimals by far more than 3:2. Refuse that
            # transport expansion with its own typed error instead of weakening
            # the hydration bound; the shared assembler applies the canonical
            # caller limit to every payload that passes this backend guard.
            return canonical_limit + (canonical_limit + 1) // 2

        event_transport_bytes = f"{jsonb_transport_bytes('event')} + octet_length(sequence::text)"
        transcript_transport_bytes = (
            f"{jsonb_transport_bytes('message')} "
            "+ octet_length(session_order::text) "
            "+ COALESCE(octet_length(interaction_id), 0)"
        )
        session_transport_bytes = " + ".join(
            f"COALESCE(octet_length(session.{column}), 0)"
            for column in (
                "id",
                "instance_id",
                "agent_name",
                "provider_name",
                "model",
                "parent_session_id",
                "causal_budget_id",
                "runtime_name",
                "runtime_version",
                "environment_name",
                "status",
            )
        )
        session_transport_bytes += (
            " + octet_length(session.run_epoch::text)"
            f" + {jsonb_transport_bytes('session.metadata')}"
            f" + {jsonb_transport_bytes('session.invocation')}"
        )
        max_record_transport_bytes = transport_limit(selected_limits.max_record_bytes)
        max_total_transport_bytes = transport_limit(selected_limits.max_total_bytes)

        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                    if spool is not None:
                        await cur.execute(
                            "SELECT set_config('statement_timeout', %s, true)",
                            (str(spool.limits.max_seconds * 1000),),
                        )
                    await cur.execute(
                        f"""
                            SELECT session.status,
                                   session.run_epoch,
                                   session.parent_session_id,
                                   ({session_transport_bytes})
                                   + COALESCE((
                                       SELECT SUM(
                                           octet_length(label.key)
                                           + octet_length(label.value)
                                       )
                                       FROM cayu_session_labels AS label
                                       WHERE label.session_id = session.id
                                   ), 0) AS transport_bytes
                            FROM cayu_sessions AS session
                            WHERE session.id = %s
                            """,
                        (session_id,),
                    )
                    session_preflight = await cur.fetchone()
                    if session_preflight is None:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.SESSION_NOT_FOUND
                        )
                    session_status = SessionStatus(session_preflight[0])
                    session_run_epoch = session_preflight[1]
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
                        and session_preflight[2] != expected_parent_session_id
                    ):
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                        )
                    session_transport_size = int(session_preflight[3])
                    if session_transport_size > max_record_transport_bytes:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED,
                            limit=max_record_transport_bytes,
                            observed=session_transport_size,
                        )

                    if observed is not None:
                        await cur.execute(
                            """
                            WITH bounded_identities AS (
                                SELECT octet_length(event_type)
                                           + octet_length(sequence::text) AS transport_bytes
                                FROM cayu_events
                                WHERE session_id = %s
                                ORDER BY sequence ASC
                                LIMIT %s
                            )
                            SELECT COUNT(*),
                                   COALESCE(MAX(transport_bytes), 0),
                                   COALESCE(SUM(transport_bytes), 0)
                            FROM bounded_identities
                            """,
                            (session_id, selected_limits.max_events + 1),
                        )
                        identity_preflight = await cur.fetchone()
                        if identity_preflight is None:
                            raise TerminalSessionEvidenceError(
                                TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                            )
                        identity_count = int(identity_preflight[0])
                        if identity_count > selected_limits.max_events:
                            raise TerminalSessionEvidenceError(
                                TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
                                limit=selected_limits.max_events,
                                observed=identity_count,
                            )
                        if identity_count != len(observed):
                            raise TerminalSessionEvidenceError(
                                TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                            )
                        identity_largest_bytes = int(identity_preflight[1])
                        if identity_largest_bytes > max_record_transport_bytes:
                            raise TerminalSessionEvidenceError(
                                TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED,
                                limit=max_record_transport_bytes,
                                observed=identity_largest_bytes,
                            )
                        identity_total_bytes = int(identity_preflight[2])
                        if identity_total_bytes > max_total_transport_bytes:
                            raise TerminalSessionEvidenceError(
                                TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED,
                                limit=max_total_transport_bytes,
                                observed=identity_total_bytes,
                            )
                        await cur.execute(
                            """
                            SELECT sequence, event_type
                            FROM cayu_events
                            WHERE session_id = %s
                            ORDER BY sequence ASC
                            LIMIT %s
                            """,
                            (session_id, identity_count),
                        )
                        identity_rows = await cur.fetchall()
                        _validate_runner_observed_event_identity_snapshot(
                            observed,
                            tuple(
                                RunnerObservedEventIdentity(
                                    session_id=session_id,
                                    sequence=row[0],
                                    event_type=row[1],
                                )
                                for row in identity_rows
                            ),
                        )

                    await cur.execute(
                        """
                        SELECT
                            CASE
                                WHEN state ? 'session_run_operation'
                                THEN jsonb_typeof(state -> 'session_run_operation')
                            END AS marker_type,
                            jsonb_typeof(
                                state #> '{session_run_operation,version}'
                            ) AS version_type,
                            state #>> '{session_run_operation,version}' AS version_value,
                            jsonb_typeof(
                                state #> '{session_run_operation,operation_id}'
                            ) AS operation_id_type,
                            octet_length(
                                state #>> '{session_run_operation,operation_id}'
                            ) AS operation_id_bytes,
                            length(btrim(COALESCE(
                                state #>> '{session_run_operation,operation_id}',
                                ''
                            ))) > 0 AS operation_id_nonblank,
                            jsonb_typeof(
                                state #> '{session_run_operation,run_epoch}'
                            ) AS run_epoch_type,
                            state #>> '{session_run_operation,run_epoch}' AS run_epoch_value,
                            state ? 'initial_transcript_pending'
                                AS initial_transcript_pending,
                            state ? 'pending_session_interrupt'
                                AS pending_session_interrupt
                        FROM cayu_checkpoints
                        WHERE session_id = %s
                        """,
                        (session_id,),
                    )
                    checkpoint_projection = await cur.fetchone()
                    marker: TerminalPublicationMarker | None = None
                    initial_transcript_pending = False
                    pending_session_interrupt = False
                    marker_stored_bytes = 0
                    if checkpoint_projection is not None:
                        initial_transcript_pending = bool(checkpoint_projection[8])
                        pending_session_interrupt = bool(checkpoint_projection[9])
                        marker_type = checkpoint_projection[0]
                        if marker_type is not None:
                            run_epoch_text = checkpoint_projection[7]
                            marker_valid = (
                                marker_type == "object"
                                and checkpoint_projection[1] == "number"
                                and checkpoint_projection[2] == "1"
                                and checkpoint_projection[3] == "string"
                                and bool(checkpoint_projection[5])
                                and checkpoint_projection[6] == "number"
                                and type(run_epoch_text) is str
                                and run_epoch_text.isascii()
                                and run_epoch_text.isdecimal()
                            )
                            if not marker_valid:
                                raise TerminalSessionEvidenceError(
                                    TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_INVALID
                                )
                            marker_run_epoch = int(run_epoch_text)
                            if not 1 <= marker_run_epoch <= MAX_DURABLE_JSON_INTEGER:
                                raise TerminalSessionEvidenceError(
                                    TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_INVALID
                                )
                            operation_id_bytes = int(checkpoint_projection[4])
                            marker_stored_bytes = operation_id_bytes + len(run_epoch_text)
                            if marker_stored_bytes > selected_limits.max_record_bytes:
                                raise TerminalSessionEvidenceError(
                                    TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
                                    limit=selected_limits.max_record_bytes,
                                )
                            await cur.execute(
                                """
                                SELECT state #>> '{session_run_operation,operation_id}'
                                FROM cayu_checkpoints
                                WHERE session_id = %s
                                """,
                                (session_id,),
                            )
                            operation_row = await cur.fetchone()
                            if operation_row is None:
                                raise TerminalSessionEvidenceError(
                                    TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                                )
                            try:
                                marker = TerminalPublicationMarker(
                                    operation_id=operation_row[0],
                                    run_epoch=marker_run_epoch,
                                )
                            except (TypeError, ValueError) as exc:
                                raise TerminalSessionEvidenceError(
                                    TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_INVALID
                                ) from exc

                    await cur.execute(
                        cast(
                            "LiteralString",
                            f"""
                            SELECT sequence,
                                   event_type,
                                   ({event_transport_bytes}) AS transport_bytes,
                                   jsonb_typeof(
                                       payload -> 'session_run_operation_id'
                                   ) AS operation_id_type,
                                   length(btrim(COALESCE(
                                       payload ->> 'session_run_operation_id',
                                       ''
                                   ))) > 0 AS operation_id_nonblank
                            FROM cayu_events
                            WHERE session_id = %s AND event_type = ANY(%s)
                            ORDER BY sequence DESC
                            LIMIT %s
                            """,
                        ),
                        (
                            session_id,
                            evidence_event_types,
                            _TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT,
                        ),
                    )
                    newest_preflight_rows = await cur.fetchall()
                    if any(
                        int(row[2]) > max_record_transport_bytes for row in newest_preflight_rows
                    ):
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED,
                            limit=max_record_transport_bytes,
                        )
                    if any(
                        row[3] not in {None, "string"} or (row[3] == "string" and not bool(row[4]))
                        for row in newest_preflight_rows
                    ):
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                        )
                    newest_sequences = [row[0] for row in newest_preflight_rows]
                    newest_evidence_records: tuple[EventRecord, ...]
                    if newest_sequences:
                        await cur.execute(
                            """
                            SELECT sequence,
                                   event_id,
                                   event_type,
                                   payload ->> 'session_run_operation_id'
                            FROM cayu_events
                            WHERE sequence = ANY(%s)
                            ORDER BY sequence DESC
                            """,
                            (newest_sequences,),
                        )
                        newest_rows = await cur.fetchall()
                        newest_evidence_records = tuple(
                            EventRecord(
                                sequence=row[0],
                                event=Event(
                                    id=row[1],
                                    type=row[2],
                                    session_id=session_id,
                                    payload=(
                                        {}
                                        if row[3] is None
                                        else {"session_run_operation_id": row[3]}
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

                    await cur.execute(
                        cast(
                            "LiteralString",
                            f"""
                            WITH bounded_events AS (
                                SELECT ({event_transport_bytes}) AS transport_bytes
                                FROM cayu_events
                                WHERE session_id = %s AND sequence <= %s
                                ORDER BY sequence ASC
                                LIMIT %s
                            )
                            SELECT COUNT(*),
                                   COALESCE(MAX(transport_bytes), 0),
                                   COALESCE(SUM(transport_bytes), 0)
                            FROM bounded_events
                            """,
                        ),
                        (session_id, terminal_record.sequence, selected_limits.max_events + 1),
                    )
                    event_preflight = await cur.fetchone()
                    await cur.execute(
                        cast(
                            "LiteralString",
                            f"""
                            WITH bounded_transcript AS (
                                SELECT ({transcript_transport_bytes}) AS transport_bytes
                                FROM cayu_transcript_messages
                                WHERE session_id = %s
                                ORDER BY session_order ASC
                                LIMIT %s
                            )
                            SELECT COUNT(*),
                                   COALESCE(MAX(transport_bytes), 0),
                                   COALESCE(SUM(transport_bytes), 0)
                            FROM bounded_transcript
                            """,
                        ),
                        (session_id, selected_limits.max_transcript_records + 1),
                    )
                    transcript_preflight = await cur.fetchone()
                    if event_preflight is None or transcript_preflight is None:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                        )
                    event_count = int(event_preflight[0])
                    transcript_count = int(transcript_preflight[0])
                    if event_count > selected_limits.max_events:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
                            limit=selected_limits.max_events,
                            observed=event_count,
                        )
                    if transcript_count > selected_limits.max_transcript_records:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.TRANSCRIPT_LIMIT_EXCEEDED,
                            limit=selected_limits.max_transcript_records,
                            observed=transcript_count,
                        )
                    largest_transport_bytes = max(
                        session_transport_size,
                        int(event_preflight[1]),
                        int(transcript_preflight[1]),
                        marker_stored_bytes,
                    )
                    if largest_transport_bytes > max_record_transport_bytes:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED,
                            limit=max_record_transport_bytes,
                            observed=largest_transport_bytes,
                        )
                    total_transport_bytes = (
                        session_transport_size
                        + int(event_preflight[2])
                        + int(transcript_preflight[2])
                        + marker_stored_bytes
                    )
                    if total_transport_bytes > max_total_transport_bytes:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED,
                            limit=max_total_transport_bytes,
                            observed=total_transport_bytes,
                        )

                    session = await self._load(cur, session_id)
                    if session is None:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                        )
                    if spool is not None:
                        after_sequence = 0
                        while True:
                            spool.check()
                            await cur.execute(
                                "SELECT sequence, event, input_contract_runtime_owned, "
                                f"file_attachment_attestations_runtime_owned, ({event_transport_bytes}), "
                                "((event->>'id') IS NOT DISTINCT FROM event_id AND "
                                "(event->>'session_id') IS NOT DISTINCT FROM session_id AND "
                                "(event->>'type') IS NOT DISTINCT FROM event_type AND "
                                "(event->>'interaction_id') IS NOT DISTINCT FROM interaction_id AND "
                                "(event->>'agent_name') IS NOT DISTINCT FROM agent_name AND "
                                "(event->>'environment_name') IS NOT DISTINCT FROM environment_name AND "
                                "(event->>'workflow_name') IS NOT DISTINCT FROM workflow_name AND "
                                "(event->>'tool_name') IS NOT DISTINCT FROM tool_name AND "
                                "(event->'payload') IS NOT DISTINCT FROM payload) "
                                "FROM cayu_events "
                                "WHERE session_id = %s AND sequence > %s AND sequence <= %s "
                                "ORDER BY sequence ASC LIMIT %s",
                                (
                                    session_id,
                                    after_sequence,
                                    terminal_record.sequence,
                                    spool.limits.batch_records,
                                ),
                            )
                            rows = await cur.fetchall()
                            if not rows:
                                break
                            spool.observe_page(
                                records=len(rows), transport_bytes=sum(row[4] for row in rows)
                            )
                            for row in rows:
                                if row[5] is not True:
                                    raise TerminalSessionEvidenceError(
                                        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                                    )
                                spool.append(
                                    "event",
                                    EventRecord(
                                        sequence=row[0],
                                        event=restore_persisted_event_authority(
                                            Event(**pg_support._json_obj(row[1])),
                                            input_contract_runtime_owned=row[2],
                                            file_attachment_attestations_runtime_owned=row[3],
                                        ),
                                    ),
                                )
                            after_sequence = rows[-1][0]
                        after_index = 0
                        while True:
                            spool.check()
                            await cur.execute(
                                f"SELECT session_order, interaction_id, message, ({transcript_transport_bytes}) "
                                "FROM cayu_transcript_messages WHERE session_id = %s "
                                "AND session_order > %s ORDER BY session_order ASC LIMIT %s",
                                (session_id, after_index, spool.limits.batch_records),
                            )
                            rows = await cur.fetchall()
                            if not rows:
                                break
                            spool.observe_page(
                                records=len(rows), transport_bytes=sum(row[3] for row in rows)
                            )
                            for row in rows:
                                spool.append(
                                    "transcript",
                                    TranscriptRecord(
                                        index=row[0] - 1,
                                        interaction_id=row[1],
                                        message=Message(**pg_support._json_obj(row[2])),
                                    ),
                                )
                            after_index = rows[-1][0]
                        spool.stage(session, marker, terminal_record, event_count, transcript_count)
                        return None
                    await cur.execute(
                        """
                        SELECT sequence, event, input_contract_runtime_owned,
                               file_attachment_attestations_runtime_owned
                        FROM cayu_events
                        WHERE session_id = %s AND sequence <= %s
                        ORDER BY sequence ASC
                        """,
                        (session_id, terminal_record.sequence),
                    )
                    event_rows = await cur.fetchall()
                    await cur.execute(
                        """
                        SELECT session_order - 1, interaction_id, message
                        FROM cayu_transcript_messages
                        WHERE session_id = %s
                        ORDER BY session_order ASC
                        """,
                        (session_id,),
                    )
                    transcript_rows = await cur.fetchall()
                    if len(event_rows) != event_count or len(transcript_rows) != transcript_count:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                        )
                    events = tuple(
                        EventRecord(
                            sequence=row[0],
                            event=restore_persisted_event_authority(
                                Event(**pg_support._json_obj(row[1])),
                                input_contract_runtime_owned=row[2],
                                file_attachment_attestations_runtime_owned=row[3],
                            ),
                        )
                        for row in event_rows
                    )
                    transcript = tuple(
                        TranscriptRecord(
                            index=row[0],
                            interaction_id=row[1],
                            message=Message(**pg_support._json_obj(row[2])),
                        )
                        for row in transcript_rows
                    )
                    return _assemble_terminal_session_evidence(
                        session=session,
                        marker=marker,
                        terminal_record=terminal_record,
                        events=events,
                        transcript=transcript,
                        limits=eager_limits,
                        allow_interrupted=allow_interrupted,
                    )
            except TerminalSessionEvidenceError:
                raise
            except (TypeError, ValueError) as exc:
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                ) from exc

    async def query_latest_interaction_events(
        self,
        session_id: str,
        *,
        before_sequence: int | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        session_id = require_clean_nonblank(session_id, "session_id")
        before_sequence, limit = _validate_interaction_page(before_sequence, limit)
        cursor_clause = "" if before_sequence is None else "AND latest.latest_event_sequence < %s"
        params: list[object] = [session_id]
        if before_sequence is not None:
            params.append(before_sequence)
        params.append(limit)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (session_id,))
            if await cur.fetchone() is None:
                raise KeyError(f"Session not found: {session_id}")
            await cur.execute(
                f"""
                    SELECT event.sequence, event.event
                    FROM cayu_interaction_latest_events AS latest
                    JOIN cayu_events AS event
                      ON event.sequence = latest.latest_event_sequence
                    WHERE latest.session_id = %s {cursor_clause}
                    ORDER BY latest.latest_event_sequence DESC
                    LIMIT %s
                    """,
                params,
            )
            rows = await cur.fetchall()
            return [
                EventRecord(sequence=row[0], event=Event(**pg_support._json_obj(row[1])))
                for row in rows
            ]

    async def _query_events(
        self,
        cur: Any,
        query: EventQuery,
        *,
        safe_insert_xid: Any,
        force_snapshot_cutoff: bool = False,
    ) -> list[EventRecord]:
        needs_snapshot_cutoff = force_snapshot_cutoff or _event_query_needs_snapshot_cutoff(query)
        if needs_snapshot_cutoff and safe_insert_xid is None:
            await cur.execute("SELECT pg_snapshot_xmin(pg_current_snapshot())")
            row = await cur.fetchone()
            if row is None:
                raise RuntimeError("Failed to read Postgres event visibility snapshot.")
            safe_insert_xid = row[0]
        extra_clauses: tuple[session_store_sql.SqlClause, ...] = ()
        if needs_snapshot_cutoff:
            # Postgres identity values are allocated at INSERT but published at COMMIT.
            # Cross-session event consumers must not advance an after_sequence cursor
            # past an event inserted by a still-open transaction with a lower identity.
            extra_clauses = (
                session_store_sql.SqlClause(
                    "cayu_events.insert_xid < %s",
                    (safe_insert_xid,),
                ),
            )
        plan = session_store_sql.build_event_query_sql(
            query,
            dialect=_SQL_DIALECT,
            extra_after_sequence_clauses=extra_clauses,
        )
        params = [*plan.params, query.limit]

        # where_sql is built only from hard-coded clause literals; all values
        # are bound via %s params, so the assembled text carries no user input.
        await cur.execute(
            cast(
                "LiteralString",
                f"""
                SELECT cayu_events.sequence,
                       cayu_events.event,
                       cayu_events.input_contract_runtime_owned,
                       cayu_events.file_attachment_attestations_runtime_owned
                FROM cayu_events
                JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
                {plan.where_sql}
                ORDER BY cayu_events.sequence {plan.order_direction}
                LIMIT %s
                """,
            ),
            params,
        )
        rows = await cur.fetchall()
        return [
            EventRecord(
                sequence=row[0],
                event=restore_persisted_event_authority(
                    Event(**pg_support._json_obj(row[1])),
                    input_contract_runtime_owned=row[2],
                    file_attachment_attestations_runtime_owned=row[3],
                ),
            )
            for row in rows
        ]

    async def _query_events_by_session_id_batches(self, query: EventQuery) -> list[EventRecord]:
        records: list[EventRecord] = []
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            safe_insert_xid = None
            needs_snapshot_cutoff = query.after_sequence is not None
            if needs_snapshot_cutoff:
                await cur.execute("SELECT pg_snapshot_xmin(pg_current_snapshot())")
                row = await cur.fetchone()
                if row is None:
                    raise RuntimeError("Failed to read Postgres event visibility snapshot.")
                safe_insert_xid = row[0]
            for batch in _event_query_session_id_batches(query.session_ids):
                records.extend(
                    await self._query_events(
                        cur,
                        session_store_sql.event_query_with_session_ids(
                            query,
                            session_ids=batch,
                        ),
                        safe_insert_xid=safe_insert_xid,
                        force_snapshot_cutoff=needs_snapshot_cutoff,
                    )
                )
        records.sort(
            key=lambda record: record.sequence,
            reverse=query.order_by.value == "sequence_desc",
        )
        return records[: query.limit]

    async def summarize_events(self, session_id: str) -> EventSummary:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            if access_bounds is not None:
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                access_bounds.require_read(await self._load(cur, session_id))
            await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (session_id,))
            if await cur.fetchone() is None:
                raise KeyError(f"Session not found: {session_id}")

            await cur.execute(
                "SELECT COUNT(*) FROM cayu_events WHERE session_id = %s",
                (session_id,),
            )
            total_row = await cur.fetchone()
            total_events = int(total_row[0]) if total_row is not None else 0

            await cur.execute(
                """
                SELECT event_type, COUNT(*)
                FROM cayu_events
                WHERE session_id = %s
                GROUP BY event_type
                ORDER BY event_type ASC
                """,
                (session_id,),
            )
            counts_by_type = {row[0]: int(row[1]) for row in await cur.fetchall()}

            await cur.execute(
                """
                SELECT sequence, event
                FROM cayu_events
                WHERE session_id = %s
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (session_id,),
            )
            latest_row = await cur.fetchone()

            return EventSummary(
                session_id=session_id,
                total_events=total_events,
                counts_by_type=counts_by_type,
                latest_event=_event_record_from_row(latest_row),
            )

    async def summarize_outcome(self, session_id: str) -> SessionOutcome:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            if access_bounds is not None:
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                access_bounds.require_read(await self._load(cur, session_id))
            session = await self._load(cur, session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")

            # Terminal and retry events are scoped to the latest session invocation:
            # only events after the most recent start/resume count, so a resumed
            # session does not surface a stale terminal event from a prior run.
            await cur.execute(
                """
                SELECT sequence, event
                FROM cayu_events
                WHERE session_id = %s
                  AND event_type = ANY(%s)
                  AND sequence > COALESCE(
                      (
                          SELECT MAX(sequence)
                          FROM cayu_events
                          WHERE session_id = %s
                            AND event_type = ANY(%s)
                      ),
                      0
                  )
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (session_id, _TERMINAL_EVENT_TYPES, session_id, _LIFECYCLE_EVENT_TYPES),
            )
            terminal_row = await cur.fetchone()

            await cur.execute(
                """
                SELECT sequence, event
                FROM cayu_events
                WHERE session_id = %s
                  AND event_type = %s
                  AND sequence > COALESCE(
                      (
                          SELECT MAX(sequence)
                          FROM cayu_events
                          WHERE session_id = %s
                            AND event_type = ANY(%s)
                      ),
                      0
                  )
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (session_id, str(EventType.MODEL_RETRY), session_id, _LIFECYCLE_EVENT_TYPES),
            )
            retry_row = await cur.fetchone()

            return session_outcome(
                session,
                terminal_event=_event_record_from_row(terminal_row),
                latest_retry_event=_event_record_from_row(retry_row),
            )

    async def list_sessions(self, query: SessionQuery | None = None) -> SessionListResult:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        if access_bounds is not None:
            return await self._access_list_sessions(access_bounds, copy_session_query(query))
        return await self._list_sessions(query, pending_interruption_cascade_only=False)

    async def query_session_topology(
        self,
        query: SessionTopologyQuery,
    ) -> SessionTopologyStoreResult:
        if type(query) is not SessionTopologyQuery:
            raise TypeError("Session topology queries must be SessionTopologyQuery instances.")
        query = query.model_copy(deep=True)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                    await cur.execute(
                        cast(
                            "LiteralString",
                            f"""
                            SELECT {pg_support.SESSION_TOPOLOGY_COLUMNS}
                            FROM cayu_sessions
                            WHERE id = %s
                            """,
                        ),
                        (query.focus_session_id,),
                    )
                    focus_row = await cur.fetchone()
                    if focus_row is None:
                        raise KeyError(f"Session not found: {query.focus_session_id}")
                    focus = pg_support.session_topology_node_from_row(focus_row)

                    ancestors = []
                    seen_ids = {focus.id}
                    parent_session_id = focus.parent_session_id
                    while parent_session_id is not None:
                        if parent_session_id in seen_ids:
                            raise SessionTopologyCycle(
                                f"Session topology contains a parent cycle at {parent_session_id}."
                            )
                        if len(ancestors) >= query.ancestor_depth_limit:
                            raise SessionTopologyDepthExceeded(
                                "Session topology exceeds the "
                                f"{query.ancestor_depth_limit}-ancestor limit."
                            )
                        await cur.execute(
                            cast(
                                "LiteralString",
                                f"""
                                SELECT {pg_support.SESSION_TOPOLOGY_COLUMNS}
                                FROM cayu_sessions
                                WHERE id = %s
                                """,
                            ),
                            (parent_session_id,),
                        )
                        parent_row = await cur.fetchone()
                        if parent_row is None:
                            raise ValueError(
                                f"Session topology references missing parent {parent_session_id}."
                            )
                        parent = pg_support.session_topology_node_from_row(parent_row)
                        ancestors.append(parent)
                        seen_ids.add(parent.id)
                        parent_session_id = parent.parent_session_id
                    ancestors.reverse()

                    expanded_parents = []
                    if query.expanded_parent_ids:
                        await cur.execute(
                            cast(
                                "LiteralString",
                                f"""
                                SELECT {pg_support.SESSION_TOPOLOGY_COLUMNS}
                                FROM cayu_sessions
                                WHERE id = ANY(%s)
                                """,
                            ),
                            (list(query.expanded_parent_ids),),
                        )
                        parents_by_id = {
                            row[0]: pg_support.session_topology_node_from_row(row)
                            for row in await cur.fetchall()
                        }
                        for parent_id in query.expanded_parent_ids:
                            parent = parents_by_id.get(parent_id)
                            if parent is None:
                                raise KeyError(f"Session not found: {parent_id}")
                            expanded_parents.append(parent)

                    candidates_by_parent = {parent.id: [] for parent in expanded_parents}
                    if expanded_parents:
                        requested_parent_ids: list[str] = []
                        cursor_created_ats: list[datetime | None] = []
                        cursor_ids: list[str | None] = []
                        for parent in expanded_parents:
                            requested_parent_ids.append(parent.id)
                            cursor = query.child_cursors.get(parent.id)
                            if cursor is None:
                                cursor_created_ats.append(None)
                                cursor_ids.append(None)
                                continue
                            cursor_created_at, cursor_id = decode_session_topology_cursor(
                                cursor,
                                parent_session_id=parent.id,
                            )
                            cursor_created_ats.append(cursor_created_at)
                            cursor_ids.append(cursor_id)
                        await cur.execute(
                            cast(
                                "LiteralString",
                                f"""
                                WITH requested_branches AS (
                                    SELECT parent_session_id, cursor_created_at, cursor_id,
                                           branch_order
                                    FROM unnest(
                                        %s::text[],
                                        %s::timestamptz[],
                                        %s::text[]
                                    ) WITH ORDINALITY AS requested(
                                        parent_session_id,
                                        cursor_created_at,
                                        cursor_id,
                                        branch_order
                                    )
                                )
                                SELECT child.*
                                FROM requested_branches AS requested
                                CROSS JOIN LATERAL (
                                    SELECT {pg_support.SESSION_TOPOLOGY_COLUMNS}
                                    FROM cayu_sessions
                                    WHERE cayu_sessions.parent_session_id =
                                          requested.parent_session_id
                                      AND (
                                          requested.cursor_created_at IS NULL
                                          OR cayu_sessions.created_at >
                                             requested.cursor_created_at
                                          OR (
                                              cayu_sessions.created_at =
                                                  requested.cursor_created_at
                                              AND cayu_sessions.id COLLATE "C" >
                                                  requested.cursor_id COLLATE "C"
                                          )
                                      )
                                    ORDER BY cayu_sessions.created_at ASC,
                                             cayu_sessions.id COLLATE "C" ASC
                                    LIMIT %s
                                ) AS child
                                ORDER BY requested.branch_order ASC,
                                         child.created_at ASC,
                                         child.id COLLATE "C" ASC
                                """,
                            ),
                            (
                                requested_parent_ids,
                                cursor_created_ats,
                                cursor_ids,
                                query.child_limit + 1,
                            ),
                        )
                        for row in await cur.fetchall():
                            candidates_by_parent[row[4]].append(
                                pg_support.session_topology_node_from_row(row)
                            )
                    result = build_session_topology_result(
                        focus=focus,
                        ancestors=ancestors,
                        expanded_parents=expanded_parents,
                        branch_candidates=(
                            candidates_by_parent[parent.id] for parent in expanded_parents
                        ),
                        child_limit=query.child_limit,
                    )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return result

    async def query_session_lineage(
        self,
        query: SessionLineageQuery,
    ) -> SessionLineageResult:
        query = copy_session_lineage_query(query)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                    await cur.execute(
                        "SELECT 1 FROM cayu_sessions WHERE id = %s",
                        (query.parent_session_id,),
                    )
                    if await cur.fetchone() is None:
                        raise KeyError(f"Session not found: {query.parent_session_id}")

                    cursor_clause = ""
                    params: list[object] = [
                        SESSION_LINEAGE_MAX_IDENTIFIER_BYTES,
                        query.parent_session_id,
                    ]
                    if query.cursor is not None:
                        cursor_created_at, cursor_id = decode_session_lineage_cursor(
                            query.cursor,
                            parent_session_id=query.parent_session_id,
                        )
                        cursor_clause = (
                            'AND (created_at > %s OR (created_at = %s AND id COLLATE "C" '
                            '> %s COLLATE "C"))'
                        )
                        params.extend((cursor_created_at, cursor_created_at, cursor_id))
                    params.append(query.limit + 1)
                    await cur.execute(
                        f"""
                        SELECT CASE
                                   WHEN octet_length(id) <= %s THEN id
                               END AS id,
                               created_at
                        FROM cayu_sessions
                        WHERE parent_session_id = %s
                          {cursor_clause}
                        ORDER BY created_at ASC, id COLLATE "C" ASC
                        LIMIT %s
                        """,
                        params,
                    )
                    rows = await cur.fetchall()
                    retained_rows = rows[: query.limit]
                    bases = tuple(
                        SessionLineageNode(
                            id=row[0],
                            parent_session_id=query.parent_session_id,
                            created_at=pg_support.to_utc(row[1]),
                        )
                        for row in retained_rows
                    )
                    grouped_origins: dict[str, list[SessionLineageOrigin]] = {
                        base.id: [] for base in bases
                    }
                    if bases:
                        await cur.execute(
                            """
                            SELECT requested.session_id, origin.sequence,
                                   origin.event_id, origin.event_type
                            FROM unnest(%s::text[]) WITH ORDINALITY AS requested(
                                session_id,
                                session_order
                            )
                            LEFT JOIN LATERAL (
                                SELECT sequence,
                                       CASE
                                           WHEN length(event_id) <= %s
                                            AND octet_length(event_id) <= %s
                                           THEN event_id
                                       END AS event_id,
                                       event_type
                                FROM cayu_events
                                WHERE cayu_events.session_id = requested.session_id
                                  AND event_type = ANY(%s)
                                ORDER BY sequence ASC
                                LIMIT %s
                            ) AS origin ON TRUE
                            ORDER BY requested.session_order ASC, origin.sequence ASC
                            """,
                            (
                                [base.id for base in bases],
                                EVENT_ID_MAX_CHARS,
                                SESSION_LINEAGE_MAX_EVENT_ID_BYTES,
                                [
                                    str(EventType.SESSION_STARTED),
                                    str(EventType.SESSION_FORKED),
                                ],
                                SESSION_LINEAGE_MAX_ORIGIN_EVENTS,
                            ),
                        )
                        for row in await cur.fetchall():
                            if row[1] is None:
                                continue
                            grouped_origins[row[0]].append(
                                SessionLineageOrigin(
                                    sequence=row[1],
                                    event_id=row[2],
                                    event_type=EventType(row[3]),
                                )
                            )
                    children = tuple(
                        SessionLineageNode(
                            id=base.id,
                            parent_session_id=base.parent_session_id,
                            created_at=base.created_at,
                            origin_events=tuple(grouped_origins[base.id]),
                        )
                        for base in bases
                    )
                    has_more = len(rows) > len(retained_rows)
                    result = SessionLineageResult(
                        parent_session_id=query.parent_session_id,
                        children=children,
                        next_cursor=(
                            encode_session_lineage_cursor(
                                query.parent_session_id,
                                children[-1],
                            )
                            if has_more and children
                            else None
                        ),
                        has_more=has_more,
                    )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return result

    async def query_child_session_lifecycle(
        self,
        query: ChildSessionLifecycleQuery,
    ) -> ChildSessionLifecyclePage:
        query = ChildSessionLifecycleQuery.model_validate(query)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                    parent = await self._load(cur, query.parent_session_id)
                    if parent is None:
                        raise KeyError(f"Session not found: {query.parent_session_id}")
                    await cur.execute(
                        """
                        SELECT child_session_id
                        FROM cayu_child_session_lifecycle_candidates
                        WHERE parent_session_id = %s
                        ORDER BY priority, sort_at, child_session_id COLLATE "C"
                        LIMIT %s
                        """,
                        (
                            parent.id,
                            query.max_children_inspected + 1,
                        ),
                    )
                    rows = await cur.fetchall()
                    retained_rows = rows[: query.max_children_inspected]
                    retained_ids = [str(row[0]) for row in retained_rows]
                    entries = []
                    unavailable_count = 0
                    lifecycle_types = [
                        str(EventType.SESSION_STARTED),
                        str(EventType.SESSION_RESUMED),
                        str(EventType.SESSION_FORKED),
                        str(EventType.SESSION_COMPLETED),
                        str(EventType.SESSION_FAILED),
                        str(EventType.SESSION_INTERRUPTED),
                    ]
                    children_by_id: dict[str, Session] = {}
                    records_by_child: dict[str, dict[EventType, EventRecord]] = {
                        child_id: {} for child_id in retained_ids
                    }
                    if retained_ids:
                        await cur.execute(
                            f"SELECT {pg_support.SESSION_COLUMNS} FROM cayu_sessions "
                            "WHERE id = ANY(%s)",
                            (retained_ids,),
                        )
                        children_by_id = {
                            str(child_row[0]): pg_support.session_from_row(
                                child_row,
                                labels={},
                            )
                            for child_row in await cur.fetchall()
                        }
                        await cur.execute(
                            """
                            SELECT latest.session_id, latest.sequence, latest.event
                            FROM (
                                SELECT DISTINCT ON (session_id, event_type)
                                       session_id, event_type, sequence, event
                                FROM cayu_events
                                WHERE session_id = ANY(%s)
                                  AND event_type = ANY(%s)
                                ORDER BY session_id, event_type, sequence DESC
                            ) AS latest
                            ORDER BY latest.session_id, latest.sequence ASC
                            """,
                            (retained_ids, lifecycle_types),
                        )
                        for event_row in await cur.fetchall():
                            event = Event(**pg_support._json_obj(event_row[2]))
                            records_by_child[str(event_row[0])][EventType(event.type)] = (
                                EventRecord(sequence=event_row[1], event=event)
                            )

                    consumption_key_by_child: dict[str, str] = {}
                    for child_id in retained_ids:
                        child = children_by_id.get(child_id)
                        if child is None or child.parent_session_id != parent.id:
                            raise RuntimeError(
                                "Postgres child-session lifecycle index is inconsistent."
                            )
                        occurrence_source = _child_session_lifecycle_occurrence(
                            child,
                            records_by_child[child_id],
                        )
                        if occurrence_source is not None:
                            _relationship, occurrence = occurrence_source
                            consumption_key_by_child[child_id] = (
                                child_session_notification_storage_key(
                                    child.instance_id,
                                    occurrence.source_id,
                                )
                            )
                    consumption_by_key: dict[str, dict[str, Any]] = {}
                    if consumption_key_by_child:
                        await cur.execute(
                            "SELECT idempotency_key, record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                            (parent.id, list(consumption_key_by_child.values())),
                        )
                        consumption_by_key = {
                            str(operation_row[0]): pg_support._json_obj(operation_row[1])
                            for operation_row in await cur.fetchall()
                        }

                    for child_id in retained_ids:
                        child = children_by_id[child_id]
                        consumption_key = consumption_key_by_child.get(child_id)
                        entry = _child_session_lifecycle_entry(
                            parent=parent,
                            child=child,
                            records_by_type=records_by_child[child_id],
                            consumption_record=(
                                None
                                if consumption_key is None
                                else consumption_by_key.get(consumption_key)
                            ),
                        )
                        if entry is None:
                            unavailable_count += 1
                        else:
                            entries.append(entry)
                    entries.sort(key=_child_session_lifecycle_entry_sort_key)
                    result = ChildSessionLifecyclePage(
                        parent_session_id=parent.id,
                        parent_session_instance_id=parent.instance_id,
                        entries=tuple(entries),
                        inspected_child_count=len(retained_rows),
                        unavailable_child_count=unavailable_count,
                        has_more=len(rows) > len(retained_rows),
                    )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return result

    async def aggregate_operational_snapshot(
        self,
        filters: SessionAggregateFilter | None = None,
    ) -> SessionOperationalSnapshot:
        filters = copy_session_aggregate_filter(filters)
        plan = session_store_sql.build_session_query_sql(
            session_query_from_aggregate_filter(filters),
            dialect=_SQL_DIALECT,
        )
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                    await cur.execute("SELECT transaction_timestamp()")
                    as_of_row = await cur.fetchone()
                    if as_of_row is None:
                        raise RuntimeError("Postgres did not return a snapshot timestamp.")
                    await cur.execute(
                        cast(
                            "LiteralString",
                            f"""
                            SELECT status, COUNT(*)
                            FROM cayu_sessions
                            {plan.filter_where_sql}
                            GROUP BY status
                            """,
                        ),
                        plan.filter_params,
                    )
                    rows = await cur.fetchall()
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

        counts = {status: 0 for status in SessionStatus}
        for row in rows:
            counts[SessionStatus(row[0])] = row[1]
        return SessionOperationalSnapshot(
            as_of=as_of_row[0],
            total_count=sum(counts.values()),
            counts_by_status=SessionStatusCounts.model_validate(counts),
            accuracy=EXACT_AGGREGATE.model_copy(),
        )

    async def aggregate_usage(self, query: UsageRollupQuery) -> UsageRollupStoreResult:
        query = copy_usage_rollup_query(query)
        plan = session_store_sql.build_session_query_sql(
            session_query_from_aggregate_filter(query.sessions),
            dialect=_SQL_DIALECT,
        )
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                    await cur.execute("SELECT transaction_timestamp()")
                    as_of_row = await cur.fetchone()
                    if as_of_row is None:
                        raise RuntimeError("Postgres did not return a snapshot timestamp.")
                result = await postgres_aggregates.aggregate_session_usage(
                    conn,
                    session_plan=plan,
                    query=query,
                    as_of=as_of_row[0],
                )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return result

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
            cursor_sql = (
                'WHERE session_id COLLATE "C" > %s OR '
                '(session_id COLLATE "C" = %s '
                'AND operation_id COLLATE "C" > %s)'
            )
            params.extend(
                [
                    query.after_session_id,
                    query.after_session_id,
                    query.after_operation_id,
                ]
            )
        params.append(query.limit)

        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                f"""
                    WITH queued_dispatch_receipts AS (
                        SELECT
                            checkpoint.session_id,
                            checkpoint.state
                                #>> '{{session_run_operation,queue_task_id}}'
                                AS queue_task_id,
                            checkpoint.state
                                #>> '{{session_run_operation,operation_id}}'
                                AS operation_id,
                            checkpoint.state
                                #>> '{{session_run_operation,terminal_event_id}}'
                                AS terminal_event_id
                        FROM cayu_checkpoints AS checkpoint
                        INNER JOIN cayu_events AS terminal_event
                            ON terminal_event.session_id = checkpoint.session_id
                           AND terminal_event.event_id = checkpoint.state
                                #>> '{{session_run_operation,terminal_event_id}}'
                        WHERE checkpoint.state
                            #> '{{session_run_operation,queue_task_id}}'
                            IS NOT NULL

                        UNION

                        SELECT
                            checkpoint.session_id,
                            receipt.value ->> 'queue_task_id' AS queue_task_id,
                            receipt.key AS operation_id,
                            receipt.value ->> 'terminal_event_id' AS terminal_event_id
                        FROM cayu_checkpoints AS checkpoint
                        CROSS JOIN LATERAL jsonb_each(
                            COALESCE(
                                checkpoint.state
                                    #> '{{queued_dispatch_terminal_receipts,receipts}}',
                                '{{}}'::jsonb
                            )
                        ) AS receipt(key, value)
                        WHERE checkpoint.state
                            #> '{{queued_dispatch_terminal_receipts,receipts}}'
                            IS NOT NULL
                    )
                    SELECT session_id, queue_task_id, operation_id, terminal_event_id
                    FROM queued_dispatch_receipts
                    {cursor_sql}
                    ORDER BY session_id COLLATE "C" ASC,
                             operation_id COLLATE "C" ASC
                    LIMIT %s
                    """,
                params,
            )
            rows = await cur.fetchall()
        return [
            QueuedDispatchTerminalReceipt(
                session_id=row[0],
                queue_task_id=row[1],
                operation_id=row[2],
                terminal_event_id=row[3],
            )
            for row in rows
        ]

    async def query_pending_actions(
        self,
        query: PendingActionQuery | None = None,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> PendingActionListResult:
        from cayu.sessions.pending_actions import (
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
        status_values = sorted(status.value for status in query.statuses)
        status_placeholders = ", ".join("%s" for _status in status_values)
        filters = [
            f"cayu_sessions.status IN ({status_placeholders})",
            "cayu_checkpoints.pending_action_metrics_ready",
            "cayu_checkpoints.pending_action_flags <> 0",
        ]
        params: list[Any] = list(status_values)
        if query.session_id is not None:
            filters.append("cayu_sessions.id = %s")
            params.append(query.session_id)
        if query.agent_name is not None:
            filters.append("cayu_sessions.agent_name = %s")
            params.append(query.agent_name)
        if query.environment_name is not None:
            filters.append("cayu_sessions.environment_name = %s")
            params.append(query.environment_name)
        if query.kind == PendingActionKind.TOOL_APPROVAL:
            filters.append("(cayu_checkpoints.pending_action_flags & 1) <> 0")
        elif query.kind == PendingActionKind.USER_INPUT:
            filters.append("(cayu_checkpoints.pending_action_flags & 2) <> 0")
        elif query.kind == PendingActionKind.DELEGATED_ACTION:
            filters.append("(cayu_checkpoints.pending_action_flags & 8) <> 0")
        if query.cursor is not None:
            cursor_dt, cursor_id = decode_session_cursor(query.cursor)
            filters.append(
                """
                (
                    cayu_sessions.updated_at < %s
                    OR (cayu_sessions.updated_at = %s AND cayu_sessions.id > %s)
                )
                """
            )
            params.extend((cursor_dt, cursor_dt, cursor_id))
        where_sql = " AND ".join(f"({clause.strip()})" for clause in filters)
        session_columns = ", ".join(
            f"cayu_sessions.{column.strip()}"
            for column in pg_support.PENDING_ACTION_SESSION_COLUMNS.split(",")
        )
        session_columns += (
            ", cayu_sessions.metadata -> 'cayu:runtime_build_provenance' "
            "AS runtime_build_provenance, cayu_sessions.instance_id"
        )
        candidate_select_sql = cast(
            "LiteralString",
            f"""
            SELECT {session_columns}
            FROM cayu_checkpoints
            JOIN cayu_sessions ON cayu_sessions.id = cayu_checkpoints.session_id
            WHERE {where_sql}
            ORDER BY cayu_sessions.updated_at DESC, cayu_sessions.id ASC
            LIMIT %s
            """,
        )
        selected_candidate_sql = """
            SELECT
                cayu_checkpoints.session_id AS id,
                jsonb_strip_nulls(jsonb_build_object(
                    'pending_tool_approval',
                    cayu_checkpoints.state -> 'pending_tool_approval',
                    'pending_user_input',
                    cayu_checkpoints.state -> 'pending_user_input',
                    'pending_tool_round',
                    cayu_checkpoints.state -> 'pending_tool_round',
                    'foreground_child_wait',
                    cayu_checkpoints.state -> 'foreground_child_wait'
                )) AS pending_state
            FROM cayu_checkpoints
            WHERE cayu_checkpoints.session_id = ANY(%s)
        """
        checkpoint_root_key = (
            "__cayu_no_checkpoint_root_guard__"
            if checkpoint_root_guard is None
            else checkpoint_root_guard.key
        )
        checkpoint_preflight_sql = f"""
            SELECT
                cayu_checkpoints.session_id,
                cayu_checkpoints.pending_action_source_bytes AS pending_state_bytes,
                cayu_checkpoints.pending_action_tool_call_count,
                jsonb_typeof(
                    cayu_checkpoints.state -> '{checkpoint_root_key}'
                ),
                left(
                    cayu_checkpoints.state ->> '{checkpoint_root_key}',
                    {CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS + 1}
                )
            FROM cayu_checkpoints
            WHERE cayu_checkpoints.session_id = ANY(%s)
        """
        projected_event_sql = "source_event.pending_action_projection"
        pending_action_ctes = f"""
            WITH candidates AS MATERIALIZED ({selected_candidate_sql}),
            candidate_tool_scopes AS MATERIALIZED (
                SELECT candidates.id AS session_id,
                    CASE
                        WHEN jsonb_typeof(
                            candidates.pending_state -> 'pending_tool_approval'
                        ) = 'object'
                        THEN candidates.pending_state -> 'pending_tool_approval'
                        WHEN jsonb_typeof(
                            candidates.pending_state -> 'pending_user_input'
                        ) = 'object'
                        THEN candidates.pending_state -> 'pending_user_input'
                        WHEN jsonb_typeof(
                            candidates.pending_state -> 'pending_tool_round'
                        ) = 'object'
                        THEN candidates.pending_state -> 'pending_tool_round'
                        ELSE NULL
                    END AS pending_tool_state
                FROM candidates
            ),
            candidate_tool_calls AS MATERIALIZED (
                SELECT
                    tool_scope.session_id,
                    pending_call ->> 'tool_call_id' AS tool_call_id
                FROM candidate_tool_scopes AS tool_scope
                CROSS JOIN LATERAL jsonb_array_elements(
                    CASE
                        WHEN jsonb_typeof(
                            tool_scope.pending_tool_state -> 'tool_calls'
                        ) = 'array'
                        THEN tool_scope.pending_tool_state -> 'tool_calls'
                        ELSE '[]'::jsonb
                    END
                ) AS pending_call
                WHERE jsonb_typeof(pending_call -> 'tool_call_id') = 'string'
            ),
            candidate_action_keys AS (
                SELECT id AS session_id,
                    encode(sha256(convert_to(
                        pending_state #>> '{{pending_tool_approval,approval_id}}',
                        'UTF8'
                    )), 'hex') AS action_key
                FROM candidates
                WHERE jsonb_typeof(
                    pending_state #> '{{pending_tool_approval,approval_id}}'
                ) = 'string'
                UNION
                SELECT id, encode(sha256(convert_to(
                    pending_state #>> '{{pending_user_input,input_id}}',
                    'UTF8'
                )), 'hex')
                FROM candidates
                WHERE jsonb_typeof(
                    pending_state #> '{{pending_user_input,input_id}}'
                ) = 'string'
                UNION
                SELECT tool_scope.session_id, encode(sha256(convert_to(
                    tool_scope.pending_tool_state ->> 'tool_round_id',
                    'UTF8'
                )), 'hex')
                FROM candidate_tool_scopes AS tool_scope
                WHERE jsonb_typeof(
                    tool_scope.pending_tool_state -> 'tool_round_id'
                ) = 'string'
                UNION
                SELECT pending_call.session_id, encode(sha256(convert_to(
                    pending_call.tool_call_id,
                    'UTF8'
                )), 'hex')
                FROM candidate_tool_calls AS pending_call
            ),
            pending_action_event_types(event_type) AS (
                VALUES
                    ('tool.call.approval_requested'),
                    ('session.awaiting_user_input'),
                    ('session.interrupted'),
                    ('session.delegated_action.updated')
            ),
            latest_barriers AS (
                SELECT candidates.id AS session_id,
                    COALESCE((
                        SELECT MAX(event.sequence)
                        FROM cayu_events AS event
                        WHERE event.session_id = candidates.id
                          AND (
                              event.event_type = 'session.resumed'
                              OR event.event_type = 'session.completed'
                              OR event.event_type = 'session.failed'
                          )
                    ), 0) AS sequence
                FROM candidates
            ),
            matched_action_events AS (
                SELECT
                    action_keys.session_id AS candidate_session_id,
                    event.sequence
                FROM candidate_action_keys AS action_keys
                CROSS JOIN pending_action_event_types AS action_type
                CROSS JOIN LATERAL (
                    SELECT candidate_event.sequence
                    FROM cayu_events AS candidate_event
                    WHERE candidate_event.session_id = action_keys.session_id
                      AND candidate_event.event_type = action_type.event_type
                      AND candidate_event.event_type IN (
                          'tool.call.approval_requested',
                          'session.awaiting_user_input',
                          'session.interrupted',
                          'session.delegated_action.updated',
                          'tool.call.started',
                          'tool.call.completed',
                          'tool.call.failed',
                          'tool.call.blocked',
                          'tool.call.approval_denied'
                      )
                      AND candidate_event.pending_action_lookup_key IS NOT NULL
                      AND candidate_event.pending_action_lookup_key = action_keys.action_key
                    ORDER BY candidate_event.sequence DESC
                    LIMIT 1
                ) AS event
            ),
            matched_ledger_events AS (
                SELECT
                    action_keys.session_id AS candidate_session_id,
                    action_keys.action_key,
                    event.sequence
                FROM candidate_action_keys AS action_keys
                JOIN candidates ON candidates.id = action_keys.session_id
                JOIN candidate_tool_scopes AS tool_scope
                    ON tool_scope.session_id = action_keys.session_id
                CROSS JOIN LATERAL (
                    SELECT candidate_event.sequence
                    FROM cayu_events AS candidate_event
                    WHERE candidate_event.session_id = action_keys.session_id
                      AND candidate_event.pending_action_lookup_key
                          = action_keys.action_key
                      AND candidate_event.event_type IN (
                          'tool.call.approval_requested',
                          'session.awaiting_user_input',
                          'session.interrupted',
                          'session.delegated_action.updated',
                          'tool.call.started',
                          'tool.call.completed',
                          'tool.call.failed',
                          'tool.call.blocked',
                          'tool.call.approval_denied'
                      )
                      AND candidate_event.event_type IN (
                          'tool.call.started',
                          'tool.call.completed',
                          'tool.call.failed',
                          'tool.call.blocked',
                          'tool.call.approval_denied'
                      )
                      AND candidate_event.pending_action_lookup_key IS NOT NULL
                      AND (
                          candidate_event.pending_action_projection
                              #>> '{{payload,tool_round_id}}'
                              = tool_scope.pending_tool_state ->> 'tool_round_id'
                          OR (
                              candidate_event.pending_action_projection
                                  #>> '{{payload,model_step_id}}'
                                  = tool_scope.pending_tool_state ->> 'model_step_id'
                              AND candidate_event.pending_action_projection
                                  #>> '{{payload,model_attempt_id}}'
                                  = tool_scope.pending_tool_state ->> 'model_attempt_id'
                          )
                      )
                    LIMIT {MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL + 1}
                ) AS event
                WHERE jsonb_typeof(
                    tool_scope.pending_tool_state
                ) = 'object'
            ),
            scope_conflict_events AS MATERIALIZED (
                SELECT
                    tool_scope.session_id AS candidate_session_id,
                    conflict.sequence
                FROM candidate_tool_scopes AS tool_scope
                CROSS JOIN LATERAL (
                    (
                        SELECT scoped_event.sequence
                        FROM cayu_events AS scoped_event
                        WHERE scoped_event.session_id = tool_scope.session_id
                          AND scoped_event.event_type IN (
                              'tool.call.started',
                              'tool.call.completed',
                              'tool.call.failed',
                              'tool.call.blocked',
                              'tool.call.approval_denied'
                          )
                          AND jsonb_typeof(
                              scoped_event.pending_action_projection
                                  #> '{{payload,tool_round_id}}'
                          ) = 'string'
                          AND scoped_event.pending_action_projection
                              #>> '{{payload,tool_round_id}}'
                              ~ '^tround_[0-9a-f]{{32}}$'
                          AND scoped_event.pending_action_projection
                              #>> '{{payload,tool_round_id}}'
                              = tool_scope.pending_tool_state ->> 'tool_round_id'
                          AND NOT COALESCE(
                              scoped_event.pending_action_projection
                                  #>> '{{payload,tool_round_id}}'
                                  = tool_scope.pending_tool_state ->> 'tool_round_id'
                              AND scoped_event.pending_action_projection
                                  #>> '{{payload,model_step_id}}'
                                  = tool_scope.pending_tool_state ->> 'model_step_id'
                              AND scoped_event.pending_action_projection
                                  #>> '{{payload,model_attempt_id}}'
                                  = tool_scope.pending_tool_state ->> 'model_attempt_id'
                              AND EXISTS (
                                  SELECT 1
                                  FROM candidate_tool_calls AS pending_call
                                  WHERE pending_call.session_id = tool_scope.session_id
                                    AND pending_call.tool_call_id
                                        = scoped_event.pending_action_projection
                                            #>> '{{payload,tool_call_id}}'
                              ),
                              FALSE
                          )
                        LIMIT 1
                    )
                    UNION
                    (
                        SELECT scoped_event.sequence
                        FROM cayu_events AS scoped_event
                        WHERE scoped_event.session_id = tool_scope.session_id
                          AND scoped_event.event_type IN (
                              'tool.call.started',
                              'tool.call.completed',
                              'tool.call.failed',
                              'tool.call.blocked',
                              'tool.call.approval_denied'
                          )
                          AND jsonb_typeof(
                              scoped_event.pending_action_projection
                                  #> '{{payload,model_step_id}}'
                          ) = 'string'
                          AND jsonb_typeof(
                              scoped_event.pending_action_projection
                                  #> '{{payload,model_attempt_id}}'
                          ) = 'string'
                          AND scoped_event.pending_action_projection
                              #>> '{{payload,model_step_id}}'
                              ~ '^mstep_[0-9a-f]{{32}}$'
                          AND scoped_event.pending_action_projection
                              #>> '{{payload,model_attempt_id}}'
                              ~ '^matt_[0-9a-f]{{32}}$'
                          AND scoped_event.pending_action_projection
                              #>> '{{payload,model_step_id}}'
                              = tool_scope.pending_tool_state ->> 'model_step_id'
                          AND scoped_event.pending_action_projection
                              #>> '{{payload,model_attempt_id}}'
                              = tool_scope.pending_tool_state ->> 'model_attempt_id'
                          AND NOT COALESCE(
                              scoped_event.pending_action_projection
                                  #>> '{{payload,tool_round_id}}'
                                  = tool_scope.pending_tool_state ->> 'tool_round_id'
                              AND scoped_event.pending_action_projection
                                  #>> '{{payload,model_step_id}}'
                                  = tool_scope.pending_tool_state ->> 'model_step_id'
                              AND scoped_event.pending_action_projection
                                  #>> '{{payload,model_attempt_id}}'
                                  = tool_scope.pending_tool_state ->> 'model_attempt_id'
                              AND EXISTS (
                                  SELECT 1
                                  FROM candidate_tool_calls AS pending_call
                                  WHERE pending_call.session_id = tool_scope.session_id
                                    AND pending_call.tool_call_id
                                        = scoped_event.pending_action_projection
                                            #>> '{{payload,tool_call_id}}'
                              ),
                              FALSE
                          )
                        LIMIT 1
                    )
                    LIMIT 1
                ) AS conflict
                WHERE jsonb_typeof(tool_scope.pending_tool_state) = 'object'
            ),
            matched_event_sequences AS (
                SELECT candidate_session_id, sequence
                FROM matched_action_events
                UNION
                SELECT candidate_session_id, sequence
                FROM matched_ledger_events
                UNION
                SELECT candidate_session_id, sequence
                FROM scope_conflict_events
                UNION
                SELECT candidates.id, event.sequence
                FROM candidates
                JOIN latest_barriers ON latest_barriers.session_id = candidates.id
                JOIN cayu_events AS event ON event.sequence = latest_barriers.sequence
            ),
            matched_events AS MATERIALIZED (
                SELECT
                    matched_event_sequences.candidate_session_id,
                    source_event.sequence,
                    source_event.pending_action_projection_bytes AS event_bytes,
                    source_event.pending_action_projection_bytes IS NOT NULL
                        AND source_event.pending_action_projection IS NOT NULL
                        AS projection_ready
                FROM matched_event_sequences
                JOIN cayu_events AS source_event
                    ON source_event.sequence = matched_event_sequences.sequence
            )
        """
        source_size_sql = f"""
            {pending_action_ctes}
            SELECT candidates.id,
                octet_length(candidates.pending_state::text)
                + COALESCE((
                    SELECT SUM(octet_length(jsonb_build_object(
                        'key', label.key,
                        'value', label.value
                    )::text))
                    FROM cayu_session_labels AS label
                    WHERE label.session_id = candidates.id
                ), 0)
                + COALESCE((
                    SELECT SUM(
                        matched_event.event_bytes
                        + length(matched_event.sequence::text)
                        + 22
                    )
                    FROM matched_events AS matched_event
                    WHERE matched_event.candidate_session_id = candidates.id
                ), 0) AS source_bytes,
                COALESCE((
                    SELECT bool_and(matched_event.projection_ready)
                    FROM matched_events AS matched_event
                    WHERE matched_event.candidate_session_id = candidates.id
                ), true) AS projections_ready,
                EXISTS (
                    SELECT 1
                    FROM matched_ledger_events AS matched_ledger
                    WHERE matched_ledger.candidate_session_id = candidates.id
                    GROUP BY matched_ledger.action_key
                    HAVING COUNT(*) > {MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL}
                ) AS ledger_too_complex,
                COALESCE((
                    SELECT jsonb_agg(
                        matched_event.sequence ORDER BY matched_event.sequence DESC
                    )
                    FROM matched_events AS matched_event
                    WHERE matched_event.candidate_session_id = candidates.id
                ), '[]'::jsonb) AS matched_event_sequences
            FROM candidates
        """
        materialize_sql = f"""
            WITH candidates AS MATERIALIZED ({selected_candidate_sql}),
            matched_events AS MATERIALIZED (
                SELECT
                    source_event.session_id AS candidate_session_id,
                    source_event.sequence,
                    {projected_event_sql} AS event
                FROM cayu_events AS source_event
                WHERE source_event.sequence = ANY(%s)
            )
            SELECT candidates.id, candidates.pending_state,
                COALESCE((
                    SELECT jsonb_agg(
                        jsonb_build_object(
                            'sequence', matched_event.sequence,
                            'event', matched_event.event
                        )
                        ORDER BY matched_event.sequence DESC
                    )
                    FROM matched_events AS matched_event
                    WHERE matched_event.candidate_session_id = candidates.id
                ), '[]'::jsonb) AS pending_events
            FROM candidates
        """

        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            # Candidate selection, byte accounting, projection reads, and labels all
            # observe one immutable snapshot. The look-ahead row is selected only
            # as bounded session metadata and never enters JSON projection work.
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            await cur.execute(candidate_select_sql, [*params, candidate_limit])
            candidate_rows = await cur.fetchall()
            has_more_candidates = len(candidate_rows) > inspected_candidate_limit
            inspected_rows = candidate_rows[:inspected_candidate_limit]
            candidate_sessions = {
                str(row[0]): pg_support.pending_action_session_from_row(row, labels={})
                for row in inspected_rows
            }
            inspected_ids = [str(row[0]) for row in inspected_rows]

            checkpoint_preflight_by_session_id: dict[str, tuple[int, int]] = {}
            if inspected_ids:
                await cur.execute(checkpoint_preflight_sql, (inspected_ids,))
                for row in await cur.fetchall():
                    scalar_text = row[4]
                    if checkpoint_root_guard is not None:
                        checkpoint_root_guard.validate(
                            str(row[0]),
                            checkpoint_root_field_projection_from_storage(
                                json_type=row[3],
                                scalar_text=scalar_text,
                            ),
                        )
                    if row[1] is not None:
                        checkpoint_preflight_by_session_id[str(row[0])] = (
                            int(row[1]),
                            int(row[2]),
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
                await cur.execute(source_size_sql, (preflight_eligible_ids,))
                for row in await cur.fetchall():
                    sequence_values = copy_durable_json_value(
                        row[4],
                        "matched event sequences",
                    )
                    if type(sequence_values) is not list or any(
                        type(sequence) is not int for sequence in sequence_values
                    ):
                        raise ValueError(
                            "Postgres pending event sequence projection must be an integer array."
                        )
                    source_metadata_by_session_id[str(row[0])] = (
                        int(row[1]),
                        sequence_values,
                    )
                    if not bool(row[2]):
                        invalid_ids.add(str(row[0]))
                    if bool(row[3]):
                        ledger_overcomplex_ids.add(str(row[0]))

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
                await cur.execute(
                    materialize_sql,
                    (materializable_ids, materializable_sequences),
                )
                for row in await cur.fetchall():
                    session_id = str(row[0])
                    records: list[EventRecord] = []
                    pending_events = copy_durable_json_value(row[2], "pending events")
                    if type(pending_events) is not list:
                        raise ValueError("Postgres pending events projection must be an array.")
                    for pending_event in pending_events:
                        if type(pending_event) is not dict:
                            raise ValueError("Postgres pending event projections must be objects.")
                        records.append(
                            EventRecord(
                                sequence=pending_event.get("sequence"),
                                event=Event(**pg_support._json_obj(pending_event.get("event"))),
                            )
                        )
                    grouped[session_id] = (
                        copy_durable_json_object(pg_support._json_obj(row[1]), "checkpoint"),
                        records,
                    )

            labels_by_session_id = await self._load_labels_for_sessions(cur, materializable_ids)
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
                            max_events_per_call=MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL,
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
                    update={"labels": labels_by_session_id.get(session_id, {})}, deep=True
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
                encode_session_cursor(last_inspected_session, SessionOrder.UPDATED_AT_DESC)
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

    async def _list_sessions(
        self,
        query: SessionQuery | None,
        *,
        pending_interruption_cascade_only: bool,
        access_bounds: _SessionAccessBounds | None = None,
    ) -> SessionListResult:
        query = copy_session_query(query)
        session_source_sql = (
            """
            (
                SELECT session_id
                FROM cayu_checkpoints
                WHERE state ? 'pending_interruption_cascade'
            ) AS pending_interruption_cascades
            INNER JOIN cayu_sessions
                ON cayu_sessions.id = pending_interruption_cascades.session_id
            """
            if pending_interruption_cascade_only
            else "cayu_sessions"
        )
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            if access_bounds is not None:
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            inactive_before = query.last_activity_before
            if query.inactive_for_seconds is not None:
                await cur.execute("SELECT clock_timestamp()")
                now_row = await cur.fetchone()
                if now_row is None or type(now_row[0]) is not datetime:
                    raise RuntimeError("Postgres did not return authoritative store time.")
                inactive_before = utc_duration_cutoff(
                    now_row[0],
                    query.inactive_for_seconds,
                )
                if inactive_before is None:
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
                access_clause=(
                    None
                    if access_bounds is None
                    else session_store_sql.session_access_clause(
                        access_bounds, dialect=_SQL_DIALECT
                    )
                ),
            )
            # Interpolations are trusted: SESSION_COLUMNS is a constant, order_sql is
            # an enum-derived literal, the clauses are hard-coded; values bind via %s.
            total_count: int | None = None
            if query.include_total_count:
                await cur.execute(
                    cast(
                        "LiteralString",
                        f"SELECT COUNT(*) FROM {session_source_sql} {plan.filter_where_sql}",
                    ),
                    plan.filter_params,
                )
                count_row = await cur.fetchone()
                total_count = count_row[0] if count_row is not None else 0
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    SELECT {pg_support.SESSION_COLUMNS}
                    FROM {session_source_sql}
                    {plan.page_where_sql}
                    ORDER BY {plan.order_sql}
                    {plan.pagination_sql}
                    """,
                ),
                plan.page_params,
            )
            rows = await cur.fetchall()
            has_more = len(rows) > query.limit
            rows = rows[: query.limit]
            labels_by_session_id = await self._load_labels_for_sessions(
                cur,
                [row[0] for row in rows],
            )
            sessions = [
                pg_support.session_from_row(
                    row,
                    labels=labels_by_session_id.get(row[0], {}),
                )
                for row in rows
            ]
        next_cursor = session_next_cursor(sessions, has_more, query.order_by)
        return SessionListResult(
            sessions=sessions, next_cursor=next_cursor, total_count=total_count
        )

    @runtime_session_query
    async def append_peer_content(
        self,
        request: PeerContentAppendRequest,
        *,
        qualify_target: Callable[[Session], None] | None = None,
        pending_transcript_cursor: int | None = None,
    ) -> PeerContentReceipt:
        if type(request) is not PeerContentAppendRequest:
            raise TypeError("Peer append requires a PeerContentAppendRequest.")
        request = PeerContentAppendRequest.model_validate(request)
        from cayu.collaboration.peer_content import resolve_peer_target

        key = request.append_key.model_dump_json()
        commitment = request.model_dump(mode="json")
        await self._ensure_ready()
        async with self._connection() as conn:
            async with conn.cursor() as cur:
                creation = request.append_key.creation_target
                if creation is not None:
                    await _creation_fence.postgres_lock(cur, creation)
                target_id, target_instance, creation_excluded = resolve_peer_target(
                    request.append_key,
                    None
                    if creation is None
                    else await _creation_fence.postgres_read(cur, creation),
                )
                from cayu.sessions.access import _query_bounds

                if _query_bounds.get() is not None:
                    resource_owners = {
                        sid: await self._load_for_update(cur, sid)
                        for sid in sorted(
                            {request.occurrence.sender_session_id}
                            | ({target_id} if target_id else set())
                        )
                    }
                    require_resource_session(
                        resource_owners.get(request.occurrence.sender_session_id), "read"
                    )
                    require_resource_session(resource_owners.get(target_id), "modify")
                for lock_key in sorted(
                    (
                        f"peer-append:{key}",
                        f"peer-operation:{request.operation_key}",
                        f"peer-consumer:{request.append_key.consumer_id}:"
                        f"{request.append_key.consumer_participant_incarnation}",
                    )
                ):
                    await cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (lock_key,)
                    )
                await cur.execute(
                    "SELECT commitment_json, receipt_json FROM cayu_peer_content_receipts WHERE append_key_json = %s OR operation_key = %s FOR UPDATE",
                    (key, request.operation_key),
                )
                row = await cur.fetchone()
                from cayu.storage._peer_attempts import historical_replay, replay_or_advance

                await cur.execute(
                    "SELECT request_json, receipt_json FROM cayu_peer_content_attempts WHERE operation_key = %s",
                    (request.operation_key,),
                )
                historical = historical_replay(request, await cur.fetchone())
                if historical is not None:
                    await conn.commit()
                    return historical
                if row is not None:
                    replay = replay_or_advance(
                        request,
                        PeerContentAppendRequest.model_validate(row[0]),
                        PeerContentReceipt.model_validate(row[1]),
                    )
                    if replay is not None:
                        await conn.commit()
                        return replay
                    await cur.execute(
                        "DELETE FROM cayu_peer_content_receipts WHERE append_key_json = %s", (key,)
                    )
                if row is None and request.replaces_operation_key is not None:
                    raise PeerContentConflict()
                from cayu.storage._peer_attempts import qualify, require_capacity

                await cur.execute(
                    """SELECT COUNT(*) FROM cayu_peer_content_receipts p
                    LEFT JOIN cayu_session_message_queue q
                    ON q.queue_id = p.receipt_json->>'queue_id'
                    WHERE p.append_key_json != %s AND NOT p.target_deleted
                    AND p.request_json->'append_key'->>'consumer_id' = %s
                    AND p.request_json->'append_key'->>'consumer_participant_incarnation' = %s
                    AND (p.receipt_json->>'status' = 'pending'
                         OR (p.receipt_json->>'status' = 'appended'
                             AND (q.status IS NULL OR q.status = 'queued')))""",
                    (
                        key,
                        request.append_key.consumer_id,
                        request.append_key.consumer_participant_incarnation,
                    ),
                )
                outstanding = (await cur.fetchone())[0]
                from cayu.storage._peer_attempts import receiving_cursor

                target_cursor = receiving_cursor(
                    request,
                    None if row is None else PeerContentReceipt.model_validate(row[1]),
                    pending_transcript_cursor,
                )
                await cur.execute(
                    "SELECT instance_id, run_epoch, status, agent_name, environment_name FROM cayu_sessions WHERE id = %s FOR UPDATE",
                    (target_id,),
                )
                session = await cur.fetchone()
                await cur.execute(
                    "SELECT participant_id, participant_incarnation, session_instance_id "
                    "FROM cayu_participant_session_bindings WHERE session_id = %s",
                    (target_id,),
                )
                target_binding = await cur.fetchone()
                target_binding_valid = target_binding is not None and tuple(target_binding) == (
                    request.append_key.consumer_id,
                    request.append_key.consumer_participant_incarnation,
                    target_instance,
                )
                await cur.execute(
                    "SELECT state FROM cayu_checkpoints WHERE session_id = %s",
                    (target_id,),
                )
                checkpoint_row = await cur.fetchone()
                checkpoint = None if checkpoint_row is None else checkpoint_row[0]
                parked_target = False
                if session is not None and session[2] in {"completed", "failed", "interrupted"}:
                    assert target_id is not None
                    from cayu.storage._peer_attempts import (
                        parked_delivery_key,
                        permits_parked_delivery_append,
                    )

                    wait_key = parked_delivery_key(
                        checkpoint, session_id=target_id, instance_id=session[0]
                    )
                    if wait_key is not None:
                        await cur.execute(
                            "SELECT record FROM cayu_session_operations WHERE session_id = %s AND idempotency_key = %s",
                            (target_id, wait_key),
                        )
                        wait_row = await cur.fetchone()
                        parked_target = permits_parked_delivery_append(
                            request,
                            checkpoint,
                            None if wait_row is None else wait_row[0],
                            session_id=target_id,
                            instance_id=session[0],
                            run_epoch=session[1],
                        )
                if session is not None and session[2] not in {
                    "completed",
                    "failed",
                    "interrupted",
                }:
                    message_queue.require_open_admission(session[2], checkpoint)
                await cur.execute(
                    "SELECT participant_id, participant_incarnation, session_instance_id "
                    "FROM cayu_participant_session_bindings WHERE session_id = %s",
                    (request.occurrence.sender_session_id,),
                )
                source_binding = await cur.fetchone()
                source_valid = (
                    source_binding is not None
                    and source_binding[0] == request.occurrence.sender_participant_id
                    and source_binding[1] == request.occurrence.sender_participant_incarnation
                    and source_binding[2] == request.occurrence.sender_session_instance_id
                )
                await cur.execute(
                    "SELECT COUNT(*) FROM cayu_transcript_messages WHERE session_id = %s",
                    (target_id,),
                )
                cursor = (await cur.fetchone())[0]
                now = await self._session_store_now(cur)
                expired = int(now.timestamp() * 1000) >= request.attempt_key.deadline_at_ms
                if expired:
                    result = PeerContentReceipt(
                        operation_key=request.operation_key,
                        append_key=request.append_key,
                        attempt_generation=request.attempt_key.attempt_generation,
                        status="excluded",
                        reason="delivery_deadline_expired",
                    )
                elif (
                    creation_excluded
                    or not source_valid
                    or (
                        session is not None
                        and not parked_target
                        and session[2] in {"completed", "failed", "interrupted"}
                    )
                ):
                    result = PeerContentReceipt(
                        operation_key=request.operation_key,
                        append_key=request.append_key,
                        attempt_generation=request.attempt_key.attempt_generation,
                        status="excluded",
                        reason="source_or_target_unavailable",
                    )
                elif session is None:
                    result = PeerContentReceipt(
                        operation_key=request.operation_key,
                        append_key=request.append_key,
                        attempt_generation=request.attempt_key.attempt_generation,
                        status="pending",
                        reason="target_not_created",
                    )
                elif target_binding is None:
                    result = PeerContentReceipt(
                        operation_key=request.operation_key,
                        append_key=request.append_key,
                        attempt_generation=request.attempt_key.attempt_generation,
                        status="pending",
                        reason="target_binding_pending",
                    )
                elif not target_binding_valid or session[0] != target_instance:
                    result = PeerContentReceipt(
                        operation_key=request.operation_key,
                        append_key=request.append_key,
                        attempt_generation=request.attempt_key.attempt_generation,
                        status="excluded",
                        reason="target_binding_mismatch",
                    )
                elif isinstance(checkpoint, dict) and "pending_tool_round" in checkpoint:
                    result = PeerContentReceipt(
                        operation_key=request.operation_key,
                        append_key=request.append_key,
                        attempt_generation=request.attempt_key.attempt_generation,
                        status="pending",
                        reason="target_busy",
                    )
                elif session[1] != request.attempt_key.target_run_epoch or cursor != target_cursor:
                    result = PeerContentReceipt(
                        operation_key=request.operation_key,
                        append_key=request.append_key,
                        attempt_generation=request.attempt_key.attempt_generation,
                        status="pending",
                        reason="target_cursor_changed",
                    )
                else:
                    require_capacity(outstanding)
                    assert target_id is not None
                    qualified_session = await self._load_for_update(cur, target_id)
                    qualify(qualify_target, qualified_session)
                    message = Message(
                        role=MessageRole.ASSISTANT,
                        content=(
                            request.occurrence.to_message_part(
                                append_key=request.append_key,
                                projection_id=request.append_key.projection_id,
                                operation_key=request.operation_key,
                            ),
                        ),
                    )
                    from cayu.collaboration.peer_content import peer_queue_id

                    assert target_id is not None and target_instance is not None
                    queue_id = peer_queue_id(request.append_key, target_id, target_instance)
                    await cur.execute(
                        "SELECT 1 FROM cayu_session_message_queue WHERE queue_id = %s "
                        "OR (session_id = %s AND idempotency_key = %s)",
                        (queue_id, target_id, request.operation_key),
                    )
                    if await cur.fetchone() is not None:
                        raise PeerContentConflict(
                            "Peer queue identity already has another authority."
                        )
                    delivery_mode = (
                        SessionMessageDeliveryMode.ON_IDLE
                        if request.wake_policy == "ordinary_continuation"
                        else SessionMessageDeliveryMode.NEXT_TURN
                    )
                    accepted_at = await self._session_store_now(cur)
                    accepted_event_id = str(uuid4())
                    await cur.execute(
                        "INSERT INTO cayu_session_message_queue (queue_id, session_id, idempotency_key, content, message_json, delivery_mode, status, accepted_run_epoch, accepted_transcript_cursor, accepted_event_id, accepted_at) VALUES (%s, %s, %s, %s, %s, %s, 'queued', %s, %s, %s, %s) RETURNING ordering_key",
                        (
                            queue_id,
                            target_id,
                            request.operation_key,
                            request.occurrence.payload.text,
                            pg_support._dumps(message.model_dump(mode="json")),
                            str(delivery_mode),
                            session[1],
                            cursor,
                            accepted_event_id,
                            accepted_at,
                        ),
                    )
                    ordering_row = await cur.fetchone()
                    if ordering_row is None:
                        raise RuntimeError("Peer queue insert did not return ordering key.")
                    ordering_key = ordering_row[0]
                    accepted_event = event_with_runtime_payload_authority(
                        Event(
                            id=accepted_event_id,
                            type=EventType.SESSION_MESSAGE_QUEUED,
                            session_id=str(target_id),
                            agent_name=session[3],
                            environment_name=session[4],
                            timestamp=accepted_at,
                            payload={
                                **_queued_session_message_event_payload(
                                    queue_id=queue_id,
                                    delivery_mode=delivery_mode,
                                    ordering_key=ordering_key,
                                    actor=None,
                                    run_epoch=session[1],
                                    transcript_cursor=cursor,
                                ),
                                "peer_occurrence_id": request.occurrence.occurrence_id,
                                "peer_provenance_sha256": request.occurrence.provenance_sha256,
                            },
                        )
                    )
                    await cur.execute(
                        "UPDATE cayu_sessions SET event_seq = event_seq + 1 "
                        "WHERE id = %s RETURNING event_seq",
                        (target_id,),
                    )
                    event_order_row = await cur.fetchone()
                    if event_order_row is None:
                        raise RuntimeError("Peer event session sequence was not advanced.")
                    await cur.execute(
                        "INSERT INTO cayu_events (session_id, session_order, event_id, "
                        "interaction_id, event_type, timestamp, agent_name, environment_name, "
                        "workflow_name, tool_name, payload, event, pending_action_lookup_key, "
                        "pending_action_projection, pending_action_projection_bytes) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            target_id,
                            event_order_row[0],
                            accepted_event.id,
                            accepted_event.interaction_id,
                            str(accepted_event.type),
                            accepted_event.timestamp,
                            accepted_event.agent_name,
                            accepted_event.environment_name,
                            accepted_event.workflow_name,
                            accepted_event.tool_name,
                            pg_support._dumps(accepted_event.payload),
                            pg_support._dumps(accepted_event.model_dump(mode="json")),
                            None,
                            None,
                            None,
                        ),
                    )
                    result = PeerContentReceipt(
                        operation_key=request.operation_key,
                        append_key=request.append_key,
                        attempt_generation=request.attempt_key.attempt_generation,
                        status="appended",
                        target_session_id=target_id,
                        target_session_instance_id=target_instance,
                        occurrence=request.occurrence,
                        queue_id=queue_id,
                    )
                if result.status == "pending":
                    require_capacity(outstanding)
                await cur.execute(
                    "INSERT INTO cayu_peer_content_receipts (append_key_json, operation_key, commitment_json, receipt_json, request_json) VALUES (%s, %s, %s, %s, %s)",
                    (
                        key,
                        request.operation_key,
                        pg_support._dumps(commitment),
                        pg_support._dumps(result.model_dump(mode="json")),
                        pg_support._dumps(request.model_dump(mode="json")),
                    ),
                )
                await cur.execute(
                    "INSERT INTO cayu_peer_content_attempts (operation_key, request_json, receipt_json) "
                    "VALUES (%s, %s, %s) ON CONFLICT(operation_key) DO UPDATE SET "
                    "request_json=EXCLUDED.request_json, receipt_json=EXCLUDED.receipt_json",
                    (
                        request.operation_key,
                        pg_support._dumps(request.model_dump(mode="json")),
                        pg_support._dumps(result.model_dump(mode="json")),
                    ),
                )
            await conn.commit()
            return result

    async def read_peer_content_attempt(
        self, request: PeerContentAppendRequest
    ) -> PeerContentReceipt | None:
        from cayu.storage._peer_attempts import exact_read

        request = PeerContentAppendRequest.model_validate(request)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT request_json, receipt_json FROM cayu_peer_content_attempts WHERE operation_key = %s",
                (request.operation_key,),
            )
            return exact_read(request, await cur.fetchone())

    async def list_pending_peer_content(self, *, after_operation_key=None, limit=32):
        """Trusted receiving-owner discovery, independent of creation settlement."""
        from cayu.collaboration.peer_content import validate_peer_discovery

        validate_peer_discovery(after_operation_key, limit)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT request_json FROM cayu_peer_content_receipts "
                "WHERE receipt_json->>'status' = 'pending' "
                "AND operation_key > %s ORDER BY operation_key LIMIT %s",
                (after_operation_key or "", limit),
            )
            return tuple(
                PeerContentAppendRequest.model_validate(row[0]) for row in await cur.fetchall()
            )

    async def read_peer_content(self, append_key: PeerAppendKey) -> PeerContentReceipt | None:
        if type(append_key) is not PeerAppendKey:
            raise TypeError("append_key must be a PeerAppendKey.")
        append_key = PeerAppendKey.model_validate(append_key)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT receipt_json FROM cayu_peer_content_receipts WHERE append_key_json = %s",
                (append_key.model_dump_json(),),
            )
            row = await cur.fetchone()
            return None if row is None else PeerContentReceipt.model_validate(row[0])

    async def record_peer_content_exposure(
        self, request: PeerContentExposureRequest
    ) -> PeerContentExposureReceipt:
        await self._ensure_ready()
        key = request.append_key.model_dump_json()
        commitment = request.model_dump(mode="json")
        async with self._connection() as conn, conn.cursor() as cur:
            for lock_key in sorted(
                (f"peer-exposure:{request.exposure_id}", f"peer-operation:{request.operation_key}")
            ):
                await cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (lock_key,)
                )
            await cur.execute(
                "SELECT commitment_json, receipt_json FROM cayu_peer_content_exposures WHERE exposure_id = %s OR operation_key = %s FOR UPDATE",
                (request.exposure_id, request.operation_key),
            )
            row = await cur.fetchone()
            if row is not None:
                prior = PeerContentExposureReceipt.model_validate(row[1])
                expected = (
                    request.identity_commitment() if prior.outcome == "pending" else commitment
                )
                if row[0] != expected:
                    raise PeerContentConflict()
                if prior.outcome == "pending":
                    receipt = PeerContentExposureReceipt(
                        operation_key=request.operation_key,
                        append_key=request.append_key,
                        exposure_id=request.exposure_id,
                        model_attempt_id=request.model_attempt_id,
                        outcome=request.outcome,
                        reason=request.reason,
                    )
                    await cur.execute(
                        "UPDATE cayu_peer_content_exposures SET commitment_json = %s, receipt_json = %s WHERE exposure_id = %s",
                        (
                            pg_support._dumps(commitment),
                            pg_support._dumps(receipt.model_dump(mode="json")),
                            request.exposure_id,
                        ),
                    )
                    await conn.commit()
                    return receipt
                await conn.commit()
                return prior.model_copy(update={"replayed": True})
            await cur.execute(
                "SELECT receipt_json FROM cayu_peer_content_receipts WHERE append_key_json = %s",
                (key,),
            )
            append = await cur.fetchone()
            if append is None or append[0].get("status") != "appended":
                raise PeerContentUnavailable("Peer content was not durably appended.")
            receipt = PeerContentExposureReceipt(
                operation_key=request.operation_key,
                append_key=request.append_key,
                exposure_id=request.exposure_id,
                model_attempt_id=request.model_attempt_id,
                outcome=request.outcome,
                reason=request.reason,
            )
            await cur.execute(
                "INSERT INTO cayu_peer_content_exposures (exposure_id, operation_key, append_key_json, commitment_json, receipt_json) VALUES (%s, %s, %s, %s, %s)",
                (
                    request.exposure_id,
                    request.operation_key,
                    pg_support._dumps(key),
                    pg_support._dumps(commitment),
                    pg_support._dumps(receipt.model_dump(mode="json")),
                ),
            )
            await conn.commit()
            return receipt

    async def begin_peer_content_exposure(
        self, request: PeerContentExposureRequest
    ) -> PeerContentExposureReceipt:
        await self._ensure_ready()
        key = request.append_key.model_dump_json()
        identity = request.identity_commitment()
        async with self._connection() as conn, conn.cursor() as cur:
            for lock_key in sorted(
                (f"peer-exposure:{request.exposure_id}", f"peer-operation:{request.operation_key}")
            ):
                await cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (lock_key,)
                )
            await cur.execute(
                "SELECT commitment_json, receipt_json FROM cayu_peer_content_exposures WHERE exposure_id = %s OR operation_key = %s FOR UPDATE",
                (request.exposure_id, request.operation_key),
            )
            row = await cur.fetchone()
            if row is not None:
                prior = PeerContentExposureReceipt.model_validate(row[1])
                expected = (
                    identity if prior.outcome == "pending" else request.model_dump(mode="json")
                )
                if row[0] != expected:
                    raise PeerContentConflict()
                await conn.commit()
                return prior.model_copy(update={"replayed": True})
            await cur.execute(
                "SELECT receipt_json FROM cayu_peer_content_receipts WHERE append_key_json = %s",
                (key,),
            )
            append = await cur.fetchone()
            if append is None or append[0].get("status") != "appended":
                raise PeerContentUnavailable("Peer content was not durably appended.")
            receipt = PeerContentExposureReceipt(
                operation_key=request.operation_key,
                append_key=request.append_key,
                exposure_id=request.exposure_id,
                model_attempt_id=request.model_attempt_id,
                outcome="pending",
            )
            await cur.execute(
                "INSERT INTO cayu_peer_content_exposures (exposure_id, operation_key, append_key_json, commitment_json, receipt_json) VALUES (%s, %s, %s, %s, %s)",
                (
                    request.exposure_id,
                    request.operation_key,
                    pg_support._dumps(key),
                    pg_support._dumps(identity),
                    pg_support._dumps(receipt.model_dump(mode="json")),
                ),
            )
            await conn.commit()
            return receipt

    async def read_peer_content_exposure(
        self, append_key: PeerAppendKey, exposure_id: str
    ) -> PeerContentExposureReceipt | None:
        if type(append_key) is not PeerAppendKey or type(exposure_id) is not str:
            raise TypeError("Invalid exposure lookup.")
        append_key = PeerAppendKey.model_validate(append_key)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT append_key_json, receipt_json FROM cayu_peer_content_exposures WHERE exposure_id = %s",
                (exposure_id,),
            )
            row = await cur.fetchone()
            if row is None or row[0] != append_key.model_dump_json():
                return None
            return PeerContentExposureReceipt.model_validate(row[1])

    async def exclude_peer_content(
        self, request: PeerContentAppendRequest, *, reason: str
    ) -> PeerContentReceipt:
        from cayu._validation import require_durable_clean_nonblank

        if type(request) is not PeerContentAppendRequest:
            raise TypeError("Peer exclusion requires a PeerContentAppendRequest.")
        request = PeerContentAppendRequest.model_validate(request)
        reason = require_durable_clean_nonblank(reason, "reason")
        from cayu.collaboration.peer_content import resolve_peer_target

        key = request.append_key.model_dump_json()
        commitment = request.model_dump(mode="json")
        result = PeerContentReceipt(
            operation_key=request.operation_key,
            append_key=request.append_key,
            attempt_generation=request.attempt_key.attempt_generation,
            status="excluded",
            reason=reason,
        )
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            creation = request.append_key.creation_target
            if creation is not None:
                await _creation_fence.postgres_lock(cur, creation)
            resolve_peer_target(
                request.append_key,
                None if creation is None else await _creation_fence.postgres_read(cur, creation),
            )
            for lock_key in sorted(
                (f"peer-append:{key}", f"peer-operation:{request.operation_key}")
            ):
                await cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (lock_key,)
                )
            await cur.execute(
                "SELECT commitment_json, receipt_json FROM cayu_peer_content_receipts "
                "WHERE append_key_json = %s OR operation_key = %s FOR UPDATE",
                (key, request.operation_key),
            )
            row = await cur.fetchone()
            from cayu.storage._peer_attempts import historical_replay

            await cur.execute(
                "SELECT request_json, receipt_json FROM cayu_peer_content_attempts WHERE operation_key = %s",
                (request.operation_key,),
            )
            historical = historical_replay(request, await cur.fetchone())
            if historical is not None:
                if historical.status == "excluded" and historical.reason != reason:
                    raise PeerContentConflict()
                await conn.commit()
                return historical
            if row is not None:
                existing_commitment = row[0]
                if isinstance(existing_commitment, str):
                    existing_commitment = json.loads(existing_commitment)
                if existing_commitment != commitment:
                    raise PeerContentConflict()
                stored = PeerContentReceipt.model_validate(row[1])
                if stored.status == "excluded" and stored.reason != reason:
                    raise PeerContentConflict()
                if stored.status != "pending":
                    await conn.commit()
                    return stored.model_copy(update={"replayed": True})
                await cur.execute(
                    "DELETE FROM cayu_peer_content_receipts WHERE append_key_json = %s", (key,)
                )
            await cur.execute(
                "INSERT INTO cayu_peer_content_receipts (append_key_json, operation_key, commitment_json, receipt_json, request_json) VALUES (%s, %s, %s, %s, %s)",
                (
                    key,
                    request.operation_key,
                    pg_support._dumps(commitment),
                    pg_support._dumps(result.model_dump(mode="json")),
                    pg_support._dumps(request.model_dump(mode="json")),
                ),
            )
            await cur.execute(
                "INSERT INTO cayu_peer_content_attempts (operation_key, request_json, receipt_json) "
                "VALUES (%s, %s, %s) ON CONFLICT(operation_key) DO UPDATE SET "
                "request_json=EXCLUDED.request_json, receipt_json=EXCLUDED.receipt_json",
                (
                    request.operation_key,
                    pg_support._dumps(request.model_dump(mode="json")),
                    pg_support._dumps(result.model_dump(mode="json")),
                ),
            )
        return result

    async def retry_pending_peer_content(
        self,
        session_id: str,
        *,
        expected_session_instance_id: str,
        expected_run_epoch: int,
        expected_transcript_cursor: int,
        admit=None,
    ) -> tuple[PeerContentReceipt, ...]:
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT request_json FROM cayu_peer_content_receipts "
                "WHERE receipt_json->>'status' = 'pending' "
                "AND (request_json->'append_key'->>'target_session_id' = %s OR EXISTS ("
                "SELECT 1 FROM cayu_session_creation_decisions c WHERE "
                "c.decision_json::jsonb->>'session_id' = %s AND "
                "c.decision_json::jsonb->>'session_instance_id' = %s AND "
                "c.decision_json::jsonb->'target' = request_json->'append_key'->'creation_target')) "
                "ORDER BY operation_key LIMIT %s",
                (
                    session_id,
                    session_id,
                    expected_session_instance_id,
                    SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
                ),
            )
            rows = await cur.fetchall()
        requests = [
            PeerContentAppendRequest.model_validate(row[0]) for row in rows if row[0] is not None
        ]
        results: list[PeerContentReceipt] = []
        for request in requests:
            if request.append_key.creation_target is not None:
                from cayu.collaboration._contracts import ExactMatch

                decision = await self.read_session_creation_decision(
                    request.append_key.creation_target
                )
                if not isinstance(decision, ExactMatch) or (
                    decision.receipt.session_id != session_id
                    or decision.receipt.session_instance_id != expected_session_instance_id
                ):
                    continue
            if (
                request.append_key.creation_target is None
                and request.append_key.target_session_instance_id != expected_session_instance_id
            ):
                continue
            if request.attempt_key.target_run_epoch != expected_run_epoch:
                continue
            if admit is None:
                raise PeerContentUnavailable(
                    "Pending delivery requires fresh export authorization."
                )
            results.append(
                await admit(request, pending_transcript_cursor=expected_transcript_cursor)
            )
        return tuple(results)

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
        await self._ensure_ready()
        async with self._connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT 1 FROM cayu_sessions WHERE id = %s FOR UPDATE",
                    (session_id,),
                )
                if await cur.fetchone() is None:
                    raise KeyError(f"Session not found: {session_id}")
                if copied_messages:
                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
                    await self._register_public_authorities(
                        cur,
                        session_id,
                        interaction_ids=(() if interaction_id is None else (interaction_id,)),
                    )
                    await _touch_session_activity(
                        cur,
                        session_id,
                        await self._session_store_now(cur),
                    )
                    await cur.executemany(
                        """
                        INSERT INTO cayu_transcript_messages
                            (session_id, interaction_id, message,
                             transcript_search_document)
                        VALUES (%s, %s, %s, %s)
                        """,
                        [
                            (
                                session_id,
                                interaction_id,
                                pg_support._dumps(message.model_dump(mode="json")),
                                _postgres_transcript_index_document(session_id, message),
                            )
                            for message in copied_messages
                        ],
                    )
            await conn.commit()

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
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    session = await self._load_for_update(cur, session_id)
                    if session is None:
                        raise KeyError(f"Session not found: {session_id}")
                    updated_at = await self._session_store_now(cur)
                    _assert_session_run_epoch(session_id, session)
                    await cur.execute(
                        "SELECT interaction_id, source_messages "
                        "FROM cayu_deferred_interaction_inputs "
                        "WHERE session_id = %s FOR UPDATE",
                        (session_id,),
                    )
                    row = await cur.fetchone()
                    if row is None or row[0] != interaction_id:
                        raise RuntimeError(
                            "Deferred interaction input changed before finalization."
                        )
                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
                    stored = deferred_interaction_input_from_storage_payload(
                        row[0],
                        pg_support._json_obj(row[1]),
                    )
                    require_deferred_initial_transcript_replacement(
                        stored,
                        expected_messages=expected,
                        replacement_messages=replacement,
                    )
                    await cur.execute(
                        "SELECT 1 FROM cayu_transcript_messages WHERE session_id = %s LIMIT 1",
                        (session_id,),
                    )
                    if await cur.fetchone() is not None:
                        from cayu.sessions._participant_execution_identity import (
                            require_initial_execution_input,
                        )
                        from cayu.storage._participant_session_records import reconstruct

                        await cur.execute(
                            f"SELECT {PARTICIPANT_BINDING_PROJECTION} FROM cayu_participant_session_bindings WHERE session_id = %s",
                            (session_id,),
                        )
                        binding_row = await cur.fetchone()
                        await cur.execute(
                            "SELECT message FROM cayu_transcript_messages WHERE session_id = %s ORDER BY session_order",
                            (session_id,),
                        )
                        current_rows = await cur.fetchall()
                        require_initial_execution_input(
                            session,
                            None if binding_row is None else reconstruct(binding_row, session),
                            [
                                Message.model_validate(pg_support._json_obj(row[0]))
                                for row in current_rows
                            ],
                            expected,
                        )
                        await cur.execute(
                            "DELETE FROM cayu_transcript_messages WHERE session_id = %s",
                            (session_id,),
                        )
                        await cur.execute(
                            "UPDATE cayu_sessions SET transcript_seq = 0 WHERE id = %s",
                            (session_id,),
                        )
                    prefix_count = _initial_transcript_prefix_count(
                        expected,
                        replacement,
                        runtime_suffix_count=runtime_suffix_count,
                    )
                    current_checkpoint = await self._load_checkpoint(cur, session_id)
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
                    await self._register_public_authorities(
                        cur,
                        session_id,
                        interaction_ids=(interaction_id,),
                    )
                    await cur.executemany(
                        "INSERT INTO cayu_transcript_messages "
                        "(session_id, interaction_id, message, "
                        "transcript_search_document) VALUES (%s, %s, %s, %s)",
                        [
                            (
                                session_id,
                                None if index < prefix_count else interaction_id,
                                pg_support._dumps(message.model_dump(mode="json")),
                                _postgres_transcript_index_document(session_id, message),
                            )
                            for index, message in enumerate(replacement)
                        ],
                    )
                    await cur.execute(
                        "DELETE FROM cayu_deferred_interaction_inputs WHERE session_id = %s",
                        (session_id,),
                    )
                    if checkpoint is None:
                        await cur.execute(
                            "DELETE FROM cayu_checkpoints WHERE session_id = %s",
                            (session_id,),
                        )
                    else:
                        await self._upsert_checkpoint(cur, session_id, checkpoint, updated_at)
                    await _touch_session_activity(cur, session_id, updated_at)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    async def materialize_deferred_interaction_input(
        self,
        session_id: str,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> bool:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    session = await self._load_for_update(cur, session_id)
                    if session is None:
                        raise KeyError(f"Session not found: {session_id}")
                    _assert_session_run_epoch(session_id, session)
                    await cur.execute(
                        "SELECT interaction_id, source_messages "
                        "FROM cayu_deferred_interaction_inputs "
                        "WHERE session_id = %s FOR UPDATE",
                        (session_id,),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        await conn.commit()
                        return False
                    if row[0] != interaction_id:
                        raise RuntimeError(
                            "Deferred interaction input belongs to another interaction."
                        )
                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
                    deferred = deferred_interaction_input_from_storage_payload(
                        row[0],
                        pg_support._json_obj(row[1]),
                    )
                    messages = deferred.source_messages
                    if interaction_id is not None:
                        await self._register_public_authorities(
                            cur,
                            session_id,
                            interaction_ids=(interaction_id,),
                        )
                    await cur.executemany(
                        "INSERT INTO cayu_transcript_messages "
                        "(session_id, interaction_id, message, "
                        "transcript_search_document) VALUES (%s, %s, %s, %s)",
                        [
                            (
                                session_id,
                                interaction_id,
                                pg_support._dumps(message.model_dump(mode="json")),
                                _postgres_transcript_index_document(session_id, message),
                            )
                            for message in messages
                        ],
                    )
                    await cur.execute(
                        "DELETE FROM cayu_deferred_interaction_inputs WHERE session_id = %s",
                        (session_id,),
                    )
                    await _touch_session_activity(
                        cur,
                        session_id,
                        await self._session_store_now(cur),
                    )
                await conn.commit()
                return True
            except Exception:
                await conn.rollback()
                raise

    async def load_deferred_interaction_input(
        self,
        session_id: str,
    ) -> DeferredInteractionInput | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            if await self._load(cur, session_id) is None:
                raise KeyError(f"Session not found: {session_id}")
            await cur.execute(
                "SELECT interaction_id, source_messages "
                "FROM cayu_deferred_interaction_inputs WHERE session_id = %s",
                (session_id,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return deferred_interaction_input_from_storage_payload(row[0], pg_support._json_obj(row[1]))

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
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    session = await self._load_for_update(cur, session_id)
                    if session is None:
                        raise KeyError(f"Session not found: {session_id}")
                    updated_at = await self._session_store_now(cur)
                    _assert_session_run_epoch(session_id, session)
                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
                    current_checkpoint = await self._load_checkpoint(cur, session_id)
                    transformed = checkpoint_transform(
                        session,
                        _copy_checkpoint_for_transform(
                            current_checkpoint,
                            session_id=session_id,
                        ),
                    )
                    if transformed is None:
                        raise ValueError("Checkpoint transform must return a checkpoint.")
                    transformed = _checkpoint_transform_result_preserving_completion_result_event_publications(
                        current_checkpoint,
                        transformed,
                        session_id=session_id,
                    )
                    await _touch_session_activity(cur, session_id, updated_at)
                    if copied_messages:
                        await self._register_public_authorities(
                            cur,
                            session_id,
                            interaction_ids=(() if interaction_id is None else (interaction_id,)),
                        )
                        await cur.executemany(
                            """
                            INSERT INTO cayu_transcript_messages
                                (session_id, interaction_id, message,
                                 transcript_search_document)
                            VALUES (%s, %s, %s, %s)
                            """,
                            [
                                (
                                    session_id,
                                    interaction_id,
                                    pg_support._dumps(message.model_dump(mode="json")),
                                    _postgres_transcript_index_document(session_id, message),
                                )
                                for message in copied_messages
                            ],
                        )
                    await self._upsert_checkpoint(cur, session_id, transformed, updated_at)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    async def load_transcript(self, session_id: str) -> list[Message]:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            if access_bounds is not None:
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                access_bounds.require_read(await self._load(cur, session_id))
            await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (session_id,))
            if await cur.fetchone() is None:
                raise KeyError(f"Session not found: {session_id}")
            await cur.execute(
                """
                SELECT message
                FROM cayu_transcript_messages
                WHERE session_id = %s
                ORDER BY sequence ASC
                """,
                (session_id,),
            )
            rows = await cur.fetchall()
            return [Message(**pg_support._json_obj(row[0])) for row in rows]

    async def load_transcript_snapshot(self, session_id: str) -> TranscriptSnapshot:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            if access_bounds is not None:
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                access_bounds.require_read(await self._load(cur, session_id))
            await cur.execute(
                """
                SELECT session.transcript_seq,
                       transcript.session_order - 1 AS transcript_index,
                       transcript.interaction_id,
                       transcript.message
                FROM cayu_sessions AS session
                LEFT JOIN cayu_transcript_messages AS transcript
                  ON transcript.session_id = session.id
                WHERE session.id = %s
                ORDER BY transcript.session_order ASC
                """,
                (session_id,),
            )
            rows = await cur.fetchall()
            if not rows:
                raise KeyError(f"Session not found: {session_id}")
            return TranscriptSnapshot(
                records=[
                    TranscriptRecord(
                        index=row[1],
                        interaction_id=row[2],
                        message=Message(**pg_support._json_obj(row[3])),
                    )
                    for row in rows
                    if row[1] is not None
                ],
                cursor=int(rows[0][0]),
            )

    async def load_transcript_cursor(self, session_id: str) -> int:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT transcript_seq FROM cayu_sessions WHERE id = %s",
                (session_id,),
            )
            row = await cur.fetchone()
            if row is None:
                raise KeyError(f"Session not found: {session_id}")
            return int(row[0])

    async def load_latest_transcript_message(
        self,
        session_id: str,
        *,
        role: MessageRole,
    ) -> TranscriptRecord | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(role, MessageRole):
            raise TypeError("role must be a MessageRole.")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT session.id,
                       transcript.session_order - 1,
                       transcript.interaction_id,
                       transcript.message
                FROM cayu_sessions AS session
                LEFT JOIN LATERAL (
                    SELECT session_order, interaction_id, message
                    FROM cayu_transcript_messages
                    WHERE session_id = session.id
                      AND message ->> 'role' = %s
                    ORDER BY session_order DESC
                    LIMIT 1
                ) AS transcript ON TRUE
                WHERE session.id = %s
                """,
                (str(role), session_id),
            )
            row = await cur.fetchone()
            if row is None:
                raise KeyError(f"Session not found: {session_id}")
            if row[1] is None:
                return None
            return TranscriptRecord(
                index=row[1],
                interaction_id=row[2],
                message=Message(**pg_support._json_obj(row[3])),
            )

    async def load_latest_transcript_text(
        self,
        session_id: str,
        *,
        role: MessageRole,
        max_chars: int,
    ) -> tuple[str, bool] | None:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(role, MessageRole):
            raise TypeError("role must be a MessageRole.")
        if type(max_chars) is not int:
            raise TypeError("max_chars must be an integer.")
        if not 1 <= max_chars <= LATEST_TRANSCRIPT_TEXT_MAX_CHARS:
            raise ValueError(f"max_chars must be between 1 and {LATEST_TRANSCRIPT_TEXT_MAX_CHARS}.")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            if access_bounds is not None:
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                access_bounds.require_read(await self._load(cur, session_id))
            await cur.execute(
                """
                SELECT session.id,
                       transcript.sequence,
                       pg_column_size(transcript.message)
                FROM cayu_sessions AS session
                LEFT JOIN LATERAL (
                    SELECT sequence, message
                    FROM cayu_transcript_messages
                    WHERE session_id = session.id
                      AND message ->> 'role' = %s
                    ORDER BY session_order DESC
                    LIMIT 1
                ) AS transcript ON TRUE
                WHERE session.id = %s
                """,
                (str(role), session_id),
            )
            row = await cur.fetchone()
            if row is None:
                raise KeyError(f"Session not found: {session_id}")
            sequence = row[1]
            if sequence is None:
                return None
            if int(row[2]) > LATEST_TRANSCRIPT_TEXT_MAX_SOURCE_BYTES:
                raise TranscriptTextReadLimitExceeded(
                    "Transcript message exceeds the bounded serialized-source limit."
                )
            await cur.execute(
                """
                WITH RECURSIVE
                source(message, part_count) AS (
                    SELECT message, jsonb_array_length(message -> 'content')
                    FROM cayu_transcript_messages
                    WHERE sequence = %s
                ),
                prefix(part_index, text_value, part_count) AS (
                    SELECT 0, ''::text, part_count
                    FROM source
                    UNION ALL
                    SELECT
                        prefix.part_index + 1,
                        left(
                            prefix.text_value ||
                            CASE
                                WHEN source.message -> 'content' -> prefix.part_index
                                     ->> 'type' = 'text'
                                THEN COALESCE(
                                    source.message -> 'content' -> prefix.part_index ->> 'text',
                                    ''
                                )
                                ELSE ''
                            END,
                            %s
                        ),
                        prefix.part_count
                    FROM prefix
                    CROSS JOIN source
                    WHERE prefix.part_index < prefix.part_count
                      AND prefix.part_index < %s
                      AND length(prefix.text_value) <= %s
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
            )
            projection = await cur.fetchone()
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

    async def load_transcript_window(
        self,
        session_id: str,
        *,
        start_index: int,
        limit: int,
    ) -> TranscriptSnapshot:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        if type(start_index) is not int:
            raise TypeError("start_index must be an integer.")
        if not 0 <= start_index <= MAX_DURABLE_JSON_INTEGER:
            raise ValueError("start_index exceeds the durable integer limit.")
        if type(limit) is not int:
            raise TypeError("limit must be an integer.")
        if not 1 <= limit <= 5000:
            raise ValueError("limit must be between 1 and 5000.")

        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            if access_bounds is not None:
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                access_bounds.require_read(await self._load(cur, session_id))
            await cur.execute(
                """
                SELECT session.transcript_seq,
                       transcript.session_order - 1 AS transcript_index,
                       transcript.interaction_id,
                       transcript.message
                FROM cayu_sessions AS session
                LEFT JOIN cayu_transcript_messages AS transcript
                  ON transcript.session_id = session.id
                 AND transcript.session_order > %s
                WHERE session.id = %s
                ORDER BY transcript.session_order ASC
                LIMIT %s
                """,
                (start_index, session_id, limit),
            )
            rows = await cur.fetchall()
            if not rows:
                raise KeyError(f"Session not found: {session_id}")
            return TranscriptSnapshot(
                records=[
                    TranscriptRecord(
                        index=row[1],
                        interaction_id=row[2],
                        message=Message(**pg_support._json_obj(row[3])),
                    )
                    for row in rows
                    if row[1] is not None
                ],
                cursor=int(rows[0][0]),
            )

    async def query_transcript(self, query: TranscriptQuery) -> TranscriptPage:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        query = copy_transcript_query(query)
        filters: list[str] = []
        filter_params: list[object] = []
        if query.role is not None:
            filters.append("message ->> 'role' = %s")
            filter_params.append(str(query.role))
        if query.interaction_id is not None:
            filters.append("interaction_id = %s")
            filter_params.append(query.interaction_id)
        filter_clause = " AND " + " AND ".join(filters) if filters else ""

        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            if access_bounds is not None:
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                access_bounds.require_read(await self._load(cur, query.session_id))
            await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (query.session_id,))
            if await cur.fetchone() is None:
                raise KeyError(f"Session not found: {query.session_id}")

            await cur.execute(
                f"""
                SELECT COUNT(*)
                FROM cayu_transcript_messages
                WHERE session_id = %s
                {filter_clause}
                """,
                [query.session_id, *filter_params],
            )
            total_row = await cur.fetchone()
            total_records = int(total_row[0]) if total_row is not None else 0

            await cur.execute(
                f"""
                SELECT session_order - 1 AS transcript_index, interaction_id, message
                FROM cayu_transcript_messages
                WHERE session_id = %s
                {filter_clause}
                ORDER BY session_order ASC
                LIMIT %s OFFSET %s
                """,
                [query.session_id, *filter_params, query.limit, query.offset],
            )
            rows = await cur.fetchall()
            records = [
                TranscriptRecord(
                    index=row[0],
                    interaction_id=row[1],
                    message=Message(**pg_support._json_obj(row[2])),
                )
                for row in rows
            ]
            return TranscriptPage(
                records=filter_transcript_records(records, include_thinking=query.include_thinking),
                total_records=total_records,
            )

    async def search_transcript(
        self,
        query: TranscriptSearchQuery,
    ) -> TranscriptSearchResult:
        query = copy_transcript_search_query(query)
        cursor = decode_transcript_search_cursor(query)
        query_document = transcript_search_query_document(query.text)
        before_filter = "".join(
            " AND (transcript.session_id <> %s OR transcript.session_order <= %s)"
            for _ in query.before_transcript_indexes
        )
        before_params = [
            value
            for session_id, before_index in query.before_transcript_indexes.items()
            for value in (session_id, before_index)
        ]
        fetch_limit = query.max_records_scanned + 1

        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                f"""
                WITH search_query AS (
                    SELECT to_tsquery('simple'::regconfig, %s) AS value
                )
                SELECT
                    transcript.session_id,
                    transcript.session_order - 1 AS transcript_index,
                    transcript.interaction_id,
                    transcript.message,
                    transcript.transcript_search_document
                FROM cayu_transcript_messages AS transcript
                CROSS JOIN search_query
                WHERE transcript.session_id = ANY(%s)
                  AND transcript.message ->> 'role' IN ('user', 'assistant')
                  AND transcript.message ->> 'role' = ANY(%s)
                  {before_filter}
                  AND to_tsvector(
                        'simple'::regconfig,
                        transcript.transcript_search_document
                      ) @@ search_query.value
                LIMIT %s
                """,
                [
                    _postgres_transcript_search_expression(query),
                    list(query.session_ids),
                    [str(role) for role in query.roles],
                    *before_params,
                    fetch_limit,
                ],
            )
            rows = await cur.fetchall()

        if len(rows) > query.max_records_scanned:
            return TranscriptSearchResult(
                query=query,
                matched_records_examined=query.max_records_scanned,
                truncated=True,
                coverage_complete=False,
            )

        candidate_heap: list[tuple[int, str, int, Any, Message]] = []
        for row_number, row in enumerate(rows, start=1):
            message = Message(**pg_support._json_obj(row[3]))
            document = transcript_search_document(message)
            if row[4] != _postgres_transcript_index_document(str(row[0]), message):
                raise RuntimeError("Postgres transcript search document is inconsistent.")
            score = transcript_search_document_score(document, query_document)
            if score <= 0:
                raise RuntimeError("Postgres transcript search index is inconsistent.")
            heapq.heappush(
                candidate_heap,
                (-score, str(row[0]), -int(row[1]), row, message),
            )
            if row_number % 256 == 0:
                await asyncio.sleep(0)

        hits: list[TranscriptSearchHit] = []
        remaining_bytes = query.max_bytes
        truncated = False
        continuation_available = False
        ranked_examined = 0
        while candidate_heap:
            negative_score, session_id, negative_transcript_index, row, message = heapq.heappop(
                candidate_heap
            )
            score = -negative_score
            transcript_index = -negative_transcript_index
            ranked_examined += 1
            if ranked_examined % 256 == 0:
                await asyncio.sleep(0)
            if cursor is not None and not transcript_search_position_after_cursor(
                raw_score=score,
                session_id=session_id,
                transcript_index=transcript_index,
                cursor=cursor,
            ):
                continue
            if len(hits) >= query.limit:
                truncated = True
                continuation_available = True
                break
            hit = transcript_search_hit_from_message(
                session_id=session_id,
                transcript_index=transcript_index,
                interaction_id=row[2],
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
            if not hit.text_complete:
                truncated = True
                continuation_available = bool(candidate_heap)
                break
            if remaining_bytes == 0:
                truncated = bool(candidate_heap)
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

    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state must be a dictionary.")
        copied = copy_durable_json_object(state, "checkpoint")
        await self._ensure_ready()
        async with self._connection() as conn:
            async with conn.cursor() as cur:
                if await self._load_for_update(cur, session_id) is None:
                    raise KeyError(f"Session not found: {session_id}")
                for owner in await self._closure_lineage_owners(cur, (session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
                updated_at = await self._session_store_now(cur)
                replacement = _replace_checkpoint_preserving_completion_result_event_publications(
                    await self._load_checkpoint(cur, session_id),
                    copied,
                    session_id=session_id,
                )
                await _touch_session_activity(cur, session_id, updated_at)
                await self._upsert_checkpoint(cur, session_id, replacement, updated_at)
            await conn.commit()

    async def transform_checkpoint(
        self,
        session_id: str,
        checkpoint_transform: CheckpointTransform,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    session = await self._load_for_update(cur, session_id)
                    if session is None:
                        raise KeyError(f"Session not found: {session_id}")
                    updated_at = await self._session_store_now(cur)
                    _assert_session_run_epoch(session_id, session)
                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
                    current = await self._load_checkpoint(cur, session_id)
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
                        await _touch_session_activity(cur, session_id, updated_at)
                        await self._upsert_checkpoint(cur, session_id, transformed, updated_at)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    async def transform_checkpoint_with_store_time(
        self,
        session_id: str,
        checkpoint_transform: StoreTimeCheckpointTransform,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    session = await self._load_for_update(cur, session_id)
                    if session is None:
                        raise KeyError(f"Session not found: {session_id}")
                    _assert_session_run_epoch(session_id, session)
                    for owner in await self._closure_lineage_owners(cur, (session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
                    current = await self._load_checkpoint(cur, session_id)
                    now = await self._session_store_now(cur)
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
                        await _touch_session_activity(cur, session_id, now)
                        await self._upsert_checkpoint(cur, session_id, transformed, now)
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            if access_bounds is not None:
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                access_bounds.require_action(await self._load(cur, session_id), "inspect_state")
            return await self._load_checkpoint(cur, session_id)

    async def load_interruption_cascade_marker(
        self,
        session_id: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        checkpoint_root_key = (
            "__cayu_no_checkpoint_root_guard__"
            if checkpoint_root_guard is None
            else checkpoint_root_guard.key
        )
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                f"""
                WITH marker AS (
                    SELECT
                        jsonb_typeof(state -> '{checkpoint_root_key}')
                            AS checkpoint_root_field_type,
                        left(
                            state ->> '{checkpoint_root_key}',
                            {CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS + 1}
                        )
                            AS checkpoint_root_field_scalar,
                        state -> 'pending_interruption_cascade' AS value
                    FROM cayu_checkpoints
                    WHERE session_id = %s
                )
                SELECT
                    checkpoint_root_field_type,
                    checkpoint_root_field_scalar,
                    jsonb_typeof(value),
                    jsonb_typeof(value -> 'attempt_id'),
                    left(value ->> 'attempt_id', 129),
                    jsonb_typeof(value -> 'interrupt_payload'),
                    jsonb_typeof(value -> 'generation'),
                    left(value ->> 'generation', 33),
                    jsonb_typeof(value -> 'failure_recorded'),
                    CASE
                        WHEN jsonb_typeof(value -> 'failure_recorded') = 'boolean'
                        THEN (value ->> 'failure_recorded')::boolean
                    END,
                    jsonb_typeof(value -> 'claim_id'),
                    left(value ->> 'claim_id', 129),
                    jsonb_typeof(value -> 'claim_expires_at'),
                    left(value ->> 'claim_expires_at', 65),
                    jsonb_typeof(value -> 'created_at'),
                    left(value ->> 'created_at', 65)
                FROM marker
                """,
                (session_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            scalar_text = row[1]
            if checkpoint_root_guard is not None:
                checkpoint_root_guard.validate(
                    session_id,
                    checkpoint_root_field_projection_from_storage(
                        json_type=row[0],
                        scalar_text=scalar_text,
                    ),
                )
            field_types = {
                "attempt_id": row[3],
                "interrupt_payload": row[5],
                "generation": row[6],
                "failure_recorded": row[8],
                "claim_id": row[10],
                "claim_expires_at": row[12],
                "created_at": row[14],
            }
            field_values = {
                "attempt_id": row[4],
                "generation": row[7],
                "failure_recorded": row[9],
                "claim_id": row[11],
                "claim_expires_at": row[13],
                "created_at": row[15],
            }
            return _project_interruption_cascade_marker_fields(
                row[2],
                field_types,
                field_values,
            )

    # -- internal helpers -------------------------------------------------

    async def _load(self, cur: Any, session_id: str) -> Session | None:
        await cur.execute(
            f"SELECT {pg_support.SESSION_COLUMNS} FROM cayu_sessions WHERE id = %s",
            (session_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return pg_support.session_from_row(
            row,
            labels=await self._load_labels(cur, session_id),
        )

    async def _load_for_update(self, cur: Any, session_id: str) -> Session | None:
        await cur.execute(
            f"SELECT {pg_support.SESSION_COLUMNS} FROM cayu_sessions WHERE id = %s FOR UPDATE",
            (session_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return pg_support.session_from_row(
            row,
            labels=await self._load_labels(cur, session_id),
        )

    async def _load_for_key_share(self, cur: Any, session_id: str) -> Session | None:
        """Load immutable parent identity while preventing delete or key replacement."""

        await cur.execute(
            f"SELECT {pg_support.SESSION_COLUMNS} FROM cayu_sessions WHERE id = %s FOR KEY SHARE",
            (session_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return pg_support.session_from_row(
            row,
            labels=await self._load_labels(cur, session_id),
        )

    async def _load_labels(self, cur: Any, session_id: str) -> dict[str, str]:
        await cur.execute(
            """
            SELECT key, value
            FROM cayu_session_labels
            WHERE session_id = %s
            ORDER BY key ASC
            """,
            (session_id,),
        )
        return {row[0]: row[1] for row in await cur.fetchall()}

    async def _load_labels_for_sessions(
        self,
        cur: Any,
        session_ids: list[str],
    ) -> dict[str, dict[str, str]]:
        if not session_ids:
            return {}
        await cur.execute(
            """
            SELECT session_id, key, value
            FROM cayu_session_labels
            WHERE session_id = ANY(%s)
            ORDER BY session_id ASC, key ASC
            """,
            (session_ids,),
        )
        labels_by_session_id: dict[str, dict[str, str]] = {
            session_id: {} for session_id in session_ids
        }
        for row in await cur.fetchall():
            labels_by_session_id[row[0]][row[1]] = row[2]
        return labels_by_session_id

    async def _reject_new_work_after_steering(
        self, cur: Any, session: Session, *, allow_completed_interaction: bool = False
    ) -> None:
        from cayu.runtime._session_steering import (
            reject_new_work_after_steering,
            steering_operation_key_from_checkpoint,
        )

        checkpoint = await self._load_checkpoint(cur, session.id)
        key = steering_operation_key_from_checkpoint(session, checkpoint)
        if key is not None:
            await cur.execute(
                "SELECT record FROM cayu_session_operations "
                "WHERE session_id = %s AND idempotency_key = %s",
                (session.id, key),
            )
            row = await cur.fetchone()
            reject_new_work_after_steering(
                session,
                checkpoint,
                None if row is None else _decode_model_completion_stage_record(row[0]),
                allow_completed_interaction=allow_completed_interaction,
            )

    async def _load_checkpoint(self, cur: Any, session_id: str) -> dict[str, Any] | None:
        await cur.execute(
            "SELECT state FROM cayu_checkpoints WHERE session_id = %s",
            (session_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return copy_durable_json_object(pg_support._json_obj(row[0]), "checkpoint")

    async def _upsert_checkpoint(
        self,
        cur: Any,
        session_id: str,
        checkpoint: dict[str, Any],
        updated_at: datetime,
    ) -> None:
        await cur.execute(
            """
            INSERT INTO cayu_checkpoints (
                session_id, state, updated_at,
                pending_action_source_bytes,
                pending_action_tool_call_count,
                pending_action_flags,
                pending_action_metrics_ready
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO UPDATE SET
                state = EXCLUDED.state,
                updated_at = EXCLUDED.updated_at,
                pending_action_source_bytes = EXCLUDED.pending_action_source_bytes,
                pending_action_tool_call_count = EXCLUDED.pending_action_tool_call_count,
                pending_action_flags = EXCLUDED.pending_action_flags,
                pending_action_metrics_ready = EXCLUDED.pending_action_metrics_ready
            """,
            _checkpoint_row_values(session_id, checkpoint, updated_at),
        )

    async def _first_existing_event_id(
        self,
        session_id: str,
        event_ids: list[str],
    ) -> str | None:
        async with self._connection() as conn, conn.cursor() as cur:
            for event_id in event_ids:
                await cur.execute(
                    "SELECT 1 FROM cayu_events WHERE session_id = %s AND event_id = %s",
                    (session_id, event_id),
                )
                if await cur.fetchone() is not None:
                    return event_id
        return None


def _new_id() -> str:
    from uuid import uuid4

    return str(uuid4())


def _checkpoint_row_values(
    session_id: str,
    checkpoint: dict[str, Any],
    updated_at: datetime,
) -> tuple[object, ...]:
    from cayu.sessions.pending_actions import pending_action_checkpoint_metrics

    checkpoint = copy_durable_json_object(checkpoint, "checkpoint")
    source_bytes, tool_call_count, flags = pending_action_checkpoint_metrics(checkpoint)
    return (
        session_id,
        pg_support._dumps(checkpoint),
        pg_support.to_utc(updated_at),
        source_bytes,
        tool_call_count,
        flags,
        True,
    )


def _decode_runtime_publication_record(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication receipt is malformed or conflicts with its key."
        )
    return value


def _decode_model_completion_stage_record(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise SessionModelCompletionStageConflict(
            "The durable model-completion stage record is malformed."
        )
    return value


def _event_record_from_row(row: tuple[Any, Any] | None) -> EventRecord | None:
    """Build an EventRecord from a ``(sequence, event)`` row, or None for a missing row."""
    if row is None:
        return None
    return EventRecord(sequence=row[0], event=Event(**pg_support._json_obj(row[1])))


def _persisted_event_side_effect_delivery_from_row(
    row: tuple[Any, ...],
) -> PersistedEventSideEffectDelivery:
    return PersistedEventSideEffectDelivery(
        session_id=row[0],
        event_id=row[1],
        event_sequence=row[2],
        status=PersistedEventSideEffectStatus(row[3]),
        attempts=row[4],
        claim_id=row[5],
        lease_expires_at=pg_support.to_utc_optional(row[6]),
        next_attempt_at=pg_support.to_utc_optional(row[7]),
        last_error=row[8],
        updated_at=pg_support.to_utc(row[9]),
    )


# Lifecycle/terminal event-type strings used to derive a session outcome. Sourced from
# the EventType enum (not hardcoded literals) so the SQL stays in sync with the contract.
# These are constants, never user input, so they are safe to read in queries via params.
_LIFECYCLE_EVENT_TYPES = [
    str(EventType.SESSION_STARTED),
    str(EventType.SESSION_RESUMED),
]
_TERMINAL_EVENT_TYPES = [
    str(EventType.SESSION_COMPLETED),
    str(EventType.SESSION_FAILED),
    str(EventType.SESSION_INTERRUPTED),
]


# Re-exported so callers can construct a pool explicitly when desired.
__all__ = [
    "AsyncConnectionPool",
    "PostgresEmbeddingKnowledgeStore",
    "PostgresEventWatcherStore",
    "PostgresKnowledgeStore",
    "PostgresSessionStore",
    "PostgresTaskStore",
]
