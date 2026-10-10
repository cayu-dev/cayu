from __future__ import annotations

import asyncio
import hmac
import json
import sqlite3
from collections.abc import AsyncIterator, Callable, Collection, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar
from uuid import uuid4

from cayu._resource_store_surface import model_store_surface
from cayu.budgets.pricing import PriceBook
from cayu.collaboration.peer_content import (
    PeerAppendKey,
    PeerContentAppendRequest,
    PeerContentExposureReceipt,
    PeerContentExposureRequest,
    PeerContentReceipt,
)
from cayu.runtime import _session_message_queue as message_queue
from cayu.runtime._cost_accounting import CostAccountingSnapshot
from cayu.runtime._usage_accounting import UsageAccountingSnapshot
from cayu.sessions import creation_fence
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
    SessionMessageInspection,
    SessionMessageQuery,
    SessionMessageSource,
    session_message_rejection,
)
from cayu.storage import _creation_fence
from cayu.storage import _sqlite_event_delivery as event_delivery_ops
from cayu.storage import _sqlite_peer_content as peer_content_ops
from cayu.storage import _sqlite_session_queries as session_queries
from cayu.storage import _sqlite_transcript as transcript_ops
from cayu.storage._context_selection_fence import SQLiteContextSelectionFenceMixin
from cayu.storage._creation_fence import SQLiteCreationFenceMixin
from cayu.storage._external_wait_sqlite import SQLiteExternalWaitMixin
from cayu.storage._phase_timing import TimedStoreLock, TimedStoreReadQueue
from cayu.storage._session_execution import SQLiteSessionExecutionMixin
from cayu.storage.targets import require_sqlite_store_allowed

if TYPE_CHECKING:
    from cayu.runtime._zero_work_interruption import (
        ZeroWorkInterruptionPublication,
        ZeroWorkInterruptionRequest,
    )
    from cayu.sessions._temporary_continuation import TemporaryServiceAdmission
    from cayu.sessions.access import _SessionAccessBounds
    from cayu.sessions.exports import SessionExportLimits, SessionExportSnapshot
    from cayu.storage.retention import (
        RetentionAuditEntry,
        RetentionAuditRecord,
        RetentionMode,
        RetentionProgressCallback,
        RetentionProtection,
        RetentionReport,
        SessionRetentionPolicy,
    )


from cayu._clock import utc_clock, utc_duration_cutoff
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    JsonUtf8SizeCounter,
    canonical_bounded_durable_json_bytes,
    copy_durable_json_object,
    copy_label_map,
    require_nonblank,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu.approvals.tools import ResolutionActor, resolution_actor_payload
from cayu.budgets.aggregates import EXACT_AGGREGATE, UsageRollupStoreResult
from cayu.events import (
    Event,
    EventType,
    event_with_runtime_payload_authority,
)
from cayu.execution_profiles import (
    ExecutionProfileDecision,
    ExecutionProfileIdentity,
    ExecutionProfileRejectionResult,
)
from cayu.execution_units import ToolRoundIdentity
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
    validate_context_exposure_carried_receipt_scope,
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
from cayu.runtime.evidence_spool import EvidenceSpool, _settled_evidence_reads_required
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
    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
    RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX,
    BudgetReservationIdentityConflict,
    CheckpointRootFieldGuard,
    CheckpointTransform,
    ForkCheckpointAuthorityDecoder,
    InteractionAttribution,
    InteractionTransitionReceiptResult,
    InteractionTransitionResult,
    InteractionTransitionSpec,
    ModelCompletionStage,
    ModelCompletionStageAbandonmentResult,
    ModelCompletionStageDispatch,
    ModelCompletionStageResult,
    ModelCompletionStageSettlementRequest,
    QueuedDispatchTerminalReceipt,
    QueuedDispatchTerminalReceiptQuery,
    QueuedInteractionProfileHandoff,
    RunRequest,
    RuntimePublicationMutation,
    RuntimePublicationReceipt,
    RuntimePublicationResult,
    SessionForkActiveModelStageConflict,
    SessionMessageQueueStatus,
    SessionModelCompletionDispatchAlreadyAuthorized,
    SessionModelCompletionStageConflict,
    SessionModelTransition,
    SessionOperationInitializer,
    SessionOperationPublication,
    SessionOperationTransform,
    SessionRunFenced,
    SessionRuntimePublicationConflict,
    SessionStatusConflict,
    SessionStore,
    StoreTimeCheckpointTransform,
    StoreTimeSessionOperationTransform,
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
    _checkpoint_after_queued_interaction_profile_handoff,
    _child_session_notification_consumption_record,
    _child_session_notification_consumption_replays,
    _completion_result_event_publication_delete_block_reason,
    _copy_historical_queued_interaction_profile_handoff,
    _copy_mcp_manifest_publication,
    _copy_optional_event_id,
    _copy_optional_execution_profile,
    _copy_optional_execution_profile_decision,
    _copy_optional_interaction_admission,
    _copy_optional_tool_capability_ceiling,
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
    _execution_profile_rejection_events_equivalent,
    _historical_queued_handoff_stage_from_records,
    _incomplete_recovery_claim_from_checkpoint,
    _initial_transcript_pending_checkpoint,
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
    _runtime_publication_json_equal,
    _runtime_publication_receipt_record,
    _runtime_publication_referenced_event_ids,
    _runtime_publication_storage_key,
    _session_metadata_after_model_transition,
    _session_metadata_after_runtime_identity_adoption,
    _session_metadata_after_tool_capability_ceiling_admission,
    _terminal_publication_delete_block_reason,
    _tool_lifecycle_publication_identity,
    _tool_round_lifecycle_event_limit,
    _validate_execution_profile_admission,
    _validate_execution_profile_rejection_session,
    _validate_inactive_for_seconds,
    _validate_interaction_transition_invocation_authority_parameters,
    _validate_interaction_transition_receipt_authority,
    _validate_interaction_transition_receipt_recovery_authority,
    _validate_interaction_transition_recovery_claim_id,
    _validate_invocation_release_settlement_receipt_authority,
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
    _validate_tool_round_checkpoint_mutation,
    _validate_tool_round_publication,
    _validate_user_input_checkpoint_mutation,
    checkpoint_root_field_projection_from_storage,
    copy_run_request,
    copy_session_user_metadata,
    deferred_interaction_input_for_run_request,
    replace_session_user_metadata,
    transform_fork_checkpoint,
)
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectClaim,
    PersistedEventSideEffectDelivery,
    PersistedEventSideEffectStatus,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.forks import (
    FORK_EXECUTION_PROFILE_METADATA_KEY,
    ForkSystemPromptReplacement,
    ProfiledSessionForkResult,
    SessionForkProfileRelationship,
    apply_fork_system_prompt_replacement,
)
from cayu.sessions.inspection import SessionInspectionIdentity
from cayu.sessions.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
    INTERACTION_TERMINAL_EVENT_TYPES,
)
from cayu.sessions.invocation import SessionInvocation
from cayu.sessions.lineage import (
    SessionLineageQuery,
    SessionLineageResult,
)
from cayu.sessions.mcp_manifest_history import (
    McpManifestBaseline,
    McpManifestBaselineLoadResult,
    McpManifestPublicationResult,
    _stored_mcp_manifest_baseline_json,
    _validate_mcp_manifest_history_keys,
    _validate_mcp_manifest_publication_state,
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
    session_query_from_aggregate_filter,
)
from cayu.sessions.records import (
    EventRecord,
    PendingActionKind,
    PendingActionSession,
    RunnerObservedEventIdentity,
    Session,
    SessionIdentity,
    SessionInvocationSnapshot,
    SessionRuntimeIdentity,
    SessionStateSnapshot,
    SessionStatus,
    TranscriptRecord,
    copy_session_identity,
    copy_session_runtime_identity,
)
from cayu.sessions.summaries import (
    EventSummary,
    SessionOperationalSnapshot,
    SessionOutcome,
    session_outcome,
)
from cayu.sessions.terminal_evidence import (
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES,
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
    SessionTopologyQuery,
    SessionTopologyStoreResult,
)
from cayu.sessions.transcript_input import (
    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
    DeferredInteractionInput,
    deferred_interaction_input_from_storage_payload,
    deferred_interaction_input_storage_payload,
    session_messages_input_contract_evidence,
)
from cayu.sessions.transcript_queries import (
    ForkTranscriptValidator,
    TranscriptPage,
    TranscriptQuery,
    TranscriptSearchQuery,
    TranscriptSearchResult,
    TranscriptSnapshot,
    fork_transcript_is_accepted,
    transcript_search_document,
)
from cayu.sessions.usage import UsageRollupQuery
from cayu.storage import _session_store_sql as session_store_sql
from cayu.storage import _sqlite_connection as sqlite_connection
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema
from cayu.storage._sqlite_connection import (
    _run_off_thread_with_connection_ownership as _run_off_thread_with_connection_ownership,
)
from cayu.storage._validated_cache import validated_row_cache
from cayu.storage.tasks_sqlite import (
    _SQLITE_TASK_MIN_REQUIRED_REVISION as _SQLITE_TASK_MIN_REQUIRED_REVISION,
)
from cayu.storage.tasks_sqlite import SQLiteTaskStore as SQLiteTaskStore
from cayu.storage.tasks_sqlite import _like_contains_pattern as _like_contains_pattern
from cayu.storage.tasks_sqlite import (
    _sqlite_interrupted_task_handoff_receipt as _sqlite_interrupted_task_handoff_receipt,
)
from cayu.storage.tasks_sqlite import (
    _sqlite_task_terminalization_receipt as _sqlite_task_terminalization_receipt,
)
from cayu.storage.tasks_sqlite import _validate_task_positive_int as _validate_task_positive_int
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

_SQLITE_NON_SESSION_MIN_REQUIRED_REVISION = 18
_SQLITE_SESSION_MIN_REQUIRED_REVISION = 113
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
        or receipt.created_at != sqlite_records.parse_datetime(row["created_at"])
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
        or exposure.created_at != sqlite_records.parse_datetime(row["created_at"])
        or exposure.updated_at != sqlite_records.parse_datetime(row["updated_at"])
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


def _alias_key_fingerprint_matches(value: object, expected: str) -> bool:
    if type(value) is not str:
        return False
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(encoded, expected.encode("ascii"))


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
            (sqlite_records.format_datetime(activity_at), session_id),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"Session not found: {session_id}")
        return
    cursor = connection.execute(
        "UPDATE cayu_sessions SET last_activity_at = ? WHERE id = ? AND run_epoch = ?",
        (sqlite_records.format_datetime(activity_at), session_id, expected_run_epoch),
    )
    if cursor.rowcount != 1:
        _raise_session_write_conflict(connection, session_id, expected_run_epoch)


@validated_row_cache
def _checkpoint_from_json(value: str) -> dict[str, Any]:
    return copy_durable_json_object(json.loads(value), "checkpoint")


def _load_checkpoint_json(connection: sqlite3.Connection, session_id: str) -> str | None:
    row = connection.execute(
        "SELECT state_json FROM cayu_checkpoints WHERE session_id = ?", (session_id,)
    ).fetchone()
    return None if row is None else row["state_json"]


def _load_checkpoint_state(
    connection: sqlite3.Connection,
    session_id: str,
) -> dict[str, Any] | None:
    value = _load_checkpoint_json(connection, session_id)
    return None if value is None else _checkpoint_from_json(value)


def _reject_new_work_after_steering(
    connection: sqlite3.Connection, session: Session, *, allow_completed_interaction: bool = False
) -> None:
    from cayu.runtime._session_steering import (
        reject_new_work_after_steering,
        steering_operation_key_from_checkpoint,
    )

    checkpoint = _load_checkpoint_state(connection, session.id)
    key = steering_operation_key_from_checkpoint(session, checkpoint)
    if key is not None:
        row = connection.execute(
            "SELECT record_json FROM cayu_session_operations "
            "WHERE session_id = ? AND idempotency_key = ?",
            (session.id, key),
        ).fetchone()
        reject_new_work_after_steering(
            session,
            checkpoint,
            None if row is None else _decode_model_completion_stage_record(row["record_json"]),
            allow_completed_interaction=allow_completed_interaction,
        )


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
        ("issued_at", sqlite_records.format_datetime(record.issued_at)),
        ("expires_at", sqlite_records.format_datetime(record.expires_at)),
        ("max_calls", record.max_calls),
        ("used_calls", record.used_calls),
        ("revoked_at", sqlite_records.format_optional_datetime(record.revoked_at)),
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
        ("bound_at", sqlite_records.format_datetime(binding.bound_at)),
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
    _touch_session_activity(connection, session_id, activity_at)
    _insert_event_rows_in_transaction(connection, session_id, events, activity_at=activity_at)


def _insert_event_rows_in_transaction(
    connection: sqlite3.Connection,
    session_id: str,
    events: Sequence[Event],
    *,
    activity_at: datetime,
) -> None:
    """Insert prepared events after the transaction owner has authorized them."""
    from cayu.sessions.pending_actions import pending_action_event_storage_values

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
                sqlite_records.format_datetime(event.timestamp),
                event.agent_name,
                event.environment_name,
                event.workflow_name,
                event.tool_name,
                sqlite_records.json_dumps(event.payload),
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
    _record_invocation_terminal_event_receipts(
        connection, session_id, events, activity_at=activity_at
    )
    event_delivery_ops.enqueue_persisted_event_side_effects(connection, session_id, events)


def _record_invocation_terminal_event_receipts(
    connection: sqlite3.Connection,
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
        session = sqlite_records.load_session(connection, session_id)
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
                    sqlite_records.json_dumps(receipt_record),
                    sqlite_records.format_datetime(activity_at),
                )
                for receipt_key, receipt_record in terminal_receipts
            ],
        )


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
        return sqlite_records.event_from_row(row)
    _append_events_in_transaction(
        connection,
        event.session_id,
        [event],
        activity_at=activity_at,
    )
    return event


_SESSION_MESSAGE_EVENT_RETENTION_SQL = """
    AND NOT (
        cayu_events.event_type IN (
            'session.message.queued', 'session.message.delivered',
            'session.message.withdrawn', 'session.message.quarantined',
            'session.message.stale', 'session.message.expired'
        )
        AND EXISTS (
            SELECT 1 FROM cayu_session_message_queue AS queued
            WHERE queued.session_id = cayu_events.session_id
              AND queued.queue_id = json_extract(cayu_events.payload_json, '$.queue_id')
        )
    )
"""


def _session_message_acceptance_events(
    connection: sqlite3.Connection,
    session_id: str,
    rows: list[Any],
    *,
    quarantine_queue_id: str | None = None,
) -> dict[str, Event]:
    """Batch-read bounded audit projections of canonical events in the queue transaction."""
    ids = [row["accepted_event_id"] for row in rows]
    if quarantine_queue_id is None and any(
        type(event_id) is not str or len(event_id) > 512 for event_id in ids
    ):
        raise SessionMessageConflict()
    if not ids:
        return {}
    projection = (
        "json_object('queue_id', json_extract(payload_json, '$.queue_id'), "
        "'source', json_extract(payload_json, '$.source'))"
    )
    predicate = (
        f"event_id IN ({', '.join('?' for _ in ids)})"
        if quarantine_queue_id is None
        else "json_extract(payload_json, '$.queue_id') = ? LIMIT 2"
    )
    events = connection.execute(
        f"SELECT event_id, CASE WHEN length(CAST({projection} AS BLOB)) <= 32768 "
        f"THEN {projection} END AS audit_json FROM cayu_events WHERE session_id = ? "
        "AND event_type = 'session.message.queued' "
        f"AND {predicate}",
        (session_id, *(ids if quarantine_queue_id is None else [quarantine_queue_id])),
    ).fetchall()
    if quarantine_queue_id is not None and len(events) != 1:
        raise SessionMessageConflict()
    if any(row["audit_json"] is None for row in events):
        raise SessionMessageConflict()
    return {
        row["event_id"]: Event(
            id=row["event_id"],
            type=EventType.SESSION_MESSAGE_QUEUED,
            session_id=session_id,
            payload=json.loads(row["audit_json"]),
        )
        for row in events
    }


def _session_message_raw_bounded(
    connection: sqlite3.Connection,
    session_id: str,
    queue_id: str,
) -> dict[str, Any]:
    """Hash oversized cells in chunks without hydrating rejected content."""
    from hashlib import sha256

    columns = (
        "ordering_key",
        "queue_id",
        "session_id",
        "idempotency_key",
        "content",
        "message_json",
        "conditions_json",
        "terminal_json",
        "delivery_mode",
        "status",
        "requested_by_json",
        "accepted_run_epoch",
        "accepted_transcript_cursor",
        "accepted_event_id",
        "accepted_at",
        "delivered_run_epoch",
        "delivered_transcript_cursor",
        "delivered_event_id",
        "delivered_at",
    )
    projection = ", ".join(
        f"CASE WHEN length(CAST({name} AS BLOB)) <= {SESSION_MESSAGE_QUEUE_STORAGE_VALUE_MAX_BYTES} "
        f"THEN {name} END AS {name}"
        for name in columns
    )
    row = connection.execute(
        f"SELECT {projection} FROM cayu_session_message_queue WHERE session_id = ? AND queue_id = ?",
        (session_id, queue_id),
    ).fetchone()
    if row is None:
        raise SessionMessageConflict()
    raw = dict(row)
    sizes = connection.execute(
        "SELECT "
        + ", ".join(f"length(CAST({name} AS BLOB))" for name in columns)
        + " FROM cayu_session_message_queue WHERE session_id = ? AND queue_id = ?",
        (session_id, queue_id),
    ).fetchone()
    for name, size in zip(columns, sizes, strict=True):
        if size is None or size <= SESSION_MESSAGE_QUEUE_STORAGE_VALUE_MAX_BYTES:
            continue
        digest = sha256()
        for offset in range(1, size + 1, 65536):
            chunk = connection.execute(
                f"SELECT substr(CAST({name} AS BLOB), ?, 65536) FROM cayu_session_message_queue "
                "WHERE session_id = ? AND queue_id = ?",
                (offset, session_id, queue_id),
            ).fetchone()[0]
            digest.update(chunk)
        storage_type = connection.execute(
            f"SELECT typeof({name}) FROM cayu_session_message_queue WHERE session_id = ? AND queue_id = ?",
            (session_id, queue_id),
        ).fetchone()[0]
        raw[name] = message_queue.OversizedStorageValue(
            digest.hexdigest(), byte_length=size, storage_type=storage_type
        )
    return raw


def _queued_session_message_from_row(row: sqlite3.Row | dict[str, Any]) -> SessionQueuedMessage:
    requested_by = row["requested_by_json"]
    message_json = row["message_json"]
    return SessionQueuedMessage(
        queue_id=row["queue_id"],
        session_id=row["session_id"],
        idempotency_key=row["idempotency_key"],
        conditions=SessionMessageConditions.model_validate(
            {} if row["conditions_json"] is None else json.loads(row["conditions_json"])
        ),
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
        accepted_at=sqlite_records.parse_datetime(row["accepted_at"]),
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
            else sqlite_records.parse_datetime(row["delivered_at"])
        ),
    )


@model_store_surface("sessions")
class SQLiteSessionStore(
    SQLiteExternalWaitMixin,
    SQLiteSessionExecutionMixin,
    SQLiteContextSelectionFenceMixin,
    SQLiteCreationFenceMixin,
    SessionStore,
):
    """SQLite-backed session store for durable local runtime state."""

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
        from cayu.storage._session_access_records import sqlite_read

        return await sqlite_read(self, bounds, session_id, kind, offset, limit, max_bytes)

    async def _access_list_sessions(
        self, bounds: _SessionAccessBounds, query: SessionQuery
    ) -> SessionListResult:
        return await session_queries.list_sessions(
            self._run_read,
            ownership_clock=self._ownership_clock,
            query=query,
            pending_interruption_cascade_only=False,
            access_bounds=bounds,
        )

    async def _access_update_labels(
        self, bounds: _SessionAccessBounds, session_id: str, labels: dict[str, str]
    ) -> Session:
        return await self.update_labels(session_id, labels, _access_bounds=bounds)

    async def _access_load_session(self, bounds: _SessionAccessBounds, session_id: str) -> Session:
        from cayu.sessions.access import SessionAccessDenied

        session_id = require_clean_nonblank(session_id, "session_id")
        clause = session_store_sql.session_access_clause(
            bounds, dialect=session_queries.SQL_DIALECT
        )

        def read(connection):
            row = connection.execute(
                f"SELECT * FROM cayu_sessions WHERE id = ? AND ({clause.sql})",
                (session_id, *clause.params),
            ).fetchone()
            if row is None:
                raise SessionAccessDenied()
            labels = sqlite_records.load_session_labels_batch(connection, [session_id])
            return bounds.require_read(
                sqlite_records.session_from_row(row, labels=labels[session_id])
            )

        def snapshot(connection):
            connection.execute("BEGIN")
            try:
                return read(connection)
            finally:
                connection.rollback()

        return await self._run_read(snapshot)

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
    supports_storage_retention: ClassVar[bool] = True
    participant_session_binding_version: ClassVar[int | None] = 1
    recipient_continuation_selection_version: ClassVar[int | None] = 1
    context_view_selection_fence_version: ClassVar[int | None] = 1
    context_view_version: ClassVar[int | None] = 1
    peer_content_version: ClassVar[int | None] = 1

    def __init__(
        self,
        path: str | Path,
        *,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
        read_only: bool = False,
        public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
        ownership_clock: Callable[[], datetime] | None = None,
    ) -> None:
        require_sqlite_store_allowed("SQLiteSessionStore")
        from cayu.runtime._cost_accounting_refresh import CostAccountingAuthority
        from cayu.runtime._usage_accounting import SessionUsageCache

        self._cost_accounting_authority = CostAccountingAuthority()
        self._session_usage_cache = SessionUsageCache()
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
        diagnostic_source_missing = sqlite_connection.diagnostic_sqlite_source_missing(db_path)
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
        self._lock = TimedStoreLock()
        self._participant_creation_lock = TimedStoreLock()
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
        # Each leased reader retains physical ownership through cancellation.
        # Private in-memory databases must continue sharing the writer.
        self._closed = False
        self._readers: list[tuple[asyncio.Lock, sqlite3.Connection]] = []
        try:
            if str(effective_db_path) == ":memory:":
                self._readers.append((self._lock, self._connection))
            else:
                for _ in range(4):
                    connection = self._connect_read_only(effective_db_path)
                    self._readers.append((TimedStoreLock(), connection))
            for _, connection in self._readers:
                connection.execute("PRAGMA temp_store = FILE")
                connection.execute("PRAGMA temp.cache_size = -2048")
        except BaseException:
            for _, connection in self._readers:
                if connection is not self._connection:
                    connection.close()
            self._connection.close()
            raise
        self._read_lock, self._read_connection = self._readers[0]
        self._available_readers: asyncio.LifoQueue[tuple[asyncio.Lock, sqlite3.Connection]] = (
            TimedStoreReadQueue()
        )
        for reader in reversed(self._readers):
            self._available_readers.put_nowait(reader)

    def durable_state_paths(self) -> tuple[Path, ...]:
        """Return the primary SQLite file for state-boundary validation."""

        if str(self.path) == ":memory:":
            return ()
        return (self.path.resolve(),)

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

    async def _terminalize_zero_work_interruption(
        self,
        request: ZeroWorkInterruptionRequest,
    ) -> ZeroWorkInterruptionPublication | None:
        """Atomically prove and terminalize zero work; unsupported stores decline."""
        from cayu.storage._zero_work_interruption import sqlite_terminalize

        return await sqlite_terminalize(self, request)

    async def _run_read(self, query: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run a cancellable read while retaining physical connection ownership."""

        def guarded(connection: sqlite3.Connection) -> _T:
            self._require_current_public_authority_configuration(connection)
            return query(connection)

        async def read_with_lease() -> _T:
            reader = await self._available_readers.get()
            try:
                if self._closed:
                    raise RuntimeError("SQLite session store is closed.")
                lock, connection = reader
                return await _run_off_thread_with_connection_ownership(
                    lock, connection, guarded, interrupt_on_cancellation=True
                )
            finally:
                self._available_readers.put_nowait(reader)

        if _settled_evidence_reads_required():
            return await read_with_lease()

        owner = asyncio.create_task(read_with_lease(), name="cayu-sqlite-read-owner")
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
                        sqlite_records.event_from_row(interaction_started_row),
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
                            sqlite_records.event_from_row(issued_row),
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
                            sqlite_records.format_datetime(record.issued_at),
                            sqlite_records.format_datetime(record.expires_at),
                            record.max_calls,
                            record.used_calls,
                            None,
                            sqlite_records.json_dumps(record.model_dump(mode="json")),
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
                if not sqlite_records.session_exists(connection, session_id):
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
                if not sqlite_records.session_exists(connection, session_id):
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
                        sqlite_records.event_from_row(event_row),
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
                        sqlite_records.format_datetime(binding.bound_at),
                        sqlite_records.json_dumps(binding.model_dump(mode="json")),
                    ),
                )
                connection.execute(
                    "UPDATE cayu_targeted_tool_grants SET used_calls = ?, record_json = ? "
                    "WHERE grant_id = ? AND used_calls = ?",
                    (
                        updated.used_calls,
                        sqlite_records.json_dumps(updated.model_dump(mode="json")),
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
                    persisted_event = sqlite_records.event_from_row(event_row)
                    validate_targeted_tool_grant_revocation_evidence(
                        record,
                        persisted_event,
                    )
                    return record, persisted_event
                for owner in self._closure_lineage_owners_unlocked(
                    (session_id,), connection=connection
                ):
                    _check_closure_lineage_owner(owner, (session_id,))
                latest_use_row = connection.execute(
                    "SELECT MAX(bound_at) AS latest_bound_at "
                    "FROM cayu_targeted_tool_grant_uses WHERE grant_id = ?",
                    (grant_id,),
                ).fetchone()
                latest_bound_at = latest_use_row["latest_bound_at"]
                if latest_bound_at is not None and (
                    sqlite_records.parse_datetime(str(latest_bound_at)) > revoked_at
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
                        sqlite_records.format_datetime(revoked_at),
                        sqlite_records.json_dumps(updated.model_dump(mode="json")),
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
                    sqlite_records.event_from_row(interaction_started_row),
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
        async with self._lock:
            self._require_current_public_authority_configuration(self._connection)
            if request.session_id is not None and request.parent_session_id == request.session_id:
                raise ValueError("Session cannot be its own parent.")
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                with self._connection:
                    created_at = self._ownership_clock()
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
                        creation_fence.require_pending(
                            creation_target,
                            _creation_fence.sqlite_read(self._connection, creation_target),
                            request_commitment=participant_request_commitment,
                            requested_session_id=request.session_id,
                        )
                    parent_session = (
                        None
                        if request.parent_session_id is None
                        else sqlite_records.load_session(
                            self._connection, request.parent_session_id
                        )
                    )
                    if request.parent_session_id is not None and parent_session is None:
                        raise ValueError(f"Parent session not found: {request.parent_session_id}")
                    if parent_session is not None:
                        for owner in self._closure_lineage_owners_unlocked((parent_session.id,)):
                            _check_closure_lineage_owner(owner, (parent_session.id,))
                    session = sqlite_records.session_from_request(
                        request,
                        identity=identity,
                        parent_session=parent_session,
                        created_at=created_at,
                    )
                    self._require_available_closure_identity_unlocked(session.id)
                    self._require_external_creation_unlocked(request)
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
                            sqlite_records.format_datetime(session.created_at),
                            sqlite_records.format_datetime(session.updated_at),
                            sqlite_records.format_datetime(session.last_activity_at),
                            session.run_epoch,
                            sqlite_records.json_dumps(session.invocation.model_dump(mode="json")),
                            sqlite_records.json_dumps(session.metadata),
                        ),
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
                        self._connection.execute(
                            """
                            INSERT INTO cayu_participant_session_bindings (
                                creation_key, request_commitment, session_id, session_instance_id,
                                application_scope, participant_owner_id, participant_owner_incarnation,
                                participant_id, participant_incarnation, lifecycle_revision,
                                configuration_revision, admission_generation, creator_commitment,
                                authorization_commitment, initial_input_commitment,
                                execution_profile_commitment, binding_json, receipt_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                                sqlite_records.json_dumps(binding.model_dump(mode="json")),
                                sqlite_records.json_dumps(receipt.model_dump(mode="json")),
                            ),
                        )
                        if recipient_selection is not None:
                            from cayu.sessions.context_views import ContextViewSelectionReceipt

                            row = self._connection.execute(
                                "SELECT receipt_json FROM cayu_context_view_selections "
                                "WHERE selection_key = ?",
                                (recipient_selection.selection_key,),
                            ).fetchone()
                            stored_selection = (
                                None
                                if row is None
                                else ContextViewSelectionReceipt.model_validate_json(row[0])
                            )
                            if (
                                stored_selection is None
                                or stored_selection != recipient_selection
                                or stored_selection.state not in {"adopted", "transferred"}
                            ):
                                raise PermissionError(
                                    "Recipient context-view ownership changed before child creation."
                                )
                        if receipt.recipient_metadata_json is not None:
                            from cayu.sessions.base import _RECIPIENT_PROVENANCE_CAPABILITY

                            if participant_provenance is not _RECIPIENT_PROVENANCE_CAPABILITY:
                                raise PermissionError(
                                    "Recipient provenance requires the trusted application boundary."
                                )
                            self._connection.executemany(
                                "INSERT INTO cayu_transcript_messages "
                                "(session_id, role, interaction_id, message_json, transcript_search_document) "
                                "VALUES (?, ?, ?, ?, ?)",
                                [
                                    (
                                        session.id,
                                        str(message.role),
                                        None,
                                        sqlite_records.json_dumps(message.model_dump(mode="json")),
                                        transcript_search_document(message),
                                    )
                                    for message in request.messages
                                ],
                            )
                        if recipient_receipt_validator is not None:
                            recipient_receipt_validator(session, receipt)
                        if creation_target is not None:
                            creation_fence.validate_binding(
                                creation_target,
                                binding,
                                requested_session_id=receipt.requested_session_id,
                            )
                            _creation_fence.sqlite_write(
                                self._connection,
                                creation_fence.created(creation_target, session, receipt),
                            )
                    elif creation_target is not None:
                        raise PermissionError("Creation targets require participant ownership.")
                    if initial_operation_records:
                        self._connection.executemany(
                            "INSERT INTO cayu_session_operations "
                            "(session_id, idempotency_key, record_json, updated_at) "
                            "VALUES (?, ?, ?, ?)",
                            [
                                (
                                    session.id,
                                    key,
                                    sqlite_records.json_dumps(record),
                                    sqlite_records.format_datetime(session.updated_at),
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
                            sqlite_records.session_label_row_values(session),
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
                                sqlite_records.format_datetime(started_event.timestamp),
                                started_event.agent_name,
                                started_event.environment_name,
                                started_event.workflow_name,
                                started_event.tool_name,
                                sqlite_records.json_dumps(started_event.payload),
                                lookup_key,
                                projection,
                                projection_bytes,
                            ),
                        )
                        event_delivery_ops.enqueue_persisted_event_side_effects(
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
                                sqlite_records.json_dumps(
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
                            sqlite_records.checkpoint_row_values(
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
                                sqlite_records.checkpoint_row_values(
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
                            sqlite_records.checkpoint_row_values(
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
        if creation_request.metadata_json is not None:
            from cayu.sessions.base import _RECIPIENT_PROVENANCE_CAPABILITY

            if recipient_provenance is not _RECIPIENT_PROVENANCE_CAPABILITY:
                raise PermissionError(
                    "Recipient provenance requires the trusted application boundary."
                )
        async with self._participant_creation_lock:
            return await self._create_participant_owned_session_unserialized(
                creation_request,
                resolved_request=resolved_request,
                identity=identity,
                binding_factory=binding_factory,
                recipient_provenance=recipient_provenance,
                recipient_selection=recipient_selection,
                creation_target=creation_target,
                recipient_receipt_validator=recipient_receipt_validator,
            )

    async def _create_participant_owned_session_unserialized(
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

        existing = await self._lookup_participant_session_creation_unserialized(creation_request)
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
                participant_request_commitment=creation_request.request_commitment,
                recipient_receipt_validator=recipient_receipt_validator,
            )
        except (sqlite3.IntegrityError, creation_fence.SessionCreationConflict):
            existing = await self._lookup_participant_session_creation_unserialized(
                creation_request
            )
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
        async with self._participant_creation_lock:
            return await self._lookup_participant_session_creation_unserialized(creation_request)

    async def _lookup_participant_session_creation_unserialized(self, creation_request):
        from cayu.sessions.context_views import ParticipantSessionCreationRequest
        from cayu.storage._participant_session_records import reconstruct

        if type(creation_request) is not ParticipantSessionCreationRequest:
            raise TypeError("Participant creation requires a typed creation request.")
        async with self._lock:
            row = self._connection.execute(
                "SELECT * FROM cayu_participant_session_bindings WHERE creation_key = ?",
                (creation_request.creation_key,),
            ).fetchone()
            if row is None:
                return None
            if row["request_commitment"] != creation_request.request_commitment:
                raise ValueError("Participant creation key conflicts with the request.")
            session = sqlite_records.load_session(self._connection, row["session_id"])
            receipt = reconstruct(dict(row), session)
            assert session is not None
            return session.model_copy(deep=True), receipt

    async def load_participant_session_binding(self, session_id):
        receipt = await self.load_participant_session_creation_receipt(session_id)
        return None if receipt is None else receipt.binding

    async def _scan_participant_session_bindings(self, participant, *, after=None, limit=32):
        from cayu.sessions._participant_discovery import prepare_scan, reference, scan_parameters
        from cayu.storage._participant_session_records import reconstruct

        query = prepare_scan(participant, after, limit)

        def read(connection):
            with connection:
                connection.execute("BEGIN")
                rows = connection.execute(
                    "SELECT * FROM cayu_participant_session_bindings WHERE "
                    "application_scope=? AND participant_owner_id=? AND "
                    "participant_owner_incarnation=? AND participant_id=? AND "
                    "participant_incarnation=? AND creation_key>? "
                    "ORDER BY creation_key LIMIT ?",
                    scan_parameters(query),
                ).fetchall()
                return tuple(
                    reference(
                        reconstruct(
                            dict(row), sqlite_records.load_session(connection, row["session_id"])
                        ),
                        query,
                    )
                    for row in rows
                )

        return await self._run_read(read)

    async def load_participant_session_creation_receipt(self, session_id):
        from cayu.storage._participant_session_records import reconstruct

        async with self._lock:
            row = self._connection.execute(
                "SELECT * FROM cayu_participant_session_bindings WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            session = sqlite_records.load_session(self._connection, session_id)
            return reconstruct(dict(row), session)

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
        from cayu.storage._participant_session_records import reconstruct

        def query(connection):
            with connection:
                connection.execute("BEGIN")
                session = sqlite_records.load_session(connection, session_id)
                row = connection.execute(
                    "SELECT * FROM cayu_participant_session_bindings WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                binding = None if row is None else reconstruct(dict(row), session).binding
                checkpoint = _load_checkpoint_state(connection, session_id)
                pointer = completed_boundary(session, binding, checkpoint)
                assert session is not None and binding is not None
                tool_receipt = None
                if pointer.tool_round_id is not None:
                    publication_id = f"tool-round:{pointer.tool_round_id}"
                    key = _runtime_publication_storage_key(publication_id)
                    row = connection.execute(
                        "SELECT record_json FROM cayu_session_operations WHERE session_id = ? AND idempotency_key = ?",
                        (session_id, key),
                    ).fetchone()
                    if row is None:
                        closures = connection.execute(
                            "SELECT * FROM cayu_events WHERE session_id = ? AND event_type = ? "
                            "AND json_extract(payload_json, '$.tool_round_id') = ? "
                            "AND (json_extract(payload_json, '$.cleared') = 1 OR json_extract(payload_json, '$.transition') = 'answered') LIMIT 2",
                            (
                                session_id,
                                EventType.SESSION_CHECKPOINTED.value,
                                pointer.tool_round_id,
                            ),
                        ).fetchall()
                        publication_id = closed_round_publication_id(
                            pointer, tuple(sqlite_records.event_from_row(item) for item in closures)
                        )
                        key = _runtime_publication_storage_key(publication_id)
                        row = connection.execute(
                            "SELECT record_json FROM cayu_session_operations WHERE session_id = ? AND idempotency_key = ?",
                            (session_id, key),
                        ).fetchone()
                    if row is not None:
                        tool_receipt = _reconstruct_runtime_publication_receipt(
                            _decode_runtime_publication_record(row["record_json"]),
                            storage_key=key,
                            session_id=session_id,
                            publication_id=publication_id,
                        )
                    publication_frontier(pointer, tool_receipt)
                    assert tool_receipt is not None
                    self._validate_runtime_publication_material(connection, tool_receipt)
                end = publication_frontier(pointer, tool_receipt)
                row = connection.execute(
                    "SELECT * FROM cayu_events WHERE session_id = ? AND event_id = ?",
                    (session_id, pointer.completion_event_id),
                ).fetchone()
                completion = None if row is None else sqlite_records.event_from_row(row)
                rows = connection.execute(
                    "SELECT session_order, interaction_id, message_json FROM cayu_transcript_messages "
                    "WHERE session_id = ? AND session_order > ? AND session_order <= ? "
                    "ORDER BY session_order",
                    (session_id, pointer.source_transcript_cursor, end),
                ).fetchall()
                records = tuple(
                    TranscriptRecord(
                        index=row["session_order"] - 1,
                        interaction_id=row["interaction_id"],
                        message=Message.model_validate_json(row["message_json"]),
                    )
                    for row in rows
                )
                publication = capture_source(
                    session, binding, checkpoint, pointer, completion, records, tool_receipt
                )
                queued = connection.execute(
                    "SELECT 1 FROM cayu_session_message_queue "
                    "WHERE session_id = ? AND status = 'queued' LIMIT 1",
                    (session_id,),
                ).fetchone()
                closure = connection.execute(
                    "SELECT 1 FROM cayu_session_closure_progress AS p "
                    "WHERE root_session_id = ? OR EXISTS "
                    "(SELECT 1 FROM json_each(p.progress_json, '$.descendants') AS child "
                    "WHERE json_extract(child.value, '$.session_id') = ?) LIMIT 1",
                    (session_id, session_id),
                ).fetchone()
                active = connection.execute(
                    "SELECT 1 FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ? LIMIT 1",
                    (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                ).fetchone()
                return CompletedTurnSnapshot(
                    publication=publication,
                    current_session=session,
                    checkpoint=checkpoint,
                    has_queued_input=queued is not None,
                    has_closure_owner=closure is not None,
                    has_active_model_stage=active is not None,
                    current_transcript_cursor=transcript_ops.transcript_cursor(
                        connection, session_id
                    ),
                )

        return await self._run_read(query)

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
        manifest_json = sqlite_records.json_dumps(manifest.model_dump(mode="json"))
        owner = manifest.source_owner
        async with self._context_view_transaction():
            existing = self._connection.execute(
                "SELECT * FROM cayu_context_views WHERE publication_key = ?",
                (publication_key,),
            ).fetchone()
            if existing is not None:
                restored = validate_context_view_manifest_storage(
                    ContextViewManifest.model_validate(json.loads(existing["manifest_json"])),
                    view_id=existing["view_id"],
                    owner_scope=existing["source_owner_scope"],
                    owner_id=existing["source_owner_id"],
                    owner_incarnation=existing["source_owner_incarnation"],
                    source_session_id=existing["source_session_id"],
                    source_session_instance_id=existing["source_session_instance_id"],
                    transcript_cursor=existing["transcript_cursor"],
                    projection_schema=existing["projection_schema"],
                    extension_set_commitment=existing["extension_set_commitment"],
                )
                if restored != manifest:
                    raise ValueError("Context-view publication key conflicts with its manifest.")
                return restored.model_copy(deep=True)
            existing_view = self._connection.execute(
                "SELECT 1 FROM cayu_context_views WHERE view_id = ?",
                (manifest.view_id,),
            ).fetchone()
            if existing_view is not None:
                raise ValueError("Context-view ID is already bound to another manifest.")
            source = sqlite_records.load_session(self._connection, manifest.source_session_id)
            if source is None or source.instance_id != manifest.source_session_instance_id:
                raise LookupError("The source session incarnation is unavailable.")
            publication_count = self._connection.execute(
                "SELECT COUNT(*) AS count FROM cayu_context_views "
                "WHERE source_owner_scope = ? AND source_owner_id = ? "
                "AND source_owner_incarnation = ?",
                (owner.application_scope, owner.owner_id, owner.incarnation),
            ).fetchone()["count"]
            if publication_count >= CONTEXT_VIEW_MAX_PUBLICATIONS_PER_OWNER:
                raise OverflowError("Context-view publication quota exceeded for the owner.")
            self._connection.execute(
                """
                INSERT INTO cayu_context_views (
                    view_id, publication_key, source_owner_scope, source_owner_id,
                    source_owner_incarnation, source_session_id, source_session_instance_id,
                    transcript_cursor, projection_schema, extension_set_commitment, manifest_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    manifest_json,
                ),
            )
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
        publication_key = require_clean_nonblank(publication_key, "publication_key")
        async with self._lock:
            row = self._connection.execute(
                "SELECT * FROM cayu_context_views WHERE publication_key = ?",
                (publication_key,),
            ).fetchone()
            if row is None:
                return None
            return validate_context_view_manifest_storage(
                ContextViewManifest.model_validate(json.loads(row["manifest_json"])),
                view_id=row["view_id"],
                owner_scope=row["source_owner_scope"],
                owner_id=row["source_owner_id"],
                owner_incarnation=row["source_owner_incarnation"],
                source_session_id=row["source_session_id"],
                source_session_instance_id=row["source_session_instance_id"],
                transcript_cursor=row["transcript_cursor"],
                projection_schema=row["projection_schema"],
                extension_set_commitment=row["extension_set_commitment"],
            ).model_copy(deep=True)

    @asynccontextmanager
    async def _context_view_transaction(self) -> AsyncIterator[None]:
        # A process-local asyncio lock alone cannot serialize independent stores.
        async with self._lock:
            with self._connection:
                self._connection.execute("BEGIN IMMEDIATE")
                yield

    def _require_context_view_lifecycle_capacity(
        self, view_id: str, *, additional_slots: int = 0
    ) -> None:
        from cayu.sessions.context_views import validate_context_view_lifecycle_capacity

        events = self._connection.execute(
            "SELECT COUNT(*) FROM cayu_context_view_lifecycle_events WHERE view_id = ?",
            (view_id,),
        ).fetchone()[0]
        unsettled = self._connection.execute(
            "SELECT COUNT(*) FROM cayu_context_view_selections WHERE view_id = ? "
            "AND state IN ('selected', 'adopted', 'transferred')",
            (view_id,),
        ).fetchone()[0]
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
        from cayu.storage._context_selection_fence import reconstruct_exclusion, sqlite_exclusion

        if type(request) is not ContextViewSelectionRequest:
            raise TypeError("Context-view selection requires a typed request.")
        request = ContextViewSelectionRequest.model_validate(request)
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
        async with self._context_view_transaction():
            require_not_excluded(
                request,
                reconstruct_exclusion(sqlite_exclusion(self._connection, request.selection_key)),
                target=target,
            )
            now_ms = int(self._ownership_clock().timestamp() * 1000)
            expired_rows = self._connection.execute(
                "SELECT selection_key, receipt_json FROM cayu_context_view_selections "
                "WHERE owner_scope = ? AND owner_id = ? AND owner_incarnation = ? "
                "AND expires_at_ms <= ? AND state = 'selected' "
                "ORDER BY (selection_key = ?) DESC, expires_at_ms, selection_key LIMIT ?",
                (
                    owner.application_scope,
                    owner.owner_id,
                    owner.incarnation,
                    now_ms,
                    request.selection_key,
                    CONTEXT_VIEW_EXPIRY_BATCH_SIZE,
                ),
            ).fetchall()
            self._connection.execute(
                """
                UPDATE cayu_context_view_selections
                SET state = 'expired',
                    receipt_json = json_set(receipt_json, '$.state', 'expired')
                WHERE selection_key IN (
                    SELECT selection_key FROM cayu_context_view_selections
                    WHERE owner_scope = ? AND owner_id = ? AND owner_incarnation = ?
                      AND expires_at_ms <= ?
                      AND state = 'selected'
                    ORDER BY (selection_key = ?) DESC, expires_at_ms, selection_key
                    LIMIT ?
                )
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
                expired_receipt = ContextViewSelectionReceipt.model_validate(
                    json.loads(expired_row["receipt_json"])
                )
                self._require_context_view_lifecycle_capacity(
                    expired_receipt.view.view_id, additional_slots=1
                )
                operation_key = (
                    f"expiry:{expired_row['selection_key']}:{expired_receipt.ownership_revision}"
                )
                event = ContextViewLifecycleEvent(
                    event_id="sha256:"
                    + sha256(f"context-view-event:{operation_key}".encode()).hexdigest(),
                    operation_key=operation_key,
                    selection_key=expired_row["selection_key"],
                    view_id=expired_receipt.view.view_id,
                    state="expired",
                    owner=expired_receipt.owner,
                    owner_participant=expired_receipt.owner_participant,
                    pin_commitment=expired_receipt.pin_commitment,
                    ownership_revision=expired_receipt.ownership_revision,
                )
                self._connection.execute(
                    "INSERT OR IGNORE INTO cayu_context_view_lifecycle_events "
                    "(event_id, operation_key, selection_key, view_id, state, owner_scope, owner_id, "
                    "owner_incarnation, pin_commitment, ownership_revision, event_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                        sqlite_records.json_dumps(event.model_dump(mode="json")),
                    ),
                )
            existing = self._connection.execute(
                "SELECT * FROM cayu_context_view_selections WHERE selection_key = ?",
                (request.selection_key,),
            ).fetchone()
            if existing is not None:
                if existing["request_commitment"] != request_commitment:
                    raise ValueError("Context-view selection key conflicts with its request.")
                receipt = validate_context_view_receipt_storage(
                    ContextViewSelectionReceipt.model_validate(
                        json.loads(existing["receipt_json"])
                    ),
                    selection_key=existing["selection_key"],
                    view_id=existing["view_id"],
                    owner_scope=existing["owner_scope"],
                    owner_id=existing["owner_id"],
                    owner_incarnation=existing["owner_incarnation"],
                    state=existing["state"],
                    pin_commitment=existing["pin_commitment"],
                    ownership_revision=existing["ownership_revision"],
                )
                if receipt.state == "selected" and receipt.expires_at_ms <= now_ms:
                    raise ValueError("Expired context-view selection lacks cleanup evidence.")
                return receipt
            source = sqlite_records.load_session(self._connection, request.source_session_id)
            if source is None or source.instance_id != request.source_session_instance_id:
                raise LookupError("Context-view source session incarnation is unavailable.")
            if target is not None:
                from cayu.sessions._context_selection_fence import require_selection_source
                from cayu.storage._participant_session_records import reconstruct

                binding_row = self._connection.execute(
                    "SELECT * FROM cayu_participant_session_bindings WHERE session_id = ?",
                    (request.source_session_id,),
                ).fetchone()
                binding = (
                    None if binding_row is None else reconstruct(dict(binding_row), source).binding
                )
                require_selection_source(target, binding, now_ms=now_ms)
            rows = self._connection.execute(
                """
                SELECT view_id FROM cayu_context_views
                WHERE source_owner_scope = ? AND source_owner_id = ?
                  AND source_owner_incarnation = ? AND source_session_id = ?
                  AND source_session_instance_id = ? AND projection_schema = ?
                  AND extension_set_commitment = ?
                  AND (? != 'exact' OR view_id = ?)
                  AND (? IS NULL OR transcript_cursor >= ?)
                  ORDER BY transcript_cursor DESC, view_id DESC
                  LIMIT ?
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
            ).fetchall()
            if not rows:
                raise LookupError("No eligible context view is available.")
            if len(rows) > request.limits.max_views:
                raise OverflowError("Context-view count exceeds the requested limit.")
            row = self._connection.execute(
                "SELECT * FROM cayu_context_views WHERE view_id = ?", (rows[0]["view_id"],)
            ).fetchone()
            selected = validate_context_view_manifest_storage(
                ContextViewManifest.model_validate(json.loads(row["manifest_json"])),
                view_id=row["view_id"],
                owner_scope=row["source_owner_scope"],
                owner_id=row["source_owner_id"],
                owner_incarnation=row["source_owner_incarnation"],
                source_session_id=row["source_session_id"],
                source_session_instance_id=row["source_session_instance_id"],
                transcript_cursor=row["transcript_cursor"],
                projection_schema=row["projection_schema"],
                extension_set_commitment=row["extension_set_commitment"],
            )
            selection_count = self._connection.execute(
                "SELECT COUNT(*) AS count FROM cayu_context_view_selections "
                "WHERE owner_scope = ? AND owner_id = ? AND owner_incarnation = ?",
                (owner.application_scope, owner.owner_id, owner.incarnation),
            ).fetchone()["count"]
            if selection_count >= CONTEXT_VIEW_MAX_SELECTIONS_PER_OWNER:
                raise OverflowError("Context-view selection quota exceeded for the owner.")
            active = self._connection.execute(
                """
                SELECT COUNT(*) AS count FROM cayu_context_view_selections
                WHERE owner_scope = ? AND owner_id = ? AND owner_incarnation = ?
                  AND state IN ('selected', 'adopted', 'transferred')
                  AND (state <> 'selected' OR expires_at_ms > ?)
                """,
                (owner.application_scope, owner.owner_id, owner.incarnation, now_ms),
            ).fetchone()["count"]
            if active >= request.limits.max_pins:
                raise OverflowError("Context-view pin count exceeds the requested limit.")
            self._require_context_view_lifecycle_capacity(selected.view_id, additional_slots=1)
            selected_bytes = context_view_manifest_bytes(selected)
            if selected_bytes > request.limits.max_view_bytes:
                raise OverflowError("Context-view manifest exceeds the requested byte limit.")
            retained_rows = self._connection.execute(
                "SELECT * FROM cayu_context_view_selections "
                "WHERE owner_scope = ? AND owner_id = ? AND owner_incarnation = ? "
                "AND state IN ('selected', 'adopted', 'transferred') "
                "AND (state <> 'selected' OR expires_at_ms > ?)",
                (owner.application_scope, owner.owner_id, owner.incarnation, now_ms),
            ).fetchall()
            retained_views = {}
            for row in retained_rows:
                receipt = validate_context_view_receipt_storage(
                    ContextViewSelectionReceipt.model_validate(json.loads(row["receipt_json"])),
                    selection_key=row["selection_key"],
                    view_id=row["view_id"],
                    owner_scope=row["owner_scope"],
                    owner_id=row["owner_id"],
                    owner_incarnation=row["owner_incarnation"],
                    state=row["state"],
                    pin_commitment=row["pin_commitment"],
                    ownership_revision=row["ownership_revision"],
                )
                retained_views[receipt.view.view_id] = receipt.view
            retained_views[selected.view_id] = selected
            retained_bytes = sum(
                context_view_manifest_bytes(view) for view in retained_views.values()
            )
            if retained_bytes > request.limits.max_retained_bytes:
                raise OverflowError("Context-view retained bytes exceed the requested limit.")
            expires_at_ms = int(
                self._ownership_clock().timestamp() * 1000
                + request.limits.max_lifetime_seconds * 1000
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
            self._connection.execute(
                """
                INSERT INTO cayu_context_view_selections (
                    selection_key, request_commitment, view_id, owner_scope, owner_id,
                    owner_incarnation, state, pin_commitment, expires_at_ms,
                    ownership_revision, receipt_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    sqlite_records.json_dumps(receipt.model_dump(mode="json")),
                ),
            )
            self._connection.commit()
            return receipt.model_copy(deep=True)

    async def lookup_context_view_selection(self, selection_key):
        from cayu.sessions.context_views import ContextViewSelectionReceipt

        if type(selection_key) is not str:
            raise TypeError("Context-view selection key must be a string.")
        async with self._lock:
            row = self._connection.execute(
                "SELECT receipt_json FROM cayu_context_view_selections WHERE selection_key = ?",
                (selection_key,),
            ).fetchone()
        if row is None:
            return None
        return ContextViewSelectionReceipt.model_validate_json(row[0])

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
            reconstruct_exclusion,
            sqlite_decision,
            sqlite_exclusion,
        )

        if type(request) is not ContextViewOwnershipRequest:
            raise TypeError("Context-view ownership requires a typed request.")
        request = ContextViewOwnershipRequest.model_validate(request)
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
        async with self._context_view_transaction():
            control = reconstruct_exclusion(
                sqlite_exclusion(self._connection, request.selection_key)
            )
            require_selection_adoption(
                request,
                sqlite_decision(self._connection, control.request) if control is not None else None,
                target=target,
            )
            operation = self._connection.execute(
                "SELECT request_commitment, receipt_json "
                "FROM cayu_context_view_ownership_operations "
                "WHERE operation_key = ?",
                (request.operation_key,),
            ).fetchone()
            if operation is not None:
                if operation["request_commitment"] != request_commitment:
                    raise ValueError("Ownership operation key conflicts with its request.")
                return ContextViewSelectionReceipt.model_validate(
                    json.loads(operation["receipt_json"])
                )
            row = self._connection.execute(
                "SELECT * FROM cayu_context_view_selections WHERE selection_key = ?",
                (request.selection_key,),
            ).fetchone()
            if row is None:
                raise LookupError("Context-view selection is unavailable.")
            receipt = validate_context_view_receipt_storage(
                ContextViewSelectionReceipt.model_validate(json.loads(row["receipt_json"])),
                selection_key=row["selection_key"],
                view_id=row["view_id"],
                owner_scope=row["owner_scope"],
                owner_id=row["owner_id"],
                owner_incarnation=row["owner_incarnation"],
                state=row["state"],
                pin_commitment=row["pin_commitment"],
                ownership_revision=row["ownership_revision"],
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
            now_ms = int(self._ownership_clock().timestamp() * 1000)
            from cayu.sessions._context_selection_fence import require_adoption_deadline

            require_adoption_deadline(target, now_ms)
            self._require_context_view_lifecycle_capacity(receipt.view.view_id)
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
                self._connection.execute(
                    "UPDATE cayu_context_view_selections SET state = 'expired', receipt_json = ? "
                    "WHERE selection_key = ? AND ownership_revision = ?",
                    (
                        sqlite_records.json_dumps(expired.model_dump(mode="json")),
                        request.selection_key,
                        request.expected_revision,
                    ),
                )
                self._connection.execute(
                    "INSERT OR IGNORE INTO cayu_context_view_lifecycle_events "
                    "(event_id, operation_key, selection_key, view_id, state, owner_scope, owner_id, "
                    "owner_incarnation, pin_commitment, ownership_revision, event_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                        sqlite_records.json_dumps(event.model_dump(mode="json")),
                    ),
                )
                self._connection.commit()
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
            updated_json = sqlite_records.json_dumps(updated.model_dump(mode="json"))
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
            event_json = sqlite_records.json_dumps(event.model_dump(mode="json"))
            self._require_context_view_lifecycle_capacity(
                updated.view.view_id, additional_slots=int(next_state != "released")
            )
            cursor = self._connection.execute(
                "UPDATE cayu_context_view_selections SET owner_scope = ?, owner_id = ?, "
                "owner_incarnation = ?, state = ?, ownership_revision = ?, receipt_json = ? "
                "WHERE selection_key = ? AND ownership_revision = ? AND state = ?",
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
            if cursor.rowcount != 1:
                raise ValueError("Context-view ownership transition lost its compare-and-set race.")
            self._connection.execute(
                "INSERT INTO cayu_context_view_ownership_operations "
                "(operation_key, selection_key, request_commitment, receipt_json) VALUES (?, ?, ?, ?)",
                (
                    request.operation_key,
                    request.selection_key,
                    request_commitment,
                    updated_json,
                ),
            )
            self._connection.execute(
                "INSERT INTO cayu_context_view_lifecycle_events "
                "(event_id, operation_key, selection_key, view_id, state, owner_scope, owner_id, "
                "owner_incarnation, pin_commitment, ownership_revision, event_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            self._connection.commit()
            return updated.model_copy(deep=True)

    async def validate_context_view_source_closure(self, session_id: str) -> None:
        async with self._lock:
            now_ms = int(self._ownership_clock().timestamp() * 1000)
            row = self._connection.execute(
                "SELECT 1 FROM cayu_context_view_selections s "
                "LEFT JOIN cayu_context_views v ON v.view_id = s.view_id "
                "WHERE (v.source_session_id = ? OR v.view_id IS NULL) "
                "AND s.state IN ('selected', 'adopted', 'transferred') "
                "AND (s.state <> 'selected' OR s.expires_at_ms > ?) "
                "LIMIT 1",
                (session_id, now_ms),
            ).fetchone()
            if row is not None:
                raise ValueError("Session has an active context-view retention pin.")

    async def validate_context_view_compaction(
        self, session_id: str, expected_transcript_cursor: int
    ) -> None:
        async with self._lock:
            self._validate_context_view_compaction_unlocked(session_id, expected_transcript_cursor)

    def _validate_context_view_compaction_unlocked(
        self, session_id: str, expected_transcript_cursor: int
    ) -> None:
        from cayu.sessions.context_views import (
            ContextViewManifest,
            require_independent_context_view_material,
            validate_context_view_manifest_storage,
        )

        now_ms = int(self._ownership_clock().timestamp() * 1000)
        rows = self._connection.execute(
            "SELECT v.* FROM cayu_context_view_selections s "
            "LEFT JOIN cayu_context_views v ON v.view_id = s.view_id "
            "WHERE (v.source_session_id = ? OR v.view_id IS NULL) "
            "AND s.state IN ('selected', 'adopted', 'transferred') "
            "AND (s.state <> 'selected' OR s.expires_at_ms > ?) "
            "AND (v.view_id IS NULL OR v.transcript_cursor <= ?)",
            (session_id, now_ms, expected_transcript_cursor),
        )
        for row in rows:
            if row["manifest_json"] is None:
                raise ValueError("Pinned context-view material is unavailable for compaction.")
            manifest = validate_context_view_manifest_storage(
                ContextViewManifest.model_validate_json(row["manifest_json"]),
                view_id=row["view_id"],
                owner_scope=row["source_owner_scope"],
                owner_id=row["source_owner_id"],
                owner_incarnation=row["source_owner_incarnation"],
                source_session_id=row["source_session_id"],
                source_session_instance_id=row["source_session_instance_id"],
                transcript_cursor=row["transcript_cursor"],
                projection_schema=row["projection_schema"],
                extension_set_commitment=row["extension_set_commitment"],
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
            for path, value in (
                ("participant_id", participant.participant_id),
                ("incarnation", participant.incarnation),
                ("owner.application_scope", participant.owner.application_scope),
                ("owner.owner_id", participant.owner.owner_id),
                ("owner.incarnation", participant.owner.incarnation),
            ):
                participant_filter += (
                    f" AND json_extract(event_json, '$.owner_participant.{path}') = ?"
                )
                parameters.append(value)
        parameters.append(limit)
        async with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM cayu_context_view_lifecycle_events "
                "WHERE view_id = ?"
                + participant_filter
                + " ORDER BY ownership_revision, event_id LIMIT ?",
                parameters,
            ).fetchall()
            return tuple(
                validate_context_view_lifecycle_storage(
                    ContextViewLifecycleEvent.model_validate(json.loads(row["event_json"])), row
                )
                for row in rows
            )

    async def read_context_view(self, view_id: str, *, source_session_id: str):
        from cayu.sessions.context_views import (
            ContextViewManifest,
            ContextViewReadback,
            validate_context_view_manifest_storage,
        )

        async with self._lock:
            row = self._connection.execute(
                "SELECT * FROM cayu_context_views WHERE view_id = ?",
                (view_id,),
            ).fetchone()
            if row is None or row["source_session_id"] != source_session_id:
                raise LookupError("Context view is unavailable.")
            return ContextViewReadback(
                view=validate_context_view_manifest_storage(
                    ContextViewManifest.model_validate(json.loads(row["manifest_json"])),
                    view_id=row["view_id"],
                    owner_scope=row["source_owner_scope"],
                    owner_id=row["source_owner_id"],
                    owner_incarnation=row["source_owner_incarnation"],
                    source_session_id=row["source_session_id"],
                    source_session_instance_id=row["source_session_instance_id"],
                    transcript_cursor=row["transcript_cursor"],
                    projection_schema=row["projection_schema"],
                    extension_set_commitment=row["extension_set_commitment"],
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

        async with self._lock:
            self._require_current_public_authority_configuration(self._connection)
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                for owner in self._closure_lineage_owners_unlocked((source_session_id,)):
                    _check_closure_lineage_owner(owner, (source_session_id,))
                self._require_available_closure_identity_unlocked(fork.id)
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
                source_transcript_cursor = transcript_ops.transcript_cursor(
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
                    sqlite_records.session_to_row_values(fork),
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
                                sqlite_records.json_dumps(record),
                                sqlite_records.format_datetime(fork.updated_at),
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
                        sqlite_records.session_label_row_values(fork),
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
                                sqlite_records.json_dumps(message.model_dump(mode="json")),
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
                        sqlite_records.checkpoint_row_values(
                            fork.id, copied_checkpoint, fork.updated_at
                        ),
                    )
                if events:
                    from cayu.sessions.pending_actions import pending_action_event_storage_values

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
                                sqlite_records.format_datetime(event.timestamp),
                                event.agent_name,
                                event.environment_name,
                                event.workflow_name,
                                event.tool_name,
                                sqlite_records.json_dumps(event.payload),
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
                    event_delivery_ops.enqueue_persisted_event_side_effects(
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
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        if access_bounds is not None:
            from cayu._resource_access_errors import ResourceAccessDenied

            try:
                return await self._access_load_session(access_bounds, session_id)
            except ResourceAccessDenied:
                return None
        session_id = require_clean_nonblank(session_id, "session_id")
        return await self._run_read(
            lambda connection: sqlite_records.load_session(connection, session_id)
        )

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
                updated_at=sqlite_records.parse_datetime(row["updated_at"]),
                last_activity_at=sqlite_records.parse_datetime(row["last_activity_at"]),
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
                for owner in self._closure_lineage_owners_unlocked(
                    (copied.session_id,), connection=connection
                ):
                    _check_closure_lineage_owner(owner, (copied.session_id,))
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
                        sqlite_records.format_datetime(copied.created_at),
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
                created_at = sqlite_records.format_datetime(after[0])
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
                for owner in self._closure_lineage_owners_unlocked(
                    (copied.session_id,), connection=connection
                ):
                    _check_closure_lineage_owner(owner, (copied.session_id,))
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
                for receipt_id in copied.carried_receipt_ids:
                    receipt_row = connection.execute(
                        "SELECT * FROM cayu_recall_receipts WHERE receipt_id = ?",
                        (receipt_id,),
                    ).fetchone()
                    if receipt_row is None:
                        raise KeyError(f"Recall receipt not found: {receipt_id}")
                    validate_context_exposure_carried_receipt_scope(
                        copied,
                        _sqlite_recall_receipt(receipt_row),
                    )
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
                        sqlite_records.format_datetime(copied.created_at),
                        sqlite_records.format_datetime(copied.updated_at),
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
                created_at = sqlite_records.format_datetime(after[0])
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
                for owner in self._closure_lineage_owners_unlocked(
                    (session_id,), connection=connection
                ):
                    _check_closure_lineage_owner(owner, (session_id,))
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
                        sqlite_records.format_datetime(updated.updated_at),
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
        return await session_queries.inspect_identity(self._run_read, session_id)

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

    async def load_session_closure_receipt(
        self, session_id: str, plan_id: str
    ) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        plan_id = require_clean_nonblank(plan_id, "plan_id")
        async with self._lock:
            row = self._connection.execute(
                "SELECT receipt_json FROM cayu_session_closure_receipts "
                "WHERE session_id = ? AND plan_id = ?",
                (session_id, plan_id),
            ).fetchone()
        return None if row is None else json.loads(row[0])

    async def load_session_closure_progress(self, session_id: str, plan_id: str):
        async with self._lock:
            row = self._connection.execute(
                "SELECT progress_json FROM cayu_session_closure_progress "
                "WHERE root_session_id = ? AND plan_id = ?",
                (session_id, plan_id),
            ).fetchone()
        return None if row is None else json.loads(row[0])

    async def save_session_closure_progress(self, progress: dict[str, Any]) -> None:
        root_id = progress.get("root_session_id")
        plan_id = progress.get("plan_id")
        payload = json.dumps(progress, ensure_ascii=False, separators=(",", ":"))
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    "SELECT progress_json FROM cayu_session_closure_progress "
                    "WHERE root_session_id = ? AND plan_id = ?",
                    (root_id, plan_id),
                ).fetchone()
                if row is not None:
                    _validate_closure_progress_update(json.loads(row[0]), progress)
                self._connection.execute(
                    "INSERT INTO cayu_session_closure_progress "
                    "(root_session_id, plan_id, progress_json) VALUES (?, ?, ?) "
                    "ON CONFLICT(root_session_id, plan_id) DO UPDATE SET progress_json=excluded.progress_json",
                    (root_id, plan_id, payload),
                )
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    def _closure_lineage_owners_unlocked(
        self, targets: Iterable[str], *, connection: sqlite3.Connection | None = None
    ) -> tuple[dict[str, Any], ...]:
        encoded = json.dumps(tuple(targets))
        executor = self._connection if connection is None else connection
        rows = executor.execute(
            "SELECT progress_json FROM cayu_session_closure_progress AS p "
            "WHERE root_session_id IN (SELECT value FROM json_each(?)) OR EXISTS "
            "(SELECT 1 FROM json_each(p.progress_json, '$.descendants') AS child "
            "WHERE json_extract(child.value, '$.session_id') IN (SELECT value FROM json_each(?)))",
            (encoded, encoded),
        ).fetchall()
        return tuple(json.loads(row[0]) for row in rows)

    async def claim_session_closure_progress(self, progress: dict[str, Any]) -> None:
        payload = json.dumps(progress, ensure_ascii=False, separators=(",", ":"))
        progress = json.loads(payload)
        targets = _closure_progress_targets(progress)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                for owner in self._closure_lineage_owners_unlocked(targets):
                    if (owner["root_session_id"], owner["plan_id"]) == (
                        progress["root_session_id"],
                        progress["plan_id"],
                    ):
                        _validate_closure_progress_update(owner, progress)
                        self._connection.rollback()
                        return
                    _check_closure_lineage_owner(owner, targets)
                if self._load_unlocked(progress["root_session_id"]) is None:
                    raise ValueError("Closure root disappeared before lineage admission.")
                for item in progress["descendants"]:
                    child = self._load_unlocked(item["session_id"])
                    if child is None or child.parent_session_id != item["parent_session_id"]:
                        raise ValueError("Child lineage changed before closure admission.")
                if progress["phase"] in {"recursive", "reject"}:
                    expected = {
                        (item["session_id"], item["parent_session_id"])
                        for item in progress["descendants"]
                    }
                    rows = self._connection.execute(
                        "SELECT id, parent_session_id FROM cayu_sessions "
                        "WHERE parent_session_id IN (SELECT value FROM json_each(?)) LIMIT ?",
                        (json.dumps(list(targets)), len(expected) + 1),
                    )
                    if {(row["id"], row["parent_session_id"]) for row in rows} != expected:
                        raise ValueError("Child lineage changed before closure admission.")
                for target_id in targets:
                    target = self._load_unlocked(target_id)
                    if target is None:
                        raise ValueError("Closure target disappeared before admission.")
                    self._require_session_erasure_quiescence_unlocked(target)
                    self._load_session_closure_records_unlocked(
                        self._connection,
                        target_id,
                        max_records=progress["max_records"],
                        max_bytes=progress["max_bytes"],
                    )
                self._connection.execute(
                    "INSERT INTO cayu_session_closure_progress "
                    "(root_session_id, plan_id, progress_json) VALUES (?, ?, ?)",
                    (progress["root_session_id"], progress["plan_id"], payload),
                )
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    def _require_available_closure_identity_unlocked(self, session_id: str) -> None:
        for owner in self._closure_lineage_owners_unlocked((session_id,)):
            _check_closure_lineage_owner(owner, (session_id,))
        if (
            self._connection.execute(
                "SELECT 1 FROM cayu_session_closure_receipts WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
            is not None
        ):
            raise ValueError("Session identity was retired by closure.")

    async def load_session_closure_tombstones(
        self, root_session_id: str, plan_id: str
    ) -> tuple[dict[str, Any], ...]:
        root_session_id = require_clean_nonblank(root_session_id, "root_session_id")
        plan_id = require_clean_nonblank(plan_id, "plan_id")
        async with self._lock:
            rows = self._connection.execute(
                "SELECT tombstone_json FROM cayu_session_closure_tombstones "
                "WHERE root_session_id = ? AND plan_id = ? ORDER BY child_session_id",
                (root_session_id, plan_id),
            ).fetchall()
        return tuple(json.loads(row["tombstone_json"]) for row in rows)

    async def detach_session_children(
        self,
        parent_session_id: str,
        child_session_ids: tuple[str, ...],
        *,
        closure_receipt: dict[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        parent_session_id = require_clean_nonblank(parent_session_id, "parent_session_id")
        child_session_ids = tuple(child_session_ids)
        root_id = closure_receipt.get("root_session_id")
        plan_id = closure_receipt.get("plan_id")
        if type(root_id) is not str or type(plan_id) is not str:
            raise ValueError("Closure detachment receipt is missing identity.")
        if len(set(child_session_ids)) != len(child_session_ids):
            raise ValueError("Detached child session IDs must be unique.")
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                existing = self._connection.execute(
                    "SELECT tombstone_json FROM cayu_session_closure_tombstones "
                    "WHERE root_session_id = ? AND plan_id = ? ORDER BY child_session_id",
                    (root_id, plan_id),
                ).fetchall()
                if existing:
                    tombstones = tuple(json.loads(row["tombstone_json"]) for row in existing)
                    _validate_session_closure_detach_replay(
                        tombstones,
                        root_id=root_id,
                        plan_id=plan_id,
                        parent_session_id=parent_session_id,
                        child_session_ids=child_session_ids,
                    )
                    self._connection.rollback()
                    return tombstones
                for owner in self._closure_lineage_owners_unlocked(child_session_ids):
                    _check_closure_lineage_owner(owner, child_session_ids)
                if not child_session_ids:
                    self._connection.rollback()
                    return ()
                placeholders = ", ".join("?" for _ in child_session_ids)
                rows = self._connection.execute(
                    f"SELECT id, parent_session_id FROM cayu_sessions WHERE id IN ({placeholders})",
                    child_session_ids,
                ).fetchall()
                if {row["id"] for row in rows} != set(child_session_ids) or any(
                    row["parent_session_id"] != parent_session_id for row in rows
                ):
                    raise ValueError("Child lineage changed before detachment.")
                detached_at = datetime.now(UTC).isoformat()
                tombstones = tuple(
                    {
                        "root_session_id": root_id,
                        "plan_id": plan_id,
                        "child_session_id": child_id,
                        "original_parent_session_id": parent_session_id,
                        "detached_at": detached_at,
                    }
                    for child_id in sorted(child_session_ids)
                )
                self._connection.execute(
                    f"UPDATE cayu_sessions SET parent_session_id = NULL, updated_at = ? "
                    f"WHERE id IN ({placeholders})",
                    (detached_at, *child_session_ids),
                )
                self._connection.executemany(
                    "INSERT INTO cayu_session_closure_tombstones "
                    "(root_session_id, plan_id, child_session_id, original_parent_session_id, "
                    "detached_at, tombstone_json) VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            item["root_session_id"],
                            item["plan_id"],
                            item["child_session_id"],
                            item["original_parent_session_id"],
                            item["detached_at"],
                            json.dumps(item, ensure_ascii=False, separators=(",", ":")),
                        )
                        for item in tombstones
                    ],
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
            return tombstones

    def _require_session_erasure_quiescence_unlocked(self, session: Session) -> None:
        """Shared admission for closure and final deletion; no mutations."""
        if (
            self._connection.execute(
                "SELECT 1 FROM cayu_external_waits WHERE session_id=? "
                "AND session_instance_id=? AND pending_handoff=1 LIMIT 1",
                (session.id, session.instance_id),
            ).fetchone()
            is not None
        ):
            raise ValueError("Session has a pending external-wait handoff.")
        from cayu._validation import DURABLE_DOCUMENT_LIMITS
        from cayu.collaboration import _session_export_store as session_exports
        from cayu.runtime._session_closure_records import require_terminal_protected_effect
        from cayu.sessions import _session_continuation_store as continuations

        session_id = session.id
        export_records: dict[str, dict[str, Any]] = {}
        continuation_records: dict[str, dict[str, Any]] = {}
        producer_records = {}
        # Allow JSON escaping/whitespace overhead; the shared validator applies
        # the durable document limit before model reconstruction.
        rows = self._connection.execute(
            "SELECT idempotency_key, CASE WHEN length(CAST(record_json AS BLOB)) <= ? "
            "THEN record_json END FROM cayu_session_operations "
            "WHERE session_id = ? AND (idempotency_key GLOB 'tool-effect:*' "
            "OR idempotency_key GLOB 'session-export:*' "
            "OR idempotency_key GLOB 'producer-output:*' "
            "OR idempotency_key GLOB 'session-continuation:*')",
            (8 * DURABLE_DOCUMENT_LIMITS.max_bytes, session_id),
        )
        try:
            for key, raw in rows:
                try:
                    value = None if raw is None else json.loads(raw)
                except (ValueError, RecursionError):
                    raise ValueError(
                        "Session closure requires settled protected tool effects."
                    ) from None
                if key.startswith(continuations.CONTINUATION_OPERATION_PREFIX):
                    continuations.collect_retained_record(continuation_records, key, value)
                elif key.startswith("producer-output:"):
                    producer_records[key] = value
                elif key.startswith(session_exports.OPERATION_PREFIX):
                    if type(value) is not dict:
                        raise ValueError("Session export retention evidence is malformed.")
                    export_records[key] = value
                else:
                    require_terminal_protected_effect(session_id, session.instance_id, key, value)
        finally:
            rows.close()
        if session.status in DELETE_BLOCKED_SESSION_STATUSES:
            raise ValueError("Session closure requires a non-running target.")
        if (
            self._connection.execute(
                "SELECT 1 FROM cayu_persisted_event_side_effects "
                "WHERE session_id = ? AND status = 'leased' LIMIT 1",
                (session_id,),
            ).fetchone()
            is not None
        ):
            raise ValueError("Session closure requires settled event side-effect deliveries.")
        checkpoint = self._load_checkpoint_unlocked(session_id)
        deletion_now = self._ownership_clock()
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
        terminal_evidence_rows = self._connection.execute(
            f"SELECT {', '.join(sqlite_records.EVENT_COLUMN_NAMES)} FROM cayu_events "
            "WHERE session_id = ? "
            f"AND event_type IN ({', '.join('?' for _ in _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES)}) "
            "ORDER BY sequence DESC LIMIT ?",
            (
                session_id,
                *(str(event_type) for event_type in _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES),
                _TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT,
            ),
        ).fetchall()
        terminal_publication_block = _terminal_publication_delete_block_reason(
            session=session,
            checkpoint=checkpoint,
            evidence_events=[sqlite_records.event_from_row(row) for row in terminal_evidence_rows],
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
        active_stage = self._connection.execute(
            "SELECT 1 FROM cayu_session_operations WHERE session_id = ? AND idempotency_key = ?",
            (
                session_id,
                MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
            ),
        ).fetchone()
        if active_stage is not None:
            raise ValueError(
                f"Cannot delete a session while a model-completion stage is active: {session_id}"
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

    async def validate_session_closure_admission(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                session = self._load_unlocked(session_id)
                if session is None:
                    raise ValueError("Closure target is unavailable.")
                self._require_session_erasure_quiescence_unlocked(session)
            finally:
                self._connection.rollback()

    @runtime_session_mutation
    async def delete_session(
        self,
        session_id: str,
        *,
        closure_receipt: dict[str, Any] | None = None,
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> None:
        await self._participant_creation_lock.acquire()
        try:
            await self._delete_session_unserialized(
                session_id,
                closure_receipt=closure_receipt,
                _access_bounds=_access_bounds,
            )
        finally:
            self._participant_creation_lock.release()

    async def _delete_session_unserialized(
        self,
        session_id: str,
        *,
        closure_receipt: dict[str, Any] | None = None,
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                if not self._delete_session_in_transaction_unlocked(
                    session_id,
                    closure_receipt=closure_receipt,
                    _access_bounds=_access_bounds,
                ):
                    self._connection.rollback()
                    return
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    async def apply_retention_policy(
        self,
        policy: SessionRetentionPolicy,
        *,
        protected_session_ids: Collection[str] = (),
        references: Mapping[RetentionProtection, Collection[str]] | None = None,
        progress: RetentionProgressCallback | None = None,
    ) -> RetentionReport:
        from cayu.storage import _session_retention as retention

        return await retention.apply_retention_policy(
            retention.SQLiteSessionRetentionBackend(self),
            policy,
            protected_session_ids=protected_session_ids,
            references=references,
            progress=progress,
        )

    async def inspect_session_retention(
        self,
        session_ids: Collection[str],
        *,
        mode: RetentionMode | None = None,
        include_store_guards: bool = True,
    ) -> dict[str, tuple[RetentionProtection, ...]]:
        from cayu.storage import _session_retention as retention
        from cayu.storage.retention import RetentionMode

        return await retention.inspect_session_protections(
            retention.SQLiteSessionRetentionBackend(self),
            session_ids,
            mode=RetentionMode.DELETE if mode is None else RetentionMode(mode),
            store_guards=include_store_guards,
        )

    async def retention_artifact_references(self) -> frozenset[str]:
        from cayu.storage import _session_retention as retention

        return await retention.session_artifact_references(
            retention.SQLiteSessionRetentionBackend(self)
        )

    async def list_retention_audits(
        self,
        *,
        limit: int = 20,
        item_id: str | None = None,
        store_kind: str | None = None,
    ) -> tuple[RetentionAuditRecord, ...]:
        from cayu.storage import _session_retention as retention

        return await retention.list_retention_audits(
            retention.SQLiteSessionRetentionBackend(self),
            limit=limit,
            item_id=None if item_id is None else require_clean_nonblank(item_id, "item_id"),
            store_kind=store_kind,
        )

    async def load_retention_audit(self, audit_id: str) -> RetentionAuditRecord | None:
        from cayu.storage import _session_retention as retention

        return await retention.load_retention_audit(
            retention.SQLiteSessionRetentionBackend(self),
            require_clean_nonblank(audit_id, "audit_id"),
        )

    async def begin_retention_audit(
        self,
        *,
        store_kind: str,
        mode: RetentionMode,
        policy: Mapping[str, Any],
        started_at: datetime,
    ) -> str:
        from cayu.storage import _session_retention as retention

        return await retention.begin_retention_audit(
            retention.SQLiteSessionRetentionBackend(self),
            store_kind=require_clean_nonblank(store_kind, "store_kind"),
            mode=mode,
            policy=dict(policy),
            started_at=started_at,
        )

    async def record_retention_entry(self, audit_id: str, entry: RetentionAuditEntry) -> None:
        from cayu.storage import _session_retention as retention

        await retention.record_retention_entry(
            retention.SQLiteSessionRetentionBackend(self), audit_id, entry
        )

    async def complete_retention_audit(
        self, audit_id: str, *, completed_at: datetime, summary: Mapping[str, Any]
    ) -> None:
        from cayu.storage import _session_retention as retention

        await retention.complete_retention_audit(
            retention.SQLiteSessionRetentionBackend(self),
            audit_id,
            completed_at=completed_at,
            summary=dict(summary),
        )

    def _require_session_deletion_admission_unlocked(
        self,
        session: Session,
        *,
        closure_receipt: dict[str, Any] | None = None,
    ) -> None:
        """Check deletion guards that do not depend on remaining child edges."""

        session_id = session.id
        for owner in self._closure_lineage_owners_unlocked((session_id,)):
            _check_closure_lineage_owner(owner, (session_id,), closure_receipt)
        if (
            self._connection.execute(
                "SELECT 1 FROM cayu_session_closure_progress AS p "
                "JOIN cayu_sessions AS child ON child.id = p.root_session_id "
                "WHERE child.parent_session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
            is not None
        ):
            raise ValueError("Session lineage is owned by an unfinished recursive closure.")
        if closure_receipt is not None and closure_receipt.get("operation") == "recursive":
            expected_parent = closure_receipt.get("original_parent_session_id")
            if type(expected_parent) is not str or session.parent_session_id != expected_parent:
                raise ValueError("Recursive closure child parent identity conflict.")
        if session.status in DELETE_BLOCKED_SESSION_STATUSES:
            raise ValueError(
                f"Cannot delete a session while it is {session.status}; "
                f"interrupt it first: {session_id}"
            )
        if (
            self.context_view_version is not None
            and self._connection.execute(
                "SELECT 1 FROM cayu_context_view_selections s "
                "LEFT JOIN cayu_context_views v ON v.view_id = s.view_id "
                "WHERE (v.source_session_id = ? OR v.view_id IS NULL) "
                "AND s.state IN ('selected', 'adopted', 'transferred') "
                "AND (s.state <> 'selected' OR s.expires_at_ms > ?) LIMIT 1",
                (session_id, int(self._ownership_clock().timestamp() * 1000)),
            ).fetchone()
            is not None
        ):
            raise ValueError("Session has an active context-view retention pin.")

    def _delete_session_in_transaction_unlocked(
        self,
        session_id: str,
        *,
        closure_receipt: dict[str, Any] | None = None,
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> bool:
        """Apply delete_session inside the caller's write transaction.

        Returns ``False`` when the session does not exist. Guards raise
        ``ValueError``; the caller owns commit and rollback.
        """

        session = self._load_unlocked(session_id)
        if _access_bounds is not None:
            _access_bounds.require_action(session, "delete")
        if session is None:
            return False
        self._require_session_deletion_admission_unlocked(session, closure_receipt=closure_receipt)
        durable_child = self._connection.execute(
            "SELECT id FROM cayu_sessions "
            "WHERE parent_session_id = ? "
            "AND json_extract(metadata_json, '$.subagent.mode') = ? "
            "ORDER BY id LIMIT 1",
            (session_id, "durable"),
        ).fetchone()
        if (closure_receipt is not None or _access_bounds is not None) and self._connection.execute(
            "SELECT 1 FROM cayu_sessions WHERE parent_session_id = ? LIMIT 1",
            (session_id,),
        ).fetchone() is not None:
            raise ValueError("Closure deletion requires no remaining child edges.")
        if durable_child is not None:
            raise ValueError(_durable_subagent_parent_delete_block_reason(durable_child["id"]))
        self._require_session_erasure_quiescence_unlocked(session)
        self._connection.execute(
            "UPDATE cayu_peer_content_receipts SET target_deleted = 1 "
            "WHERE json_extract(receipt_json, '$.status') = 'appended' "
            "AND json_extract(receipt_json, '$.target_session_id') = ? "
            "AND json_extract(receipt_json, '$.target_session_instance_id') = ?",
            (session.id, session.instance_id),
        )
        # ON DELETE CASCADE removes events/labels/checkpoint/transcript;
        # the self-FK is ON DELETE SET NULL so children keep loading.
        self._connection.execute(
            "DELETE FROM cayu_sessions WHERE id = ?",
            (session_id,),
        )
        if closure_receipt is not None:
            receipt_plan_id = closure_receipt.get("plan_id")
            if type(receipt_plan_id) is not str:
                raise ValueError("Session closure receipt is missing plan identity.")
            self._connection.execute(
                "INSERT INTO cayu_session_closure_receipts "
                "(session_id, plan_id, committed_at, receipt_json) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id, plan_id) DO UPDATE SET receipt_json = excluded.receipt_json",
                (
                    session_id,
                    receipt_plan_id,
                    datetime.now(UTC).isoformat(),
                    json.dumps(closure_receipt, ensure_ascii=False, separators=(",", ":")),
                ),
            )
        return True

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
        updated_at = self._ownership_clock()
        expected_run_epoch = _current_session_run_epoch(session_id)
        async with self._lock:
            with self._connection:
                self._connection.execute("BEGIN IMMEDIATE")
                if _access_bounds is not None:
                    access_session = self._load_unlocked(session_id)
                    _access_bounds.require_label_update(access_session, new_labels)
                for owner in self._closure_lineage_owners_unlocked((session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
                epoch_clause = "" if expected_run_epoch is None else " AND run_epoch = ?"
                params: list[object] = [
                    sqlite_records.format_datetime(updated_at),
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
                if _access_bounds is not None:
                    audit = _access_bounds.label_audit(access_session, new_labels, updated_at)
                    if audit is not None:
                        key, record = audit
                        self._connection.execute(
                            "INSERT INTO cayu_session_operations (session_id, idempotency_key, record_json, updated_at) VALUES (?, ?, ?, ?)",
                            (
                                session_id,
                                key,
                                sqlite_records.json_dumps(record),
                                sqlite_records.format_datetime(updated_at),
                            ),
                        )
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                from cayu.sessions._invocation_lifecycle import (
                    require_invocation_lifecycle_release_capacity,
                )

                require_invocation_lifecycle_release_capacity(
                    self._load_checkpoint_unlocked(session_id),
                    loaded,
                )
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
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                if _access_bounds is not None:
                    _access_bounds.require_action(self._load_unlocked(session_id), "modify")
                updated_at = self._ownership_clock()
                row = self._connection.execute(
                    "SELECT run_epoch, metadata_json FROM cayu_sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch_value(session_id, row["run_epoch"])
                for owner in self._closure_lineage_owners_unlocked((session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
                new_metadata = replace_session_user_metadata(
                    json.loads(row["metadata_json"]),
                    user_metadata,
                )
                self._connection.execute(
                    "UPDATE cayu_sessions SET metadata_json = ?, updated_at = ? WHERE id = ?",
                    (
                        sqlite_records.json_dumps(new_metadata),
                        sqlite_records.format_datetime(updated_at),
                        session_id,
                    ),
                )
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                from cayu.sessions._invocation_lifecycle import (
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
                for owner in self._closure_lineage_owners_unlocked((session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
                updated_at = self._ownership_clock()
                expected_run_epoch = _current_session_run_epoch(session_id)
                admission_source = (
                    self._load_unlocked(session_id) if to_status is SessionStatus.RUNNING else None
                )
                placeholders = ", ".join("?" for _ in allowed_statuses)
                params: list[object] = [
                    str(to_status),
                    sqlite_records.format_datetime(updated_at),
                    sqlite_records.format_datetime(updated_at),
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
                    if admission_source is None:
                        raise KeyError(f"Session not found: {session_id}")
                    self._require_external_wait_admission_unlocked(admission_source)
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

        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                updated_at = self._ownership_clock()
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, loaded)
                for owner in self._closure_lineage_owners_unlocked((session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
                if loaded.status not in allowed_statuses:
                    raise SessionStatusConflict(
                        f"Session status transition not allowed: {loaded.status} -> {to_status}"
                    )
                if to_status is SessionStatus.RUNNING:
                    self._require_external_wait_admission_unlocked(loaded)
                if expected_latest_interaction_event_id is not None:
                    latest_interaction_row = self._connection.execute(
                        "SELECT event.latest_event_sequence, retained.event_id "
                        "FROM cayu_interaction_latest_events AS event "
                        "JOIN cayu_events AS retained "
                        "ON retained.sequence = event.latest_event_sequence "
                        "WHERE event.session_id = ? "
                        "ORDER BY event.latest_event_sequence DESC LIMIT 1",
                        (session_id,),
                    ).fetchone()
                    if (
                        latest_interaction_row is None
                        or latest_interaction_row["event_id"]
                        != expected_latest_interaction_event_id
                    ):
                        raise SessionRunFenced(
                            "Session latest interaction changed before the status transition."
                        )
                if require_no_active_model_completion_dispatch:
                    active_row = self._connection.execute(
                        "SELECT record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key = ?",
                        (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                    ).fetchone()
                    if active_row is not None:
                        active_marker = _reconstruct_active_model_completion_stage_record(
                            _decode_model_completion_stage_record(active_row["record_json"]),
                            session_id=session_id,
                        )
                        dispatch_row = self._connection.execute(
                            "SELECT 1 FROM cayu_session_operations "
                            "WHERE session_id = ? AND idempotency_key = ?",
                            (
                                session_id,
                                _model_completion_stage_dispatch_storage_key(
                                    active_marker.stage_id
                                ),
                            ),
                        ).fetchone()
                        if dispatch_row is not None:
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
                    binding_row = self._connection.execute(
                        "SELECT * FROM cayu_participant_session_bindings WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()
                    if binding_row is not None:
                        from cayu.sessions._participant_execution_identity import (
                            require_initial_execution_input,
                        )
                        from cayu.storage._participant_session_records import reconstruct

                        transcript_rows = self._connection.execute(
                            "SELECT message_json FROM cayu_transcript_messages WHERE session_id = ? ORDER BY session_order",
                            (session_id,),
                        ).fetchall()
                        require_initial_execution_input(
                            loaded,
                            reconstruct(dict(binding_row), loaded),
                            [Message.model_validate_json(row[0]) for row in transcript_rows],
                            admission[2],
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
                        transcript_ops.transcript_cursor(self._connection, session_id),
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
                    updated_at = self._ownership_clock()
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
                    sqlite_records.format_datetime(updated_at),
                    sqlite_records.format_datetime(updated_at),
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
                            sqlite_records.json_dumps(transition_metadata),
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
                            sqlite_records.json_dumps(transition_metadata),
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
                            sqlite_records.json_dumps(transition_metadata),
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
                failover_keys = _model_failover_admission_storage_keys(
                    current_checkpoint, prepared_execution_profile
                )
                if failover_keys:
                    route_placeholders = ", ".join("?" for _ in failover_keys)
                    route_rows = self._connection.execute(
                        "SELECT idempotency_key, record_json FROM cayu_session_operations "
                        f"WHERE session_id = ? AND idempotency_key IN ({route_placeholders})",
                        (session_id, *failover_keys),
                    ).fetchall()
                    transformed_checkpoint = _model_failover_checkpoint_after_profile_admission(
                        source_session=loaded,
                        admitted_session=transitioned,
                        source_checkpoint=current_checkpoint,
                        admitted_checkpoint=transformed_checkpoint,
                        candidate_profile=prepared_execution_profile,
                        records={
                            row["idempotency_key"]: _decode_model_completion_stage_record(
                                row["record_json"]
                            )
                            for row in route_rows
                        },
                        transcript_cursor=transcript_ops.transcript_cursor(
                            self._connection, session_id
                        ),
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
                        raise ValueError("Result checkpoint transform must return a checkpoint.")
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
                    service_rows = self._connection.execute(
                        "SELECT idempotency_key, record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key IN (?, ?)",
                        (session_id, parent_key, child_key),
                    ).fetchall()
                    service_records = {
                        row["idempotency_key"]: json.loads(row["record_json"])
                        for row in service_rows
                    }
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
                    self._connection.executemany(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record_json, updated_at) VALUES (?, ?, ?, ?) "
                        "ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                        "record_json = excluded.record_json, updated_at = excluded.updated_at",
                        [
                            (
                                session_id,
                                key,
                                sqlite_records.json_dumps(record),
                                sqlite_records.format_datetime(updated_at),
                            )
                            for key, record in publication.operation_records.items()
                        ],
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
                        sqlite_records.checkpoint_row_values(
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
                                sqlite_records.format_datetime(admission_event.timestamp),
                                admission_event.agent_name,
                                admission_event.environment_name,
                                admission_event.workflow_name,
                                admission_event.tool_name,
                                sqlite_records.json_dumps(admission_event.payload),
                                lookup_key,
                                projection,
                                projection_bytes,
                            ),
                        )
                    if admission_events:
                        event_delivery_ops.enqueue_persisted_event_side_effects(
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
                                sqlite_records.json_dumps(
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
                                    sqlite_records.json_dumps(message.model_dump(mode="json")),
                                    transcript_search_document(message),
                                )
                                for message in source_messages
                            ],
                        )
                self._connection.commit()
            except sqlite3.IntegrityError as exc:
                transaction_failure = sqlite_connection._settle_failed_transaction(
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
                transaction_failure = sqlite_connection._settle_failed_transaction(
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
                    existing = sqlite_records.event_from_row(existing_row)
                    if not _execution_profile_rejection_events_equivalent(
                        existing,
                        copied_event,
                    ):
                        raise ValueError(
                            f"Execution-profile rejection id was reused: {copied_event.id}"
                        )
                    self._connection.commit()
                    return ExecutionProfileRejectionResult(event=existing, replayed=True)

                for owner in self._closure_lineage_owners_unlocked((session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
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
                        sqlite_records.format_datetime(copied_event.timestamp),
                        copied_event.agent_name,
                        copied_event.environment_name,
                        copied_event.workflow_name,
                        copied_event.tool_name,
                        sqlite_records.json_dumps(copied_event.payload),
                        lookup_key,
                        projection,
                        projection_bytes,
                    ),
                )
                event_delivery_ops.enqueue_persisted_event_side_effects(
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
                lease_identity = self._execution_identity(session_id)
                if self._has_live_execution_owner_unlocked(lease_identity, now):
                    self._connection.commit()
                    return None
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
                        sqlite_records.format_datetime(now),
                        session_id,
                        *(str(status) for status in allowed_statuses),
                        sqlite_records.format_datetime(inactive_before),
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
                if inactive_for_seconds is not None and self._has_live_execution_owner_unlocked(
                    loaded, now
                ):
                    self._connection.commit()
                    return None
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
                for owner in self._closure_lineage_owners_unlocked((session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
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
                    sqlite_records.checkpoint_row_values(session_id, transformed, now),
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
                for owner in self._closure_lineage_owners_unlocked((session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
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
                    (sqlite_records.format_datetime(updated_at), session_id),
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
                    sqlite_records.checkpoint_row_values(
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
        checkpoint_mutation: dict[str, Any] | None = None,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        mutation = _prepare_queue_completion_checkpoint_mutation(checkpoint_mutation, to_status)
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
                for owner in self._closure_lineage_owners_unlocked((session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
                pending = self._connection.execute(
                    "SELECT 1 FROM cayu_session_message_queue "
                    "WHERE session_id = ? AND status = 'queued' LIMIT 1",
                    (session_id,),
                ).fetchone()
                if pending is not None:
                    raise SessionQueuedMessagesPending(
                        f"Session has durable queued messages: {session_id}"
                    )
                if mutation is not None:
                    checkpoint = _apply_queue_completion_checkpoint_mutation(
                        loaded, mutation, _load_checkpoint_state(self._connection, session_id)
                    )
                    if checkpoint is None:
                        raise ValueError("Queue completion mutation cannot delete its checkpoint.")
                    self._connection.execute(
                        "INSERT INTO cayu_checkpoints (session_id, state_json, updated_at, "
                        "pending_action_source_bytes, pending_action_tool_call_count, "
                        "pending_action_flags, pending_action_metrics_ready) VALUES (?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(session_id) DO UPDATE SET state_json = excluded.state_json, "
                        "updated_at = excluded.updated_at, "
                        "pending_action_source_bytes = excluded.pending_action_source_bytes, "
                        "pending_action_tool_call_count = excluded.pending_action_tool_call_count, "
                        "pending_action_flags = excluded.pending_action_flags, "
                        "pending_action_metrics_ready = excluded.pending_action_metrics_ready",
                        sqlite_records.checkpoint_row_values(session_id, checkpoint, updated_at),
                    )
                cursor = self._connection.execute(
                    "UPDATE cayu_sessions SET status = ?, updated_at = ?, "
                    "last_activity_at = ?, run_epoch = run_epoch + ? WHERE id = ?",
                    (
                        str(to_status),
                        sqlite_records.format_datetime(updated_at),
                        sqlite_records.format_datetime(updated_at),
                        1 if to_status == SessionStatus.RUNNING and mutation is None else 0,
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

        def statement(connection: sqlite3.Connection) -> InteractionTransitionResult:
            try:
                connection.execute("BEGIN IMMEDIATE")
                loaded = sqlite_records.load_session(connection, session_id)
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
                    f"SELECT {', '.join(sqlite_records.EVENT_COLUMN_NAMES)} FROM cayu_events "
                    "WHERE session_id = ? AND event_id = ?",
                    (session_id, copied_event.id),
                ).fetchone()
                existing_terminal_row = (
                    None
                    if copied_terminal_event is None
                    else connection.execute(
                        f"SELECT {', '.join(sqlite_records.EVENT_COLUMN_NAMES)} FROM cayu_events "
                        "WHERE session_id = ? AND event_id = ?",
                        (session_id, copied_terminal_event.id),
                    ).fetchone()
                )
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
                    if (
                        existing_row is not None
                        and sqlite_records.event_from_row(existing_row) != receipt.event
                    ):
                        raise RuntimeError(
                            "Interaction transition receipt conflicts with retained event history."
                        )
                    if copied_terminal_event is not None and (
                        receipt.terminal_event != copied_terminal_event
                        or (
                            existing_terminal_row is not None
                            and sqlite_records.event_from_row(existing_terminal_row)
                            != receipt.terminal_event
                        )
                    ):
                        raise RuntimeError(
                            "Interaction transition receipt conflicts with its terminal session event."
                        )
                    connection.commit()
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
                for owner in self._closure_lineage_owners_unlocked(
                    (session_id,), connection=connection
                ):
                    _check_closure_lineage_owner(owner, (session_id,))
                current_checkpoint = _load_checkpoint_state(connection, session_id)
                if terminalization_only:
                    from cayu.runtime._durable_model_terminalization import (
                        require_terminalization_checkpoint,
                        require_terminalization_plan_owner,
                    )

                    require_terminalization_checkpoint(loaded, current_checkpoint)

                    require_terminalization_plan_owner(
                        current_checkpoint, terminalization_plan_ownership, self._ownership_clock()
                    )
                    if any(
                        connection.execute(
                            f"SELECT 1 FROM {table} WHERE {column} = ? "
                            + (
                                "AND status = 'queued' "
                                if table == "cayu_session_message_queue"
                                else ""
                            )
                            + "LIMIT 1",
                            (session_id,),
                        ).fetchone()
                        is not None
                        for table, column in (
                            ("cayu_sessions", "parent_session_id"),
                            ("cayu_deferred_interaction_inputs", "session_id"),
                            ("cayu_session_message_queue", "session_id"),
                        )
                    ):
                        raise SessionRunFenced("Model terminalization has dependent work.")
                settled_checkpoint = _checkpoint_after_exact_invocation_terminal_decision(
                    current_checkpoint,
                    session=loaded,
                    expected=terminal_decision,
                )
                active_recovery_claim_id = _active_unexpired_incomplete_recovery_claim_id(
                    current_checkpoint,
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
                    from cayu.sessions._invocation_lifecycle import (
                        require_invocation_command_authority,
                        require_released_invocation_command_authority,
                    )

                    assert expected_session_instance_id is not None
                    checkpoint = current_checkpoint
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
                            run_epochs=frozenset({expected_active_invocation_profile.run_epoch}),
                            active_profile=expected_active_invocation_profile,
                            events=tuple(
                                event
                                for event in (copied_event, copied_terminal_event)
                                if event is not None
                            ),
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
                from cayu.runtime._session_steering import (
                    interaction_completion_steering_key,
                    prepare_interaction_completion_steering_record,
                )

                steering_key = interaction_completion_steering_key(
                    loaded, current_checkpoint, copied_event
                )
                completion_record = None
                if steering_key is not None:
                    steering_row = connection.execute(
                        "SELECT record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key = ?",
                        (session_id, steering_key),
                    ).fetchone()
                    completion_record = prepare_interaction_completion_steering_record(
                        loaded,
                        current_checkpoint,
                        copied_event,
                        None
                        if steering_row is None
                        else _decode_model_completion_stage_record(steering_row["record_json"]),
                        keeps_running=queued or target_status is SessionStatus.RUNNING,
                    )
                updated_at = self._ownership_clock()
                formatted_updated_at = sqlite_records.format_datetime(updated_at)
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
                            sqlite_records.checkpoint_row_values(
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
                            sqlite_records.json_dumps(settlement_record),
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
                committed_events = [
                    event for event in (copied_event, copied_terminal_event) if event is not None
                ]
                for committed_event in committed_events:
                    lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                        committed_event
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
                            committed_event.id,
                            committed_event.interaction_id,
                            str(committed_event.type),
                            sqlite_records.format_datetime(committed_event.timestamp),
                            committed_event.agent_name,
                            committed_event.environment_name,
                            committed_event.workflow_name,
                            committed_event.tool_name,
                            sqlite_records.json_dumps(committed_event.payload),
                            lookup_key,
                            projection,
                            projection_bytes,
                        ),
                    )
                _record_invocation_terminal_event_receipts(
                    connection, session_id, committed_events, activity_at=updated_at
                )
                event_delivery_ops.enqueue_persisted_event_side_effects(
                    connection,
                    session_id,
                    committed_events,
                )
                if terminal_decision is not None:
                    assert settled_checkpoint is not None
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
                        sqlite_records.checkpoint_row_values(
                            session_id,
                            settled_checkpoint,
                            updated_at,
                        ),
                    )
                transitioned = sqlite_records.load_session(connection, session_id)
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
                connection.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record_json, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        session_id,
                        receipt_storage_key,
                        sqlite_records.json_dumps(receipt_record),
                        formatted_updated_at,
                    ),
                )
                if completion_record is not None:
                    assert steering_key is not None
                    connection.execute(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record_json, updated_at) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            session_id,
                            steering_key,
                            sqlite_records.json_dumps(completion_record),
                            formatted_updated_at,
                        ),
                    )
                connection.commit()
                return InteractionTransitionResult(
                    session=transitioned,
                    event=copied_event,
                    terminal_event=copied_terminal_event,
                    status_changed=not queued,
                )
            except BaseException as primary:
                transaction_failure = sqlite_connection._settle_failed_transaction(
                    connection,
                    primary,
                )
                if transaction_failure is not primary:
                    raise transaction_failure from None
                raise

        return await self._run_write(statement)

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

        def statement(
            connection: sqlite3.Connection,
        ) -> InteractionTransitionReceiptResult | None:
            selected_event_columns = ", ".join(
                f"retained.{column} AS {column}" for column in sqlite_records.EVENT_COLUMN_NAMES
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
            if retained_event_exists and sqlite_records.event_from_row(row) != receipt.event:
                raise RuntimeError(
                    "Interaction transition receipt conflicts with retained event history."
                )
            return InteractionTransitionReceiptResult(
                session=receipt.session,
                transition=_interaction_transition_spec_from_receipt(receipt),
                status_changed=receipt.status_changed,
            )

        return await self._run_read(statement)

    async def _load_historical_interaction_settlement_record(
        self, session_id: str, event_id: str
    ) -> dict[str, Any] | None:
        key = _interaction_transition_storage_key(event_id)

        def statement(connection: sqlite3.Connection) -> dict[str, Any] | None:
            row = connection.execute(
                "SELECT record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key = ?",
                (session_id, key),
            ).fetchone()
            return (
                None
                if row is None
                else copy_durable_json_object(json.loads(row["record_json"]), "settlement")
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
                f"retained.{column} AS {column}" for column in sqlite_records.EVENT_COLUMN_NAMES
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
            current_session = sqlite_records.load_session(connection, session_id)
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
            if retained_event_exists and sqlite_records.event_from_row(row) != receipt.event:
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

        def statement(connection: sqlite3.Connection) -> Any:
            with sqlite_connection._transaction(connection):
                session = sqlite_records.load_session(connection, copied.session_id)
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
                            else sqlite_records.event_from_row(terminal_event_row)
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
                    sqlite_records.checkpoint_row_values(
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
                existing = connection.execute(
                    "SELECT 1 FROM cayu_budget_reservation_identities WHERE reservation_id = ?",
                    (reservation_id,),
                ).fetchone()
                if existing is None:
                    for owner in self._closure_lineage_owners_unlocked(
                        (publication_session_id,), connection=connection
                    ):
                        _check_closure_lineage_owner(owner, (publication_session_id,))
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
                if not sqlite_records.session_exists(connection, session_id):
                    raise KeyError(f"Session not found: {session_id}")
                return

            try:
                connection.execute("BEGIN IMMEDIATE")
                if not sqlite_records.session_exists(connection, session_id):
                    raise KeyError(f"Session not found: {session_id}")
                activity_at = self._ownership_clock()
                for owner in self._closure_lineage_owners_unlocked((session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
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

    async def append_tool_effect_conflict(self, request: object) -> Event:
        from cayu.runtime._tool_effect_conflicts import (
            copy_tool_effect_conflict_audit,
            reconcile_tool_effect_conflict_event,
        )

        audit = copy_tool_effect_conflict_audit(request)
        session_id = audit.executing.intent.session_id

        def statement(connection: sqlite3.Connection) -> Event:
            try:
                connection.execute("BEGIN IMMEDIATE")
                session = self._load_unlocked(session_id)
                if session is None:
                    raise KeyError("Tool effect audit session is unavailable.")
                row = connection.execute(
                    "SELECT record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, audit.storage_key),
                ).fetchone()
                current = None if row is None else json.loads(row["record_json"])
                event = audit.prepare_event(session, current, now=self._ownership_clock())
                existing = connection.execute(
                    "SELECT * FROM cayu_events WHERE session_id = ? AND event_id = ?",
                    (session_id, event.id),
                ).fetchone()
                if existing is not None:
                    event = reconcile_tool_effect_conflict_event(
                        event, sqlite_records.event_from_row(existing)
                    )
                else:
                    for owner in self._closure_lineage_owners_unlocked(
                        (session_id,), connection=connection
                    ):
                        _check_closure_lineage_owner(owner, (session_id,))
                    # Evidence authority was established above; do not touch the
                    # current run's liveness or weaken the ordinary append fence.
                    _insert_event_rows_in_transaction(
                        connection, session_id, [event], activity_at=event.timestamp
                    )
                connection.commit()
                return event
            except BaseException:
                connection.rollback()
                raise

        return await self._run_write(statement)

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

        def statement(connection: sqlite3.Connection) -> bool:
            try:
                connection.execute("BEGIN IMMEDIATE")
                if not sqlite_records.session_exists(connection, session_id):
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

                for owner in self._closure_lineage_owners_unlocked(
                    (session_id,), connection=connection
                ):
                    _check_closure_lineage_owner(owner, (session_id,))
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
                        sqlite_records.format_datetime(copied_event.timestamp),
                        copied_event.agent_name,
                        copied_event.environment_name,
                        copied_event.workflow_name,
                        copied_event.tool_name,
                        sqlite_records.json_dumps(copied_event.payload),
                        lookup_key,
                        projection,
                        projection_bytes,
                    ),
                )
                event_delivery_ops.enqueue_persisted_event_side_effects(
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
        from cayu.sessions.pending_actions import pending_action_event_storage_values

        session_id, expected, updates, copied_events = _copy_mcp_manifest_publication(
            session_id,
            expected_generations=expected_generations,
            baseline_updates=baseline_updates,
            events=events,
        )

        def statement(connection: sqlite3.Connection) -> McpManifestPublicationResult:
            try:
                connection.execute("BEGIN IMMEDIATE")
                session = sqlite_records.load_session(connection, session_id)
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
                for owner in self._closure_lineage_owners_unlocked(
                    (session_id,), connection=connection
                ):
                    _check_closure_lineage_owner(owner, (session_id,))
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
                            sqlite_records.format_datetime(event.timestamp),
                            event.agent_name,
                            event.environment_name,
                            event.workflow_name,
                            event.tool_name,
                            sqlite_records.json_dumps(event.payload),
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
                event_delivery_ops.enqueue_persisted_event_side_effects(
                    connection,
                    session_id,
                    copied_events,
                )
                updated_at = sqlite_records.format_datetime(self._ownership_clock())
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
                            sqlite_records.json_dumps(baseline.model_dump(mode="json")),
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

    async def claim_first_persisted_event_side_effect(
        self, expected: PersistedEventSideEffectDelivery
    ) -> PersistedEventSideEffectClaim | None:
        return await event_delivery_ops.claim_first_persisted_event_side_effect(
            self._run_write, expected, ownership_clock=self._ownership_clock
        )

    async def claim_persisted_event_side_effect(
        self,
        *,
        session_id: str | None = None,
        event_id: str | None = None,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectClaim | None:
        return await event_delivery_ops.claim_persisted_event_side_effect(
            self._run_write,
            session_id=session_id,
            event_id=event_id,
            lease_seconds=lease_seconds,
            ownership_clock=self._ownership_clock,
        )

    async def get_persisted_event_side_effect_delivery(
        self, *, session_id: str, event_id: str
    ) -> PersistedEventSideEffectDelivery | None:
        return await event_delivery_ops.get_persisted_event_side_effect_delivery(
            self._run_read, session_id=session_id, event_id=event_id
        )

    async def retire_failed_first_event_delivery(
        self, expected: PersistedEventSideEffectDelivery
    ) -> PersistedEventSideEffectDelivery | None:
        return await event_delivery_ops.retire_failed_first_event_delivery(
            self._run_write, expected, ownership_clock=self._ownership_clock
        )

    async def mark_persisted_event_side_effect_delivered(
        self, claim: PersistedEventSideEffectClaim
    ) -> PersistedEventSideEffectDelivery:
        return await event_delivery_ops.mark_persisted_event_side_effect_delivered(
            self._run_write, claim, ownership_clock=self._ownership_clock
        )

    async def mark_persisted_event_side_effect_failed(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        error: str,
        max_attempts: int,
        retry_delay_seconds: float,
    ) -> PersistedEventSideEffectDelivery:
        return await event_delivery_ops.mark_persisted_event_side_effect_failed(
            self._run_write,
            claim,
            error=error,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
            ownership_clock=self._ownership_clock,
        )

    async def defer_persisted_event_side_effect(
        self, claim: PersistedEventSideEffectClaim
    ) -> PersistedEventSideEffectDelivery:
        return await event_delivery_ops.defer_persisted_event_side_effect(
            self._run_write, claim, ownership_clock=self._ownership_clock
        )

    async def renew_persisted_event_side_effect(
        self, claim: PersistedEventSideEffectClaim, *, lease_seconds: float = 300.0
    ) -> PersistedEventSideEffectDelivery:
        return await event_delivery_ops.renew_persisted_event_side_effect(
            self._run_write,
            claim,
            lease_seconds=lease_seconds,
            ownership_clock=self._ownership_clock,
        )

    async def get_persisted_event_side_effect_health(self) -> PersistedEventSideEffectHealth:
        return await event_delivery_ops.get_persisted_event_side_effect_health(
            self._run_read, ownership_clock=self._ownership_clock
        )

    async def query_persisted_event_side_effect_deliveries(
        self, query: PersistedEventSideEffectQuery
    ) -> PersistedEventSideEffectPage:
        return await event_delivery_ops.query_persisted_event_side_effect_deliveries(
            self._run_read, query, ownership_clock=self._ownership_clock
        )

    async def list_persisted_event_side_effect_deliveries(
        self,
        *,
        statuses: set[PersistedEventSideEffectStatus] | None = None,
        claimable_only: bool = False,
        after_sequence: int | None = None,
        limit: int = 100,
    ) -> list[PersistedEventSideEffectDelivery]:
        return await event_delivery_ops.list_persisted_event_side_effect_deliveries(
            self._run_read,
            statuses=statuses,
            claimable_only=claimable_only,
            after_sequence=after_sequence,
            limit=limit,
            ownership_clock=self._ownership_clock,
        )

    def _session_message_source_unlocked(
        self,
        connection: sqlite3.Connection,
        session: Session,
        *,
        include_transcript_digest: bool,
        include_checkpoint_digest: bool,
    ) -> SessionMessageSource:
        cursor = transcript_ops.transcript_cursor(connection, session.id)
        transcript_digest = None
        if include_transcript_digest:
            hasher = message_queue.SourceTranscriptHasher(cursor)
            for row in connection.execute(
                "SELECT session_order, message_json FROM cayu_transcript_messages "
                "WHERE session_id = ? ORDER BY session_order",
                (session.id,),
            ):
                hasher.add(row[0] - 1, Message.model_validate_json(row[1]))
            transcript_digest = hasher.hexdigest()
        checkpoint = None
        if include_checkpoint_digest:
            checkpoint = _load_checkpoint_state(connection, session.id)
        return message_queue.source_snapshot(
            session,
            cursor,
            transcript_sha256=transcript_digest,
            checkpoint=checkpoint,
            include_checkpoint_digest=include_checkpoint_digest,
        )

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

        def query(connection: sqlite3.Connection) -> SessionMessageSource:
            connection.execute("BEGIN")
            try:
                session = sqlite_records.load_session(connection, session_id)
                require_resource_session(session, "read")
                if session is None:
                    raise KeyError("Session not found.")
                message_queue.require_authorized_session_instance(
                    session, expected_authorized_session_instance_id
                )
                result = self._session_message_source_unlocked(
                    connection,
                    session,
                    include_transcript_digest=include_transcript_digest,
                    include_checkpoint_digest=include_checkpoint_digest,
                )
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise

        return await self._run_read(query)

    @runtime_session_query
    async def inspect_session_messages(
        self,
        query: SessionMessageQuery,
        *,
        expected_authorized_session_instance_id: str | None = None,
    ) -> SessionMessageInspection:
        query = message_queue.copy_inspection_query(query)

        def read(connection: sqlite3.Connection) -> SessionMessageInspection:
            connection.execute("BEGIN")
            try:
                session = sqlite_records.load_session(connection, query.session_id)
                require_resource_session(session, "read")
                if session is None:
                    raise KeyError("Session not found.")
                message_queue.require_authorized_session_instance(
                    session, expected_authorized_session_instance_id
                )
                maximum = 0
                if query.cursor is None:
                    maximum = connection.execute(
                        "SELECT COALESCE(MAX(ordering_key), 0) FROM cayu_session_message_queue "
                        "WHERE session_id = ?",
                        (session.id,),
                    ).fetchone()[0]
                boundary = message_queue.inspection_boundary(session, query.cursor, maximum)
                rows = connection.execute(
                    "WITH ordered AS (SELECT queue_id, ordering_key, "
                    "CASE delivery_mode WHEN 'next_turn' THEN 0 WHEN 'on_idle' THEN 1 ELSE 2 END "
                    "AS priority FROM cayu_session_message_queue "
                    "WHERE session_id = ? AND ordering_key <= ?) "
                    "SELECT queue_id, ordering_key, priority FROM ordered "
                    "WHERE (priority, ordering_key) > (?, ?) "
                    "ORDER BY priority, ordering_key LIMIT ?",
                    (
                        session.id,
                        boundary.through_ordering_key,
                        boundary.after_priority,
                        boundary.after_ordering_key,
                        query.limit + 1,
                    ),
                ).fetchall()
                raw_rows = [
                    _session_message_raw_bounded(connection, query.session_id, row["queue_id"])
                    for row in rows[: query.limit]
                ]
                records = tuple(
                    message_queue.inspect_record(
                        raw, lambda raw=raw: _queued_session_message_from_row(raw)
                    )
                    for raw in raw_rows
                )
                result = SessionMessageInspection(
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    records=records,
                    next_cursor=(
                        message_queue.inspection_next_cursor(
                            boundary,
                            rows[query.limit - 1]["priority"],
                            records[-1].ordering_key,
                        )
                        if len(rows) > query.limit
                        else None
                    ),
                )
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise

        return await self._run_read(read)

    @runtime_session_query
    async def apply_session_message_action(
        self,
        request: SessionMessageActionRequest,
    ) -> SessionMessageActionResult:
        request = SessionMessageActionRequest(**message_queue.action_material(request))

        def statement(connection: sqlite3.Connection) -> SessionMessageActionResult:
            connection.execute("BEGIN IMMEDIATE")
            try:
                session = self._load_unlocked(request.session_id)
                require_resource_session(session, "modify")
                if session is None or session.instance_id != request.session_instance_id:
                    raise SessionMessageConflict()
                raw = _session_message_raw_bounded(connection, session.id, request.queue_id)
                accepted_events = _session_message_acceptance_events(
                    connection,
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
                replay = message_queue.replay_action(raw, request, accepted_event=accepted_event)
                record = message_queue.inspect_record(
                    raw, lambda: _queued_session_message_from_row(raw)
                )
                if replay is not None:
                    connection.commit()
                    return SessionMessageActionResult(record=record, event=replay, replayed=True)
                for owner in self._closure_lineage_owners_unlocked(
                    (session.id,), connection=connection
                ):
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
                # Independent receipt evidence survives event retention and malformed content.
                delivered = connection.execute(
                    "SELECT 1 FROM cayu_session_message_deliveries, json_each(queue_ids_json) "
                    "WHERE session_id = ? AND json_each.value = ? LIMIT 1",
                    (session.id, request.queue_id),
                ).fetchone()
                if delivered is not None or (
                    request.action == "withdraw" and record.validity != "valid"
                ):
                    raise SessionMessageConflict()
                status = SessionMessageQueueStatus(
                    "withdrawn" if request.action == "withdraw" else "quarantined"
                )
                now = self._ownership_clock()
                event = message_queue.terminal_event(
                    session,
                    raw,
                    status,
                    now,
                    actor=request.requested_by,
                    accepted_event=accepted_event,
                )
                proof = sqlite_records.json_dumps(
                    message_queue.terminal_receipt(status, event, request)
                )
                connection.execute(
                    "UPDATE cayu_session_message_queue SET status = ?, terminal_json = ? "
                    "WHERE session_id = ? AND queue_id = ? AND status = 'queued'",
                    (str(status), proof, session.id, request.queue_id),
                )
                _append_events_in_transaction(connection, session.id, [event], activity_at=now)
                updated = _session_message_raw_bounded(connection, session.id, request.queue_id)
                result = SessionMessageActionResult(
                    record=message_queue.inspect_record(
                        updated, lambda: _queued_session_message_from_row(updated)
                    ),
                    event=event,
                )
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise

        return await self._run_write(statement)

    @runtime_session_query
    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
        *,
        expected_authorized_target_instance_id: str | None = None,
    ) -> EnqueueSessionMessageResult:
        request = copy_enqueue_session_message_request(request)

        def statement(connection: sqlite3.Connection) -> EnqueueSessionMessageResult:
            from cayu.sessions.pending_actions import pending_action_event_storage_values

            try:
                connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(request.session_id)
                require_resource_session(loaded, "modify")
                if loaded is None:
                    raise KeyError(f"Session not found: {request.session_id}")
                if expected_authorized_target_instance_id is not None and (
                    type(expected_authorized_target_instance_id) is not str
                    or loaded.instance_id != expected_authorized_target_instance_id
                ):
                    raise SessionMessageConflict()
                if request.conditions.source is not None:
                    source = self._load_unlocked(request.conditions.source.session_id)
                    require_resource_session(source, "read")
                    if (
                        source is None
                        or source.instance_id != request.conditions.source.session_instance_id
                    ):
                        raise SessionMessageConflict()
                existing_row = connection.execute(
                    "SELECT * FROM cayu_session_message_queue "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (request.session_id, request.idempotency_key),
                ).fetchone()
                if existing_row is not None:
                    existing = _queued_session_message_from_row(existing_row)
                    _validate_equivalent_queued_session_message(existing, request)
                    event_row = connection.execute(
                        f"SELECT {', '.join(sqlite_records.EVENT_COLUMN_NAMES)} FROM cayu_events "
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
                        event=sqlite_records.event_from_row(event_row),
                        replayed=True,
                    )
                for owner in self._closure_lineage_owners_unlocked((request.session_id,)):
                    _check_closure_lineage_owner(owner, (request.session_id,))
                checkpoint = _load_checkpoint_state(connection, request.session_id)
                message_queue.require_open_admission(loaded.status, checkpoint)
                if request.conditions.source is not None:
                    expected_source = request.conditions.source
                    source = self._load_unlocked(expected_source.session_id)
                    require_resource_session(source, "read")
                    if (
                        source is None
                        or self._session_message_source_unlocked(
                            connection,
                            source,
                            include_transcript_digest=expected_source.transcript_sha256 is not None,
                            include_checkpoint_digest=expected_source.checkpoint_sha256 is not None,
                        )
                        != expected_source
                    ):
                        raise SessionMessageConflict()
                transcript_cursor = transcript_ops.transcript_cursor(connection, request.session_id)
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
                            else sqlite_records.json_dumps(request.message.model_dump(mode="json"))
                        ),
                        str(request.delivery_mode),
                        (
                            None
                            if request.requested_by is None
                            else sqlite_records.json_dumps(
                                resolution_actor_payload(request.requested_by)
                            )
                        ),
                        loaded.run_epoch,
                        transcript_cursor,
                        accepted_event_id,
                        sqlite_records.format_datetime(accepted_at),
                    ),
                )
                ordering_key = cursor.lastrowid
                if type(ordering_key) is not int:
                    raise RuntimeError("SQLite queue insert did not return an ordering key.")
                connection.execute(
                    "UPDATE cayu_session_message_queue SET conditions_json = ? WHERE queue_id = ?",
                    (
                        sqlite_records.json_dumps(request.conditions.model_dump(mode="json")),
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
                        sqlite_records.format_datetime(accepted_event.timestamp),
                        accepted_event.agent_name,
                        accepted_event.environment_name,
                        accepted_event.workflow_name,
                        accepted_event.tool_name,
                        sqlite_records.json_dumps(accepted_event.payload),
                        lookup_key,
                        projection,
                        projection_bytes,
                    ),
                )
                event_delivery_ops.enqueue_persisted_event_side_effects(
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
        reject_only: bool = False,
        delivery_id: str | None = None,
        eligible_through: int | None = None,
        limit: int = SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
        interaction_id: str | None = None,
        interaction_started_event: Event | None = None,
        profile_handoff: QueuedInteractionProfileHandoff | None = None,
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

        def statement(connection: sqlite3.Connection) -> SessionMessageDeliveryBatch:
            from cayu.sessions.pending_actions import pending_action_event_storage_values

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
                        or bool(delivery_row["reject_only"]) != reject_only
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
                    if replayed_messages and profile_handoff is not None:
                        receipt_row = connection.execute(
                            "SELECT record_json FROM cayu_session_operations "
                            "WHERE session_id = ? AND idempotency_key = ?",
                            (
                                session_id,
                                _interaction_transition_storage_key(
                                    profile_handoff.predecessor_settlement_event_id
                                ),
                            ),
                        ).fetchone()
                        if receipt_row is None:
                            raise SessionRunFenced(
                                "Queued interaction handoff lost its predecessor "
                                "settlement receipt."
                            )
                        active_model_stage = None
                        stage_dispatch = None
                        active_row = connection.execute(
                            "SELECT record_json FROM cayu_session_operations "
                            "WHERE session_id = ? AND idempotency_key = ?",
                            (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                        ).fetchone()
                        if active_row is not None:
                            active_record = _decode_model_completion_stage_record(
                                active_row["record_json"]
                            )
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
                            stage_rows = connection.execute(
                                "SELECT idempotency_key, record_json "
                                "FROM cayu_session_operations WHERE session_id = ? "
                                "AND idempotency_key IN (?, ?, ?)",
                                (
                                    session_id,
                                    preparation_key,
                                    terminal_key,
                                    dispatch_key,
                                ),
                            ).fetchall()
                            stage_records = {
                                row["idempotency_key"]: (
                                    _decode_model_completion_stage_record(row["record_json"])
                                )
                                for row in stage_rows
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
                        repaired_checkpoint = _checkpoint_after_queued_interaction_profile_handoff(
                            loaded,
                            _load_checkpoint_state(connection, session_id),
                            profile_handoff,
                            settlement_record=copy_durable_json_object(
                                json.loads(receipt_row["record_json"]),
                                "interaction transition receipt",
                            ),
                            replayed_delivery=True,
                            active_model_stage=active_model_stage,
                            stage_dispatch=stage_dispatch,
                        )
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
                                pending_action_source_bytes = excluded.pending_action_source_bytes,
                                pending_action_tool_call_count =
                                    excluded.pending_action_tool_call_count,
                                pending_action_flags = excluded.pending_action_flags,
                                pending_action_metrics_ready =
                                    excluded.pending_action_metrics_ready
                            """,
                            sqlite_records.checkpoint_row_values(
                                session_id,
                                repaired_checkpoint,
                                self._ownership_clock(),
                            ),
                        )
                    connection.commit()
                    return SessionMessageDeliveryBatch(
                        messages=tuple(replayed_messages),
                        events=tuple(replayed_events),
                        delivery_id=delivery_id,
                        interaction_id=interaction_id,
                        eligible_through=delivery_row["eligible_through"],
                        has_more=bool(delivery_row["has_more"]),
                        replayed=True,
                        active_invocation_profile=(
                            None
                            if not replayed_messages or profile_handoff is None
                            else profile_handoff.target_active_profile
                        ),
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
                reject_only_more = False
                if reject_only:
                    # Eligible rows remain pending. Scan in bounded pages so they cannot hide
                    # an expired record behind the first delivery-sized prefix.
                    rows = []
                    scan_now = self._ownership_clock()
                    scan_cursor = transcript_ops.transcript_cursor(connection, session_id)
                    for mode in ("next_turn", "on_idle") if include_on_idle else ("next_turn",):
                        after = 0
                        while len(rows) < limit + 1:
                            page = connection.execute(
                                "SELECT * FROM cayu_session_message_queue "
                                "WHERE session_id = ? AND status = 'queued' AND delivery_mode = ? "
                                "AND ordering_key > ? AND ordering_key <= ? "
                                "ORDER BY ordering_key LIMIT 100",
                                (session_id, mode, after, boundary),
                            ).fetchall()
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
                            after = page[-1]["ordering_key"]
                        if len(rows) == limit + 1:
                            break
                    rows = rows[:limit]
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
                                else sqlite_records.json_dumps(
                                    interaction_started_event.model_dump(mode="json")
                                )
                            ),
                            sqlite_records.format_datetime(self._ownership_clock()),
                        ),
                    )
                    connection.execute(
                        "UPDATE cayu_session_message_deliveries SET reject_only = ? WHERE delivery_id = ?",
                        (reject_only, delivery_id),
                    )
                    connection.commit()
                    return SessionMessageDeliveryBatch(
                        delivery_id=delivery_id,
                        interaction_id=interaction_id,
                        eligible_through=boundary,
                        has_more=False,
                    )
                transcript_cursor = transcript_ops.transcript_cursor(connection, session_id)
                delivered_at = scan_now if reject_only else self._ownership_clock()
                accepted_events = _session_message_acceptance_events(connection, session_id, rows)
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
                        dict(row),
                        rejection,
                        delivered_at,
                        accepted_event=accepted_events.get(row["accepted_event_id"]),
                        actor=queued.requested_by,
                        interaction_id=interaction_id,
                    )
                    rejection_events.append(event)
                    connection.execute(
                        "UPDATE cayu_session_message_queue SET status = ?, terminal_json = ? "
                        "WHERE queue_id = ? AND status = 'queued'",
                        (
                            str(rejection),
                            sqlite_records.json_dumps(
                                message_queue.terminal_receipt(rejection, event)
                            ),
                            queued.queue_id,
                        ),
                    )
                rows = deliverable_rows
                rebound_checkpoint: dict[str, Any] | None = None
                if rows and profile_handoff is not None:
                    _reject_new_work_after_steering(
                        connection, loaded, allow_completed_interaction=True
                    )
                    receipt_row = connection.execute(
                        "SELECT record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key = ?",
                        (
                            session_id,
                            _interaction_transition_storage_key(
                                profile_handoff.predecessor_settlement_event_id
                            ),
                        ),
                    ).fetchone()
                    if receipt_row is None:
                        raise SessionRunFenced(
                            "Queued interaction handoff lost its predecessor settlement receipt."
                        )
                    rebound_checkpoint = _checkpoint_after_queued_interaction_profile_handoff(
                        loaded,
                        _load_checkpoint_state(connection, session_id),
                        profile_handoff,
                        settlement_record=copy_durable_json_object(
                            json.loads(receipt_row["record_json"]),
                            "interaction transition receipt",
                        ),
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
                                    dict(row), accepted_events.get(row["accepted_event_id"])
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
                            sqlite_records.json_dumps(message.model_dump(mode="json")),
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
                            sqlite_records.format_datetime(delivered_at),
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
                            sqlite_records.format_datetime(event.timestamp),
                            event.agent_name,
                            event.environment_name,
                            event.workflow_name,
                            event.tool_name,
                            sqlite_records.json_dumps(event.payload),
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
                event_delivery_ops.enqueue_persisted_event_side_effects(
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
                has_more = reject_only_more if reject_only else remaining is not None
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
                        has_more,
                        (
                            None
                            if interaction_started_event is None
                            else sqlite_records.json_dumps(
                                interaction_started_event.model_dump(mode="json")
                            )
                        ),
                        sqlite_records.json_dumps(
                            [message.queue_id for message in updated_messages]
                        ),
                        sqlite_records.json_dumps(
                            [event.model_dump(mode="json") for event in persisted_events]
                        ),
                        sqlite_records.format_datetime(delivered_at),
                    ),
                )
                connection.execute(
                    "UPDATE cayu_session_message_deliveries SET reject_only = ? WHERE delivery_id = ?",
                    (reject_only, delivery_id),
                )
                if rebound_checkpoint is not None:
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
                            pending_action_source_bytes = excluded.pending_action_source_bytes,
                            pending_action_tool_call_count =
                                excluded.pending_action_tool_call_count,
                            pending_action_flags = excluded.pending_action_flags,
                            pending_action_metrics_ready = excluded.pending_action_metrics_ready
                        """,
                        sqlite_records.checkpoint_row_values(
                            session_id,
                            rebound_checkpoint,
                            delivered_at,
                        ),
                    )
                connection.commit()
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
                connection.rollback()
                raise

        return await self._run_write(statement)

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

        def statement(connection: sqlite3.Connection) -> ActiveInvocationExecutionProfile:
            try:
                connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, loaded)
                delivery_row = connection.execute(
                    "SELECT session_id, interaction_id, interaction_started_event_json, "
                    "queue_ids_json FROM cayu_session_message_deliveries "
                    "WHERE delivery_id = ?",
                    (target.interaction_id,),
                ).fetchone()
                stored_started_event = (
                    None
                    if delivery_row is None
                    or delivery_row["interaction_started_event_json"] is None
                    else Event.model_validate_json(delivery_row["interaction_started_event_json"])
                )
                if (
                    delivery_row is None
                    or delivery_row["session_id"] != session_id
                    or delivery_row["interaction_id"] != target.interaction_id
                    or stored_started_event != interaction_started_event
                    or not json.loads(delivery_row["queue_ids_json"])
                ):
                    raise SessionRunFenced(
                        "Historical queued interaction handoff lacks its exact delivery receipt."
                    )
                receipt_row = connection.execute(
                    "SELECT record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (
                        session_id,
                        _interaction_transition_storage_key(
                            profile_handoff.predecessor_settlement_event_id
                        ),
                    ),
                ).fetchone()
                if receipt_row is None:
                    raise SessionRunFenced(
                        "Historical queued interaction handoff lost its predecessor settlement."
                    )
                active_row = connection.execute(
                    "SELECT record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                ).fetchone()
                stage_records: dict[str, Any] = {}
                if active_row is not None:
                    active_record = _decode_model_completion_stage_record(active_row["record_json"])
                    marker = _reconstruct_active_model_completion_stage_record(
                        active_record,
                        session_id=session_id,
                    )
                    _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
                        session_id, marker.stage_id
                    )
                    dispatch_key = _model_completion_stage_dispatch_storage_key(marker.stage_id)
                    rows = connection.execute(
                        "SELECT idempotency_key, record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key IN (?, ?, ?)",
                        (session_id, preparation_key, terminal_key, dispatch_key),
                    ).fetchall()
                    stage_records = {
                        row["idempotency_key"]: _decode_model_completion_stage_record(
                            row["record_json"]
                        )
                        for row in rows
                    }
                    stage_records[MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY] = active_record
                active_model_stage, stage_dispatch = _historical_queued_handoff_stage_from_records(
                    session_id,
                    stage_records,
                )
                repaired_checkpoint = _checkpoint_after_queued_interaction_profile_handoff(
                    loaded,
                    _load_checkpoint_state(connection, session_id),
                    profile_handoff,
                    settlement_record=copy_durable_json_object(
                        json.loads(receipt_row["record_json"]),
                        "interaction transition receipt",
                    ),
                    replayed_delivery=True,
                    active_model_stage=active_model_stage,
                    stage_dispatch=stage_dispatch,
                )
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
                        pending_action_source_bytes = excluded.pending_action_source_bytes,
                        pending_action_tool_call_count = excluded.pending_action_tool_call_count,
                        pending_action_flags = excluded.pending_action_flags,
                        pending_action_metrics_ready = excluded.pending_action_metrics_ready
                    """,
                    sqlite_records.checkpoint_row_values(
                        session_id,
                        repaired_checkpoint,
                        self._ownership_clock(),
                    ),
                )
                connection.commit()
                return target.model_copy(deep=True)
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
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        idempotency_key = _reject_reserved_runtime_publication_key(
            idempotency_key,
            "idempotency_key",
            browser_control_read=True,
        )
        _require_session_export_target(session_id, idempotency_key)

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

        from cayu.storage._session_access_records import sqlite_owner_read

        return await self._run_read(
            lambda connection: sqlite_owner_read(connection, access_bounds, session_id, query)
        )

    async def _load_runtime_publication_receipt_record(
        self,
        session_id: str,
        storage_key: str,
        publication_id: str,
    ) -> dict[str, Any] | None:
        def query(connection: sqlite3.Connection) -> dict[str, Any] | None:
            connection.execute("BEGIN")
            try:
                if not sqlite_records.session_exists(connection, session_id):
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
                    f"SELECT {', '.join(sqlite_records.EVENT_COLUMN_NAMES)} FROM cayu_events "
                    f"WHERE session_id = ? AND event_id IN ({placeholders})",
                    (receipt.session_id, *requested_event_ids),
                ).fetchall()
                events_by_id = {row["event_id"]: sqlite_records.event_from_row(row) for row in rows}
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
            if not sqlite_records.session_exists(connection, session_id):
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
            if not sqlite_records.session_exists(connection, session_id):
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
            if not sqlite_records.session_exists(connection, session_id):
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
                if not sqlite_records.session_exists(connection, session_id):
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
                    checkpoint=self._load_checkpoint_unlocked(session_id),
                    current_transcript_cursor=transcript_ops.transcript_cursor(
                        connection, session_id
                    ),
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
                            sqlite_records.json_dumps(dispatch_record),
                            sqlite_records.format_datetime(published_at),
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
                        child = sqlite_records.load_session(connection, claim.child_session_id)
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
                        event_record = sqlite_records.event_record_from_row(event_row)
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
                                    sqlite_records.json_dumps(material),
                                    sqlite_records.format_datetime(published_at),
                                ),
                            )
                formatted_at = sqlite_records.format_datetime(published_at)
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
                failover_source_keys = _model_failover_predecessor_storage_keys(prepared)
                if failover_source_keys:
                    source_placeholders = ", ".join("?" for _ in failover_source_keys)
                    source_rows = connection.execute(
                        "SELECT idempotency_key, record_json FROM cayu_session_operations "
                        f"WHERE session_id = ? AND idempotency_key IN ({source_placeholders})",
                        (session_id, *failover_source_keys),
                    ).fetchall()
                    records.update(
                        {
                            row["idempotency_key"]: _decode_model_completion_stage_record(
                                row["record_json"]
                            )
                            for row in source_rows
                        }
                    )
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
                    _model_failover_preparation_checkpoint(
                        prepared,
                        session=loaded,
                        checkpoint=self._load_checkpoint_unlocked(session_id),
                        current_transcript_cursor=transcript_ops.transcript_cursor(
                            connection, session_id
                        ),
                        active=active,
                        records=records,
                        replayed=True,
                    )
                    expected_selection = _model_failover_selection_event(
                        prepared, session=loaded, prepared_at=stage.prepared_at
                    )
                    if expected_selection is not None:
                        event_row = connection.execute(
                            "SELECT * FROM cayu_events WHERE session_id = ? AND event_id = ?",
                            (session_id, expected_selection.id),
                        ).fetchone()
                        _validate_model_failover_selection_replay(
                            expected_selection,
                            None if event_row is None else sqlite_records.event_from_row(event_row),
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
                current_cursor = transcript_ops.transcript_cursor(connection, session_id)
                if current_cursor != prepared.expected_transcript_cursor:
                    raise ValueError(
                        "Session source transcript cursor is stale: expected "
                        f"{prepared.expected_transcript_cursor}, current {current_cursor}."
                    )

                if active is None:
                    _reject_new_work_after_steering(connection, loaded)
                route_checkpoint = _model_failover_preparation_checkpoint(
                    prepared,
                    session=loaded,
                    checkpoint=self._load_checkpoint_unlocked(session_id),
                    current_transcript_cursor=current_cursor,
                    active=active,
                    records=records,
                    replayed=False,
                )
                prepared_at = _next_runtime_publication_timestamp(loaded)
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
                formatted_at = sqlite_records.format_datetime(prepared_at)
                if route_checkpoint is not None:
                    connection.execute(
                        "INSERT INTO cayu_checkpoints (session_id, state_json, updated_at, "
                        "pending_action_source_bytes, pending_action_tool_call_count, "
                        "pending_action_flags, pending_action_metrics_ready) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(session_id) DO UPDATE SET "
                        "state_json = excluded.state_json, updated_at = excluded.updated_at, "
                        "pending_action_source_bytes = excluded.pending_action_source_bytes, "
                        "pending_action_tool_call_count = excluded.pending_action_tool_call_count, "
                        "pending_action_flags = excluded.pending_action_flags, "
                        "pending_action_metrics_ready = excluded.pending_action_metrics_ready",
                        sqlite_records.checkpoint_row_values(
                            session_id, route_checkpoint, prepared_at
                        ),
                    )
                connection.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record_json, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        session_id,
                        prepared.preparation_storage_key,
                        sqlite_records.json_dumps(record),
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
                            sqlite_records.json_dumps(retry_settlement_record),
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
                        sqlite_records.json_dumps(active_record),
                        formatted_at,
                    ),
                )
                cursor = connection.execute(
                    "UPDATE cayu_sessions SET updated_at = ?, last_activity_at = ? WHERE id = ?",
                    (formatted_at, formatted_at, session_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"Session not found: {session_id}")
                if selection_event is not None:
                    _append_events_in_transaction(
                        connection, session_id, (selection_event,), activity_at=prepared_at
                    )
                connection.commit()
                return ModelCompletionStageResult(
                    stage=stage,
                    replayed=False,
                    dispatch_authorized=True,
                    prepared_events=() if selection_event is None else (selection_event,),
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
                if prepared.recovery_fence is not None:
                    active_row = connection.execute(
                        "SELECT record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key = ?",
                        (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                    ).fetchone()
                    dispatch_row = connection.execute(
                        "SELECT record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key = ?",
                        (session_id, _model_completion_stage_dispatch_storage_key(stage.stage_id)),
                    ).fetchone()
                    _validate_model_completion_stage_recovery_fence(
                        prepared.recovery_fence,
                        session=loaded,
                        checkpoint=self._load_checkpoint_unlocked(session_id),
                        stage=stage,
                        active_record=(
                            None
                            if active_row is None
                            else _decode_model_completion_stage_record(active_row["record_json"])
                        ),
                        dispatch_record=(
                            None
                            if dispatch_row is None
                            else _decode_model_completion_stage_record(dispatch_row["record_json"])
                        ),
                        now=self._ownership_clock(),
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
                formatted_at = sqlite_records.format_datetime(completed_at)
                connection.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record_json, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        session_id,
                        prepared.terminal_storage_key,
                        sqlite_records.json_dumps(terminal_record),
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
                formatted_at = sqlite_records.format_datetime(abandoned_at)
                connection.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record_json, updated_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                    "record_json = excluded.record_json, updated_at = excluded.updated_at",
                    (
                        session_id,
                        prepared.abandonment_storage_key,
                        sqlite_records.json_dumps(abandonment_record),
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
        from cayu.sessions.pending_actions import (
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

                for owner in self._closure_lineage_owners_unlocked((session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))

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

                if request.argument_continuity is not None:
                    from cayu.sessions._argument_continuity import STORAGE_KEY, append_record

                    private_row = connection.execute(
                        "SELECT record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key = ?",
                        (session_id, STORAGE_KEY),
                    ).fetchone()
                    operation_mutation_records[STORAGE_KEY] = append_record(
                        None if private_row is None else json.loads(private_row["record_json"]),
                        continuity=request.argument_continuity,
                        request_digest=prepared.request_digest,
                        session=loaded,
                        messages=request.transcript_messages,
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
                transcript_start_cursor = transcript_ops.transcript_cursor(connection, session_id)
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
                        f"SELECT {', '.join(sqlite_records.EVENT_COLUMN_NAMES)} FROM cayu_events "
                        f"WHERE session_id = ? AND event_id IN ({placeholders})",
                        (session_id, *referenced_event_ids),
                    ).fetchall()
                    durable_referenced_events = {
                        row["event_id"]: sqlite_records.event_from_row(row) for row in rows
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
                    event_type_placeholders = ", ".join("?" for _ in lifecycle_event_types)
                    rows = connection.execute(
                        f"SELECT {', '.join(sqlite_records.EVENT_COLUMN_NAMES)} FROM cayu_events "
                        "INDEXED BY idx_cayu_events_pending_action_lookup "
                        f"WHERE session_id = ? AND pending_action_lookup_key IN "
                        "(SELECT value FROM json_each(?)) AND event_type IN "
                        f"({event_type_placeholders}) AND "
                        f"({session_queries.PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
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
                            json.dumps(lookup_keys),
                            *lifecycle_event_types,
                            execution_identity.tool_round_id,
                            execution_identity.model_step_id,
                            execution_identity.model_attempt_id,
                            _tool_round_lifecycle_event_limit(tool_call_ids) + 1,
                        ),
                    ).fetchall()
                    if len(rows) > _tool_round_lifecycle_event_limit(tool_call_ids):
                        raise ValueError(
                            "Tool-round lifecycle evidence exceeds the publication limit."
                        )
                    durable_tool_events = [sqlite_records.event_from_row(row) for row in rows]
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
                        sqlite_records.json_dumps(message_payload),
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
                            sqlite_records.format_datetime(event.timestamp),
                            event.agent_name,
                            event.environment_name,
                            event.workflow_name,
                            event.tool_name,
                            sqlite_records.json_dumps(event_payload["payload"]),
                            lookup_key,
                            projection,
                            projection_bytes,
                        )
                    )

                published_at = _next_runtime_publication_timestamp(loaded)
                checkpoint_values = (
                    None
                    if stored_target_checkpoint is None or not request.mutation.operations
                    else sqlite_records.checkpoint_row_values(
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
                receipt_json = sqlite_records.json_dumps(
                    _runtime_publication_receipt_record(receipt)
                )
                formatted_published_at = sqlite_records.format_datetime(published_at)

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
                    event_delivery_ops.enqueue_persisted_event_side_effects(
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
                                sqlite_records.json_dumps(record),
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
                            sqlite_records.json_dumps(winner),
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

    async def _prepare_temporary_side_service(self, preparation):
        from cayu.sessions._side_service_preparation import (
            SidePreparationSnapshot,
            plan_preparation,
            preparation_record_keys,
            prepare_selection,
        )

        prepared = prepare_selection(preparation)
        keys = preparation_record_keys(prepared)

        def statement(connection):
            try:
                connection.execute("BEGIN IMMEDIATE")
                snapshots = {}
                for session_id in sorted(keys):
                    session = self._load_unlocked(session_id)
                    if session is None:
                        raise KeyError("Side-service session is unavailable.")
                    _assert_session_run_epoch(session_id, session)
                    for owner in self._closure_lineage_owners_unlocked((session_id,)):
                        _check_closure_lineage_owner(owner, (session_id,))
                    records = {}
                    for key in keys[session_id]:
                        row = connection.execute(
                            "SELECT record_json FROM cayu_session_operations WHERE session_id = ? AND idempotency_key = ?",
                            (session_id, key),
                        ).fetchone()
                        if row is not None:
                            records[key] = json.loads(row["record_json"])
                    snapshots[session_id] = SidePreparationSnapshot(
                        session, self._load_checkpoint_unlocked(session_id), records
                    )
                now = self._ownership_clock()
                plans = plan_preparation(prepared, snapshots, now)
                # Serialize and validate every row before starting either write.
                checkpoints = {
                    session_id: sqlite_records.checkpoint_row_values(
                        session_id, plan.checkpoint, now
                    )
                    for session_id, plan in plans.items()
                }
                operations = [
                    (
                        session_id,
                        key,
                        sqlite_records.json_dumps(record),
                        sqlite_records.format_datetime(now),
                    )
                    for session_id, plan in plans.items()
                    for key, record in plan.operation_records.items()
                ]
                for session_id, values in checkpoints.items():
                    connection.execute(
                        "INSERT INTO cayu_checkpoints (session_id, state_json, updated_at, pending_action_source_bytes, "
                        "pending_action_tool_call_count, pending_action_flags, pending_action_metrics_ready) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET "
                        "state_json=excluded.state_json, updated_at=excluded.updated_at, "
                        "pending_action_source_bytes=excluded.pending_action_source_bytes, "
                        "pending_action_tool_call_count=excluded.pending_action_tool_call_count, "
                        "pending_action_flags=excluded.pending_action_flags, pending_action_metrics_ready=excluded.pending_action_metrics_ready",
                        values,
                    )
                    _touch_session_activity(connection, session_id, now)
                connection.executemany(
                    "INSERT INTO cayu_session_operations (session_id, idempotency_key, record_json, updated_at) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                    "record_json=excluded.record_json, updated_at=excluded.updated_at",
                    operations,
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

        await self._run_write(statement)

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

        def statement(connection: sqlite3.Connection) -> Session:
            try:
                connection.execute("BEGIN IMMEDIATE")
                updated_at = self._ownership_clock()
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, loaded)
                for owner in self._closure_lineage_owners_unlocked((session_id,)):
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
                current_cursor = transcript_ops.transcript_cursor(connection, session_id)
                if (
                    expected_transcript_cursor is not None
                    and current_cursor != expected_transcript_cursor
                ):
                    raise ValueError(
                        "Session source transcript cursor is stale: expected "
                        f"{expected_transcript_cursor}, current {current_cursor}."
                    )
                if context_view_compaction_cursor is not None:
                    self._validate_context_view_compaction_unlocked(
                        session_id, context_view_compaction_cursor
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
                            placeholders = ",".join("?" for _ in indices)
                            rows = connection.execute(
                                "SELECT session_order - 1 AS transcript_index, "
                                "interaction_id, message_json FROM cayu_transcript_messages "
                                f"WHERE session_id = ? AND session_order IN ({placeholders}) "
                                "ORDER BY session_order",
                                (session_id, *(index + 1 for index in indices)),
                            ).fetchall()
                            selected_rows = tuple(
                                TranscriptRecord(
                                    index=row["transcript_index"],
                                    interaction_id=row["interaction_id"],
                                    message=Message(**json.loads(row["message_json"])),
                                )
                                for row in rows
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
                            sqlite_records.format_datetime(event.timestamp),
                            event.agent_name,
                            event.environment_name,
                            event.workflow_name,
                            event.tool_name,
                            sqlite_records.json_dumps(event.payload),
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
                    sqlite_records.checkpoint_row_values(session_id, transformed, updated_at),
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
                                sqlite_records.json_dumps(record),
                                sqlite_records.format_datetime(updated_at),
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
                    _record_invocation_terminal_event_receipts(
                        connection, session_id, copied_events, activity_at=updated_at
                    )
                    event_delivery_ops.enqueue_persisted_event_side_effects(
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

    async def load_session_closure_records(
        self, session_id: str, *, max_records: int, max_bytes: int
    ) -> dict[str, Any]:
        def query(connection: sqlite3.Connection) -> dict[str, Any]:
            with connection:
                connection.execute("BEGIN")
                return self._load_session_closure_records_unlocked(
                    connection, session_id, max_records=max_records, max_bytes=max_bytes
                )

        return await self._run_read(query)

    def _load_session_closure_records_unlocked(
        self, connection: sqlite3.Connection, session_id: str, *, max_records: int, max_bytes: int
    ) -> dict[str, Any]:
        from cayu.runtime._session_closure_records import (
            ClosureRecordsBuilder,
            ClosureRecordsTooLarge,
        )
        from cayu.storage._session_closure_sql import closure_size_statement

        session_id = require_clean_nonblank(session_id, "session_id")
        builder = ClosureRecordsBuilder(max_records=max_records, max_bytes=max_bytes)

        statement, source_count = closure_size_statement(postgres=False)
        count, size = connection.execute(
            statement, (session_id, max_records + 1) * source_count
        ).fetchone()
        if count > max_records or size > max_bytes:
            raise ClosureRecordsTooLarge()
        session = sqlite_records.load_session(connection, session_id)
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
        builder.add_class(
            "recall_receipts",
            (
                _sqlite_recall_receipt(row)
                for row in connection.execute(
                    "SELECT * FROM cayu_recall_receipts WHERE session_id = ? "
                    "ORDER BY created_at, receipt_id LIMIT ?",
                    (session_id, max_records + 1),
                )
            ),
        )
        builder.add_class(
            "context_exposures",
            (
                _sqlite_context_exposure(row)
                for row in connection.execute(
                    "SELECT * FROM cayu_context_exposures WHERE session_id = ? "
                    "ORDER BY created_at, exposure_id LIMIT ?",
                    (session_id, max_records + 1),
                )
            ),
        )
        builder.add_class(
            "recall_item_exposures",
            (
                json.loads(row[0])
                for row in connection.execute(
                    "SELECT item.item_json FROM cayu_recall_item_exposures AS item "
                    "JOIN cayu_context_exposures AS exposure "
                    "ON exposure.exposure_id = item.exposure_id "
                    "WHERE exposure.session_id = ? "
                    "ORDER BY exposure.created_at, exposure.exposure_id, item.ordinal LIMIT ?",
                    (session_id, max_records + 1),
                )
            ),
        )
        builder.add_class(
            "events",
            (
                EventRecord(sequence=row["sequence"], event=sqlite_records.event_from_row(row))
                for row in connection.execute(
                    "SELECT * FROM cayu_events WHERE session_id = ? ORDER BY sequence LIMIT ?",
                    (session_id, max_records + 1),
                )
            ),
        )
        builder.add_class(
            "transcript",
            (
                {
                    "transcript_index": row["session_order"] - 1,
                    "interaction_id": row["interaction_id"],
                    "message": json.loads(row["message_json"]),
                }
                for row in connection.execute(
                    "SELECT session_order, interaction_id, message_json FROM cayu_transcript_messages WHERE session_id = ? ORDER BY session_order LIMIT ?",
                    (session_id, max_records + 1),
                )
            ),
        )
        checkpoint = _load_checkpoint_state(connection, session_id)
        builder.add_class("checkpoint", () if checkpoint is None else (checkpoint,))
        builder.add_class(
            "queued_messages",
            (
                {
                    "message": _queued_session_message_from_row(row),
                    "terminal": None
                    if row["terminal_json"] is None
                    else json.loads(row["terminal_json"]),
                }
                for row in connection.execute(
                    "SELECT * FROM cayu_session_message_queue WHERE session_id = ? ORDER BY ordering_key LIMIT ?",
                    (session_id, max_records + 1),
                )
            ),
        )
        builder.add_class(
            "session_operations",
            (
                {
                    "idempotency_key": row["idempotency_key"],
                    "record": json.loads(row["record_json"]),
                }
                for row in connection.execute(
                    "SELECT idempotency_key, record_json FROM cayu_session_operations WHERE session_id = ? ORDER BY idempotency_key LIMIT ?",
                    (session_id, max_records + 1),
                )
            ),
        )
        builder.add_class(
            "event_side_effect_deliveries",
            (
                event_delivery_ops.delivery_from_row(row)
                for row in connection.execute(
                    "SELECT * FROM cayu_persisted_event_side_effects WHERE session_id = ? ORDER BY event_sequence LIMIT ?",
                    (session_id, max_records + 1),
                )
            ),
        )

        def delivery_record(row):
            record = {
                key.removesuffix("_json"): json.loads(row[key])
                if key.endswith("_json") and row[key] is not None
                else row[key]
                for key in dict(row)
                if key != "created_at"
            }
            for key in ("include_on_idle", "has_more", "reject_only"):
                if type(record[key]) is not int or record[key] not in (0, 1):
                    raise ValueError("Invalid stored queue delivery boolean.")
                record[key] = bool(record[key])
            return record

        builder.add_class(
            "queue_deliveries",
            (
                delivery_record(row)
                for row in connection.execute(
                    "SELECT * FROM cayu_session_message_deliveries WHERE session_id = ? ORDER BY created_at, delivery_id LIMIT ?",
                    (session_id, max_records + 1),
                )
            ),
        )
        builder.add_class(
            "deferred_interaction_inputs",
            (
                deferred_interaction_input_from_storage_payload(
                    row["interaction_id"], json.loads(row["source_messages_json"])
                )
                for row in connection.execute(
                    "SELECT interaction_id, source_messages_json FROM cayu_deferred_interaction_inputs WHERE session_id = ? LIMIT ?",
                    (session_id, max_records + 1),
                )
            ),
        )

        def project_grant(row):
            codec = self.public_authority_alias_codec
            if codec is None:
                raise RuntimeError("Closure grant export requires an authority alias codec.")
            return targeted_tool_grant_with_active_reference(
                _targeted_tool_grant_from_row(row), codec
            )

        builder.add_class(
            "targeted_tool_grants",
            (
                project_grant(row)
                for row in connection.execute(
                    "SELECT * FROM cayu_targeted_tool_grants WHERE session_id = ? ORDER BY issued_at, grant_id LIMIT ?",
                    (session_id, max_records + 1),
                )
            ),
        )
        builder.add_class(
            "targeted_tool_grant_uses",
            (
                _targeted_tool_use_from_row(row)
                for row in connection.execute(
                    "SELECT * FROM cayu_targeted_tool_grant_uses WHERE session_id = ? ORDER BY bound_at, use_id LIMIT ?",
                    (session_id, max_records + 1),
                )
            ),
        )
        return builder.finish()

    async def _complete_native_producer_cleanup(self, registration, *, authority):
        from cayu.storage._producer_cleanup import sqlite_cleanup

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer cleanup is not qualified.")
        return await sqlite_cleanup(self, registration, authority=authority, commit=True)

    async def _retire_native_producer_cleanup(self, retirement, *, authority, limit):
        from cayu.storage._producer_retirement import sqlite_retirement

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer retirement is not qualified.")
        return await sqlite_retirement(self, retirement, authority=authority, limit=limit)

    async def _read_completed_native_producer_cleanup(self, registration):
        from cayu.storage._producer_cleanup import sqlite_cleanup

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer cleanup readback is not qualified.")
        return await sqlite_cleanup(self, registration)

    async def _read_native_producer_release(self, command):
        from cayu.storage._producer_observation import sqlite_observation

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer release readback is not qualified.")
        return await sqlite_observation(self, command)

    async def _read_native_producer_progress(self, command, *, kind):
        from cayu.storage._producer_observation import sqlite_observation

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer progress readback is not qualified.")
        return await sqlite_observation(self, command, kind=kind)

    async def _read_native_producer_attachment(self, command):
        from cayu.storage._producer_observation import sqlite_observation

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer attachment readback is not qualified.")
        return await sqlite_observation(self, command, attachment_only=True)

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

        def query(connection: sqlite3.Connection) -> SessionExportSnapshot | None:
            with connection:
                connection.execute("BEGIN")
                statement, parameter_count = export_size_statement(postgres=False)
                sizes = connection.execute(statement, (session_id,) * parameter_count).fetchone()
                builder.preflight_bytes(int(sizes[0]), int(sizes[1]))
                session = sqlite_records.load_session(connection, session_id)
                if session is None:
                    return None
                cursor = transcript_ops.transcript_cursor(connection, session_id)
                checkpoint = _load_checkpoint_state(connection, session_id)
                rows = connection.execute(
                    "SELECT * FROM cayu_events WHERE session_id = ? ORDER BY sequence",
                    (session_id,),
                )
                while page := rows.fetchmany(SESSION_EXPORT_PAGE_SIZE):
                    for row in page:
                        builder.event(
                            EventRecord(
                                sequence=row["sequence"], event=sqlite_records.event_from_row(row)
                            )
                        )
                rows = connection.execute(
                    "SELECT session_order, interaction_id, message_json FROM cayu_transcript_messages "
                    "WHERE session_id = ? ORDER BY session_order",
                    (session_id,),
                )
                while page := rows.fetchmany(SESSION_EXPORT_PAGE_SIZE):
                    for row in page:
                        builder.message(
                            TranscriptRecord(
                                index=row["session_order"] - 1,
                                interaction_id=row["interaction_id"],
                                message=Message(**json.loads(row["message_json"])),
                            )
                        )
                row = connection.execute(
                    "SELECT interaction_id, source_messages_json FROM cayu_deferred_interaction_inputs "
                    "WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                deferred = (
                    None
                    if row is None
                    else deferred_interaction_input_from_storage_payload(
                        row["interaction_id"],
                        json.loads(row["source_messages_json"]),
                    )
                )
                codec = self.public_authority_alias_codec
                grants = []
                uses = []
                rows = connection.execute(
                    "SELECT * FROM cayu_targeted_tool_grants WHERE session_id = ? ORDER BY issued_at, grant_id",
                    (session_id,),
                )
                while page := rows.fetchmany(SESSION_EXPORT_PAGE_SIZE):
                    for row in page:
                        if codec is None:
                            raise RuntimeError(
                                "Exporting targeted grants requires an authority alias codec."
                            )
                        grant = targeted_tool_grant_with_active_reference(
                            _targeted_tool_grant_from_row(row), codec
                        )
                        builder.charge(grant.model_dump(mode="json"))
                        grants.append(grant)
                rows = connection.execute(
                    "SELECT * FROM cayu_targeted_tool_grant_uses WHERE session_id = ? ORDER BY bound_at, use_id",
                    (session_id,),
                )
                while page := rows.fetchmany(SESSION_EXPORT_PAGE_SIZE):
                    for row in page:
                        use = _targeted_tool_use_from_row(row)
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

        return await self._run_read(query)

    async def load_events(self, session_id: str) -> list[Event]:
        return await session_queries.load_events(self._run_read, session_id)

    async def load_user_input_supersession_events(
        self,
        session_id: str,
        input_id: str,
    ) -> list[Event]:
        return await session_queries.load_user_input_supersession_events(
            self._run_read, session_id, input_id
        )

    async def load_tool_round_lifecycle_events(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
    ) -> list[Event]:
        return await session_queries.load_tool_round_lifecycle_events(
            self._run_read, session_id, tool_call_ids
        )

    async def load_tool_round_lifecycle_events_for_round(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
        *,
        tool_round_identity: ToolRoundIdentity,
    ) -> list[Event]:
        return await session_queries.load_tool_round_lifecycle_events_for_round(
            self._run_read, session_id, tool_call_ids, tool_round_identity=tool_round_identity
        )

    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        return await session_queries.query_events(self._run_read, query)

    async def read_usage_accounting(
        self, query: EventQuery, *, by_session: bool = False, by_identity: bool = False
    ) -> UsageAccountingSnapshot:
        return await session_queries.read_usage_accounting(
            self._run_read,
            query,
            by_session=by_session,
            by_identity=by_identity,
            usage_cache=self._session_usage_cache,
        )

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
        return await session_queries.read_cost_accounting(
            self._run_read,
            query,
            pricing,
            currency=currency,
            details=details,
            by_session=by_session,
            additional_events=additional_events,
            max_detail_bytes=max_detail_bytes,
            previous=previous,
            cost_authority=self._cost_accounting_authority,
        )

    async def event_exists(self, query: EventQuery) -> bool:
        return await session_queries.event_exists(self._run_read, query)

    async def query_events_bounded(
        self,
        query: EventQuery,
        *,
        max_bytes: int,
    ) -> list[EventRecord]:
        return await session_queries.query_events_bounded(
            self._run_read, query, max_bytes=max_bytes
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

        def read(connection: sqlite3.Connection) -> Session | None:
            connection.execute("BEGIN")
            try:
                row = connection.execute(
                    "SELECT (COALESCE(length(CAST(s.id AS BLOB)), 0) + COALESCE(length(CAST(s.instance_id AS BLOB)), 0) + COALESCE(length(CAST(s.agent_name AS BLOB)), 0) + COALESCE(length(CAST(s.provider_name AS BLOB)), 0) + COALESCE(length(CAST(s.model AS BLOB)), 0) + COALESCE(length(CAST(s.parent_session_id AS BLOB)), 0) + COALESCE(length(CAST(s.causal_budget_id AS BLOB)), 0) + COALESCE(length(CAST(s.runtime_name AS BLOB)), 0) + COALESCE(length(CAST(s.runtime_version AS BLOB)), 0) + COALESCE(length(CAST(s.environment_name AS BLOB)), 0) + COALESCE(length(CAST(s.status AS BLOB)), 0) + COALESCE(length(CAST(s.created_at AS BLOB)), 0) + COALESCE(length(CAST(s.updated_at AS BLOB)), 0) + COALESCE(length(CAST(s.last_activity_at AS BLOB)), 0) + COALESCE(length(CAST(s.run_epoch AS BLOB)), 0) + COALESCE(length(CAST(s.invocation_json AS BLOB)), 0) + COALESCE(length(CAST(s.metadata_json AS BLOB)), 0)) + COALESCE((SELECT SUM(length(CAST(key AS BLOB)) "
                    "+ length(CAST(value AS BLOB))) FROM cayu_session_labels "
                    "WHERE session_id = s.id), 0) FROM cayu_sessions s WHERE id = ?",
                    (session_id,),
                ).fetchone()
                if row is None:
                    return None
                if row[0] > max_bytes:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED, limit=max_bytes
                    )
                session = sqlite_records.load_session(connection, session_id)
                if (
                    session is not None
                    and compact_json_utf8_size(session.model_dump(mode="json")) > max_bytes
                ):
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED, limit=max_bytes
                    )
                return session
            finally:
                connection.rollback()

        return await self._run_read(read)

    async def export_terminal_session_evidence(
        self,
        session_id: str,
        *,
        spool: EvidenceSpool,
    ) -> None:
        """Copy one stable terminal snapshot into caller-owned bounded backing."""
        async with asyncio.timeout(spool.limits.max_seconds):
            await spool.run_owned(
                self._load_terminal_session_evidence(
                    session_id,
                    limits=None,
                    observed_interrupted_events=None,
                    expected_interrupted_parent_session_id=None,
                    require_interrupted_proof=False,
                    spool=spool,
                )
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
        resolved_limits = eager_limits if spool is None else spool.limits
        observed, expected_parent_session_id = _copy_runner_owned_interruption_proof(
            session_id,
            observed_events=observed_interrupted_events,
            expected_parent_session_id=expected_interrupted_parent_session_id,
            limits=eager_limits,
            required=require_interrupted_proof,
        )
        allow_interrupted = observed is not None or expected_parent_session_id is not None
        evidence_event_types = tuple(
            str(event_type) for event_type in _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES
        )
        evidence_type_placeholders = ", ".join("?" for _ in evidence_event_types)
        event_columns = ", ".join(sqlite_records.EVENT_COLUMN_NAMES)
        event_stored_bytes = " + ".join(
            [
                "length(CAST(sequence AS TEXT))",
                *(
                    f"COALESCE(length(CAST({column} AS BLOB)), 0)"
                    for column in sqlite_records.EVENT_COLUMN_NAMES
                ),
            ]
        )
        session_stored_bytes = " + ".join(
            f"COALESCE(length(CAST(session.{column} AS BLOB)), 0)"
            for column in (
                "id",
                "instance_id",
                "invocation_json",
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

        def run_query(connection: sqlite3.Connection) -> TerminalSessionEvidence | None:
            limits = resolved_limits
            if spool is not None:
                connection.set_progress_handler(lambda: int(spool.should_interrupt()), 1000)
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

                session = sqlite_records.load_session(connection, session_id)
                if session is None:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                    )
                if spool is not None:
                    cursor = connection.execute(
                        f"SELECT sequence, {event_columns}, ({event_stored_bytes}) AS transport_bytes FROM cayu_events "
                        "WHERE session_id = ? AND sequence <= ? ORDER BY sequence ASC",
                        (session_id, terminal_record.sequence),
                    )
                    while True:
                        spool.check()
                        rows = cursor.fetchmany(spool.limits.batch_records)
                        if not rows:
                            break
                        spool.observe_page(
                            records=len(rows),
                            transport_bytes=sum(row["transport_bytes"] for row in rows),
                        )
                        for row in rows:
                            spool.append(
                                "event",
                                EventRecord(
                                    sequence=row["sequence"],
                                    event=sqlite_records.event_from_row(row),
                                ),
                            )
                    cursor.close()
                    cursor = connection.execute(
                        f"SELECT session_order - 1 AS transcript_index, interaction_id, message_json, ({transcript_stored_bytes}) AS transport_bytes "
                        "FROM cayu_transcript_messages WHERE session_id = ? ORDER BY session_order ASC",
                        (session_id,),
                    )
                    while True:
                        spool.check()
                        rows = cursor.fetchmany(spool.limits.batch_records)
                        if not rows:
                            break
                        spool.observe_page(
                            records=len(rows),
                            transport_bytes=sum(row["transport_bytes"] for row in rows),
                        )
                        for row in rows:
                            spool.append(
                                "transcript",
                                TranscriptRecord(
                                    index=row["transcript_index"],
                                    interaction_id=row["interaction_id"],
                                    message=Message(**json.loads(row["message_json"])),
                                ),
                            )
                    cursor.close()
                    spool.stage(session, marker, terminal_record, event_count, transcript_count)
                    return None
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
                    EventRecord(sequence=row["sequence"], event=sqlite_records.event_from_row(row))
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
                    limits=eager_limits,
                    allow_interrupted=allow_interrupted,
                )
            except TerminalSessionEvidenceError:
                raise
            except (json.JSONDecodeError, sqlite3.DatabaseError, TypeError, ValueError) as exc:
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                ) from exc
            finally:
                if spool is not None:
                    connection.set_progress_handler(None, 0)
                connection.rollback()

        return await self._run_read(run_query)

    async def query_latest_interaction_events(
        self,
        session_id: str,
        *,
        before_sequence: int | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        return await session_queries.query_latest_interaction_events(
            self._run_read, session_id, before_sequence=before_sequence, limit=limit
        )

    async def summarize_events(self, session_id: str) -> EventSummary:
        return await session_queries.summarize_events(self._run_read, session_id)

    async def summarize_outcome(self, session_id: str) -> SessionOutcome:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection: sqlite3.Connection) -> SessionOutcome:
            session = sqlite_records.load_session(connection, session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")

            terminal_row = connection.execute(
                f"""
                SELECT sequence, {", ".join(sqlite_records.EVENT_COLUMN_NAMES)}
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
                SELECT sequence, {", ".join(sqlite_records.EVENT_COLUMN_NAMES)}
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
                terminal_event=sqlite_records.event_record_from_row(terminal_row),
                latest_retry_event=sqlite_records.event_record_from_row(retry_row),
            )

        from cayu.storage._session_access_records import sqlite_owner_read

        return await self._run_read(
            lambda connection: sqlite_owner_read(connection, access_bounds, session_id, query)
        )

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
        Queue lifecycle events and their sequence/side-effect records are retained
        while the event-owned session and queue identity still name a queue row,
        including terminal rows. Queue JSON is not parsed to establish this pin.
        Returns the number of events deleted.
        """
        if not isinstance(before, datetime):
            raise TypeError("prune_events 'before' must be a datetime.")
        cutoff = sqlite_records.format_datetime(before)
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")

        def statement(connection: sqlite3.Connection) -> int:
            if session_id is not None and not sqlite_records.session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            publication_key_pattern = RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX + "*"
            with connection:
                if session_id is None:
                    cursor = connection.execute(
                        f"""
                        DELETE FROM cayu_events
                        WHERE timestamp < ?
                          AND event_type NOT IN (?, ?, ?, ?)
                          {_SESSION_MESSAGE_EVENT_RETENTION_SQL}
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
                        f"""
                        DELETE FROM cayu_events
                        WHERE session_id = ? AND timestamp < ?
                          AND event_type NOT IN (?, ?, ?, ?)
                          {_SESSION_MESSAGE_EVENT_RETENTION_SQL}
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

        return await transcript_ops.compact_transcript(
            self._run_write, session_id, keep_last=keep_last
        )

    async def list_sessions(self, query: SessionQuery | None = None) -> SessionListResult:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        if access_bounds is not None:
            return await self._access_list_sessions(access_bounds, copy_session_query(query))
        return await session_queries.list_sessions(
            self._run_read,
            ownership_clock=self._ownership_clock,
            query=query,
            pending_interruption_cascade_only=False,
        )

    async def query_session_topology(
        self,
        query: SessionTopologyQuery,
    ) -> SessionTopologyStoreResult:
        return await session_queries.query_session_topology(self._run_read, query)

    async def query_session_lineage(
        self,
        query: SessionLineageQuery,
    ) -> SessionLineageResult:
        return await session_queries.query_session_lineage(self._run_read, query)

    async def query_child_session_lifecycle(
        self,
        query: ChildSessionLifecycleQuery,
    ) -> ChildSessionLifecyclePage:
        return await session_queries.query_child_session_lifecycle(self._run_read, query)

    async def aggregate_operational_snapshot(
        self,
        filters: SessionAggregateFilter | None = None,
    ) -> SessionOperationalSnapshot:
        filters = copy_session_aggregate_filter(filters)
        plan = session_store_sql.build_session_query_sql(
            session_query_from_aggregate_filter(filters),
            dialect=session_queries.SQL_DIALECT,
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
                as_of=sqlite_records.parse_datetime(rows[0]["as_of"]),
                total_count=sum(counts.values()),
                counts_by_status=SessionStatusCounts.model_validate(counts),
                accuracy=EXACT_AGGREGATE.model_copy(),
            )

        return await self._run_read(query_snapshot)

    async def aggregate_usage(self, query: UsageRollupQuery) -> UsageRollupStoreResult:
        return await session_queries.aggregate_usage(self._run_read, query)

    async def list_sessions_with_pending_interruption_cascade(
        self,
        query: SessionQuery | None = None,
    ) -> SessionListResult:
        return await session_queries.list_sessions(
            self._run_read,
            ownership_clock=self._ownership_clock,
            query=query,
            pending_interruption_cascade_only=True,
        )

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
        status_placeholders = ", ".join("?" for _status in status_values)
        filters = [
            f"cayu_sessions.status IN ({status_placeholders})",
            "cayu_checkpoints.pending_action_metrics_ready = 1",
            "cayu_checkpoints.pending_action_flags <> 0",
        ]
        params: list[Any] = list(status_values)
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
        elif query.kind == PendingActionKind.DELEGATED_ACTION:
            filters.append("(cayu_checkpoints.pending_action_flags & 8) <> 0")
        if query.cursor is not None:
            cursor_dt, cursor_id = decode_session_cursor(query.cursor)
            cursor_value = sqlite_records.format_datetime(cursor_dt)
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
                cayu_sessions.instance_id,
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
                    ),
                    'foreground_child_wait',
                    json_extract(
                        cayu_checkpoints.state_json,
                        '$.foreground_child_wait'
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
                    ('session.interrupted'),
                    ('session.delegated_action.updated')
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
                              'session.delegated_action.updated',
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
                              'session.delegated_action.updated',
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
                    row["id"]: sqlite_records.pending_action_session_from_row(row, labels={})
                    for row in inspected_rows
                }
                inspected_ids = [row["id"] for row in inspected_rows]
                selected_ids_json = sqlite_records.json_dumps(inspected_ids)

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
                        (sqlite_records.json_dumps(preflight_eligible_ids),),
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
                            sqlite_records.json_dumps(materializable_ids),
                            sqlite_records.json_dumps(materializable_sequences),
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

                labels_by_session_id = sqlite_records.load_session_labels_batch(
                    connection, materializable_ids
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

    async def append_transcript_messages(
        self,
        session_id: str,
        messages: list[Message],
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        return await transcript_ops.append_transcript_messages(
            self._run_write,
            session_id,
            messages,
            interaction_id=interaction_id,
            ownership_clock=self._ownership_clock,
            closure_owners=self._closure_lineage_owners_unlocked,
            touch_activity=_touch_session_activity,
        )

    @runtime_session_query
    async def append_peer_content(
        self,
        request: PeerContentAppendRequest,
        *,
        qualify_target: Callable[[Session], None] | None = None,
        pending_transcript_cursor: int | None = None,
    ) -> PeerContentReceipt:
        return await peer_content_ops.append_peer_content(
            self._run_write,
            request,
            qualify_target=qualify_target,
            pending_transcript_cursor=pending_transcript_cursor,
            load_unlocked=self._load_unlocked,
            ownership_clock=self._ownership_clock,
        )

    async def read_peer_content_attempt(
        self, request: PeerContentAppendRequest
    ) -> PeerContentReceipt | None:
        return await peer_content_ops.read_peer_content_attempt(self._run_read, request)

    async def list_pending_peer_content(self, *, after_operation_key=None, limit=32):
        """Trusted receiving-owner discovery, independent of creation settlement."""
        return await peer_content_ops.list_pending_peer_content(
            self._run_read, after_operation_key=after_operation_key, limit=limit
        )

    async def read_peer_content(self, append_key: PeerAppendKey) -> PeerContentReceipt | None:
        return await peer_content_ops.read_peer_content(self._run_read, append_key)

    async def record_peer_content_exposure(
        self, request: PeerContentExposureRequest
    ) -> PeerContentExposureReceipt:
        return await peer_content_ops.record_peer_content_exposure(self._run_write, request)

    async def begin_peer_content_exposure(
        self, request: PeerContentExposureRequest
    ) -> PeerContentExposureReceipt:
        return await peer_content_ops.begin_peer_content_exposure(self._run_write, request)

    async def read_peer_content_exposure(
        self, append_key: PeerAppendKey, exposure_id: str
    ) -> PeerContentExposureReceipt | None:
        return await peer_content_ops.read_peer_content_exposure(
            self._run_read, append_key, exposure_id
        )

    async def exclude_peer_content(
        self, request: PeerContentAppendRequest, *, reason: str
    ) -> PeerContentReceipt:
        return await peer_content_ops.exclude_peer_content(self._run_write, request, reason=reason)

    async def retry_pending_peer_content(
        self,
        session_id: str,
        *,
        expected_session_instance_id: str,
        expected_run_epoch: int,
        expected_transcript_cursor: int,
        admit=None,
    ) -> tuple[PeerContentReceipt, ...]:
        return await peer_content_ops.retry_pending_peer_content(
            self._run_read,
            session_id,
            expected_session_instance_id=expected_session_instance_id,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
            admit=admit,
            read_creation_decision=self.read_session_creation_decision,
        )

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
        return await transcript_ops.replace_initial_transcript_messages(
            self._run_write,
            session_id,
            expected_messages,
            replacement_messages,
            interaction_id=interaction_id,
            checkpoint_transform=checkpoint_transform,
            runtime_suffix_count=runtime_suffix_count,
            ownership_clock=self._ownership_clock,
            load_session=self._load_unlocked,
            load_checkpoint=self._load_checkpoint_unlocked,
            closure_owners=self._closure_lineage_owners_unlocked,
            touch_activity=_touch_session_activity,
        )

    async def materialize_deferred_interaction_input(
        self,
        session_id: str,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> bool:
        return await transcript_ops.materialize_deferred_interaction_input(
            self._run_write,
            session_id,
            interaction_id=interaction_id,
            ownership_clock=self._ownership_clock,
            load_session=self._load_unlocked,
            closure_owners=self._closure_lineage_owners_unlocked,
            touch_activity=_touch_session_activity,
        )

    async def load_deferred_interaction_input(
        self,
        session_id: str,
    ) -> DeferredInteractionInput | None:
        return await transcript_ops.load_deferred_interaction_input(self._run_read, session_id)

    async def append_transcript_messages_and_transform_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint_transform: CheckpointTransform,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        return await transcript_ops.append_transcript_messages_and_transform_checkpoint(
            self._run_write,
            session_id,
            messages,
            checkpoint_transform,
            interaction_id=interaction_id,
            ownership_clock=self._ownership_clock,
            load_session=self._load_unlocked,
            load_checkpoint=self._load_checkpoint_unlocked,
            closure_owners=self._closure_lineage_owners_unlocked,
            touch_activity=_touch_session_activity,
        )

    async def load_transcript(self, session_id: str) -> list[Message]:
        return await transcript_ops.load_transcript(self._run_read, session_id)

    async def load_transcript_snapshot(self, session_id: str) -> TranscriptSnapshot:
        return await transcript_ops.load_transcript_snapshot(self._run_read, session_id)

    async def load_transcript_cursor(self, session_id: str) -> int:
        return await transcript_ops.load_transcript_cursor(self._run_read, session_id)

    async def load_latest_transcript_message(
        self,
        session_id: str,
        *,
        role: MessageRole,
    ) -> TranscriptRecord | None:
        return await transcript_ops.load_latest_transcript_message(
            self._run_read, session_id, role=role
        )

    async def load_latest_transcript_text(
        self,
        session_id: str,
        *,
        role: MessageRole,
        max_chars: int,
    ) -> tuple[str, bool] | None:
        return await transcript_ops.load_latest_transcript_text(
            self._run_read, session_id, role=role, max_chars=max_chars
        )

    async def load_transcript_window(
        self,
        session_id: str,
        *,
        start_index: int,
        limit: int,
    ) -> TranscriptSnapshot:
        return await transcript_ops.load_transcript_window(
            self._run_read, session_id, start_index=start_index, limit=limit
        )

    async def query_transcript(self, query: TranscriptQuery) -> TranscriptPage:
        return await transcript_ops.query_transcript(self._run_read, query)

    async def search_transcript(
        self,
        query: TranscriptSearchQuery,
    ) -> TranscriptSearchResult:
        return await transcript_ops.search_transcript(self._run_read, query)

    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state must be a dictionary.")
        checkpoint = copy_durable_json_object(state, "checkpoint")

        def statement(connection: sqlite3.Connection) -> None:
            try:
                connection.execute("BEGIN IMMEDIATE")
                updated_at = self._ownership_clock()
                if not sqlite_records.session_exists(connection, session_id):
                    raise KeyError(f"Session not found: {session_id}")
                for owner in self._closure_lineage_owners_unlocked((session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
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
                    sqlite_records.checkpoint_row_values(session_id, replacement, updated_at),
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
                for owner in self._closure_lineage_owners_unlocked((session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
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
                        sqlite_records.checkpoint_row_values(session_id, transformed, updated_at),
                    )
                connection.commit()
            except BaseException as primary:
                transaction_failure = sqlite_connection._settle_failed_transaction(
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
                for owner in self._closure_lineage_owners_unlocked((session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
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
                        sqlite_records.checkpoint_row_values(session_id, transformed, now),
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

        await self._run_write(statement)

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        from cayu.storage._session_access_records import sqlite_owner_read

        value = await self._run_read(
            lambda connection: sqlite_owner_read(
                connection,
                access_bounds,
                session_id,
                lambda conn: _load_checkpoint_json(conn, session_id),
                action="inspect_state",
            )
        )

        if value is None:
            return None
        return await asyncio.to_thread(_checkpoint_from_json, value)

    async def load_execution_snapshot_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection):
            row = connection.execute(
                "SELECT json_extract(state_json, '$.checkpoint_schema_version') AS version, "
                "json_extract(state_json, '$.execution_snapshots') AS snapshots "
                "FROM cayu_checkpoints WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            projection = {}
            if row["version"] is not None:
                projection["checkpoint_schema_version"] = row["version"]
            if row["snapshots"] is not None:
                projection["execution_snapshots"] = json.loads(row["snapshots"])
            return projection

        return await self._run_read(query)

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
        self._closed = True
        async with self._lock:
            for lock, connection in self._readers:
                if connection is not self._connection:
                    async with lock:
                        connection.close()
            self._connection.close()

    def _connect(self, path: Path) -> sqlite3.Connection:
        return sqlite_connection.connect(path)

    def _connect_read_only(self, path: Path) -> sqlite3.Connection:
        if sqlite_support.current_diagnostic_store_inspection() is not None:
            return sqlite_connection.connect_read_only_inspection(path)
        return sqlite_connection.connect(path, read_only=True)

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
        return sqlite_records.load_session(self._connection, session_id)

    def _load_labels_unlocked(self, session_id: str) -> dict[str, str]:
        return sqlite_records.load_session_labels(self._connection, session_id)

    def _session_exists_unlocked(self, session_id: str) -> bool:
        return sqlite_records.session_exists(self._connection, session_id)

    def _first_existing_event_id_unlocked(
        self,
        session_id: str,
        event_ids: list[str],
    ) -> str | None:
        return _first_existing_event_id(self._connection, session_id, event_ids)
