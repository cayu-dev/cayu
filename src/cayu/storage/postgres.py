from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Literal, LiteralString, cast
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
from cayu.deadlines import ExecutionDeadline, ExecutionDeadlineRenewal
from cayu.runtime._cost_accounting import CostAccountingSnapshot
from cayu.runtime._usage_accounting import UsageAccountingSnapshot
from cayu.session_metadata import SummaryTone, summary_metadata
from cayu.sessions import _deadline_replacement as deadline_replacement
from cayu.sessions import creation_fence
from cayu.sessions._access_rules import runtime_session_mutation, runtime_session_query
from cayu.sessions._closure_records import (
    _check_closure_lineage_owner,
    _closure_progress_targets,
    _validate_closure_progress_update,
    _validate_session_closure_detach_replay,
)
from cayu.sessions._event_snapshot import EventSnapshotConsumer
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectHealth,
    PersistedEventSideEffectPage,
    PersistedEventSideEffectQuery,
)
from cayu.sessions.messaging import (
    SessionMessageActionRequest,
    SessionMessageActionResult,
    SessionMessageInspection,
    SessionMessageQuery,
    SessionMessageSource,
)
from cayu.sessions.records import update_session_annotations
from cayu.storage import _creation_fence
from cayu.storage import _postgres_base as postgres_base
from cayu.storage import _postgres_context_views as context_view_ops
from cayu.storage import _postgres_event_delivery as event_delivery_ops
from cayu.storage import _postgres_event_publication as event_publication_ops
from cayu.storage import _postgres_memory_evidence as memory_evidence_ops
from cayu.storage import _postgres_model_stage as model_stage_ops
from cayu.storage import _postgres_peer_content as peer_content_ops
from cayu.storage import _postgres_public_authority as public_authority_ops
from cayu.storage import _postgres_publication as publication_ops
from cayu.storage import _postgres_session_creation as creation_ops
from cayu.storage import _postgres_session_messages as message_ops
from cayu.storage import _postgres_session_queries as session_queries
from cayu.storage import _postgres_support as pg_support
from cayu.storage import _postgres_terminal_evidence as terminal_evidence_ops
from cayu.storage import _postgres_tool_grants as tool_grants_ops
from cayu.storage import _postgres_transcript as transcript_ops
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
    from cayu.sessions._access_rules import SessionReadBounds, _SessionAccessBounds
    from cayu.sessions._temporary_continuation import TemporaryServiceAdmission
    from cayu.sessions.exports import SessionExportLimits, SessionExportSnapshot
    from cayu.storage.retention import (
        RetentionAuditEntry,
        RetentionAuditRecord,
        RetentionMode,
        RetentionProgressCallback,
        RetentionProtection,
        RetentionReport,
        SessionRetentionPolicy,
        StorageGrowthReport,
    )

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
    JsonUtf8SizeCounter,
    copy_durable_json_object,
    copy_durable_json_value,
    copy_label_map,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu.budgets.aggregates import EXACT_AGGREGATE, UsageRollupStoreResult
from cayu.budgets.base import BudgetReservationIdentityConflict
from cayu.events import (
    Event,
    EventType,
)
from cayu.execution_profiles import (
    ExecutionProfileDecision,
    ExecutionProfileIdentity,
    ExecutionProfileRejectionResult,
)
from cayu.execution_units import (
    ToolRoundIdentity,
)
from cayu.memory.evidence import (
    ContextExposure,
    ContextExposurePage,
    ContextExposureTransitionRequest,
    RecallEvidenceQuery,
    RecallItemExposure,
    RecallReceipt,
    RecallReceiptPage,
)
from cayu.messages import Message, MessageRole
from cayu.runtime._child_session_notifications import (
    ChildSessionLifecyclePage,
    ChildSessionLifecycleQuery,
)
from cayu.runtime.evidence_spool import EvidenceSpool
from cayu.runtime.public_authority import PublicAuthorityAliasCodec
from cayu.runtime.service_manifest import RuntimeStoreDurability
from cayu.sessions._checkpoint_mutations import (
    RuntimePublicationMutation,
    _apply_runtime_publication_checkpoint_mutation,
)
from cayu.sessions._checkpoint_preservation import (
    _checkpoint_transform_result_preserving_completion_result_event_publications,
    _completion_result_event_publication_delete_block_reason,
    _copy_checkpoint_for_transform,
    _replace_checkpoint_preserving_completion_result_event_publications,
)
from cayu.sessions._completion_finalization import (
    _apply_queue_completion_checkpoint_mutation,
    _prepare_queue_completion_checkpoint_mutation,
)
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
)
from cayu.sessions._execution_profile_rules import (
    SessionModelTransition,
    _copy_optional_execution_profile,
    _copy_optional_execution_profile_decision,
    _copy_session_model_transition,
    _execution_profile_rejection_events_equivalent,
    _prepare_execution_profile_rejection,
    _session_metadata_after_model_transition,
    _session_metadata_after_runtime_identity_adoption,
    _session_metadata_after_tool_capability_ceiling_admission,
    _validate_execution_profile_admission,
    _validate_execution_profile_rejection_session,
    _validate_session_model_transition,
)
from cayu.sessions._interaction_receipts import (
    _checkpoint_after_exact_invocation_terminal_decision,
    _interaction_transition_receipt_record,
    _interaction_transition_spec_from_receipt,
    _interaction_transition_storage_key,
    _invocation_terminal_event_receipt_record,
    _invocation_terminal_event_storage_key,
    _load_interaction_transition_receipt,
    _reconstruct_interaction_transition_receipt,
    _require_invocation_release_recovery_claim,
    _require_invocation_release_settlement_record,
    _require_invocation_release_terminal_session_event,
    _validate_interaction_transition_invocation_authority_parameters,
    _validate_interaction_transition_receipt_authority,
    _validate_interaction_transition_receipt_recovery_authority,
    _validate_interaction_transition_recovery_claim_id,
    _validate_invocation_release_settlement_receipt_authority,
)
from cayu.sessions._interaction_transitions import (
    InteractionTransitionReceiptResult,
    InteractionTransitionResult,
    InteractionTransitionSpec,
    QueuedInteractionProfileHandoff,
    _copy_transition_interaction_admission,
    _prepare_interaction_transition,
    _prepare_interaction_transition_receipt_lookup,
)
from cayu.sessions._invocation_scope import (
    INHERIT_INTERACTION,
    InteractionAttribution,
    _activate_session_run_fence,
    _assert_session_run_epoch,
    _assert_session_run_epoch_value,
    _current_session_run_epoch,
    _deactivate_session_interaction,
    _deactivate_session_run_fence,
)
from cayu.sessions._invocation_terminal_decision import InvocationTerminalDecision
from cayu.sessions._model_completion_publication import ModelCompletionStageSettlementRequest
from cayu.sessions._model_failover_rules import (
    _model_failover_admission_storage_keys,
    _model_failover_checkpoint_after_profile_admission,
)
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
    ModelCompletionStage,
    ModelCompletionStageAbandonmentResult,
    ModelCompletionStageDispatch,
    ModelCompletionStageResult,
    SessionModelCompletionDispatchAlreadyAuthorized,
    SessionModelCompletionStageConflict,
)
from cayu.sessions._model_stage_rules import (
    _model_completion_stage_dispatch_storage_key,
    _model_completion_stage_settlement_record,
    _model_completion_stage_settlement_storage_key,
    _model_completion_stage_storage_identity,
    _model_completion_stage_winner_storage_key,
    _ModelCompletionStagePromotionContext,
    _PreparedModelCompletionStage,
    _PreparedModelCompletionStageAbandonment,
    _PreparedModelCompletionStageTerminal,
    _reconstruct_active_model_completion_stage,
    _reconstruct_active_model_completion_stage_record,
    _validate_model_completion_stage_for_settlement,
)
from cayu.sessions._operation_records import (
    _active_unexpired_session_operation_id,
    _reject_reserved_runtime_publication_key,
)
from cayu.sessions._provider_operation_cancellation_claim import (
    active_provider_operation_cancellation_claim_from_checkpoint,
)
from cayu.sessions._runtime_publication import (
    RuntimePublicationResult,
    SessionOperationInitializer,
    SessionOperationTransform,
    StoreTimeSessionOperationTransform,
    _runtime_publication_storage_key,
)
from cayu.sessions._runtime_publication_rules import (
    _PreparedRuntimePublication,
    _reconstruct_runtime_publication_receipt,
)
from cayu.sessions._terminal_evidence import (
    QueuedDispatchTerminalReceipt,
    QueuedDispatchTerminalReceiptQuery,
    _queued_dispatch_terminal_receipts_from_checkpoint,
    _session_run_operation_from_checkpoint,
)
from cayu.sessions.authority import (
    CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS,
    CheckpointRootFieldGuard,
    CheckpointValueAuthority,
    SessionRunFenced,
    checkpoint_root_field_projection_from_storage,
)
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectClaim,
    PersistedEventSideEffectDelivery,
    PersistedEventSideEffectStatus,
    _copy_optional_event_id,
    restore_persisted_event_authority,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.exports import _require_session_export_target
from cayu.sessions.forks import (
    ForkCheckpointAuthorityDecoder,
    ForkSystemPromptReplacement,
    ProfiledSessionForkResult,
    SessionForkProfileRelationship,
    _copy_profiled_fork_authority,
)
from cayu.sessions.inspection import SessionInspectionIdentity
from cayu.sessions.invocation import SessionInvocation
from cayu.sessions.lineage import (
    DELETE_BLOCKED_SESSION_STATUSES,
    SessionLineageQuery,
    SessionLineageResult,
    _durable_subagent_parent_delete_block_reason,
)
from cayu.sessions.mcp_manifest_history import (
    McpManifestBaseline,
    McpManifestBaselineLoadResult,
    McpManifestPublicationResult,
)
from cayu.sessions.messaging import (
    SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
    EnqueueSessionMessageRequest,
    EnqueueSessionMessageResult,
    SessionMessageDeliveryBatch,
    SessionQueuedMessagesPending,
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
    _validate_inactive_for_seconds,
    copy_session_aggregate_filter,
    copy_session_query,
    decode_session_cursor,
    encode_session_cursor,
    session_query_from_aggregate_filter,
)
from cayu.sessions.records import (
    CheckpointTransform,
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
    SessionStatusConflict,
    StoreTimeCheckpointTransform,
    TranscriptRecord,
    _validate_status_set,
    copy_session_runtime_identity,
    copy_session_user_metadata,
)
from cayu.sessions.recovery import (
    _active_unexpired_incomplete_recovery_claim_id,
    _incomplete_recovery_claim_from_checkpoint,
    _project_interruption_cascade_marker_fields,
    _require_live_incomplete_recovery_claim_for_run_epoch_transfer,
)
from cayu.sessions.requests import (
    RunRequest,
    _copy_optional_tool_capability_ceiling,
)
from cayu.sessions.store import SessionStore
from cayu.sessions.summaries import (
    EventSummary,
    SessionOperationalSnapshot,
    SessionOutcome,
    session_outcome,
)
from cayu.sessions.terminal_evidence import (
    _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES,
    _TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT,
    TerminalSessionEvidence,
    TerminalSessionEvidenceError,
    TerminalSessionEvidenceErrorCode,
    TerminalSessionEvidenceLimits,
    _terminal_publication_delete_block_reason,
)
from cayu.sessions.topology import (
    SessionTopologyQuery,
    SessionTopologyStoreResult,
)
from cayu.sessions.transcript_input import (
    DeferredInteractionInput,
    deferred_interaction_input_from_storage_payload,
    deferred_interaction_input_storage_payload,
)
from cayu.sessions.transcript_queries import (
    BoundedTranscriptWindow,
    ForkTranscriptValidator,
    TranscriptPage,
    TranscriptQuery,
    TranscriptSearchQuery,
    TranscriptSearchResult,
    TranscriptSnapshot,
)
from cayu.sessions.usage import UsageRollupQuery
from cayu.storage import _session_store_sql as session_store_sql
from cayu.storage import migration_authority
from cayu.storage import migrations as schema
from cayu.storage._participant_bindings_schema import (
    PARTICIPANT_BINDING_PROJECTION,
)
from cayu.tools.exposure import ToolCapabilityCeiling
from cayu.tools.grants import (
    TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS,
    TargetedToolGrantIssueResult,
    TargetedToolGrantReconstructionResult,
    TargetedToolGrantRecord,
    TargetedToolGrantStateSnapshot,
    TargetedToolUseBinding,
    TargetedToolUseRequest,
    TargetedToolUseResult,
    copy_targeted_tool_grant_record,
    targeted_tool_grant_with_active_reference,
)

_POSTGRES_SESSION_MIN_REQUIRED_REVISION = 129


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
        from cayu.sessions._access_rules import _creation_bounds

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
        self, bounds: SessionReadBounds, query: SessionQuery
    ) -> SessionListResult:
        return await session_queries.list_sessions(
            self._ready_connection,
            query,
            pending_interruption_cascade_only=False,
            access_bounds=bounds,
        )

    async def _access_update_labels(
        self, bounds: _SessionAccessBounds, session_id: str, labels: dict[str, str]
    ) -> Session:
        return await self.update_labels(session_id, labels, _access_bounds=bounds)

    async def _access_load_session(self, bounds: SessionReadBounds, session_id: str) -> Session:
        from cayu.sessions._access_rules import require_clause_admitted
        from cayu.sessions.access import SessionAccessDenied

        session_id = require_clean_nonblank(session_id, "session_id")
        clause = session_store_sql.session_access_clause(
            bounds, dialect=session_queries.SQL_DIALECT
        )
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
            return require_clause_admitted(bounds, pg_support.session_from_row(row, labels=labels))

    supports_usage_aggregates: ClassVar[bool] = True
    supports_private_argument_continuity: ClassVar[bool] = True
    supports_mcp_manifest_history: ClassVar[bool] = True
    supports_public_authority_aliases: ClassVar[bool] = True
    supports_targeted_tool_grants: ClassVar[bool] = True
    supports_session_topology: ClassVar[bool] = True
    supports_session_lineage: ClassVar[bool] = True
    supports_supervision_events: ClassVar[bool] = True
    supports_fork_controller_reads: ClassVar[bool] = True
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
    supports_session_closure_finished_only: ClassVar[bool] = True
    supports_session_closure_progress: ClassVar[bool] = True
    supports_storage_retention: ClassVar[bool] = True
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
    def can_register_public_authority_aliases(self) -> bool:
        """Whether this store instance can persist new public authority aliases."""

        return not self._read_only

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

        return await public_authority_ops.register_public_authority_alias(
            self._connection,
            public_alias,
            field_name=field_name,
            private_value=private_value,
            scope_session_id=scope_session_id,
            codec=self.public_authority_alias_codec,
            ensure_ready=self._ensure_ready,
        )

    async def resolve_public_authority_alias(
        self,
        public_alias: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> str | None:
        """Resolve one exact alias through its indexed authority scope."""

        return await public_authority_ops.resolve_public_authority_alias(
            self._connection,
            public_alias,
            field_name=field_name,
            scope_session_id=scope_session_id,
            codec=self.public_authority_alias_codec,
            ensure_ready=self._ensure_ready,
        )

    async def public_authority_private_value_exists(
        self,
        private_value: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> bool:
        return await public_authority_ops.public_authority_private_value_exists(
            self._connection,
            private_value,
            field_name=field_name,
            scope_session_id=scope_session_id,
            codec=self.public_authority_alias_codec,
            ensure_ready=self._ensure_ready,
        )

    async def issue_targeted_tool_grants(
        self,
        session_id: str,
        *,
        expected_run_epoch: int,
        records: tuple[TargetedToolGrantRecord, ...],
        events: tuple[Event, ...],
    ) -> TargetedToolGrantIssueResult:
        return await tool_grants_ops.issue_targeted_tool_grants(
            self._connection,
            session_id,
            expected_run_epoch=expected_run_epoch,
            records=records,
            events=events,
            ensure_ready=self._ensure_ready,
            get_codec=lambda: self.public_authority_alias_codec,
            append_events=self._append_events_with_cursor,
            append_event_once=self._append_event_once_with_cursor,
            register_alias=public_authority_ops.register_alias_row,
            decode_grant=_targeted_tool_grant_from_postgres_row,
            validate_use_counts=_validate_targeted_tool_use_counts,
        )

    async def list_targeted_tool_grants(
        self,
        session_id: str,
        *,
        interaction_id: str | None = None,
        limit: int = TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS,
    ) -> tuple[TargetedToolGrantRecord, ...]:
        return await tool_grants_ops.list_targeted_tool_grants(
            self._connection,
            session_id,
            interaction_id=interaction_id,
            limit=limit,
            ensure_ready=self._ensure_ready,
            get_codec=lambda: self.public_authority_alias_codec,
            decode_grant=_targeted_tool_grant_from_postgres_row,
            validate_use_counts=_validate_targeted_tool_use_counts,
        )

    async def load_targeted_tool_grant_state(
        self, session_id: str
    ) -> TargetedToolGrantStateSnapshot:
        return await tool_grants_ops.load_targeted_tool_grant_state(
            self._connection,
            session_id,
            ensure_ready=self._ensure_ready,
            get_codec=lambda: self.public_authority_alias_codec,
            decode_grant=_targeted_tool_grant_from_postgres_row,
            decode_use=_targeted_tool_use_from_postgres_row,
        )

    async def bind_targeted_tool_grant_use(
        self, request: TargetedToolUseRequest, *, observed_at: datetime
    ) -> TargetedToolUseResult:
        return await tool_grants_ops.bind_targeted_tool_grant_use(
            self._connection,
            request,
            observed_at=observed_at,
            ensure_ready=self._ensure_ready,
            get_codec=lambda: self.public_authority_alias_codec,
            append_events=self._append_events_with_cursor,
            append_event_once=self._append_event_once_with_cursor,
            decode_grant=_targeted_tool_grant_from_postgres_row,
            decode_use=_targeted_tool_use_from_postgres_row,
            validate_use_counts=_validate_targeted_tool_use_counts,
        )

    async def revoke_targeted_tool_grant(
        self,
        tool_ref: str,
        *,
        session_id: str,
        expected_run_epoch: int,
        reason: str,
        revoked_at: datetime,
    ) -> TargetedToolGrantRecord | None:
        return await tool_grants_ops.revoke_targeted_tool_grant(
            self._connection,
            tool_ref,
            session_id=session_id,
            expected_run_epoch=expected_run_epoch,
            reason=reason,
            revoked_at=revoked_at,
            ensure_ready=self._ensure_ready,
            closure_owners=self._closure_lineage_owners,
            get_codec=lambda: self.public_authority_alias_codec,
            append_events=self._append_events_with_cursor,
            decode_grant=_targeted_tool_grant_from_postgres_row,
            validate_use_counts=_validate_targeted_tool_use_counts,
        )

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
        return await tool_grants_ops.reconstruct_targeted_tool_grants(
            self._connection,
            session_id,
            expected_run_epoch=expected_run_epoch,
            interaction_id=interaction_id,
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
            ensure_ready=self._ensure_ready,
            append_event_once=self._append_event_once_with_cursor,
            decode_grant=_targeted_tool_grant_from_postgres_row,
            validate_use_counts=_validate_targeted_tool_use_counts,
        )

    async def _ensure_ready(self) -> None:
        await super()._ensure_ready()
        if not self._public_authority_aliases_reconciled:
            async with self._public_authority_alias_backfill_lock:
                if not self._public_authority_aliases_reconciled:
                    await public_authority_ops.reconcile_keys(
                        self._connection,
                        codec=self.public_authority_alias_codec,
                        read_only=self._read_only,
                        acquire_schema_lock=postgres_base._acquire_schema_transaction_lock,
                    )
                    self._public_authority_aliases_reconciled = True
        await public_authority_ops.require_current_configuration(
            self._connection, codec=self.public_authority_alias_codec
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
        return await creation_ops.create(
            self._connection,
            request,
            identity=identity,
            interaction_started_event=interaction_started_event,
            interaction_source_messages=interaction_source_messages,
            checkpoint_transform=checkpoint_transform,
            result_checkpoint_transform=result_checkpoint_transform,
            operation_initializer=operation_initializer,
            participant_binding_factory=participant_binding_factory,
            participant_request_commitment=participant_request_commitment,
            participant_provenance=participant_provenance,
            recipient_selection=recipient_selection,
            creation_target=creation_target,
            recipient_receipt_validator=recipient_receipt_validator,
            ensure_ready=self._ensure_ready,
            store_now=self._session_store_now,
            lock_closure_lineage=self._lock_closure_lineage,
            closure_owners=self._closure_lineage_owners,
            require_available_closure_identity=self._require_available_closure_identity,
            require_external_creation=self._require_external_creation,
            load_for_key_share=self._load_for_key_share,
            load_checkpoint=self._load_checkpoint,
            upsert_checkpoint=self._upsert_checkpoint,
            register_event_authorities=partial(
                public_authority_ops.register_event_authorities,
                codec=self.public_authority_alias_codec,
            ),
            jsonb=Jsonb,
            unique_violation=UniqueViolation,
            foreign_key_violation=ForeignKeyViolation,
        )

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
            from cayu.sessions._access_rules import (
                _RECIPIENT_PROVENANCE_CAPABILITY,
            )

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
                        publication_ops.decode_record(row[0]),
                        storage_key=key,
                        session_id=session_id,
                        publication_id=publication_id,
                    )
                publication_frontier(pointer, tool_receipt)
                assert tool_receipt is not None
                await publication_ops.validate_material(cur, tool_receipt, lock_events=False)
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
                current_transcript_cursor=await transcript_ops.transcript_cursor(cur, session_id),
            )

    async def publish_context_view(self, manifest, *, publication_key):
        return await context_view_ops.publish_context_view(
            self._connection,
            manifest,
            publication_key=publication_key,
            ensure_ready=self._ensure_ready,
            load_session=self._load,
            jsonb=Jsonb,
            unique_violation=UniqueViolation,
        )

    async def lookup_context_view_publication(self, publication_key):
        return await context_view_ops.lookup_context_view_publication(
            self._connection, publication_key, ensure_ready=self._ensure_ready
        )

    async def select_context_view(self, request):
        return await self._select_context_view(request)

    async def _select_context_view(self, request, *, target=None):
        return await context_view_ops.select_context_view(
            self._connection,
            request,
            target=target,
            ensure_ready=self._ensure_ready,
            store_now=self._clock,
            load_for_update=self._load_for_update,
            jsonb=Jsonb,
        )

    async def lookup_context_view_selection(self, selection_key):
        return await context_view_ops.lookup_context_view_selection(
            self._connection, selection_key, ensure_ready=self._ensure_ready
        )

    async def transition_context_view_ownership(self, request):
        return await self._transition_context_view_ownership(request)

    async def _transition_context_view_ownership(self, request, *, target=None):
        return await context_view_ops.transition_context_view_ownership(
            self._connection,
            request,
            target=target,
            ensure_ready=self._ensure_ready,
            store_now=self._clock,
            jsonb=Jsonb,
            unique_violation=UniqueViolation,
        )

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
        return await context_view_ops.read_context_view_lifecycle_events(
            self._connection,
            view_id,
            limit=limit,
            owner_participant=owner_participant,
            ensure_ready=self._ensure_ready,
            jsonb=Jsonb,
        )

    async def read_context_view(self, view_id: str, *, source_session_id: str):
        return await context_view_ops.read_context_view(
            self._connection,
            view_id,
            source_session_id=source_session_id,
            ensure_ready=self._ensure_ready,
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
        return await creation_ops.create_fork(
            self._connection,
            source_session_id=source_session_id,
            fork=fork,
            source_statuses=source_statuses,
            transcript_cursor=transcript_cursor,
            checkpoint_transform=checkpoint_transform,
            system_prompt_replacement=system_prompt_replacement,
            expected_source_run_epoch=expected_source_run_epoch,
            transcript_validator=transcript_validator,
            profile_relationship=profile_relationship,
            events=events,
            checkpoint_authority_decoder=checkpoint_authority_decoder,
            operation_initializer=operation_initializer,
            ensure_ready=self._ensure_ready,
            store_now=self._session_store_now,
            lock_closure_lineage=self._lock_closure_lineage,
            closure_owners=self._closure_lineage_owners,
            require_available_closure_identity=self._require_available_closure_identity,
            load_checkpoint=self._load_checkpoint,
            register_event_authorities=partial(
                public_authority_ops.register_event_authorities,
                codec=self.public_authority_alias_codec,
            ),
            register_authorities=partial(
                public_authority_ops.register_authorities, codec=self.public_authority_alias_codec
            ),
            load_session=self._load,
            load_for_update=self._load_for_update,
            supports_model_failover=self._supports_model_failover_stage_protocol,
            unique_violation=UniqueViolation,
            checkpoint_values=_checkpoint_row_values,
        )

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
        return await memory_evidence_ops.create_recall_receipt(
            self._connection,
            receipt,
            ensure_ready=self._ensure_ready,
            lock_closure_lineage=self._lock_closure_lineage,
            closure_owners=self._closure_lineage_owners,
        )

    async def load_recall_receipt(self, session_id: str, receipt_id: str) -> RecallReceipt | None:
        return await memory_evidence_ops.load_recall_receipt(
            self._connection, session_id, receipt_id, ensure_ready=self._ensure_ready
        )

    async def list_recall_receipts(self, query: RecallEvidenceQuery) -> RecallReceiptPage:
        return await memory_evidence_ops.list_recall_receipts(
            self._connection, query, ensure_ready=self._ensure_ready
        )

    async def create_context_exposure(
        self, exposure: ContextExposure, item_exposures: tuple[RecallItemExposure, ...] = ()
    ) -> ContextExposure:
        return await memory_evidence_ops.create_context_exposure(
            self._connection,
            exposure,
            item_exposures,
            ensure_ready=self._ensure_ready,
            lock_closure_lineage=self._lock_closure_lineage,
            closure_owners=self._closure_lineage_owners,
        )

    async def load_context_exposure(
        self, session_id: str, exposure_id: str
    ) -> ContextExposure | None:
        return await memory_evidence_ops.load_context_exposure(
            self._connection, session_id, exposure_id, ensure_ready=self._ensure_ready
        )

    async def load_recall_item_exposures(
        self, session_id: str, exposure_id: str
    ) -> tuple[RecallItemExposure, ...]:
        return await memory_evidence_ops.load_recall_item_exposures(
            self._connection, session_id, exposure_id, ensure_ready=self._ensure_ready
        )

    async def list_context_exposures(self, query: RecallEvidenceQuery) -> ContextExposurePage:
        return await memory_evidence_ops.list_context_exposures(
            self._connection, query, ensure_ready=self._ensure_ready
        )

    async def transition_context_exposure(
        self, session_id: str, exposure_id: str, request: ContextExposureTransitionRequest
    ) -> ContextExposure:
        return await memory_evidence_ops.transition_context_exposure(
            self._connection,
            session_id,
            exposure_id,
            request,
            ensure_ready=self._ensure_ready,
            lock_closure_lineage=self._lock_closure_lineage,
            closure_owners=self._closure_lineage_owners,
        )

    async def inspect_identity(self, session_id: str) -> SessionInspectionIdentity:
        return await session_queries.inspect_identity(self._ready_connection, session_id)

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

    async def claim_session_closure_progress(
        self, progress: dict[str, Any], *, finished_only: bool = False
    ) -> None:
        progress = deepcopy(progress)
        targets = _closure_progress_targets(progress)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await self._lock_closure_lineage(cur)
            if finished_only:
                root = await self._load_for_update(cur, progress["root_session_id"])
                if root is None or root.status not in {
                    SessionStatus.COMPLETED,
                    SessionStatus.FAILED,
                }:
                    raise ValueError("Only completed or failed sessions can be deleted.")
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
        from cayu.sessions import _session_continuation_store as continuations
        from cayu.sessions._closure_records import require_terminal_protected_effect

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
                    deleted = await self._delete_session_in_transaction(
                        cur,
                        session_id,
                        closure_receipt=closure_receipt,
                        _access_bounds=_access_bounds,
                    )
                if not deleted:
                    await conn.rollback()
                    return
                await conn.commit()
            except Exception:
                await conn.rollback()
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
            retention.PostgresSessionRetentionBackend(self),
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
            retention.PostgresSessionRetentionBackend(self),
            session_ids,
            mode=RetentionMode.DELETE if mode is None else RetentionMode(mode),
            store_guards=include_store_guards,
        )

    async def retention_artifact_references(self) -> frozenset[str]:
        from cayu.storage import _session_retention as retention

        return await retention.session_artifact_references(
            retention.PostgresSessionRetentionBackend(self)
        )

    async def inspect_storage_growth(self) -> StorageGrowthReport:
        """Read-only size, age and retention-history summary of this store."""

        from cayu.storage import _session_retention as retention

        return await retention.inspect_storage_growth(
            retention.PostgresSessionRetentionBackend(self)
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
            retention.PostgresSessionRetentionBackend(self),
            limit=limit,
            item_id=None if item_id is None else require_clean_nonblank(item_id, "item_id"),
            store_kind=store_kind,
        )

    async def load_retention_audit(self, audit_id: str) -> RetentionAuditRecord | None:
        from cayu.storage import _session_retention as retention

        return await retention.load_retention_audit(
            retention.PostgresSessionRetentionBackend(self),
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
            retention.PostgresSessionRetentionBackend(self),
            store_kind=require_clean_nonblank(store_kind, "store_kind"),
            mode=mode,
            policy=dict(policy),
            started_at=started_at,
        )

    async def record_retention_entry(self, audit_id: str, entry: RetentionAuditEntry) -> None:
        from cayu.storage import _session_retention as retention

        await retention.record_retention_entry(
            retention.PostgresSessionRetentionBackend(self), audit_id, entry
        )

    async def complete_retention_audit(
        self, audit_id: str, *, completed_at: datetime, summary: Mapping[str, Any]
    ) -> None:
        from cayu.storage import _session_retention as retention

        await retention.complete_retention_audit(
            retention.PostgresSessionRetentionBackend(self),
            audit_id,
            completed_at=completed_at,
            summary=dict(summary),
        )

    async def _require_session_deletion_admission(
        self,
        cur: Any,
        session: Session,
        *,
        closure_receipt: dict[str, Any] | None = None,
    ) -> None:
        """Check deletion guards that do not depend on remaining child edges."""

        session_id = session.id
        for owner in await self._closure_lineage_owners(cur, (session_id,)):
            _check_closure_lineage_owner(owner, (session_id,), closure_receipt)
        await cur.execute(
            "SELECT 1 FROM cayu_session_closure_progress AS p "
            "JOIN cayu_sessions AS child ON child.id = p.root_session_id "
            "WHERE child.parent_session_id = %s LIMIT 1",
            (session_id,),
        )
        if await cur.fetchone() is not None:
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

    async def _delete_session_in_transaction(
        self,
        cur: Any,
        session_id: str,
        *,
        closure_receipt: dict[str, Any] | None = None,
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> bool:
        """Apply delete_session inside the caller's transaction.

        Returns ``False`` when the session does not exist. Guards raise
        ``ValueError``; the caller owns commit and rollback.
        """

        await self._lock_closure_lineage(cur)
        await cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"context-view-session:{session_id}",),
        )
        session = await self._load_for_update(cur, session_id)
        if _access_bounds is not None:
            _access_bounds.require_action(session, "delete")
        if session is None:
            return False
        await self._require_session_deletion_admission(
            cur, session, closure_receipt=closure_receipt
        )
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
            raise ValueError(_durable_subagent_parent_delete_block_reason(durable_child[0]))
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
    async def set_summary(
        self,
        session_id: str,
        summary: str,
        *,
        tone: SummaryTone = "neutral",
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> Session:
        return await self._update_display_or_user_metadata(
            session_id,
            None,
            summary_patch=summary_metadata(summary, tone),
            _access_bounds=_access_bounds,
        )

    async def _access_set_summary(self, bounds, session_id, summary, tone):
        return await self.set_summary(session_id, summary, tone=tone, _access_bounds=bounds)

    @runtime_session_mutation
    async def update_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any],
        *,
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> Session:
        return await self._update_display_or_user_metadata(
            session_id,
            metadata,
            _access_bounds=_access_bounds,
        )

    async def _update_display_or_user_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any] | None,
        *,
        summary_patch: dict[str, Any] | None = None,
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        user_metadata = None if metadata is None else copy_session_user_metadata(metadata)
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
                    new_metadata = update_session_annotations(
                        pg_support._json_obj(row[1]), user_metadata, summary_patch
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

    replaces_execution_deadline_in_admitted_run: ClassVar[bool] = True

    async def replace_execution_deadline(
        self,
        session_id: str,
        deadline: ExecutionDeadline,
        *,
        expected_session_instance_id: str | None,
        expected_run_epoch: int,
        renewal: ExecutionDeadlineRenewal | None = None,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        deadline, renewal = deadline_replacement.require_replacement_deadline(deadline, renewal)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT run_epoch, instance_id, status, metadata FROM cayu_sessions "
                        "WHERE id = %s FOR UPDATE",
                        (session_id,),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        raise KeyError(f"Session not found: {session_id}")
                    _assert_session_run_epoch_value(session_id, row[0])
                    deadline_replacement.require_execution_deadline_replacement(
                        session_id,
                        status=SessionStatus(row[2]),
                        session_instance_id=row[1],
                        run_epoch=row[0],
                        expected_session_instance_id=expected_session_instance_id,
                        expected_run_epoch=expected_run_epoch,
                    )
                    new_metadata = deadline_replacement.metadata_with_execution_deadline(
                        pg_support._json_obj(row[3]), deadline, renewal
                    )
                    await cur.execute(
                        "UPDATE cayu_sessions SET metadata = %s, updated_at = %s WHERE id = %s",
                        (
                            pg_support._dumps(new_metadata),
                            await self._session_store_now(cur),
                            session_id,
                        ),
                    )
                    loaded = await self._load(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    from cayu.sessions._invocation_lifecycle import (
                        require_invocation_lifecycle_release_capacity,
                    )

                    require_invocation_lifecycle_release_capacity(
                        await self._load_checkpoint(cur, session_id), loaded
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
            interaction_id=(None if admission is None else admission.interaction_id),
            interaction_is_new=(admission is not None and admission.started_event is not None),
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
                                model_stage_ops.decode_stage_record(active_row[0]),
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
                        and admission.defer_source
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
                                list(admission.source_messages),
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
                            await transcript_ops.transcript_cursor(cur, session_id),
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
                    if admission is not None and admission.started_event is not None:
                        admission_events.append(admission.started_event)

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
                            row[0]: model_stage_ops.decode_stage_record(row[1])
                            for row in await cur.fetchall()
                        }
                        transformed_checkpoint = _model_failover_checkpoint_after_profile_admission(
                            source_session=loaded,
                            admitted_session=transitioned,
                            source_checkpoint=current_checkpoint,
                            admitted_checkpoint=transformed_checkpoint,
                            candidate_profile=prepared_execution_profile,
                            records=route_records,
                            transcript_cursor=await transcript_ops.transcript_cursor(
                                cur, session_id
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
                        interaction_id = admission.interaction_id
                        source_messages = list(admission.source_messages)
                        defer_source = admission.defer_source
                        await public_authority_ops.register_authorities(
                            cur,
                            session_id,
                            interaction_ids=(interaction_id,),
                            codec=self.public_authority_alias_codec,
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
                            await event_delivery_ops.enqueue_persisted_event_side_effects(
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
                                        transcript_ops.index_document(session_id, message),
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
                            *(
                                [admission.started_event.id]
                                if admission.started_event is not None
                                else []
                            ),
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
                    await public_authority_ops.register_event_authorities(
                        cur, session_id, [copied_event], codec=self.public_authority_alias_codec
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
                    await event_delivery_ops.enqueue_persisted_event_side_effects(
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
                        )
                        from cayu.sessions._model_stage_rules import (
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
                        active_record = model_stage_ops.decode_stage_record(active_row[0])
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
                            row[0]: model_stage_ops.decode_stage_record(row[1])
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
                            row[0]: model_stage_ops.decode_stage_record(row[1])
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
                    await public_authority_ops.register_event_authorities(
                        cur, session_id, committed_events, codec=self.public_authority_alias_codec
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
                    await event_delivery_ops.enqueue_persisted_event_side_effects(
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
        self, *, reservation_id: str, publication_session_id: str, publication_id: str
    ) -> None:
        return await event_publication_ops.claim_budget_reservation_identity(
            self._connection,
            reservation_id=reservation_id,
            publication_session_id=publication_session_id,
            publication_id=publication_id,
            ensure_ready=self._ensure_ready,
            closure_owners=self._closure_lineage_owners,
            raise_write_conflict=_raise_session_write_conflict,
        )

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
        await public_authority_ops.register_event_authorities(
            cur, session_id, copied_events, codec=self.public_authority_alias_codec
        )
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
        await event_delivery_ops.enqueue_persisted_event_side_effects(
            cur, session_id, copied_events
        )

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
        return await event_publication_ops.append_events(
            self._connection,
            session_id,
            events,
            ensure_ready=self._ensure_ready,
            load_session=self._load_for_update,
            closure_owners=self._closure_lineage_owners,
            append_events=self._append_events_with_cursor,
            first_existing_event_id=self._first_existing_event_id,
            unique_violation=UniqueViolation,
        )

    async def append_tool_effect_conflict(self, request: object) -> Event:
        return await event_publication_ops.append_tool_effect_conflict(
            self._connection,
            request,
            ensure_ready=self._ensure_ready,
            load_session=self._load_for_update,
            store_now=self._session_store_now,
            closure_owners=self._closure_lineage_owners,
            lock_closure=self._lock_closure_lineage,
            insert_events=self._insert_event_rows_with_cursor,
        )

    async def append_workflow_step_started(
        self, session_id: str, event: Event, *, workflow_name: str, attempt_id: str
    ) -> bool:
        return await event_publication_ops.append_workflow_step_started(
            self._connection,
            session_id,
            event,
            workflow_name=workflow_name,
            attempt_id=attempt_id,
            ensure_ready=self._ensure_ready,
            load_session=self._load_for_update,
            store_now=self._session_store_now,
            closure_owners=self._closure_lineage_owners,
            lock_closure=self._lock_closure_lineage,
            register_event_authorities=partial(
                public_authority_ops.register_event_authorities,
                codec=self.public_authority_alias_codec,
            ),
            first_existing_event_id=self._first_existing_event_id,
            raise_write_conflict=_raise_session_write_conflict,
            unique_violation=UniqueViolation,
        )

    async def load_mcp_manifest_baselines(
        self, history_keys: tuple[str, ...]
    ) -> McpManifestBaselineLoadResult:
        return await event_publication_ops.load_mcp_manifest_baselines(
            self._connection, history_keys, ensure_ready=self._ensure_ready
        )

    async def compare_and_publish_mcp_manifest_checks(
        self,
        session_id: str,
        *,
        expected_generations: dict[str, int | None],
        baseline_updates: dict[str, McpManifestBaseline],
        events: list[Event],
    ) -> McpManifestPublicationResult:
        return await event_publication_ops.compare_and_publish_mcp_manifest_checks(
            self._connection,
            session_id,
            expected_generations=expected_generations,
            baseline_updates=baseline_updates,
            events=events,
            ensure_ready=self._ensure_ready,
            load_session=self._load_for_update,
            store_now=self._session_store_now,
            closure_owners=self._closure_lineage_owners,
            lock_closure=self._lock_closure_lineage,
            register_event_authorities=partial(
                public_authority_ops.register_event_authorities,
                codec=self.public_authority_alias_codec,
            ),
            first_existing_event_id=self._first_existing_event_id,
            raise_write_conflict=_raise_session_write_conflict,
            unique_violation=UniqueViolation,
        )

    async def claim_first_persisted_event_side_effect(
        self, expected: PersistedEventSideEffectDelivery
    ) -> PersistedEventSideEffectClaim | None:
        return await event_delivery_ops.claim_first_persisted_event_side_effect(
            self._ready_connection, expected, lock_closure_lineage=self._lock_closure_lineage
        )

    async def claim_persisted_event_side_effect(
        self,
        *,
        session_id: str | None = None,
        event_id: str | None = None,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectClaim | None:
        return await event_delivery_ops.claim_persisted_event_side_effect(
            self._ready_connection,
            session_id=session_id,
            event_id=event_id,
            lease_seconds=lease_seconds,
            lock_closure_lineage=self._lock_closure_lineage,
        )

    async def get_persisted_event_side_effect_delivery(
        self, *, session_id: str, event_id: str
    ) -> PersistedEventSideEffectDelivery | None:
        return await event_delivery_ops.get_persisted_event_side_effect_delivery(
            self._ready_connection, session_id=session_id, event_id=event_id
        )

    async def retire_failed_first_event_delivery(
        self, expected: PersistedEventSideEffectDelivery
    ) -> PersistedEventSideEffectDelivery | None:
        return await event_delivery_ops.retire_failed_first_event_delivery(
            self._ready_connection, expected, store_now=self._session_store_now
        )

    async def mark_persisted_event_side_effect_delivered(
        self, claim: PersistedEventSideEffectClaim
    ) -> PersistedEventSideEffectDelivery:
        return await event_delivery_ops.mark_persisted_event_side_effect_delivered(
            self._ready_connection, claim
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
            self._ready_connection,
            claim,
            error=error,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )

    async def defer_persisted_event_side_effect(
        self, claim: PersistedEventSideEffectClaim
    ) -> PersistedEventSideEffectDelivery:
        return await event_delivery_ops.defer_persisted_event_side_effect(
            self._ready_connection, claim
        )

    async def renew_persisted_event_side_effect(
        self, claim: PersistedEventSideEffectClaim, *, lease_seconds: float = 300.0
    ) -> PersistedEventSideEffectDelivery:
        return await event_delivery_ops.renew_persisted_event_side_effect(
            self._ready_connection, claim, lease_seconds=lease_seconds
        )

    async def get_persisted_event_side_effect_health(self) -> PersistedEventSideEffectHealth:
        return await event_delivery_ops.get_persisted_event_side_effect_health(
            self._ready_connection
        )

    async def query_persisted_event_side_effect_deliveries(
        self, query: PersistedEventSideEffectQuery
    ) -> PersistedEventSideEffectPage:
        return await event_delivery_ops.query_persisted_event_side_effect_deliveries(
            self._ready_connection, query
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
            self._ready_connection,
            statuses=statuses,
            claimable_only=claimable_only,
            after_sequence=after_sequence,
            limit=limit,
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
        return await message_ops.snapshot_session_message_source(
            self._connection,
            session_id,
            include_transcript_digest=include_transcript_digest,
            include_checkpoint_digest=include_checkpoint_digest,
            expected_authorized_session_instance_id=expected_authorized_session_instance_id,
            ensure_ready=self._ensure_ready,
            load_checkpoint=self._load_checkpoint,
            load_labels=self._load_labels,
        )

    @runtime_session_query
    async def inspect_session_messages(
        self,
        query: SessionMessageQuery,
        *,
        expected_authorized_session_instance_id: str | None = None,
    ) -> SessionMessageInspection:
        return await message_ops.inspect_session_messages(
            self._connection,
            query,
            expected_authorized_session_instance_id=expected_authorized_session_instance_id,
            ensure_ready=self._ensure_ready,
            load_labels=self._load_labels,
        )

    @runtime_session_query
    async def apply_session_message_action(
        self, request: SessionMessageActionRequest
    ) -> SessionMessageActionResult:
        return await message_ops.apply_session_message_action(
            self._connection,
            request,
            ensure_ready=self._ensure_ready,
            load_session=self._load_for_update,
            store_now=self._session_store_now,
            closure_owners=self._closure_lineage_owners,
            append_events=self._append_events_with_cursor,
        )

    @runtime_session_query
    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
        *,
        expected_authorized_target_instance_id: str | None = None,
    ) -> EnqueueSessionMessageResult:
        return await message_ops.enqueue_session_message(
            self._connection,
            request,
            expected_authorized_target_instance_id=expected_authorized_target_instance_id,
            ensure_ready=self._ensure_ready,
            load_session=self._load_for_update,
            store_now=self._session_store_now,
            closure_owners=self._closure_lineage_owners,
            load_checkpoint=self._load_checkpoint,
            register_event_authorities=partial(
                public_authority_ops.register_event_authorities,
                codec=self.public_authority_alias_codec,
            ),
        )

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
        return await message_ops.deliver_queued_session_messages(
            self._connection,
            session_id,
            include_on_idle=include_on_idle,
            reject_only=reject_only,
            delivery_id=delivery_id,
            eligible_through=eligible_through,
            limit=limit,
            interaction_id=interaction_id,
            interaction_started_event=interaction_started_event,
            profile_handoff=profile_handoff,
            ensure_ready=self._ensure_ready,
            load_session=self._load_for_update,
            store_now=self._session_store_now,
            load_checkpoint=self._load_checkpoint,
            register_event_authorities=partial(
                public_authority_ops.register_event_authorities,
                codec=self.public_authority_alias_codec,
            ),
            register_authorities=partial(
                public_authority_ops.register_authorities, codec=self.public_authority_alias_codec
            ),
            reject_steering=self._reject_new_work_after_steering,
            upsert_checkpoint=self._upsert_checkpoint,
            decode_stage_record=model_stage_ops.decode_stage_record,
        )

    async def repair_queued_interaction_profile_handoff(
        self,
        session_id: str,
        *,
        interaction_started_event: Event,
        profile_handoff: QueuedInteractionProfileHandoff,
    ) -> ActiveInvocationExecutionProfile:
        return await message_ops.repair_queued_interaction_profile_handoff(
            self._connection,
            session_id,
            interaction_started_event=interaction_started_event,
            profile_handoff=profile_handoff,
            ensure_ready=self._ensure_ready,
            load_session=self._load_for_update,
            store_now=self._session_store_now,
            load_checkpoint=self._load_checkpoint,
            upsert_checkpoint=self._upsert_checkpoint,
            decode_stage_record=model_stage_ops.decode_stage_record,
        )

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
        return await publication_ops.load_receipt_record(
            session_id,
            storage_key,
            publication_id,
            connect=self._connection,
            ensure_ready=self._ensure_ready,
        )

    async def _load_model_completion_stage_records(
        self,
        session_id: str,
        preparation_storage_key: str,
        terminal_storage_key: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        return await model_stage_ops.load_stage_records(
            session_id,
            preparation_storage_key,
            terminal_storage_key,
            connect=self._connection,
            ensure_ready=self._ensure_ready,
        )

    async def _load_model_completion_stage_settlement_record(
        self,
        session_id: str,
        settlement_storage_key: str,
    ) -> dict[str, Any] | None:
        return await model_stage_ops.load_settlement_record(
            session_id,
            settlement_storage_key,
            connect=self._connection,
            ensure_ready=self._ensure_ready,
        )

    async def _load_model_completion_stage_dispatch_record(
        self,
        session_id: str,
        dispatch_storage_key: str,
    ) -> dict[str, Any] | None:
        return await model_stage_ops.load_dispatch_record(
            session_id,
            dispatch_storage_key,
            connect=self._connection,
            ensure_ready=self._ensure_ready,
        )

    async def _load_active_model_completion_stage_records(
        self,
        session_id: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        return await model_stage_ops.load_active_stage_records(
            session_id, connect=self._connection, ensure_ready=self._ensure_ready
        )

    async def _mark_model_completion_stage_dispatched_atomic(
        self,
        session_id: str,
        *,
        stage: ModelCompletionStage,
        consume_child_session_notifications: bool,
    ) -> ModelCompletionStageDispatch:
        return await model_stage_ops.mark_dispatched(
            session_id,
            stage=stage,
            consume_child_session_notifications=consume_child_session_notifications,
            connect=self._connection,
            ensure_ready=self._ensure_ready,
            load_session=self._load,
            load_for_update=self._load_for_update,
            load_checkpoint=self._load_checkpoint,
        )

    async def _prepare_model_completion_stage_atomic(
        self,
        prepared: _PreparedModelCompletionStage,
    ) -> ModelCompletionStageResult:
        return await model_stage_ops.prepare(
            prepared,
            connect=self._connection,
            ensure_ready=self._ensure_ready,
            load_for_update=self._load_for_update,
            load_checkpoint=self._load_checkpoint,
            append_events=self._append_events_with_cursor,
            reject_steering=self._reject_new_work_after_steering,
            upsert_checkpoint=self._upsert_checkpoint,
        )

    async def _complete_model_completion_stage_atomic(
        self,
        prepared: _PreparedModelCompletionStageTerminal,
    ) -> ModelCompletionStageResult:
        return await model_stage_ops.complete(
            prepared,
            connect=self._connection,
            ensure_ready=self._ensure_ready,
            load_for_update=self._load_for_update,
            load_checkpoint=self._load_checkpoint,
            store_now=self._session_store_now,
        )

    async def _abandon_model_completion_stage_atomic(
        self,
        prepared: _PreparedModelCompletionStageAbandonment,
    ) -> ModelCompletionStageAbandonmentResult:
        return await model_stage_ops.abandon(
            prepared,
            connect=self._connection,
            ensure_ready=self._ensure_ready,
            load_for_update=self._load_for_update,
        )

    async def _promote_model_completion_stage_atomic(
        self,
        *,
        session_id: str,
        stage_id: str,
        preparation_storage_key: str,
        terminal_storage_key: str,
        expected_run_epoch: int,
    ) -> RuntimePublicationResult:
        return await model_stage_ops.promote(
            session_id=session_id,
            stage_id=stage_id,
            preparation_storage_key=preparation_storage_key,
            terminal_storage_key=terminal_storage_key,
            expected_run_epoch=expected_run_epoch,
            load_stage=self.load_model_completion_stage,
            publish=self._publish_runtime_publication_atomic,
        )

    async def _publish_runtime_publication_atomic(
        self,
        prepared: _PreparedRuntimePublication,
        *,
        _model_completion_stage: _ModelCompletionStagePromotionContext | None = None,
    ) -> RuntimePublicationResult:
        return await publication_ops.publish_runtime(
            prepared,
            _model_completion_stage=_model_completion_stage,
            connect=self._connection,
            ensure_ready=self._ensure_ready,
            load_for_update=self._load_for_update,
            load_checkpoint=self._load_checkpoint,
            closure_owners=self._closure_lineage_owners,
            public_authority_alias_codec=self.public_authority_alias_codec,
            checkpoint_row_values=_checkpoint_row_values,
            unique_violation=UniqueViolation,
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
        return await publication_ops.publish_checkpoint_events(
            session_id,
            checkpoint_transform=checkpoint_transform,
            store_time_checkpoint_transform=store_time_checkpoint_transform,
            operation_idempotency_key=operation_idempotency_key,
            operation_transform=operation_transform,
            store_time_operation_transform=store_time_operation_transform,
            operation_commit_guard=operation_commit_guard,
            operation_commit_time_guard=operation_commit_time_guard,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
            context_view_compaction_cursor=context_view_compaction_cursor,
            preserve_completion_result_publications=preserve_completion_result_publications,
            unique_violation=UniqueViolation,
            connect=self._connection,
            ensure_ready=self._ensure_ready,
            load_for_update=self._load_for_update,
            load_checkpoint=self._load_checkpoint,
            store_now=self._session_store_now,
            closure_owners=self._closure_lineage_owners,
            public_authority_alias_codec=self.public_authority_alias_codec,
            first_existing_event_id=self._first_existing_event_id,
            publish_budget_identities=self._publish_budget_reservation_identities,
            record_terminal_receipts=self._record_invocation_terminal_event_receipts,
            upsert_checkpoint=self._upsert_checkpoint,
            validate_compaction=self._validate_context_view_compaction,
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
        from cayu.sessions._closure_records import (
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
                memory_evidence_ops.recall_receipt_from_row,
            ),
            (
                "context_exposures",
                "SELECT exposure_id, session_id, interaction_id, model_step_id, model_attempt_id, "
                "provider_attempt_id, state, state_revision, created_at, updated_at, exposure_json, "
                "document_bytes FROM cayu_context_exposures WHERE session_id = %s "
                "ORDER BY created_at, exposure_id LIMIT %s",
                memory_evidence_ops.context_exposure_from_row,
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
                f"SELECT {message_ops.QUEUE_COLUMNS} FROM cayu_session_message_queue WHERE session_id = %s ORDER BY ordering_key LIMIT %s",
                lambda row: {
                    "message": message_ops.message_from_row(row),
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
                event_delivery_ops.delivery_from_row,
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
            cursor = await transcript_ops.transcript_cursor(cur, session_id)
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
        return await session_queries.load_events(
            self._ready_connection, session_id, load_session=self._load
        )

    async def load_user_input_supersession_events(
        self,
        session_id: str,
        input_id: str,
    ) -> list[Event]:
        return await session_queries.load_user_input_supersession_events(
            self._ready_connection, session_id, input_id
        )

    async def load_tool_round_lifecycle_events(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
    ) -> list[Event]:
        return await session_queries.load_tool_round_lifecycle_events(
            self._ready_connection, session_id, tool_call_ids
        )

    async def load_tool_round_lifecycle_events_for_round(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
        *,
        tool_round_identity: ToolRoundIdentity,
    ) -> list[Event]:
        return await session_queries.load_tool_round_lifecycle_events_for_round(
            self._ready_connection,
            session_id,
            tool_call_ids,
            tool_round_identity=tool_round_identity,
        )

    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        return await session_queries.query_events(self._ready_connection, query)

    async def event_exists(self, query: EventQuery) -> bool:
        return await session_queries.event_exists(self._ready_connection, query)

    async def read_store_identity(self) -> str:
        identity = None
        async with self._ready_connection() as conn, conn.cursor() as cur:
            await cur.execute("SELECT to_regclass('cayu_store_identity')")
            registered = await cur.fetchone()
            if registered is not None and registered[0] is not None:
                await cur.execute("SELECT store_id FROM cayu_store_identity WHERE singleton = 1")
                row = await cur.fetchone()
                identity = None if row is None else str(row[0])
        if identity is None:
            raise NotImplementedError(
                "This database has no store identity; run `cayu storage migrate`."
            )
        return identity

    async def read_usage_accounting(
        self, query: EventQuery, *, by_session: bool = False, by_identity: bool = False
    ) -> UsageAccountingSnapshot:
        return await session_queries.read_usage_accounting(
            self._ready_connection,
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
        fork_lineage_owner: str | None = None,
        without_reservation_for: str | None = None,
    ) -> CostAccountingSnapshot:
        return await session_queries.read_cost_accounting(
            self._ready_connection,
            query,
            pricing,
            currency=currency,
            details=details,
            by_session=by_session,
            additional_events=additional_events,
            max_detail_bytes=max_detail_bytes,
            previous=previous,
            cost_authority=self._cost_accounting_authority,
            fork_lineage_owner=fork_lineage_owner,
            without_reservation_for=without_reservation_for,
        )

    async def fold_event_snapshot(
        self,
        query: EventQuery,
        consume: EventSnapshotConsumer,
        *,
        max_bytes: int,
        max_pages: int | None = None,
    ) -> int:
        return await session_queries.fold_event_snapshot(
            self._ready_connection,
            query,
            consume,
            max_bytes=max_bytes,
            max_pages=max_pages,
        )

    async def query_events_bounded(
        self,
        query: EventQuery,
        *,
        max_bytes: int,
    ) -> list[EventRecord]:
        return await session_queries.query_events_bounded(
            self._ready_connection, query, max_bytes=max_bytes
        )

    async def load_terminal_session_evidence(
        self,
        session_id: str,
        *,
        limits: TerminalSessionEvidenceLimits | None = None,
    ) -> TerminalSessionEvidence:
        return await terminal_evidence_ops.load_terminal_session_evidence(
            self._connection,
            session_id,
            limits=limits,
            ensure_ready=self._ensure_ready,
            load_session=self._load,
        )

    async def load_runner_owned_interrupted_evidence(
        self,
        session_id: str,
        *,
        observed_events: tuple[RunnerObservedEventIdentity, ...] | None = None,
        expected_parent_session_id: str | None = None,
        limits: TerminalSessionEvidenceLimits | None = None,
    ) -> TerminalSessionEvidence:
        return await terminal_evidence_ops.load_runner_owned_interrupted_evidence(
            self._connection,
            session_id,
            observed_events=observed_events,
            expected_parent_session_id=expected_parent_session_id,
            limits=limits,
            ensure_ready=self._ensure_ready,
            load_session=self._load,
        )

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
        return await terminal_evidence_ops.export_terminal_session_evidence(
            self._connection,
            session_id,
            spool=spool,
            ensure_ready=self._ensure_ready,
            load_session=self._load,
        )

    async def query_latest_interaction_events(
        self,
        session_id: str,
        *,
        before_sequence: int | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        return await session_queries.query_latest_interaction_events(
            self._ready_connection, session_id, before_sequence=before_sequence, limit=limit
        )

    async def summarize_events(self, session_id: str) -> EventSummary:
        return await session_queries.summarize_events(
            self._ready_connection, session_id, load_session=self._load
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
                terminal_event=session_queries.event_record_from_row(terminal_row),
                latest_retry_event=session_queries.event_record_from_row(retry_row),
            )

    async def list_sessions(self, query: SessionQuery | None = None) -> SessionListResult:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        if access_bounds is not None:
            return await self._access_list_sessions(access_bounds, copy_session_query(query))
        return await session_queries.list_sessions(
            self._ready_connection, query, pending_interruption_cascade_only=False
        )

    async def query_session_topology(
        self,
        query: SessionTopologyQuery,
    ) -> SessionTopologyStoreResult:
        return await session_queries.query_session_topology(self._ready_connection, query)

    async def query_session_lineage(
        self,
        query: SessionLineageQuery,
    ) -> SessionLineageResult:
        return await session_queries.query_session_lineage(self._ready_connection, query)

    async def query_child_session_lifecycle(
        self,
        query: ChildSessionLifecycleQuery,
    ) -> ChildSessionLifecyclePage:
        return await session_queries.query_child_session_lifecycle(
            self._ready_connection, query, load_session=self._load
        )

    async def aggregate_operational_snapshot(
        self,
        filters: SessionAggregateFilter | None = None,
    ) -> SessionOperationalSnapshot:
        filters = copy_session_aggregate_filter(filters)
        plan = session_store_sql.build_session_query_sql(
            session_query_from_aggregate_filter(filters),
            dialect=session_queries.SQL_DIALECT,
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
        return await session_queries.aggregate_usage(self._ready_connection, query)

    async def list_sessions_with_pending_interruption_cascade(
        self,
        query: SessionQuery | None = None,
    ) -> SessionListResult:
        return await session_queries.list_sessions(
            self._ready_connection, query, pending_interruption_cascade_only=True
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
            "AS runtime_build_provenance, cayu_sessions.instance_id, "
            "cayu_sessions.metadata -> 'cayu:execution_deadline' AS execution_deadline, "
            "cayu_sessions.metadata ->> 'cayu:execution_deadline_origin' "
            "AS execution_deadline_origin"
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

            labels_by_session_id = await session_queries.load_session_labels_batch(
                cur, materializable_ids
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

    @runtime_session_query
    async def append_peer_content(
        self,
        request: PeerContentAppendRequest,
        *,
        qualify_target: Callable[[Session], None] | None = None,
        pending_transcript_cursor: int | None = None,
    ) -> PeerContentReceipt:
        return await peer_content_ops.append_peer_content(
            self._connection,
            request,
            qualify_target=qualify_target,
            pending_transcript_cursor=pending_transcript_cursor,
            ensure_ready=self._ensure_ready,
            load_for_update=self._load_for_update,
            session_store_now=self._session_store_now,
        )

    async def read_peer_content_attempt(
        self, request: PeerContentAppendRequest
    ) -> PeerContentReceipt | None:
        return await peer_content_ops.read_peer_content_attempt(
            self._connection, request, ensure_ready=self._ensure_ready
        )

    async def list_pending_peer_content(self, *, after_operation_key=None, limit=32):
        """Trusted receiving-owner discovery, independent of creation settlement."""
        return await peer_content_ops.list_pending_peer_content(
            self._connection,
            after_operation_key=after_operation_key,
            limit=limit,
            ensure_ready=self._ensure_ready,
        )

    async def read_peer_content(self, append_key: PeerAppendKey) -> PeerContentReceipt | None:
        return await peer_content_ops.read_peer_content(
            self._connection, append_key, ensure_ready=self._ensure_ready
        )

    async def record_peer_content_exposure(
        self, request: PeerContentExposureRequest
    ) -> PeerContentExposureReceipt:
        return await peer_content_ops.record_peer_content_exposure(
            self._connection, request, ensure_ready=self._ensure_ready
        )

    async def begin_peer_content_exposure(
        self, request: PeerContentExposureRequest
    ) -> PeerContentExposureReceipt:
        return await peer_content_ops.begin_peer_content_exposure(
            self._connection, request, ensure_ready=self._ensure_ready
        )

    async def read_peer_content_exposure(
        self, append_key: PeerAppendKey, exposure_id: str
    ) -> PeerContentExposureReceipt | None:
        return await peer_content_ops.read_peer_content_exposure(
            self._connection, append_key, exposure_id, ensure_ready=self._ensure_ready
        )

    async def exclude_peer_content(
        self, request: PeerContentAppendRequest, *, reason: str
    ) -> PeerContentReceipt:
        return await peer_content_ops.exclude_peer_content(
            self._connection, request, reason=reason, ensure_ready=self._ensure_ready
        )

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
            self._connection,
            session_id,
            expected_session_instance_id=expected_session_instance_id,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
            admit=admit,
            ensure_ready=self._ensure_ready,
            read_creation_decision=self.read_session_creation_decision,
        )

    async def append_transcript_messages(
        self,
        session_id: str,
        messages: list[Message],
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        return await transcript_ops.append_transcript_messages(
            self._ready_connection,
            session_id,
            messages,
            interaction_id=interaction_id,
            store_now=self._session_store_now,
            closure_owners=self._closure_lineage_owners,
            register_authorities=partial(
                public_authority_ops.register_authorities, codec=self.public_authority_alias_codec
            ),
            touch_activity=_touch_session_activity,
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
            self._ready_connection,
            session_id,
            expected_messages,
            replacement_messages,
            interaction_id=interaction_id,
            checkpoint_transform=checkpoint_transform,
            runtime_suffix_count=runtime_suffix_count,
            load_for_update=self._load_for_update,
            store_now=self._session_store_now,
            load_checkpoint=self._load_checkpoint,
            closure_owners=self._closure_lineage_owners,
            register_authorities=partial(
                public_authority_ops.register_authorities, codec=self.public_authority_alias_codec
            ),
            upsert_checkpoint=self._upsert_checkpoint,
            touch_activity=_touch_session_activity,
        )

    async def materialize_deferred_interaction_input(
        self,
        session_id: str,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> bool:
        return await transcript_ops.materialize_deferred_interaction_input(
            self._ready_connection,
            session_id,
            interaction_id=interaction_id,
            load_for_update=self._load_for_update,
            store_now=self._session_store_now,
            closure_owners=self._closure_lineage_owners,
            register_authorities=partial(
                public_authority_ops.register_authorities, codec=self.public_authority_alias_codec
            ),
            touch_activity=_touch_session_activity,
        )

    async def load_deferred_interaction_input(
        self,
        session_id: str,
    ) -> DeferredInteractionInput | None:
        return await transcript_ops.load_deferred_interaction_input(
            self._ready_connection, session_id, load_session=self._load
        )

    async def append_transcript_messages_and_transform_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint_transform: CheckpointTransform,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        return await transcript_ops.append_transcript_messages_and_transform_checkpoint(
            self._ready_connection,
            session_id,
            messages,
            checkpoint_transform,
            interaction_id=interaction_id,
            load_for_update=self._load_for_update,
            store_now=self._session_store_now,
            load_checkpoint=self._load_checkpoint,
            closure_owners=self._closure_lineage_owners,
            register_authorities=partial(
                public_authority_ops.register_authorities, codec=self.public_authority_alias_codec
            ),
            upsert_checkpoint=self._upsert_checkpoint,
            touch_activity=_touch_session_activity,
        )

    async def load_transcript(self, session_id: str) -> list[Message]:
        return await transcript_ops.load_transcript(
            self._ready_connection, session_id, load_session=self._load
        )

    async def load_transcript_snapshot(self, session_id: str) -> TranscriptSnapshot:
        return await transcript_ops.load_transcript_snapshot(
            self._ready_connection, session_id, load_session=self._load
        )

    async def load_transcript_cursor(self, session_id: str) -> int:
        return await transcript_ops.load_transcript_cursor(self._ready_connection, session_id)

    async def load_latest_transcript_message(
        self,
        session_id: str,
        *,
        role: MessageRole,
    ) -> TranscriptRecord | None:
        return await transcript_ops.load_latest_transcript_message(
            self._ready_connection, session_id, role=role
        )

    async def load_latest_transcript_text(
        self,
        session_id: str,
        *,
        role: MessageRole,
        max_chars: int,
    ) -> tuple[str, bool] | None:
        return await transcript_ops.load_latest_transcript_text(
            self._ready_connection,
            session_id,
            role=role,
            max_chars=max_chars,
            load_session=self._load,
        )

    async def load_transcript_window(
        self,
        session_id: str,
        *,
        start_index: int,
        limit: int,
    ) -> TranscriptSnapshot:
        return await transcript_ops.load_transcript_window(
            self._ready_connection,
            session_id,
            start_index=start_index,
            limit=limit,
            load_session=self._load,
        )

    async def load_bounded_transcript_window(
        self,
        session_id: str,
        *,
        start_index: int,
        limit: int,
        max_record_bytes: int,
        max_total_bytes: int,
    ) -> BoundedTranscriptWindow:
        return await transcript_ops.load_bounded_transcript_window(
            self._ready_connection,
            session_id,
            start_index=start_index,
            limit=limit,
            max_record_bytes=max_record_bytes,
            max_total_bytes=max_total_bytes,
            load_session=self._load,
        )

    async def query_transcript(self, query: TranscriptQuery) -> TranscriptPage:
        return await transcript_ops.query_transcript(
            self._ready_connection, query, load_session=self._load
        )

    async def search_transcript(
        self,
        query: TranscriptSearchQuery,
    ) -> TranscriptSearchResult:
        return await transcript_ops.search_transcript(self._ready_connection, query)

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

    async def load_execution_snapshot_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT state -> 'checkpoint_schema_version', state -> 'execution_snapshots' "
                "FROM cayu_checkpoints WHERE session_id = %s",
                (session_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            projection = {}
            if row[0] is not None:
                projection["checkpoint_schema_version"] = row[0]
            if row[1] is not None:
                projection["execution_snapshots"] = row[1]
            return projection

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
                None if row is None else model_stage_ops.decode_stage_record(row[0]),
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
