from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from typing import Any, ClassVar, Literal, LiteralString, cast
from uuid import uuid4

try:
    from psycopg.errors import (
        DeadlockDetected,
        DuplicateTable,
        ForeignKeyViolation,
        UniqueViolation,
    )
    from psycopg_pool import AsyncConnectionPool
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
    raise RuntimeError(
        "Cayu's Postgres stores require the optional psycopg packages. "
        'Install them with `pip install "cayu[postgres]"`.'
    ) from exc

from cayu._validation import (
    EXECUTION_UNIT_ID_MAX_CHARS,
    JsonUtf8SizeCounter,
    copy_durable_json_object,
    copy_durable_json_value,
    copy_label_map,
    require_durable_nonblank,
    require_nonblank,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu.core.billing import BillingIdentity, copy_billing_identity
from cayu.core.events import Event, EventType
from cayu.core.messages import Message, MessageRole
from cayu.embeddings import TextEmbeddingProvider, TextEmbeddingRequest
from cayu.runtime.aggregates import EXACT_AGGREGATE, UsageRollupStoreResult
from cayu.runtime.approvals import (
    _PENDING_TOOL_APPROVAL_EVENT_PROJECTION_KEYS,
    ResolutionActor,
    resolution_actor_payload,
)
from cayu.runtime.budgets import (
    DEFAULT_RESERVATION_TTL_SECONDS,
    BudgetLedger,
    BudgetLimit,
    BudgetReconciliation,
    BudgetReconciliationPricing,
    BudgetReservationRecord,
    BudgetReservationResult,
    BudgetSettlementCursor,
    BudgetSettlementFallback,
    BudgetSettlementRecord,
    _budget_reservation_amount,
    _budget_settlement_record,
    _clock_or_utc_now,
    _copy_budget_settlement_cursor,
    _EffectiveBudgetLimit,
    _ensure_effective_budget_limit,
    _expired_reservation_reason,
    _reconciled_record,
    _reconciliation_from_record,
    _released_record,
    _reservation_is_expired,
    _reservation_result,
    _utc_datetime,
    _validate_amount,
    _validate_reservation_id_batch,
    _validate_reservation_ttl,
    _validate_settlement_page_limit,
    copy_budget_settlement_fallback,
    new_budget_reservation_id,
)
from cayu.runtime.event_watchers import (
    EventWatcherClaim,
    EventWatcherDeadLetter,
    EventWatcherDelivery,
    EventWatcherDeliveryStatus,
    EventWatcherState,
    EventWatcherStore,
    copy_event_watcher_claim,
    copy_event_watcher_record,
)
from cayu.runtime.execution_units import (
    ModelAttemptIdentity,
    ToolRoundIdentity,
    copy_model_attempt_identity,
    copy_tool_round_identity,
)
from cayu.runtime.sessions import (
    _TOOL_ROUND_LIFECYCLE_EVENT_TYPES,
    DELETE_BLOCKED_SESSION_STATUSES,
    FORK_TRANSCRIPT_VALIDATION_ERROR,
    INHERIT_INTERACTION,
    MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL,
    MAX_PENDING_ACTION_RESULT_BYTES,
    MAX_PENDING_ACTION_TOOL_CALLS,
    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
    RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS,
    SESSION_INSPECTION_LABEL_LIMIT,
    SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
    BudgetReservationIdentityConflict,
    CheckpointTransform,
    DeferredInteractionInput,
    EnqueueSessionMessageRequest,
    EnqueueSessionMessageResult,
    EventQuery,
    EventQueryResultTooLarge,
    EventRecord,
    EventSummary,
    ForkTranscriptValidator,
    InteractionAttribution,
    InteractionTransitionResult,
    McpManifestBaseline,
    McpManifestBaselineLoadResult,
    McpManifestPublicationResult,
    ModelCompletionStageAbandonmentResult,
    ModelCompletionStageResult,
    PendingActionIssue,
    PendingActionKind,
    PendingActionListResult,
    PendingActionQuery,
    PendingActionSession,
    PersistedEventSideEffectClaim,
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectDelivery,
    PersistedEventSideEffectStatus,
    RunRequest,
    RuntimePublicationReceipt,
    RuntimePublicationResult,
    Session,
    SessionAggregateFilter,
    SessionIdentity,
    SessionInspectionIdentity,
    SessionListResult,
    SessionMessageDeliveryBatch,
    SessionMessageQueueStatus,
    SessionModelCompletionStageConflict,
    SessionOperationalSnapshot,
    SessionOperationPublication,
    SessionOperationTransform,
    SessionOrder,
    SessionOutcome,
    SessionQuery,
    SessionQueuedMessage,
    SessionQueuedMessagesPending,
    SessionRunFenced,
    SessionRuntimePublicationConflict,
    SessionStateSnapshot,
    SessionStatus,
    SessionStatusConflict,
    SessionStatusCounts,
    SessionStore,
    SessionTopologyCycle,
    SessionTopologyDepthExceeded,
    SessionTopologyQuery,
    SessionTopologyStoreResult,
    TranscriptPage,
    TranscriptQuery,
    TranscriptRecord,
    UsageRollupQuery,
    _activate_session_run_fence,
    _active_model_completion_stage_record,
    _active_unexpired_incomplete_recovery_claim_id,
    _active_unexpired_session_operation_id,
    _apply_runtime_publication_checkpoint_mutation,
    _assert_session_run_epoch,
    _assert_session_run_epoch_value,
    _build_runtime_publication_receipt,
    _checkpoint_after_initial_transcript_publication,
    _copy_mcp_manifest_publication,
    _copy_optional_interaction_admission,
    _copy_queued_interaction_started_event,
    _copy_session_event_batch,
    _copy_transition_interaction_admission,
    _current_session_run_epoch,
    _deactivate_session_interaction,
    _deactivate_session_run_fence,
    _initial_transcript_pending_checkpoint,
    _interaction_transition_receipt_record,
    _interaction_transition_storage_key,
    _model_completion_stage_abandonment_record,
    _model_completion_stage_preparation_record,
    _model_completion_stage_storage_identity,
    _model_completion_stage_terminal_record,
    _model_completion_stage_winner_record,
    _model_completion_stage_winner_storage_key,
    _model_completion_terminal_advances_last_activity,
    _ModelCompletionStagePromotionContext,
    _next_runtime_publication_timestamp,
    _prepare_interaction_transition,
    _prepare_model_completion_stage_promotion,
    _prepare_session_fork_request,
    _PreparedModelCompletionStage,
    _PreparedModelCompletionStageAbandonment,
    _PreparedModelCompletionStageTerminal,
    _PreparedRuntimePublication,
    _project_interruption_cascade_marker_fields,
    _queued_session_message_event_payload,
    _reconstruct_active_model_completion_stage,
    _reconstruct_active_model_completion_stage_record,
    _reconstruct_interaction_transition_receipt,
    _reconstruct_model_completion_stage,
    _reconstruct_model_completion_stage_abandonment,
    _reconstruct_runtime_publication_receipt,
    _reject_reserved_runtime_publication_key,
    _replay_model_completion_stage_abandonment,
    _replay_promoted_model_completion_stage,
    _runtime_publication_json_equal,
    _runtime_publication_receipt_record,
    _runtime_publication_referenced_event_ids,
    _runtime_publication_storage_key,
    _stored_mcp_manifest_baseline,
    _tool_round_publication_identity,
    _validate_equivalent_queued_session_message,
    _validate_interaction_page,
    _validate_mcp_manifest_history_keys,
    _validate_mcp_manifest_publication_state,
    _validate_message_delivery_eligible_through,
    _validate_model_completion_active_marker_for_preparation,
    _validate_model_completion_active_marker_for_promotion,
    _validate_model_completion_preparation_replay_state,
    _validate_model_completion_promotion_replay_active_marker,
    _validate_model_completion_stage_for_abandonment,
    _validate_model_completion_stage_preparation_replay,
    _validate_model_completion_stage_publication,
    _validate_model_completion_stage_repreparation,
    _validate_model_completion_stage_terminal_replay,
    _validate_runtime_publication_durable_material,
    _validate_runtime_publication_event_references,
    _validate_runtime_publication_replay_receipt,
    _validate_session_fork_source,
    _validate_session_operation_record_keys,
    _validate_status_set,
    _validate_tool_round_call_ids,
    _validate_tool_round_checkpoint_mutation,
    _validate_tool_round_publication,
    build_session_topology_result,
    copy_enqueue_session_message_request,
    copy_event_query,
    copy_run_request,
    copy_session_aggregate_filter,
    copy_session_identity,
    copy_session_query,
    copy_session_user_metadata,
    copy_transcript_messages,
    copy_transcript_query,
    copy_usage_rollup_query,
    decode_session_cursor,
    decode_session_topology_cursor,
    encode_session_cursor,
    enforce_pending_action_result_size,
    filter_transcript_records,
    fork_transcript_is_accepted,
    replace_session_user_metadata,
    resolve_interaction_attribution,
    session_next_cursor,
    session_outcome,
    session_query_from_aggregate_filter,
    transform_fork_checkpoint,
    validate_persisted_event_side_effect_error,
)
from cayu.runtime.tasks import (
    TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    Task,
    TaskAggregateFilter,
    TaskCreate,
    TaskOperationalSnapshot,
    TaskOrder,
    TaskQuery,
    TaskStatus,
    TaskStatusCounts,
    TaskStore,
    TaskTopologyInconsistent,
    TaskTopologyNode,
    TaskTopologyQuery,
    TaskTopologyStoreResult,
    _allocate_task_topology_branch_limits,
    _bounded_optional_task_topology_parent_id,
    _copy_optional_status_payload,
    _copy_optional_status_reason,
    _ensure_can_hold_task,
    _ensure_can_resume_task,
    _ensure_can_transition,
    _ensure_claim_query_supported,
    _ensure_owned_active_task_lease,
    _raise_task_claim_attach_error,
    _running_task_from_create,
    _task_from_create,
    _validate_task_topology_ancestry,
    build_task_topology_result,
    copy_task_aggregate_filter,
    copy_task_create,
    copy_task_query,
    decode_task_topology_cursor,
    task_query_from_aggregate_filter,
)
from cayu.storage import _postgres_aggregates as postgres_aggregates
from cayu.storage import _postgres_support as pg_support
from cayu.storage import _session_store_sql as session_store_sql
from cayu.storage import migrations as schema
from cayu.storage.memory import (
    DEFAULT_KNOWLEDGE_LIMIT,
    DEFAULT_KNOWLEDGE_MAX_BYTES,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeFacet,
    KnowledgeHit,
    KnowledgeListGroup,
    KnowledgeListItem,
    KnowledgeListQuery,
    KnowledgeListResult,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeSearchResult,
    KnowledgeStatus,
    KnowledgeStore,
    KnowledgeVisibility,
    _knowledge_chunk_content_hash,
    _score_entry,
    _search_result_from_scored_embeddings,
    _semantic_query_text,
    _validate_nonnegative_float,
    _validate_positive_int,
    _validate_unit_float,
    copy_knowledge_chunk,
    copy_knowledge_entry,
    copy_knowledge_list_query,
    copy_knowledge_query,
)

# A fixed 63-bit advisory-lock key. Every Cayu store sharing a database takes this
# lock before touching schema, so concurrent creators/migrators (the production
# PostgresSessionStore + PostgresTaskStore on one pool) serialize: one runs the
# DDL, the rest wait and then validate (ADR 0001, Decision 4). The value is the
# ASCII bytes of "cayuschm" masked to stay positive (signed bigint); its only
# requirement is being a stable constant unlikely to collide with app locks.
_SCHEMA_ADVISORY_LOCK_KEY = 0x6361_7975_7363_686D & 0x7FFF_FFFF_FFFF_FFFF
_SCHEMA_ADVISORY_LOCK_POLL_SECONDS = 0.25
_POSTGRES_MIN_REQUIRED_REVISION = 18
_POSTGRES_SESSION_MIN_REQUIRED_REVISION = 26
_EVENT_QUERY_SESSION_IDS_BATCH_SIZE = 500
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
_SQL_DIALECT = session_store_sql.SessionStoreSqlDialect(
    placeholder="%s",
    contains_style="postgres_ilike",
    datetime_param=pg_support.to_utc,
)
_KNOWLEDGE_SEARCH_PAGE_SIZE = 500
_KNOWLEDGE_SEARCH_TOKEN_RE = re.compile(r"\w+")
_PGVECTOR_HNSW_VECTOR_MAX_DIMENSIONS = 2000
_PGVECTOR_SEMANTIC_CANDIDATE_MULTIPLIER = 8
# Identifies the embedding space (model + preprocessing + normalization) a stored vector belongs to.
# Writes stamp it and reads filter on it, so bumping this constant after changing the embedding recipe
# segregates old vectors (they stop matching and are re-embedded / pruned) instead of silently mixing
# two spaces. The column is added now, while cheap, so a future bump needs no re-migration.
_EMBEDDING_SPACE_VERSION = 1
# Upper bound on chunks a single semantic search will lazily embed when it finds
# entries whose write-time embedding was deferred (provider outage). The
# missing-embedding LEFT JOIN returns nothing in steady state, so this cap only
# bites while backfilling a write that flag-and-continued.
_PGVECTOR_LAZY_BACKFILL_LIMIT = 500

logger = logging.getLogger(__name__)
_PGVECTOR_SCHEMA_ADVISORY_LOCK_KEY = 0x6361_7975_7665_6374 & 0x7FFF_FFFF_FFFF_FFFF
_TASK_RETURNING_COLUMNS = (
    "task.id, task.type, task.title, task.description, task.status, task.session_id, "
    "task.parent_task_id, task.assigned_agent_name, task.worker_id, task.lease_expires_at, "
    "task.status_reason, task.status_payload, task.input, task.result, task.error, task.metadata, "
    "task.created_at, task.updated_at, task.started_at, task.completed_at"
)
_SESSION_MESSAGE_QUEUE_COLUMNS = (
    "ordering_key, queue_id, session_id, idempotency_key, content, delivery_mode, status, "
    "requested_by, accepted_run_epoch, accepted_transcript_cursor, accepted_event_id, "
    "accepted_at, delivered_run_epoch, delivered_transcript_cursor, delivered_event_id, "
    "delivered_at"
)


def _queued_session_message_from_row(row: Any) -> SessionQueuedMessage:
    requested_by = row[7]
    return SessionQueuedMessage(
        ordering_key=row[0],
        queue_id=row[1],
        session_id=row[2],
        idempotency_key=row[3],
        content=row[4],
        delivery_mode=row[5],
        status=row[6],
        requested_by=(
            None
            if requested_by is None
            else ResolutionActor.model_validate(_json_obj(requested_by))
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


def _ilike_contains_pattern(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


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


@dataclass(frozen=True)
class PostgresEmbeddingBackfillResult:
    """Result of a bounded Postgres knowledge embedding backfill."""

    scanned_chunks: int
    embedded_chunks: int
    skipped_current_chunks: int
    limit: int
    refresh_existing: bool


def _event_query_session_id_batches(
    session_ids: tuple[str, ...],
) -> list[tuple[str, ...]]:
    return [
        session_ids[index : index + _EVENT_QUERY_SESSION_IDS_BATCH_SIZE]
        for index in range(0, len(session_ids), _EVENT_QUERY_SESSION_IDS_BATCH_SIZE)
    ]


def _event_query_with_session_ids(
    query: EventQuery,
    *,
    session_ids: tuple[str, ...],
) -> EventQuery:
    return EventQuery(
        session_ids=session_ids,
        event_id=query.event_id,
        interaction_id=query.interaction_id,
        causal_budget_id=query.causal_budget_id,
        event_type=query.event_type,
        event_types=query.event_types,
        exclude_event_types=query.exclude_event_types,
        agent_name=query.agent_name,
        environment_name=query.environment_name,
        workflow_name=query.workflow_name,
        tool_name=query.tool_name,
        since=query.since,
        until=query.until,
        after_sequence=query.after_sequence,
        before_sequence=query.before_sequence,
        limit=query.limit,
        order_by=query.order_by,
    )


def _event_query_is_single_session(query: EventQuery) -> bool:
    return query.session_id is not None or len(query.session_ids) == 1


def _event_query_needs_snapshot_cutoff(query: EventQuery) -> bool:
    return query.after_sequence is not None and not _event_query_is_single_session(query)


# Per-revision forward-migration DDL, keyed by revision number. The baseline
# (revision 1) is applied from pg_support.SCHEMA_STATEMENTS, so it is not listed
# here; future additive/breaking revisions append their ALTER/CREATE statements.
_MIGRATION_STEPS: dict[int, tuple[str, ...]] = {
    2: (
        """
        CREATE TABLE IF NOT EXISTS cayu_session_labels (
            session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (session_id, key)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_session_labels_key_value_session "
        "ON cayu_session_labels(key, value, session_id)",
    ),
    3: (
        """
        CREATE TABLE IF NOT EXISTS cayu_event_watcher_state (
            watcher_name TEXT PRIMARY KEY,
            cursor_sequence BIGINT NOT NULL,
            pending_event_id TEXT,
            pending_event_sequence BIGINT,
            pending_attempt INTEGER NOT NULL,
            pending_claim_id TEXT,
            delivery_status TEXT,
            lease_expires_at TIMESTAMPTZ,
            last_error TEXT,
            dead_lettered_count INTEGER NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_event_watcher_state_delivery "
        "ON cayu_event_watcher_state(delivery_status, lease_expires_at)",
    ),
    4: (
        "ALTER TABLE cayu_tasks ADD COLUMN worker_id TEXT",
        "ALTER TABLE cayu_tasks ADD COLUMN lease_expires_at TIMESTAMPTZ",
        "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_worker_id ON cayu_tasks(worker_id)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_status_lease "
        "ON cayu_tasks(status, lease_expires_at)",
    ),
    5: (
        "ALTER TABLE cayu_tasks ADD COLUMN status_reason TEXT",
        "ALTER TABLE cayu_tasks ADD COLUMN status_payload JSONB",
    ),
    6: (
        """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_entries (
            id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            text TEXT NOT NULL,
            kind TEXT NOT NULL,
            visibility TEXT NOT NULL,
            status TEXT NOT NULL,
            created_by_type TEXT NOT NULL,
            created_by TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            source_type TEXT,
            source_uri TEXT,
            source_id TEXT,
            source_hash TEXT,
            importance DOUBLE PRECISION,
            importance_source TEXT,
            confidence DOUBLE PRECISION,
            last_used_at TIMESTAMPTZ,
            expires_at TIMESTAMPTZ,
            title TEXT,
            metadata JSONB NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_labels (
            entry_id TEXT NOT NULL REFERENCES cayu_knowledge_entries(id) ON DELETE CASCADE,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (entry_id, key)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_aspects (
            entry_id TEXT NOT NULL REFERENCES cayu_knowledge_entries(id) ON DELETE CASCADE,
            aspect TEXT NOT NULL,
            PRIMARY KEY (entry_id, aspect)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_impact_targets (
            entry_id TEXT NOT NULL REFERENCES cayu_knowledge_entries(id) ON DELETE CASCADE,
            impact_target TEXT NOT NULL,
            PRIMARY KEY (entry_id, impact_target)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_chunks (
            id TEXT PRIMARY KEY,
            entry_id TEXT NOT NULL REFERENCES cayu_knowledge_entries(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            content_hash TEXT,
            source_uri TEXT,
            metadata JSONB NOT NULL,
            UNIQUE (entry_id, chunk_index)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entries_namespace_status "
        "ON cayu_knowledge_entries(namespace, status)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entries_kind "
        "ON cayu_knowledge_entries(kind)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entries_visibility "
        "ON cayu_knowledge_entries(visibility)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entries_source "
        "ON cayu_knowledge_entries(source_type, source_id)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entries_expires_at "
        "ON cayu_knowledge_entries(expires_at)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_labels_key_value_entry "
        "ON cayu_knowledge_labels(key, value, entry_id)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_aspects_aspect_entry "
        "ON cayu_knowledge_aspects(aspect, entry_id)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_impact_targets_target_entry "
        "ON cayu_knowledge_impact_targets(impact_target, entry_id)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_chunks_entry_index "
        "ON cayu_knowledge_chunks(entry_id, chunk_index)",
    ),
    7: (
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entries_title_fts "
        "ON cayu_knowledge_entries USING GIN (to_tsvector('simple', COALESCE(title, '')))",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entries_text_fts "
        "ON cayu_knowledge_entries USING GIN (to_tsvector('simple', text))",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_chunks_text_fts "
        "ON cayu_knowledge_chunks USING GIN (to_tsvector('simple', text))",
    ),
    8: (
        """
        CREATE TABLE IF NOT EXISTS cayu_budget_reservations (
            reservation_id TEXT PRIMARY KEY,
            scope TEXT NOT NULL,
            budget_key TEXT,
            budget_window TEXT NOT NULL,
            currency TEXT NOT NULL,
            session_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            provider_name TEXT NOT NULL,
            model TEXT NOT NULL,
            reserved_amount NUMERIC NOT NULL,
            actual_amount NUMERIC,
            status TEXT NOT NULL,
            reason TEXT,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_budget_reservations_scope "
        "ON cayu_budget_reservations(scope, budget_key, budget_window, currency, status)",
    ),
    10: (
        # Per-session monotonic counter that append_events advances with a single
        # UPDATE ... RETURNING (replacing the row-lock + COALESCE(MAX()) scan).
        # IF NOT EXISTS keeps the greenfield-through-migrations path a no-op, since
        # the baseline schema already declares the column.
        "ALTER TABLE cayu_sessions ADD COLUMN IF NOT EXISTS event_seq BIGINT NOT NULL DEFAULT 0",
        # Seed the counter from the highest existing session_order so the first
        # post-migration append continues the sequence instead of colliding with
        # already-stored rows.
        """
        UPDATE cayu_sessions AS s
        SET event_seq = COALESCE(
            (SELECT MAX(e.session_order) FROM cayu_events AS e WHERE e.session_id = s.id),
            0
        )
        """,
    ),
    11: (
        """
        CREATE TABLE IF NOT EXISTS cayu_event_watcher_dead_letters (
            watcher_name TEXT NOT NULL,
            event_sequence BIGINT NOT NULL,
            event_id TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            error TEXT NOT NULL,
            dead_lettered_at TIMESTAMPTZ NOT NULL,
            resolved_at TIMESTAMPTZ,
            PRIMARY KEY (watcher_name, event_sequence)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_event_watcher_dead_letters_unresolved "
        "ON cayu_event_watcher_dead_letters(watcher_name, resolved_at, event_sequence)",
    ),
    # Add the embedding-space version column so the standard `cayu storage migrate` deploy step (which
    # runs this table via PostgresSessionStore) reaches an existing cayu_knowledge_embeddings table.
    # `IF EXISTS` makes it a no-op when the embeddings table was never created (embedding store unused).
    12: (
        "ALTER TABLE IF EXISTS cayu_knowledge_embeddings "
        "ADD COLUMN IF NOT EXISTS embedding_space_version INTEGER NOT NULL DEFAULT 1",
    ),
    13: (
        "ALTER TABLE cayu_events "
        "ADD COLUMN IF NOT EXISTS insert_xid xid8 NOT NULL DEFAULT pg_current_xact_id()",
        "CREATE INDEX IF NOT EXISTS idx_cayu_events_insert_xid ON cayu_events(insert_xid)",
    ),
    14: (
        "ALTER TABLE cayu_sessions ADD COLUMN IF NOT EXISTS "
        "last_activity_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP",
        "ALTER TABLE cayu_sessions ADD COLUMN IF NOT EXISTS run_epoch BIGINT NOT NULL DEFAULT 0",
    ),
    15: (
        "CREATE INDEX IF NOT EXISTS idx_cayu_checkpoints_pending_interruption_cascade "
        "ON cayu_checkpoints(session_id) "
        "WHERE state ? 'pending_interruption_cascade'",
    ),
    17: (
        "ALTER TABLE cayu_events ADD COLUMN IF NOT EXISTS pending_action_lookup_key TEXT",
        "ALTER TABLE cayu_events ADD COLUMN IF NOT EXISTS pending_action_projection JSONB",
        "ALTER TABLE cayu_events ADD COLUMN IF NOT EXISTS pending_action_projection_bytes BIGINT",
        "ALTER TABLE cayu_checkpoints ADD COLUMN IF NOT EXISTS pending_action_source_bytes BIGINT",
        "ALTER TABLE cayu_checkpoints ADD COLUMN IF NOT EXISTS "
        "pending_action_tool_call_count INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE cayu_checkpoints ADD COLUMN IF NOT EXISTS "
        "pending_action_flags INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE cayu_checkpoints ADD COLUMN IF NOT EXISTS "
        "pending_action_metrics_ready BOOLEAN NOT NULL DEFAULT FALSE",
    ),
    18: (
        """
        CREATE TABLE IF NOT EXISTS cayu_session_operations (
            session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            idempotency_key TEXT NOT NULL,
            record JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (session_id, idempotency_key)
        )
        """,
    ),
    19: (
        """
        CREATE TABLE IF NOT EXISTS cayu_session_message_queue (
            ordering_key BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            queue_id TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            idempotency_key TEXT NOT NULL,
            content TEXT NOT NULL,
            delivery_mode TEXT NOT NULL,
            status TEXT NOT NULL,
            requested_by JSONB,
            accepted_run_epoch BIGINT NOT NULL,
            accepted_transcript_cursor BIGINT NOT NULL,
            accepted_event_id TEXT NOT NULL,
            accepted_at TIMESTAMPTZ NOT NULL,
            delivered_run_epoch BIGINT,
            delivered_transcript_cursor BIGINT,
            delivered_event_id TEXT,
            delivered_at TIMESTAMPTZ,
            UNIQUE (session_id, idempotency_key)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_session_message_queue_delivery "
        "ON cayu_session_message_queue(session_id, status, delivery_mode, ordering_key)",
    ),
    20: (
        """
        CREATE TABLE IF NOT EXISTS cayu_persisted_event_side_effects (
            session_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            event_sequence BIGINT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            claim_id TEXT,
            lease_expires_at TIMESTAMPTZ,
            next_attempt_at TIMESTAMPTZ,
            last_error TEXT,
            updated_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (session_id, event_id),
            FOREIGN KEY (session_id, event_id)
                REFERENCES cayu_events(session_id, event_id) ON DELETE CASCADE
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_persisted_event_side_effects_delivery "
        "ON cayu_persisted_event_side_effects"
        "(status, next_attempt_at, lease_expires_at, event_sequence)",
    ),
    21: (
        "ALTER TABLE IF EXISTS cayu_budget_reservations "
        "ADD COLUMN IF NOT EXISTS billing_identity JSONB",
    ),
    22: (
        """
        CREATE TABLE IF NOT EXISTS cayu_mcp_manifest_baselines (
            history_key TEXT PRIMARY KEY,
            generation BIGINT NOT NULL CHECK (generation >= 1),
            baseline JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """,
    ),
    26: (
        "ALTER TABLE cayu_events ADD COLUMN IF NOT EXISTS interaction_id TEXT",
        "ALTER TABLE cayu_transcript_messages ADD COLUMN IF NOT EXISTS interaction_id TEXT",
        "ALTER TABLE cayu_sessions ADD COLUMN IF NOT EXISTS "
        "transcript_seq BIGINT NOT NULL DEFAULT 0",
        "ALTER TABLE cayu_transcript_messages ADD COLUMN IF NOT EXISTS session_order BIGINT",
        "ALTER TABLE cayu_transcript_messages ALTER COLUMN session_order SET NOT NULL",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_transcript_session_order "
        "ON cayu_transcript_messages(session_id, session_order)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_transcript_interaction_order "
        "ON cayu_transcript_messages(session_id, interaction_id, session_order)",
        """
        CREATE OR REPLACE FUNCTION cayu_assign_transcript_order()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.session_order IS NOT NULL THEN
                RAISE EXCEPTION
                    'cayu_transcript_messages.session_order is runtime-owned';
            END IF;
            UPDATE cayu_sessions
            SET transcript_seq = transcript_seq + 1
            WHERE id = NEW.session_id
            RETURNING transcript_seq INTO NEW.session_order;
            IF NEW.session_order IS NULL THEN
                RAISE EXCEPTION 'transcript session does not exist: %', NEW.session_id;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """,
        "DROP TRIGGER IF EXISTS cayu_assign_transcript_order ON cayu_transcript_messages",
        """
        CREATE TRIGGER cayu_assign_transcript_order
        BEFORE INSERT ON cayu_transcript_messages
        FOR EACH ROW EXECUTE FUNCTION cayu_assign_transcript_order()
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_interaction_latest_events (
            session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            interaction_id TEXT NOT NULL,
            latest_event_sequence BIGINT NOT NULL
                REFERENCES cayu_events(sequence) ON DELETE CASCADE,
            PRIMARY KEY (session_id, interaction_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_interaction_latest_events_page "
        "ON cayu_interaction_latest_events(session_id, latest_event_sequence DESC)",
        """
        CREATE OR REPLACE FUNCTION cayu_track_interaction_latest_event()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.interaction_id IS NOT NULL AND NEW.event_type = ANY(ARRAY[
                'interaction.started', 'interaction.resumed', 'interaction.paused',
                'interaction.completed', 'interaction.failed', 'interaction.interrupted'
            ]) THEN
                INSERT INTO cayu_interaction_latest_events (
                    session_id, interaction_id, latest_event_sequence
                ) VALUES (NEW.session_id, NEW.interaction_id, NEW.sequence)
                ON CONFLICT (session_id, interaction_id) DO UPDATE SET
                    latest_event_sequence = EXCLUDED.latest_event_sequence
                WHERE EXCLUDED.latest_event_sequence
                    > cayu_interaction_latest_events.latest_event_sequence;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """,
        "DROP TRIGGER IF EXISTS cayu_track_interaction_latest_event ON cayu_events",
        """
        CREATE TRIGGER cayu_track_interaction_latest_event
        AFTER INSERT ON cayu_events
        FOR EACH ROW EXECUTE FUNCTION cayu_track_interaction_latest_event()
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_deferred_interaction_inputs (
            session_id TEXT PRIMARY KEY REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            interaction_id TEXT NOT NULL,
            source_messages JSONB NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_session_message_deliveries (
            delivery_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            interaction_id TEXT,
            include_on_idle BOOLEAN NOT NULL,
            requested_eligible_through BIGINT,
            eligible_through BIGINT NOT NULL,
            batch_limit INTEGER NOT NULL,
            has_more BOOLEAN NOT NULL,
            interaction_started_event JSONB,
            queue_ids JSONB NOT NULL,
            events JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_session_message_deliveries_session "
        "ON cayu_session_message_deliveries(session_id, created_at)",
    ),
    23: (
        "ALTER TABLE IF EXISTS cayu_budget_reservations "
        "ADD COLUMN IF NOT EXISTS budget_limit_id TEXT",
        "CREATE INDEX IF NOT EXISTS idx_cayu_budget_reservations_limit "
        "ON cayu_budget_reservations(budget_limit_id, status, updated_at)",
        "ALTER TABLE IF EXISTS cayu_budget_reservations "
        "ADD COLUMN IF NOT EXISTS model_step_id TEXT",
        "ALTER TABLE IF EXISTS cayu_budget_reservations "
        "ADD COLUMN IF NOT EXISTS model_attempt_id TEXT",
        "CREATE INDEX IF NOT EXISTS idx_cayu_budget_reservations_model_attempt "
        "ON cayu_budget_reservations(model_attempt_id, budget_limit_id, status)",
        """
        CREATE TABLE IF NOT EXISTS cayu_budget_reservation_identities (
            reservation_id TEXT PRIMARY KEY,
            publication_session_id TEXT NOT NULL,
            publication_id TEXT NOT NULL,
            published BOOLEAN NOT NULL
        )
        """,
        """
        INSERT INTO cayu_budget_reservation_identities (
            reservation_id,
            publication_session_id,
            publication_id,
            published
        )
        SELECT
            payload ->> 'reservation_id',
            session_id,
            event_id,
            TRUE
        FROM cayu_events
        WHERE event_type = 'budget.reserved'
          AND jsonb_typeof(payload -> 'reservation_id') = 'string'
        ON CONFLICT (reservation_id) DO NOTHING
        """,
    ),
    25: (
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM cayu_budget_reservations
                WHERE status = 'active'
            ) THEN
                RAISE EXCEPTION USING MESSAGE =
                    'Schema revision 25 cannot migrate active budget reservations because '
                    'their dispatch state is unknown. Drain or explicitly settle every active '
                    'reservation, then retry the migration.';
            END IF;
        END
        $$
        """,
        "ALTER TABLE IF EXISTS cayu_budget_reservations "
        "ADD COLUMN IF NOT EXISTS environment_name TEXT",
        "ALTER TABLE IF EXISTS cayu_budget_reservations "
        "ADD COLUMN IF NOT EXISTS settlement_event_payload JSONB NOT NULL "
        "DEFAULT '{}'::jsonb",
        "ALTER TABLE IF EXISTS cayu_budget_reservations "
        "ADD COLUMN IF NOT EXISTS settlement_fallback JSONB",
        """
        UPDATE cayu_budget_reservations
        SET settlement_fallback = jsonb_build_object(
            'settled_at', to_jsonb(created_at),
            'reconciliation_reason',
                'model completion settlement evidence was not publishable; '
                'charged reserved amount',
            'release_reason', 'reservation released before provider dispatch',
            'expiration_reason', NULL
        )
        WHERE settlement_fallback IS NULL
        """,
        "ALTER TABLE IF EXISTS cayu_budget_reservations "
        "ALTER COLUMN settlement_fallback SET NOT NULL",
        "ALTER TABLE IF EXISTS cayu_budget_reservations ADD COLUMN IF NOT EXISTS dispatch_id TEXT",
        "ALTER TABLE IF EXISTS cayu_budget_reservations "
        "ADD COLUMN IF NOT EXISTS dispatched_at TIMESTAMPTZ",
        """
        CREATE TABLE IF NOT EXISTS cayu_budget_settlements (
            settlement_id TEXT PRIMARY KEY,
            reservation_id TEXT NOT NULL UNIQUE
                REFERENCES cayu_budget_reservations(reservation_id),
            session_id TEXT NOT NULL,
            settled_at TIMESTAMPTZ NOT NULL,
            settlement_json JSONB NOT NULL,
            event_published BOOLEAN NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_budget_settlements_pending "
        "ON cayu_budget_settlements"
        "(session_id, event_published, settled_at, settlement_id)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_budget_settlements_pending_global "
        "ON cayu_budget_settlements"
        "(event_published, settled_at, settlement_id)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_budget_reservation_identities_session "
        "ON cayu_budget_reservation_identities(publication_session_id, reservation_id)",
    ),
}

_REVISION_17_PENDING_TOOL_CALL_COUNT_SQL = """
    GREATEST(
        CASE
            WHEN jsonb_typeof(
                target.state #> '{pending_tool_approval,tool_calls}'
            ) = 'array'
            THEN jsonb_array_length(
                target.state #> '{pending_tool_approval,tool_calls}'
            )
            ELSE 0
        END,
        CASE
            WHEN jsonb_typeof(
                target.state #> '{pending_user_input,tool_calls}'
            ) = 'array'
            THEN jsonb_array_length(
                target.state #> '{pending_user_input,tool_calls}'
            )
            ELSE 0
        END,
        CASE
            WHEN jsonb_typeof(
                target.state #> '{pending_tool_round,tool_calls}'
            ) = 'array'
            THEN jsonb_array_length(
                target.state #> '{pending_tool_round,tool_calls}'
            )
            ELSE 0
        END
    )
"""

_REVISION_17_CHECKPOINT_BACKFILL_SQL = f"""
    WITH batch AS MATERIALIZED (
        SELECT session_id
        FROM cayu_checkpoints
        WHERE NOT pending_action_metrics_ready
          AND (%s::text IS NULL OR session_id > %s)
        ORDER BY session_id
        LIMIT 100
        FOR UPDATE SKIP LOCKED
    )
    UPDATE cayu_checkpoints AS target
    SET pending_action_flags =
            CASE WHEN target.state -> 'pending_tool_approval' IS NOT NULL
                  AND target.state -> 'pending_tool_approval' <> 'null'::jsonb
                THEN 1 ELSE 0 END
            + CASE WHEN target.state -> 'pending_user_input' IS NOT NULL
                  AND target.state -> 'pending_user_input' <> 'null'::jsonb
                THEN 2 ELSE 0 END
            + CASE WHEN target.state -> 'pending_tool_round' IS NOT NULL
                  AND target.state -> 'pending_tool_round' <> 'null'::jsonb
                THEN 4 ELSE 0 END,
        pending_action_source_bytes = CASE
            WHEN ({_REVISION_17_PENDING_TOOL_CALL_COUNT_SQL})
                > {MAX_PENDING_ACTION_TOOL_CALLS}
            THEN 0
            WHEN (target.state -> 'pending_tool_approval' IS NOT NULL
                  AND target.state -> 'pending_tool_approval' <> 'null'::jsonb)
              OR (target.state -> 'pending_user_input' IS NOT NULL
                  AND target.state -> 'pending_user_input' <> 'null'::jsonb)
              OR (target.state -> 'pending_tool_round' IS NOT NULL
                  AND target.state -> 'pending_tool_round' <> 'null'::jsonb)
            THEN octet_length(jsonb_strip_nulls(jsonb_build_object(
                'pending_tool_approval', target.state -> 'pending_tool_approval',
                'pending_user_input', target.state -> 'pending_user_input',
                'pending_tool_round', target.state -> 'pending_tool_round'
            ))::text)
            ELSE NULL
        END,
        pending_action_tool_call_count = ({_REVISION_17_PENDING_TOOL_CALL_COUNT_SQL}),
        pending_action_metrics_ready = TRUE
    FROM batch
    WHERE target.session_id = batch.session_id
    RETURNING target.session_id
"""

_REVISION_17_EVENT_BACKFILL_SMALL_EVENT_BYTES = 1024 * 1024
_REVISION_17_APPROVAL_PROJECTION_KEYS_SQL = ", ".join(
    f"'{key}'" for key in _PENDING_TOOL_APPROVAL_EVENT_PROJECTION_KEYS
)
_REVISION_17_APPROVAL_PROJECTION_SQL = f"""
    (
        SELECT COALESCE(
            jsonb_object_agg(approval_field.key, approval_field.value),
            '{{}}'::jsonb
        )
        FROM jsonb_each(payload -> 'approval') AS approval_field
        WHERE approval_field.key IN ({_REVISION_17_APPROVAL_PROJECTION_KEYS_SQL})
    )
"""


def _revision_17_event_backfill_sql(*, source_predicate: str, batch_limit: int) -> str:
    return f"""
    WITH batch AS MATERIALIZED (
        SELECT sequence, event_type, payload, event
        FROM cayu_events
        WHERE pending_action_projection_bytes IS NULL
          AND sequence > %s
          AND ({source_predicate})
          AND event_type IN (
              'tool.call.approval_requested',
              'session.awaiting_user_input',
              'session.interrupted',
              'session.resumed',
              'session.completed',
              'session.failed',
              'tool.call.started',
              'tool.call.completed',
              'tool.call.failed',
              'tool.call.blocked',
              'tool.call.approval_denied'
        )
        ORDER BY sequence
        LIMIT {batch_limit}
        FOR UPDATE SKIP LOCKED
    ),
    projected AS MATERIALIZED (
        SELECT
            sequence,
            CASE
                WHEN event_type IN (
                    'tool.call.started',
                    'tool.call.completed',
                    'tool.call.failed',
                    'tool.call.blocked',
                    'tool.call.approval_denied'
                )
                  AND jsonb_typeof(payload -> 'tool_call_id') = 'string'
                  AND payload ->> 'tool_call_id' !~ '^[[:space:]]*$'
                THEN payload ->> 'tool_call_id'
                WHEN jsonb_typeof(payload -> 'approval_id') = 'string'
                  AND payload ->> 'approval_id' !~ '^[[:space:]]*$'
                THEN payload ->> 'approval_id'
                WHEN jsonb_typeof(payload #> '{{approval,approval_id}}') = 'string'
                  AND payload #>> '{{approval,approval_id}}' !~ '^[[:space:]]*$'
                THEN payload #>> '{{approval,approval_id}}'
                WHEN jsonb_typeof(payload -> 'input_id') = 'string'
                  AND payload ->> 'input_id' !~ '^[[:space:]]*$'
                THEN payload ->> 'input_id'
                WHEN jsonb_typeof(payload #> '{{user_input,input_id}}') = 'string'
                  AND payload #>> '{{user_input,input_id}}' !~ '^[[:space:]]*$'
                THEN payload #>> '{{user_input,input_id}}'
                WHEN jsonb_typeof(payload -> 'tool_call_id') = 'string'
                  AND payload ->> 'tool_call_id' !~ '^[[:space:]]*$'
                THEN payload ->> 'tool_call_id'
                WHEN jsonb_typeof(payload -> 'tool_round_id') = 'string'
                  AND payload ->> 'tool_round_id' !~ '^[[:space:]]*$'
                THEN payload ->> 'tool_round_id'
                ELSE NULL
            END AS lookup_id,
            jsonb_set(
                event,
                '{{payload}}',
                CASE
                    WHEN event_type = 'tool.call.approval_requested' THEN
                        jsonb_strip_nulls(jsonb_build_object(
                            'approval_id', payload -> 'approval_id',
                            'tool_call_id', payload -> 'tool_call_id',
                            'model_step_id', payload -> 'model_step_id',
                            'model_attempt_id', payload -> 'model_attempt_id',
                            'tool_round_id', payload -> 'tool_round_id',
                            'approval', CASE
                                WHEN jsonb_typeof(payload -> 'approval') = 'object'
                                THEN {_REVISION_17_APPROVAL_PROJECTION_SQL}
                                ELSE NULL
                            END
                        ))
                    WHEN event_type = 'session.awaiting_user_input' THEN
                        jsonb_strip_nulls(jsonb_build_object(
                            'input_id', payload -> 'input_id',
                            'tool_call_id', payload -> 'tool_call_id',
                            'question', payload -> 'question',
                            'options', payload -> 'options',
                            'model_step_id', payload -> 'model_step_id',
                            'model_attempt_id', payload -> 'model_attempt_id',
                            'tool_round_id', payload -> 'tool_round_id'
                        ))
                    WHEN event_type = 'session.interrupted' THEN
                        jsonb_strip_nulls(jsonb_build_object(
                            'interruption_type', payload -> 'interruption_type',
                            'manual_recovery_required', payload -> 'manual_recovery_required',
                            'approval_id', payload -> 'approval_id',
                            'tool_call_id', payload -> 'tool_call_id',
                            'model_step_id', payload -> 'model_step_id',
                            'model_attempt_id', payload -> 'model_attempt_id',
                            'tool_round_id', payload -> 'tool_round_id',
                            'error', payload -> 'error',
                            'message', payload -> 'message',
                            'tool_name', payload -> 'tool_name',
                            'tool_evidence_conflict', payload -> 'tool_evidence_conflict',
                            'approval', CASE
                                WHEN jsonb_typeof(payload -> 'approval') = 'object'
                                THEN {_REVISION_17_APPROVAL_PROJECTION_SQL}
                                ELSE NULL
                            END,
                            'user_input', CASE
                                WHEN jsonb_typeof(payload -> 'user_input') = 'object'
                                THEN jsonb_strip_nulls(jsonb_build_object(
                                    'input_id', payload #> '{{user_input,input_id}}',
                                    'tool_call_id', payload #> '{{user_input,tool_call_id}}',
                                    'question', payload #> '{{user_input,question}}',
                                    'options', payload #> '{{user_input,options}}'
                                ))
                                ELSE NULL
                            END
                        ))
                    WHEN event_type IN (
                        'tool.call.started',
                        'tool.call.completed',
                        'tool.call.failed',
                        'tool.call.blocked',
                        'tool.call.approval_denied'
                    ) THEN jsonb_strip_nulls(jsonb_build_object(
                        'tool_call_id', payload -> 'tool_call_id',
                        'model_step_id', payload -> 'model_step_id',
                        'model_attempt_id', payload -> 'model_attempt_id',
                        'tool_round_id', payload -> 'tool_round_id',
                        'manual_recovery', payload -> 'manual_recovery',
                        '__cayu_terminal_result_valid__',
                        CASE WHEN event_type = 'tool.call.started' THEN NULL ELSE COALESCE(
                        jsonb_typeof(payload -> 'result') = 'object'
                        AND (payload -> 'result')
                            - ARRAY['content', 'structured', 'artifacts', 'is_error']
                            = '{{}}'::jsonb
                        AND (
                            NOT ((payload -> 'result') ? 'content')
                            OR jsonb_typeof(payload #> '{{result,content}}') = 'string'
                        )
                        AND (
                            NOT ((payload -> 'result') ? 'structured')
                            OR jsonb_typeof(payload #> '{{result,structured}}')
                                IN ('object', 'null')
                        )
                        AND (
                            NOT ((payload -> 'result') ? 'artifacts')
                            OR (
                                jsonb_typeof(payload #> '{{result,artifacts}}') = 'array'
                                AND NOT EXISTS (
                                    SELECT 1
                                    FROM jsonb_array_elements(
                                        CASE
                                            WHEN jsonb_typeof(
                                                payload #> '{{result,artifacts}}'
                                            ) = 'array'
                                            THEN payload #> '{{result,artifacts}}'
                                            ELSE '[]'::jsonb
                                        END
                                    ) AS artifact
                                    WHERE jsonb_typeof(artifact) <> 'object'
                                )
                            )
                        )
                        AND (
                            NOT ((payload -> 'result') ? 'is_error')
                            OR jsonb_typeof(payload #> '{{result,is_error}}') = 'boolean'
                        ), FALSE) END
                    ))
                    WHEN event_type = 'session.resumed' THEN
                        jsonb_strip_nulls(jsonb_build_object(
                            'model_step_id', payload -> 'model_step_id',
                            'model_attempt_id', payload -> 'model_attempt_id',
                            'tool_round_id', payload -> 'tool_round_id'
                        ))
                    WHEN event_type = 'session.failed' THEN
                        jsonb_strip_nulls(jsonb_build_object(
                            'tool_evidence_conflict', payload -> 'tool_evidence_conflict'
                        ))
                    ELSE '{{}}'::jsonb
                END,
                true
            ) AS projection
        FROM batch
    ),
    measured AS MATERIALIZED (
        SELECT sequence, lookup_id, projection, octet_length(projection::text) AS bytes
        FROM projected
    )
    UPDATE cayu_events AS target
    SET pending_action_lookup_key = CASE
            WHEN measured.lookup_id IS NULL THEN NULL
            ELSE encode(sha256(convert_to(measured.lookup_id, 'UTF8')), 'hex')
        END,
        pending_action_projection = CASE
            WHEN measured.bytes <= {MAX_PENDING_ACTION_RESULT_BYTES}
            THEN measured.projection
            ELSE jsonb_build_object(
                'type', measured.projection -> 'type',
                'session_id', 'cayu_oversized_pending_action_projection',
                'interaction_id', measured.projection -> 'interaction_id',
                'id', 'cayu_oversized_pending_action_projection',
                'timestamp', measured.projection -> 'timestamp',
                'agent_name', NULL,
                'environment_name', NULL,
                'workflow_name', NULL,
                'tool_name', NULL,
                'payload', jsonb_strip_nulls(jsonb_build_object(
                    'model_step_id', CASE
                        WHEN jsonb_typeof(
                            measured.projection #> '{{payload,model_step_id}}'
                        ) = 'string'
                          AND char_length(
                              measured.projection #>> '{{payload,model_step_id}}'
                          ) <= {EXECUTION_UNIT_ID_MAX_CHARS}
                        THEN measured.projection #> '{{payload,model_step_id}}'
                        ELSE NULL
                    END,
                    'model_attempt_id', CASE
                        WHEN jsonb_typeof(
                            measured.projection #> '{{payload,model_attempt_id}}'
                        ) = 'string'
                          AND char_length(
                              measured.projection #>> '{{payload,model_attempt_id}}'
                          ) <= {EXECUTION_UNIT_ID_MAX_CHARS}
                        THEN measured.projection #> '{{payload,model_attempt_id}}'
                        ELSE NULL
                    END,
                    'tool_round_id', CASE
                        WHEN jsonb_typeof(
                            measured.projection #> '{{payload,tool_round_id}}'
                        ) = 'string'
                          AND char_length(
                              measured.projection #>> '{{payload,tool_round_id}}'
                          ) <= {EXECUTION_UNIT_ID_MAX_CHARS}
                        THEN measured.projection #> '{{payload,tool_round_id}}'
                        ELSE NULL
                    END,
                    '__cayu_pending_action_projection_bytes__',
                    {MAX_PENDING_ACTION_RESULT_BYTES + 1}
                ))
            )
        END,
        pending_action_projection_bytes = CASE
            WHEN measured.bytes <= {MAX_PENDING_ACTION_RESULT_BYTES}
            THEN measured.bytes
            ELSE {MAX_PENDING_ACTION_RESULT_BYTES + 1}
        END
    FROM measured
    WHERE target.sequence = measured.sequence
    RETURNING target.sequence
"""


_REVISION_17_EVENT_BACKFILL_SMALL_SQL = _revision_17_event_backfill_sql(
    source_predicate=(
        f"octet_length(event::text) <= {_REVISION_17_EVENT_BACKFILL_SMALL_EVENT_BYTES}"
    ),
    batch_limit=25,
)
_REVISION_17_EVENT_BACKFILL_LARGE_SQL = _revision_17_event_backfill_sql(
    source_predicate=(
        f"octet_length(event::text) > {_REVISION_17_EVENT_BACKFILL_SMALL_EVENT_BYTES}"
    ),
    batch_limit=1,
)


def _revision_17_event_backfill_remaining_sql(source_predicate: str) -> str:
    return f"""
        SELECT EXISTS(
            SELECT 1
            FROM cayu_events
            WHERE pending_action_projection_bytes IS NULL
              AND ({source_predicate})
              AND event_type IN (
                  'tool.call.approval_requested',
                  'session.awaiting_user_input',
                  'session.interrupted',
                  'session.resumed',
                  'session.completed',
                  'session.failed',
                  'tool.call.started',
                  'tool.call.completed',
                  'tool.call.failed',
                  'tool.call.blocked',
                  'tool.call.approval_denied'
              )
        )
    """


_REVISION_17_EVENT_BACKFILL_SMALL_REMAINING_SQL = _revision_17_event_backfill_remaining_sql(
    f"octet_length(event::text) <= {_REVISION_17_EVENT_BACKFILL_SMALL_EVENT_BYTES}"
)
_REVISION_17_EVENT_BACKFILL_LARGE_REMAINING_SQL = _revision_17_event_backfill_remaining_sql(
    f"octet_length(event::text) > {_REVISION_17_EVENT_BACKFILL_SMALL_EVENT_BYTES}"
)


# These revisions cannot run inside the schema transaction. The baseline still
# creates the same indexes normally because its tables are empty; existing hot
# databases use CONCURRENTLY so checkpoint and event writes remain available
# during upgrades.
def _normalize_postgres_index_expression(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.lower().replace('"', "")
    normalized = normalized.replace("::text[]", "").replace("::text", "")
    return re.sub(r"[\s()]", "", normalized)


@dataclass(frozen=True)
class _ConcurrentIndexMigration:
    index_name: str
    table_name: str
    key_definitions: tuple[str, ...]
    predicate_definition: str | None
    create_statement: str
    drop_statement: str
    required_key_collations: tuple[str | None, ...] = ()
    unique: bool = False


_CONCURRENT_INDEX_MIGRATIONS: dict[int, tuple[_ConcurrentIndexMigration, ...]] = {
    16: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_events_session_sequence",
            table_name="cayu_events",
            key_definitions=("session_id", "sequence"),
            predicate_definition=None,
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cayu_events_session_sequence "
                "ON cayu_events(session_id, sequence)"
            ),
            drop_statement=("DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_events_session_sequence"),
        ),
    ),
    17: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_checkpoints_pending_control_action",
            table_name="cayu_checkpoints",
            key_definitions=("session_id",),
            predicate_definition=("pending_action_flags <> 0"),
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_checkpoints_pending_control_action "
                "ON cayu_checkpoints(session_id) WHERE pending_action_flags <> 0"
            ),
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_checkpoints_pending_control_action"
            ),
        ),
        _ConcurrentIndexMigration(
            index_name="idx_cayu_events_pending_action_barrier",
            table_name="cayu_events",
            key_definitions=("session_id", "sequence"),
            predicate_definition="""
                event_type = 'session.resumed'
                OR event_type = 'session.completed'
                OR event_type = 'session.failed'
            """,
            create_statement="""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS
                    idx_cayu_events_pending_action_barrier
                ON cayu_events(session_id, sequence)
                WHERE event_type = 'session.resumed'
                   OR event_type = 'session.completed'
                   OR event_type = 'session.failed'
            """,
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_events_pending_action_barrier"
            ),
        ),
        _ConcurrentIndexMigration(
            index_name="idx_cayu_events_pending_action_lookup",
            table_name="cayu_events",
            key_definitions=(
                "session_id",
                "pending_action_lookup_key",
                "event_type",
                "sequence",
            ),
            predicate_definition="""
                event_type = ANY (ARRAY[
                    'tool.call.approval_requested',
                    'session.awaiting_user_input',
                    'session.interrupted',
                    'tool.call.started',
                    'tool.call.completed',
                    'tool.call.failed',
                    'tool.call.blocked',
                    'tool.call.approval_denied'
                ])
                AND pending_action_lookup_key IS NOT NULL
            """,
            create_statement="""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS
                    idx_cayu_events_pending_action_lookup
                ON cayu_events(
                    session_id,
                    pending_action_lookup_key,
                    event_type,
                    sequence
                )
                WHERE event_type IN (
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
            """,
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_events_pending_action_lookup"
            ),
        ),
    ),
    26: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_events_session_interaction_sequence",
            table_name="cayu_events",
            key_definitions=("session_id", "interaction_id", "sequence"),
            predicate_definition=None,
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_events_session_interaction_sequence "
                "ON cayu_events(session_id, interaction_id, sequence)"
            ),
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_events_session_interaction_sequence"
            ),
        ),
        _ConcurrentIndexMigration(
            index_name="idx_cayu_transcript_messages_session_interaction_sequence",
            table_name="cayu_transcript_messages",
            key_definitions=("session_id", "interaction_id", "sequence"),
            predicate_definition=None,
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_transcript_messages_session_interaction_sequence "
                "ON cayu_transcript_messages(session_id, interaction_id, sequence)"
            ),
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS "
                "idx_cayu_transcript_messages_session_interaction_sequence"
            ),
        ),
    ),
    23: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_events_budget_reservation_identity",
            table_name="cayu_events",
            key_definitions=("payload ->> 'reservation_id'",),
            predicate_definition="""
                event_type = 'budget.reserved'
                AND jsonb_typeof(payload -> 'reservation_id') = 'string'
            """,
            create_statement="""
                CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS
                    idx_cayu_events_budget_reservation_identity
                ON cayu_events ((payload ->> 'reservation_id'))
                WHERE event_type = 'budget.reserved'
                  AND jsonb_typeof(payload -> 'reservation_id') = 'string'
            """,
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_events_budget_reservation_identity"
            ),
            unique=True,
        ),
        _ConcurrentIndexMigration(
            index_name="idx_cayu_events_pending_action_round_scope",
            table_name="cayu_events",
            key_definitions=(
                "session_id",
                "pending_action_projection #>> '{payload,tool_round_id}'",
                "sequence",
            ),
            predicate_definition="""
                event_type = ANY (ARRAY[
                    'tool.call.started',
                    'tool.call.completed',
                    'tool.call.failed',
                    'tool.call.blocked',
                    'tool.call.approval_denied'
                ])
                AND jsonb_typeof(
                    pending_action_projection #> '{payload,tool_round_id}'
                ) = 'string'
                AND pending_action_projection #>> '{payload,tool_round_id}'
                    ~ '^tround_[0-9a-f]{32}$'
            """,
            create_statement="""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS
                    idx_cayu_events_pending_action_round_scope
                ON cayu_events(
                    session_id,
                    (pending_action_projection #>> '{payload,tool_round_id}'),
                    sequence
                )
                WHERE event_type IN (
                    'tool.call.started',
                    'tool.call.completed',
                    'tool.call.failed',
                    'tool.call.blocked',
                    'tool.call.approval_denied'
                )
                  AND jsonb_typeof(
                      pending_action_projection #> '{payload,tool_round_id}'
                  ) = 'string'
                  AND pending_action_projection #>> '{payload,tool_round_id}'
                      ~ '^tround_[0-9a-f]{32}$'
            """,
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_events_pending_action_round_scope"
            ),
        ),
        _ConcurrentIndexMigration(
            index_name="idx_cayu_events_pending_action_attempt_scope",
            table_name="cayu_events",
            key_definitions=(
                "session_id",
                "pending_action_projection #>> '{payload,model_step_id}'",
                "pending_action_projection #>> '{payload,model_attempt_id}'",
                "sequence",
            ),
            predicate_definition="""
                event_type = ANY (ARRAY[
                    'tool.call.started',
                    'tool.call.completed',
                    'tool.call.failed',
                    'tool.call.blocked',
                    'tool.call.approval_denied'
                ])
                AND jsonb_typeof(
                    pending_action_projection #> '{payload,model_step_id}'
                ) = 'string'
                AND jsonb_typeof(
                    pending_action_projection #> '{payload,model_attempt_id}'
                ) = 'string'
                AND pending_action_projection #>> '{payload,model_step_id}'
                    ~ '^mstep_[0-9a-f]{32}$'
                AND pending_action_projection #>> '{payload,model_attempt_id}'
                    ~ '^matt_[0-9a-f]{32}$'
            """,
            create_statement="""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS
                    idx_cayu_events_pending_action_attempt_scope
                ON cayu_events(
                    session_id,
                    (pending_action_projection #>> '{payload,model_step_id}'),
                    (pending_action_projection #>> '{payload,model_attempt_id}'),
                    sequence
                )
                WHERE event_type IN (
                    'tool.call.started',
                    'tool.call.completed',
                    'tool.call.failed',
                    'tool.call.blocked',
                    'tool.call.approval_denied'
                )
                  AND jsonb_typeof(
                      pending_action_projection #> '{payload,model_step_id}'
                  ) = 'string'
                  AND jsonb_typeof(
                      pending_action_projection #> '{payload,model_attempt_id}'
                  ) = 'string'
                  AND pending_action_projection #>> '{payload,model_step_id}'
                      ~ '^mstep_[0-9a-f]{32}$'
                  AND pending_action_projection #>> '{payload,model_attempt_id}'
                      ~ '^matt_[0-9a-f]{32}$'
            """,
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_events_pending_action_attempt_scope"
            ),
        ),
    ),
    24: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_sessions_parent_created_id",
            table_name="cayu_sessions",
            key_definitions=("parent_session_id", "created_at", "id"),
            predicate_definition=None,
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_sessions_parent_created_id "
                "ON cayu_sessions(parent_session_id, created_at, id)"
            ),
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_sessions_parent_created_id"
            ),
        ),
    ),
    27: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_tasks_session_created_id",
            table_name="cayu_tasks",
            key_definitions=("session_id", "created_at", "id"),
            predicate_definition=None,
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_tasks_session_created_id "
                'ON cayu_tasks(session_id, created_at, id COLLATE "C")'
            ),
            drop_statement=("DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_tasks_session_created_id"),
            required_key_collations=(None, None, "C"),
        ),
        _ConcurrentIndexMigration(
            index_name="idx_cayu_tasks_parent_created_id",
            table_name="cayu_tasks",
            key_definitions=("parent_task_id", "created_at", "id"),
            predicate_definition=None,
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_tasks_parent_created_id "
                'ON cayu_tasks(parent_task_id, created_at, id COLLATE "C")'
            ),
            drop_statement=("DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_tasks_parent_created_id"),
            required_key_collations=(None, None, "C"),
        ),
    ),
}


def _required_concurrent_indexes(revision: int) -> tuple[_ConcurrentIndexMigration, ...]:
    return tuple(
        index
        for index_revision, indexes in sorted(_CONCURRENT_INDEX_MIGRATIONS.items())
        if index_revision <= revision
        for index in indexes
    )


async def read_schema_state(cur: Any) -> schema.SchemaState:
    """Read the recorded schema state from an open cursor without applying DDL.

    Returns :data:`schema.UNINITIALIZED` (rather than raising) when the
    bookkeeping table is absent, so it is safe to call against any database.
    """
    # to_regclass returns NULL (not an error) when the table is absent, so an
    # uninitialized database reads as UNINITIALIZED rather than raising.
    await cur.execute("SELECT to_regclass('cayu_schema_migrations')")
    registered = await cur.fetchone()
    if registered is None or registered[0] is None:
        return schema.SchemaState(revision=schema.UNINITIALIZED, compatible_from=0)
    await cur.execute(
        "SELECT revision, compatible_from FROM cayu_schema_migrations "
        "ORDER BY revision DESC LIMIT 1"
    )
    latest = await cur.fetchone()
    if latest is None:
        return schema.SchemaState(revision=schema.UNINITIALIZED, compatible_from=0)
    return schema.SchemaState(revision=latest[0], compatible_from=latest[1])


async def _reject_populated_pre_interaction_database(cur: Any) -> None:
    await cur.execute("SELECT EXISTS(SELECT 1 FROM cayu_sessions)")
    row = await cur.fetchone()
    if row is not None and row[0] is True:
        raise schema.SchemaTooOld(
            "Storage revision 26 is a clean prerelease break and cannot migrate a "
            "populated Cayu session database. Recreate the Cayu database before "
            "starting this build."
        )


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


async def _disable_prepared_statements(conn: Any) -> None:
    """Pool ``configure`` hook: disable psycopg3 server-side prepared statements.

    Required when the store's own pool runs behind a transaction-pooling pgbouncer
    (e.g. Fly Managed Postgres), where prepared statements raise
    "prepared statement ... already exists". Harmless on a direct connection.
    """
    conn.prepare_threshold = None


async def _configure_store_connection(conn: Any) -> None:
    await _disable_prepared_statements(conn)


async def _acquire_schema_transaction_lock(
    conn: Any,
    cur: Any,
    *,
    read_only: bool = False,
) -> None:
    """Acquire the schema lock without leaving a waiting transaction open.

    Concurrent index DDL holds the same key as a session advisory lock. A
    blocking transaction-lock request would retain its virtual transaction ID
    while waiting, which ``CREATE INDEX CONCURRENTLY`` can in turn wait on and
    deadlock. End every unsuccessful try before polling again.
    """
    while True:
        await cur.execute(
            "SELECT pg_try_advisory_xact_lock(%s)",
            (_SCHEMA_ADVISORY_LOCK_KEY,),
        )
        row = await cur.fetchone()
        if row is not None and row[0] is True:
            return
        await conn.rollback()
        if read_only:
            await conn.execute("SET TRANSACTION READ ONLY")
        await asyncio.sleep(_SCHEMA_ADVISORY_LOCK_POLL_SECONDS)


class _PostgresStoreBase:
    """Shared async connection-pool management for Postgres-backed stores.

    The pool is created eagerly (closed) and opened lazily on first use so that
    it is bound to the event loop that actually drives the store. This mirrors
    the way the SQLite store opens its connection in ``__init__`` while keeping
    psycopg's async pool happy about running inside a live loop.
    """

    _min_required_revision = _POSTGRES_MIN_REQUIRED_REVISION
    _supports_read_only = False

    def __init__(
        self,
        conninfo: str | None = None,
        *,
        pool: AsyncConnectionPool | None = None,
        min_size: int = 1,
        max_size: int = 8,
        schema_mode: schema.SchemaMode = schema.SchemaMode.VALIDATE,
        read_only: bool = False,
    ) -> None:
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")
        if type(read_only) is not bool:
            raise TypeError("read_only must be a bool.")
        if read_only and not self._supports_read_only:
            raise ValueError("read_only is only supported by PostgresSessionStore.")
        if read_only and schema_mode is not schema.SchemaMode.VALIDATE:
            raise ValueError("Read-only Postgres stores require schema_mode=VALIDATE.")
        self._schema_mode = schema_mode
        self._read_only = read_only
        if pool is not None:
            if read_only:
                raise ValueError("read_only requires a store-owned Postgres connection pool.")
            if not isinstance(pool, AsyncConnectionPool):
                raise TypeError("pool must be an AsyncConnectionPool.")
            self._pool = pool
            self._owns_pool = False
            self._conninfo = None
        else:
            if type(conninfo) is not str:
                raise TypeError("conninfo must be a string.")
            self._conninfo = require_nonblank(conninfo, "conninfo")
            self._pool = AsyncConnectionPool(
                self._conninfo,
                min_size=min_size,
                max_size=max_size,
                open=False,
                # Disable server-side prepared statements so the store works behind
                # a transaction-pooling pgbouncer (e.g. Fly Managed Postgres), where
                # prepared statements raise "prepared statement already exists".
                configure=_configure_store_connection,
            )
            self._owns_pool = True
        self._open_lock = asyncio.Lock()
        self._opened = False
        self._schema_ready = False

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[Any]:
        async with self._pool.connection() as conn:
            if self._read_only:
                # Keep the guard in the same transaction as the store operation.
                # Session defaults are not stable behind transaction-pooled PgBouncer.
                await conn.execute("SET TRANSACTION READ ONLY")
            yield conn

    async def ensure_schema(self) -> None:
        """Open the pool and reconcile the schema now (per ``schema_mode``).

        Normally reconciliation happens lazily on first use; the ``cayu storage``
        CLI calls this to run a ``migrate`` (or ``validate``) as an explicit step.
        """
        await self._ensure_ready()

    async def _ensure_ready(self) -> None:
        if self._opened and self._schema_ready:
            return
        async with self._open_lock:
            if not self._opened:
                await self._pool.open()
                self._opened = True
            if not self._schema_ready:
                await self._reconcile_schema()
                self._schema_ready = True

    async def _reconcile_schema(self) -> None:
        """Reconcile the database schema with this binary per ``schema_mode``.

        Concurrent stores serialize transactional schema work on one
        transaction-scoped advisory lock (ADR 0001, Decision 4). Concurrent index
        DDL necessarily runs outside that transaction, so it polls the same key as
        a short-lived session advisory lock while validating or building each
        index. The lock is held only on this dedicated migration connection, which
        keeps normal store traffic safe behind transaction-pooled PgBouncer:

        - ``validate``: read the recorded revision and fail fast unless this binary
          can operate against it. Never runs DDL.
        - ``create``: initialize the baseline schema on an empty database; otherwise
          validate. The dev/test/local default.
        - ``migrate``: apply pending forward revisions under the lock, then validate.
        """
        if self._schema_mode is schema.SchemaMode.MIGRATE:
            await self._migrate_schema()
            return

        async with self._connection() as conn:
            async with conn.cursor() as cur:
                await _acquire_schema_transaction_lock(
                    conn,
                    cur,
                    read_only=self._read_only,
                )
                if self._schema_mode is not schema.SchemaMode.VALIDATE:
                    await cur.execute(pg_support.MIGRATIONS_TABLE_DDL)
                state = await self._read_schema_state(cur)
                if self._schema_mode is schema.SchemaMode.VALIDATE:
                    schema.validate(
                        state,
                        app_min_supported=self._min_required_revision,
                    )
                    await self._validate_postgres_schema(cur, state)
                elif self._schema_mode is schema.SchemaMode.CREATE:
                    if state.revision == schema.UNINITIALIZED:
                        await self._apply_pending(cur, state)
                    else:
                        schema.validate(
                            state,
                            app_min_supported=self._min_required_revision,
                        )
                        await self._validate_postgres_schema(cur, state)
            await conn.commit()

    async def _migrate_schema(self) -> None:
        while True:
            concurrent_revision: schema.Revision | None = None
            concurrent_indexes: tuple[_ConcurrentIndexMigration, ...] = ()
            recorded_indexes: tuple[_ConcurrentIndexMigration, ...] = ()
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await _acquire_schema_transaction_lock(conn, cur)
                    await cur.execute(pg_support.MIGRATIONS_TABLE_DDL)
                    state = await self._read_schema_state(cur)
                    current = state.revision
                    if (
                        current != schema.UNINITIALIZED
                        and current < 26
                        and any(revision.revision == 26 for revision in schema.pending(current))
                    ):
                        # Reject before applying any earlier pending revision so
                        # the clean break cannot leave a database half-migrated.
                        await _reject_populated_pre_interaction_database(cur)
                    if current == schema.UNINITIALIZED:
                        await self._apply_baseline(cur)
                        current = schema.BASELINE_REVISION
                    pending = schema.pending(current)
                    if not pending:
                        current_state = await self._read_schema_state(cur)
                        schema.validate(
                            current_state,
                            app_min_supported=self._min_required_revision,
                        )
                        self._validate_postgres_revision(current_state)
                        if current_state.revision >= 23:
                            await self._validate_budget_reservation_identity_registry(
                                cur,
                                require=True,
                            )
                        recorded_indexes = _required_concurrent_indexes(current_state.revision)
                    else:
                        revision = pending[0]
                        concurrent_indexes = _CONCURRENT_INDEX_MIGRATIONS.get(
                            revision.revision,
                            (),
                        )
                        if concurrent_indexes:
                            # A revision may pair small transactional objects
                            # with hot-table indexes that must be built outside a
                            # transaction. Record it only after both phases pass.
                            for statement in _MIGRATION_STEPS.get(revision.revision, ()):
                                await cur.execute(cast("LiteralString", statement))
                            concurrent_revision = revision
                        else:
                            for statement in _MIGRATION_STEPS.get(revision.revision, ()):
                                await cur.execute(cast("LiteralString", statement))
                            await self._record_revision(cur, revision)
                await conn.commit()
            if concurrent_revision is None:
                if not pending:
                    for index in recorded_indexes:
                        async with self._pool.connection() as conn:
                            await self._ensure_concurrent_index(conn, index)
                    return
                continue

            if concurrent_revision.revision == 17:
                await self._backfill_revision_seventeen()

            for index in concurrent_indexes:
                async with self._pool.connection() as conn:
                    await self._ensure_concurrent_index(
                        conn,
                        index,
                    )

            # Record the revision only after every non-transactional object is
            # valid. A competing migrator may have recorded it while this process
            # built or waited for the same index, so re-read under the xact lock.
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await _acquire_schema_transaction_lock(conn, cur)
                state = await self._read_schema_state(cur)
                if state.revision < concurrent_revision.revision:
                    if concurrent_revision.revision == 23:
                        await self._validate_budget_reservation_identity_registry(
                            cur,
                            require=True,
                            verify_event_ownership=True,
                        )
                    await self._record_revision(cur, concurrent_revision)
                await conn.commit()

    async def _backfill_revision_seventeen(self) -> None:
        await self._run_resumable_checkpoint_backfill(
            _REVISION_17_CHECKPOINT_BACKFILL_SQL,
            "SELECT EXISTS(SELECT 1 FROM cayu_checkpoints WHERE NOT pending_action_metrics_ready)",
        )
        await self._run_resumable_sequence_backfill(
            _REVISION_17_EVENT_BACKFILL_SMALL_SQL,
            _REVISION_17_EVENT_BACKFILL_SMALL_REMAINING_SQL,
        )
        await self._run_resumable_sequence_backfill(
            _REVISION_17_EVENT_BACKFILL_LARGE_SQL,
            _REVISION_17_EVENT_BACKFILL_LARGE_REMAINING_SQL,
        )

    async def _run_resumable_checkpoint_backfill(
        self,
        batch_sql: str,
        remaining_sql: str,
    ) -> None:
        after_session_id: str | None = None
        while True:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    cast("LiteralString", batch_sql),
                    (after_session_id, after_session_id),
                )
                updated = await cur.fetchall()
                if updated:
                    after_session_id = max(str(row[0]) for row in updated)
                    await conn.commit()
                    continue
                await cur.execute(cast("LiteralString", remaining_sql))
                row = await cur.fetchone()
                remaining = row is not None and row[0] is True
                await conn.commit()
            if not remaining:
                return
            # Catch rows skipped behind the local cursor because another
            # migrator held them. A crash simply restarts this scan from zero.
            after_session_id = None
            await asyncio.sleep(0.05)

    async def _run_resumable_sequence_backfill(
        self,
        batch_sql: str,
        remaining_sql: str,
    ) -> None:
        after_sequence = 0
        while True:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(cast("LiteralString", batch_sql), (after_sequence,))
                updated = await cur.fetchall()
                if updated:
                    after_sequence = max(int(row[0]) for row in updated)
                    await conn.commit()
                    continue
                await cur.execute(cast("LiteralString", remaining_sql))
                row = await cur.fetchone()
                remaining = row is not None and row[0] is True
                await conn.commit()
            if not remaining:
                return
            # Catch rows skipped behind the local cursor because another
            # migrator held them. A crash simply restarts this scan from zero.
            after_sequence = 0
            await asyncio.sleep(0.05)

    def _validate_postgres_revision(self, state: schema.SchemaState) -> None:
        if state.revision < self._min_required_revision:
            raise schema.SchemaTooOld(
                f"Postgres schema is at revision {state.revision}; this build requires "
                f">= {self._min_required_revision}. Run `cayu storage migrate` before "
                "starting."
            )

    async def _validate_postgres_schema(self, cur: Any, state: schema.SchemaState) -> None:
        self._validate_postgres_revision(state)
        if state.revision >= 23:
            await self._validate_budget_reservation_identity_registry(
                cur,
                require=True,
            )
        for index in _required_concurrent_indexes(state.revision):
            existing = await self._concurrent_index_state(cur, index)
            if existing is None:
                raise RuntimeError(
                    f"Required Cayu Postgres index is missing: {index.index_name}. "
                    "Run `cayu storage migrate` to repair the schema."
                )
            valid, building = existing
            if not valid or building:
                raise RuntimeError(
                    f"Required Cayu Postgres index is not ready: {index.index_name}. "
                    "Run `cayu storage migrate` to repair the schema."
                )

    async def _validate_budget_reservation_identity_registry(
        self,
        cur: Any,
        *,
        require: bool,
        verify_event_ownership: bool = False,
    ) -> bool:
        table_name = "cayu_budget_reservation_identities"
        await cur.execute("SELECT to_regclass(%s)", (table_name,))
        registered = await cur.fetchone()
        if registered is None or registered[0] is None:
            if require:
                raise RuntimeError(
                    f"Required Cayu Postgres table is missing: {table_name}. "
                    "Restore the permanent reservation ownership registry from "
                    "a known-good backup."
                )
            return False
        await cur.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = %s
            ORDER BY ordinal_position
            """,
            (table_name,),
        )
        columns = tuple(await cur.fetchall())
        await cur.execute(
            """
            SELECT pg_get_constraintdef(constraint_record.oid)
            FROM pg_catalog.pg_constraint AS constraint_record
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.oid = constraint_record.conrelid
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = table_record.relnamespace
            WHERE namespace.nspname = current_schema()
              AND table_record.relname = %s
              AND constraint_record.contype = 'p'
            """,
            (table_name,),
        )
        primary_keys = tuple(row[0] for row in await cur.fetchall())
        expected_columns = (
            ("reservation_id", "text", "NO"),
            ("publication_session_id", "text", "NO"),
            ("publication_id", "text", "NO"),
            ("published", "boolean", "NO"),
        )
        if columns != expected_columns or primary_keys != ("PRIMARY KEY (reservation_id)",):
            raise RuntimeError(
                f"Postgres schema object {table_name!r} conflicts with Cayu's "
                "reservation identity contract. Restore the required ownership "
                "registry from a known-good backup."
            )
        if not verify_event_ownership:
            return True
        await cur.execute(
            """
            SELECT 1
            FROM cayu_events AS event
            LEFT JOIN cayu_budget_reservation_identities AS identity
              ON identity.reservation_id = event.payload ->> 'reservation_id'
            WHERE event.event_type = 'budget.reserved'
              AND jsonb_typeof(event.payload -> 'reservation_id') = 'string'
              AND (
                  identity.reservation_id IS NULL
                  OR identity.publication_session_id <> event.session_id
                  OR identity.publication_id <> event.event_id
                  OR NOT identity.published
              )
            LIMIT 1
            """
        )
        if await cur.fetchone() is not None:
            raise RuntimeError(
                "Postgres budget reservation events disagree with the permanent "
                "reservation ownership registry."
            )
        return True

    async def _read_schema_state(self, cur: Any) -> schema.SchemaState:
        return await read_schema_state(cur)

    async def _apply_baseline(self, cur: Any) -> None:
        for statement in pg_support.SCHEMA_STATEMENTS:
            await cur.execute(statement)
        await self._record_revision(cur, schema.revision(schema.BASELINE_REVISION))

    async def _apply_pending(self, cur: Any, state: schema.SchemaState) -> None:
        current = state.revision
        if (
            current != schema.UNINITIALIZED
            and current < 26
            and any(revision.revision == 26 for revision in schema.pending(current))
        ):
            await _reject_populated_pre_interaction_database(cur)
        if current == schema.UNINITIALIZED:
            await self._apply_baseline(cur)
            current = schema.BASELINE_REVISION
        for rev in schema.pending(current):
            for statement in _MIGRATION_STEPS.get(rev.revision, ()):
                await cur.execute(statement)
            await self._record_revision(cur, rev)

    async def _ensure_concurrent_index(
        self,
        conn: Any,
        index: _ConcurrentIndexMigration,
    ) -> None:
        await conn.set_autocommit(True)
        lock_acquired = False
        try:
            # CREATE INDEX CONCURRENTLY cannot run under the transaction-level
            # schema lock. Poll a session lock with pg_try_advisory_lock: a
            # blocking advisory-lock statement would hold a virtual xid while
            # it waits and can deadlock the winning CREATE INDEX CONCURRENTLY.
            # Each failed try completes its autocommit transaction before sleep.
            while not lock_acquired:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT pg_try_advisory_lock(%s)",
                        (_SCHEMA_ADVISORY_LOCK_KEY,),
                    )
                    row = await cur.fetchone()
                    lock_acquired = row is not None and row[0] is True
                if not lock_acquired:
                    await asyncio.sleep(_SCHEMA_ADVISORY_LOCK_POLL_SECONDS)

            while True:
                async with conn.cursor() as cur:
                    existing = await self._concurrent_index_state(
                        cur,
                        index,
                    )
                    if existing == (True, False):
                        return
                    if existing is not None and existing[1]:
                        await asyncio.sleep(_SCHEMA_ADVISORY_LOCK_POLL_SECONDS)
                        continue
                    if existing is not None:
                        await cur.execute(index.drop_statement)
                    try:
                        await cur.execute(index.create_statement)
                    except (DeadlockDetected, DuplicateTable):
                        continue
                    except UniqueViolation as exc:
                        if not index.unique:
                            continue
                        raise RuntimeError(
                            f"Postgres migration cannot create required unique index "
                            f"{index.index_name}: durable rows contain duplicate identities."
                        ) from exc
                    created = await self._concurrent_index_state(
                        cur,
                        index,
                    )
                    if created == (True, False):
                        return
                    if created is None or not created[1]:
                        raise RuntimeError(
                            f"Postgres migration did not create a valid index: {index.index_name}"
                        )
        finally:
            if lock_acquired:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT pg_advisory_unlock(%s)",
                        (_SCHEMA_ADVISORY_LOCK_KEY,),
                    )
            await conn.set_autocommit(False)

    async def _concurrent_index_state(
        self,
        cur: Any,
        index: _ConcurrentIndexMigration,
    ) -> tuple[bool, bool] | None:
        await cur.execute(
            """
            SELECT
                index_definition.indexrelid IS NOT NULL,
                COALESCE(index_definition.indisvalid, FALSE),
                COALESCE(
                    table_class.relnamespace = namespace.oid
                    AND table_class.relname = %s,
                    FALSE
                ),
                COALESCE(access_method.amname = 'btree', FALSE),
                ARRAY(
                    SELECT pg_get_indexdef(
                        index_definition.indexrelid,
                        key_position,
                        FALSE
                    )
                    FROM generate_series(
                        1,
                        index_definition.indnkeyatts
                    ) AS key_position
                    ORDER BY key_position
                ),
                ARRAY(
                    SELECT index_collation.collname
                    FROM unnest(index_definition.indcollation::oid[])
                         WITH ORDINALITY AS key_collation(
                             collation_oid,
                             key_position
                         )
                    LEFT JOIN pg_catalog.pg_collation AS index_collation
                      ON index_collation.oid = key_collation.collation_oid
                    WHERE key_collation.key_position
                          <= index_definition.indnkeyatts
                    ORDER BY key_collation.key_position
                ),
                pg_get_expr(
                    index_definition.indpred,
                    index_definition.indrelid,
                    FALSE
                ),
                COALESCE(index_definition.indisunique, FALSE),
                EXISTS (
                    SELECT 1
                    FROM pg_catalog.pg_stat_progress_create_index AS progress
                    WHERE progress.index_relid = index_class.oid
                )
            FROM pg_catalog.pg_class AS index_class
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = index_class.relnamespace
            LEFT JOIN pg_catalog.pg_index AS index_definition
              ON index_definition.indexrelid = index_class.oid
            LEFT JOIN pg_catalog.pg_class AS table_class
              ON table_class.oid = index_definition.indrelid
            LEFT JOIN pg_catalog.pg_am AS access_method
              ON access_method.oid = index_class.relam
            WHERE namespace.nspname = current_schema()
              AND index_class.relname = %s
            """,
            (index.table_name, index.index_name),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        key_definitions = tuple(
            _normalize_postgres_index_expression(str(value)) for value in (row[4] or [])
        )
        expected_keys = tuple(
            _normalize_postgres_index_expression(value) for value in index.key_definitions
        )
        key_collations = tuple(None if value is None else str(value) for value in (row[5] or []))
        required_key_collations = index.required_key_collations
        collations_match = not required_key_collations or (
            len(key_collations) == len(required_key_collations)
            and all(
                required is None or actual == required
                for actual, required in zip(
                    key_collations,
                    required_key_collations,
                    strict=True,
                )
            )
        )
        predicate = _normalize_postgres_index_expression(row[6])
        expected_predicate = _normalize_postgres_index_expression(index.predicate_definition)
        expected_definition = (
            bool(row[0])
            and bool(row[2])
            and bool(row[3])
            and key_definitions == expected_keys
            and collations_match
            and predicate == expected_predicate
            and bool(row[7]) is index.unique
        )
        if not expected_definition:
            columns = ", ".join(
                (f'{key} COLLATE "{collation}"' if collation is not None else key)
                for key, collation in zip(
                    index.key_definitions,
                    index.required_key_collations or (None,) * len(index.key_definitions),
                    strict=True,
                )
            )
            index_kind = "unique B-tree" if index.unique else "B-tree"
            raise RuntimeError(
                f"Postgres schema object {index.index_name!r} conflicts with the required "
                f"{index_kind} index on {index.table_name}({columns}). Remove or rename the "
                "conflicting object, then rerun `cayu storage migrate`."
            )
        return bool(row[1]), bool(row[8])

    async def _record_revision(self, cur: Any, rev: schema.Revision) -> None:
        await cur.execute(
            "INSERT INTO cayu_schema_migrations "
            "(revision, kind, compatible_from, checksum, applied_at) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (revision) DO NOTHING",
            (rev.revision, str(rev.kind), rev.compatible_from, None, datetime.now(UTC)),
        )

    async def close(self) -> None:
        if self._owns_pool and self._opened:
            await self._pool.close()
            self._opened = False


class PostgresEventWatcherStore(_PostgresStoreBase, EventWatcherStore):
    """Postgres-backed durable watcher state for hosted multi-worker apps."""

    async def load_state(self, watcher_name: str) -> EventWatcherState:
        watcher_name = require_clean_nonblank(watcher_name, "watcher_name")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                    watcher_name,
                    cursor_sequence,
                    pending_event_id,
                    pending_event_sequence,
                    pending_attempt,
                    pending_claim_id,
                    delivery_status,
                    lease_expires_at,
                    last_error,
                    dead_lettered_count,
                    updated_at
                FROM cayu_event_watcher_state
                WHERE watcher_name = %s
                """,
                (watcher_name,),
            )
            row = await cur.fetchone()
            if row is None:
                return EventWatcherState(watcher_name=watcher_name)
            return _event_watcher_state_from_row(row)

    async def claim_event(
        self,
        *,
        watcher_name: str,
        record: EventRecord,
        lease_seconds: float,
    ) -> EventWatcherClaim | None:
        watcher_name = require_clean_nonblank(watcher_name, "watcher_name")
        record = copy_event_watcher_record(record)
        lease_seconds = _validate_positive_float(lease_seconds, "lease_seconds")
        await self._ensure_ready()
        now = datetime.now(UTC)
        async with self._pool.connection() as conn, conn.cursor() as cur:
            state = await self._load_watcher_state_for_update(cur, watcher_name, now=now)
            if state.cursor_sequence >= record.sequence:
                await conn.commit()
                return None
            if (
                state.delivery_status is EventWatcherDeliveryStatus.LEASED
                and state.lease_expires_at is not None
                and state.lease_expires_at > now
            ):
                await conn.commit()
                return None

            attempt = (
                state.pending_attempt + 1
                if state.pending_event_id == record.event.id
                and state.pending_event_sequence == record.sequence
                else 1
            )
            claim = EventWatcherClaim(
                watcher_name=watcher_name,
                event_id=record.event.id,
                event_sequence=record.sequence,
                attempt=attempt,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
            )
            await self._upsert_watcher_state(
                cur,
                state.model_copy(
                    update={
                        "pending_event_id": claim.event_id,
                        "pending_event_sequence": claim.event_sequence,
                        "pending_attempt": claim.attempt,
                        "pending_claim_id": claim.claim_id,
                        "delivery_status": EventWatcherDeliveryStatus.LEASED,
                        "lease_expires_at": claim.lease_expires_at,
                        "last_error": None,
                        "updated_at": now,
                    },
                    deep=True,
                ),
            )
            await conn.commit()
            return claim

    async def mark_success(self, claim: EventWatcherClaim) -> EventWatcherDelivery:
        claim = copy_event_watcher_claim(claim)
        await self._ensure_ready()
        now = datetime.now(UTC)
        async with self._pool.connection() as conn, conn.cursor() as cur:
            state = await self._matching_watcher_state_for_update(cur, claim, now=now)
            updated = state.model_copy(
                update={
                    "cursor_sequence": claim.event_sequence,
                    "pending_event_id": None,
                    "pending_event_sequence": None,
                    "pending_attempt": 0,
                    "pending_claim_id": None,
                    "delivery_status": EventWatcherDeliveryStatus.SUCCEEDED,
                    "lease_expires_at": None,
                    "last_error": None,
                    "updated_at": now,
                },
                deep=True,
            )
            await self._upsert_watcher_state(cur, updated)
            await conn.commit()
            return _event_watcher_delivery_from_claim(
                claim,
                status=EventWatcherDeliveryStatus.SUCCEEDED,
                cursor_sequence=updated.cursor_sequence,
            )

    async def mark_failure(
        self,
        claim: EventWatcherClaim,
        *,
        error: str,
        max_attempts: int,
    ) -> EventWatcherDelivery:
        claim = copy_event_watcher_claim(claim)
        error = _clean_error(error)
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be an integer greater than or equal to 1.")
        await self._ensure_ready()
        now = datetime.now(UTC)
        async with self._pool.connection() as conn, conn.cursor() as cur:
            state = await self._matching_watcher_state_for_update(cur, claim, now=now)
            if claim.attempt >= max_attempts:
                updated = state.model_copy(
                    update={
                        "cursor_sequence": claim.event_sequence,
                        "pending_event_id": None,
                        "pending_event_sequence": None,
                        "pending_attempt": 0,
                        "pending_claim_id": None,
                        "delivery_status": EventWatcherDeliveryStatus.DEAD_LETTERED,
                        "lease_expires_at": None,
                        "last_error": error,
                        "dead_lettered_count": state.dead_lettered_count + 1,
                        "updated_at": now,
                    },
                    deep=True,
                )
                status = EventWatcherDeliveryStatus.DEAD_LETTERED
                await self._insert_dead_letter(
                    cur,
                    EventWatcherDeadLetter(
                        watcher_name=claim.watcher_name,
                        event_id=claim.event_id,
                        event_sequence=claim.event_sequence,
                        attempts=claim.attempt,
                        error=error,
                        dead_lettered_at=now,
                    ),
                )
            else:
                updated = state.model_copy(
                    update={
                        "delivery_status": EventWatcherDeliveryStatus.FAILED,
                        "pending_claim_id": None,
                        "lease_expires_at": None,
                        "last_error": error,
                        "updated_at": now,
                    },
                    deep=True,
                )
                status = EventWatcherDeliveryStatus.FAILED
            await self._upsert_watcher_state(cur, updated)
            await conn.commit()
            return _event_watcher_delivery_from_claim(
                claim,
                status=status,
                cursor_sequence=updated.cursor_sequence,
                error=error,
            )

    async def list_dead_letters(
        self,
        watcher_name: str,
        *,
        include_resolved: bool = False,
        limit: int = 100,
    ) -> list[EventWatcherDeadLetter]:
        watcher_name = require_clean_nonblank(watcher_name, "watcher_name")
        limit = _validate_dead_letter_limit(limit)
        await self._ensure_ready()
        clause = "" if include_resolved else "AND resolved_at IS NULL"
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT
                    watcher_name,
                    event_id,
                    event_sequence,
                    attempts,
                    error,
                    dead_lettered_at,
                    resolved_at
                FROM cayu_event_watcher_dead_letters
                WHERE watcher_name = %s
                {clause}
                ORDER BY event_sequence ASC
                LIMIT %s
                """,
                (watcher_name, limit),
            )
            rows = await cur.fetchall()
            return [_event_watcher_dead_letter_from_row(row) for row in rows]

    async def resolve_dead_letter(
        self,
        watcher_name: str,
        event_sequence: int,
    ) -> EventWatcherDeadLetter:
        watcher_name = require_clean_nonblank(watcher_name, "watcher_name")
        event_sequence = _validate_event_sequence(event_sequence)
        await self._ensure_ready()
        now = datetime.now(UTC)
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                    watcher_name,
                    event_id,
                    event_sequence,
                    attempts,
                    error,
                    dead_lettered_at,
                    resolved_at
                FROM cayu_event_watcher_dead_letters
                WHERE watcher_name = %s AND event_sequence = %s
                FOR UPDATE
                """,
                (watcher_name, event_sequence),
            )
            row = await cur.fetchone()
            if row is None:
                raise ValueError(
                    f"No dead-letter record for watcher {watcher_name!r} "
                    f"at sequence {event_sequence}."
                )
            record = _event_watcher_dead_letter_from_row(row)
            if record.resolved_at is None:
                await cur.execute(
                    """
                    UPDATE cayu_event_watcher_dead_letters
                    SET resolved_at = %s
                    WHERE watcher_name = %s AND event_sequence = %s
                    """,
                    (now, watcher_name, event_sequence),
                )
                record = record.model_copy(update={"resolved_at": now}, deep=True)
            await conn.commit()
            return record

    async def _insert_dead_letter(self, cur: Any, dead_letter: EventWatcherDeadLetter) -> None:
        await cur.execute(
            """
            INSERT INTO cayu_event_watcher_dead_letters (
                watcher_name,
                event_sequence,
                event_id,
                attempts,
                error,
                dead_lettered_at,
                resolved_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (watcher_name, event_sequence) DO UPDATE SET
                event_id = excluded.event_id,
                attempts = excluded.attempts,
                error = excluded.error,
                dead_lettered_at = excluded.dead_lettered_at,
                resolved_at = excluded.resolved_at
            """,
            (
                dead_letter.watcher_name,
                dead_letter.event_sequence,
                dead_letter.event_id,
                dead_letter.attempts,
                dead_letter.error,
                pg_support.to_utc(dead_letter.dead_lettered_at),
                pg_support.to_utc_optional(dead_letter.resolved_at),
            ),
        )

    async def _load_watcher_state_for_update(
        self,
        cur: Any,
        watcher_name: str,
        *,
        now: datetime,
    ) -> EventWatcherState:
        await cur.execute(
            """
            INSERT INTO cayu_event_watcher_state (
                watcher_name,
                cursor_sequence,
                pending_attempt,
                dead_lettered_count,
                updated_at
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (watcher_name) DO NOTHING
            """,
            (watcher_name, 0, 0, 0, now),
        )
        await cur.execute(
            """
            SELECT
                watcher_name,
                cursor_sequence,
                pending_event_id,
                pending_event_sequence,
                pending_attempt,
                pending_claim_id,
                delivery_status,
                lease_expires_at,
                last_error,
                dead_lettered_count,
                updated_at
            FROM cayu_event_watcher_state
            WHERE watcher_name = %s
            FOR UPDATE
            """,
            (watcher_name,),
        )
        row = await cur.fetchone()
        if row is None:
            raise RuntimeError(f"Failed to initialize event watcher state: {watcher_name}")
        return _event_watcher_state_from_row(row)

    async def _matching_watcher_state_for_update(
        self,
        cur: Any,
        claim: EventWatcherClaim,
        *,
        now: datetime,
    ) -> EventWatcherState:
        state = await self._load_watcher_state_for_update(cur, claim.watcher_name, now=now)
        if state.pending_claim_id != claim.claim_id:
            raise ValueError("Watcher claim is no longer active.")
        if state.pending_event_id != claim.event_id:
            raise ValueError("Watcher claim event_id does not match active claim.")
        if state.pending_event_sequence != claim.event_sequence:
            raise ValueError("Watcher claim sequence does not match active claim.")
        if state.pending_attempt != claim.attempt:
            raise ValueError("Watcher claim attempt does not match active claim.")
        return state

    async def _upsert_watcher_state(self, cur: Any, state: EventWatcherState) -> None:
        await cur.execute(
            """
            INSERT INTO cayu_event_watcher_state (
                watcher_name,
                cursor_sequence,
                pending_event_id,
                pending_event_sequence,
                pending_attempt,
                pending_claim_id,
                delivery_status,
                lease_expires_at,
                last_error,
                dead_lettered_count,
                updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (watcher_name) DO UPDATE SET
                cursor_sequence = excluded.cursor_sequence,
                pending_event_id = excluded.pending_event_id,
                pending_event_sequence = excluded.pending_event_sequence,
                pending_attempt = excluded.pending_attempt,
                pending_claim_id = excluded.pending_claim_id,
                delivery_status = excluded.delivery_status,
                lease_expires_at = excluded.lease_expires_at,
                last_error = excluded.last_error,
                dead_lettered_count = excluded.dead_lettered_count,
                updated_at = excluded.updated_at
            """,
            (
                state.watcher_name,
                state.cursor_sequence,
                state.pending_event_id,
                state.pending_event_sequence,
                state.pending_attempt,
                state.pending_claim_id,
                None if state.delivery_status is None else str(state.delivery_status),
                pg_support.to_utc_optional(state.lease_expires_at),
                state.last_error,
                state.dead_lettered_count,
                pg_support.to_utc(state.updated_at),
            ),
        )


def _budget_advisory_lock_key(limit: _EffectiveBudgetLimit) -> int:
    """Stable 63-bit advisory-lock key for one effective budget limit."""

    material = f"cayu_budget_reservations|{limit.budget_limit_id}"
    digest = sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


class PostgresBudgetLedger(_PostgresStoreBase, BudgetLedger):
    """Postgres-backed atomic budget reservation ledger for multi-worker apps.

    ``reserve`` serializes per budget (scope/key/window/currency) under a
    transaction-scoped advisory lock, so concurrent workers on separate
    connections cannot jointly overshoot ``max_estimated_cost``; ``reconcile``
    and ``release`` row-lock the reservation with ``SELECT ... FOR UPDATE``.
    The ``cayu_budget_reservations`` table is owned by the shared migration
    machinery (ADR 0001 revision 8).
    """

    _min_required_revision = 25

    def __init__(
        self,
        conninfo: str | None = None,
        *,
        pool: AsyncConnectionPool | None = None,
        min_size: int = 1,
        max_size: int = 8,
        schema_mode: schema.SchemaMode = schema.SchemaMode.VALIDATE,
        clock: Callable[[], datetime] | None = None,
        reservation_ttl_seconds: int | None = DEFAULT_RESERVATION_TTL_SECONDS,
    ) -> None:
        super().__init__(
            conninfo,
            pool=pool,
            min_size=min_size,
            max_size=max_size,
            schema_mode=schema_mode,
        )
        self._clock = _clock_or_utc_now(clock)
        self._reservation_ttl_seconds = _validate_reservation_ttl(reservation_ttl_seconds)

    @property
    def reservation_ttl_seconds(self) -> int | None:
        return self._reservation_ttl_seconds

    async def claim_reservation_identity(
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
        await self._ensure_ready()
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
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
                        RETURNING reservation_id
                        """,
                        (reservation_id, publication_session_id, publication_id),
                    )
                    inserted = await cur.fetchone()
                    if inserted is None:
                        await cur.execute(
                            """
                            SELECT publication_session_id, publication_id
                            FROM cayu_budget_reservation_identities
                            WHERE reservation_id = %s
                            """,
                            (reservation_id,),
                        )
                        existing = await cur.fetchone()
                        if existing is None:
                            raise RuntimeError(
                                "Budget reservation identity claim disappeared during conflict."
                            )
                        if (existing[0], existing[1]) != (
                            publication_session_id,
                            publication_id,
                        ):
                            raise BudgetReservationIdentityConflict(
                                "Budget ledger reused a reservation identity."
                            )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def reserve(
        self,
        *,
        reservation_id: str | None = None,
        limit: BudgetLimit,
        session_id: str,
        agent_name: str,
        provider_name: str,
        model: str,
        model_attempt_identity: ModelAttemptIdentity,
        environment_name: str | None = None,
        settlement_event_payload: dict[str, Any] | None = None,
        settlement_fallback: BudgetSettlementFallback | None = None,
        requested_amount: Decimal | None = None,
        billing_identity: BillingIdentity | None = None,
        effective_at: datetime | None = None,
    ) -> BudgetReservationResult:
        reservation_id = (
            new_budget_reservation_id()
            if reservation_id is None
            else require_clean_nonblank(reservation_id, "reservation_id")
        )
        limit = _ensure_effective_budget_limit(
            limit,
            identity_namespace="app_policy",
        )
        session_id = require_clean_nonblank(session_id, "session_id")
        agent_name = require_clean_nonblank(agent_name, "agent_name")
        provider_name = require_clean_nonblank(provider_name, "provider_name")
        model = require_clean_nonblank(model, "model")
        model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
        durable_billing_identity = copy_billing_identity(billing_identity)
        await self._ensure_ready()
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT pg_advisory_xact_lock(%s)",
                        (_budget_advisory_lock_key(limit),),
                    )
                    now = self._clock()
                    durable_settlement_fallback = (
                        BudgetSettlementFallback(
                            settled_at=now,
                            expiration_reason=(
                                None
                                if self._reservation_ttl_seconds is None
                                else _expired_reservation_reason(self._reservation_ttl_seconds)
                            ),
                        )
                        if settlement_fallback is None
                        else copy_budget_settlement_fallback(settlement_fallback)
                    )
                    pricing_effective_at = (
                        now if effective_at is None else _utc_datetime(effective_at, "effective_at")
                    )
                    requested = (
                        _budget_reservation_amount(
                            limit=limit,
                            provider_name=provider_name,
                            model=model,
                            effective_at=pricing_effective_at,
                            billing_identity=durable_billing_identity,
                        )
                        if requested_amount is None
                        else _validate_amount(requested_amount, "requested_amount")
                    )
                    await self._reap_expired(cur, now, limit=limit)
                    current = await self._used_amount(cur, limit, now=now)
                    projected = current + requested
                    if projected > limit.max_estimated_cost:
                        # Reaping is an independent terminal transition with
                        # its own outbox evidence. Preserve it even when the
                        # new reservation is rejected.
                        await conn.commit()
                        return _reservation_result(
                            limit=limit,
                            model_attempt_identity=model_attempt_identity,
                            accepted=False,
                            requested=requested,
                            actual=projected,
                            message=(
                                "Budget reservation failed: "
                                f"{projected} > {limit.max_estimated_cost} {limit.currency}."
                            ),
                        )
                    record = BudgetReservationRecord(
                        reservation_id=reservation_id,
                        budget_limit_id=limit.budget_limit_id,
                        model_step_id=model_attempt_identity.model_step_id,
                        model_attempt_id=model_attempt_identity.model_attempt_id,
                        scope=limit.scope,
                        key=limit.key,
                        window=limit.window,
                        currency=limit.currency,
                        session_id=session_id,
                        agent_name=agent_name,
                        environment_name=environment_name,
                        provider_name=provider_name,
                        model=model,
                        billing_identity=durable_billing_identity,
                        settlement_event_payload=settlement_event_payload or {},
                        settlement_fallback=durable_settlement_fallback,
                        reserved_amount=requested,
                        created_at=now,
                        updated_at=now,
                    )
                    try:
                        await self._insert_record(cur, record)
                    except UniqueViolation as exc:
                        if (
                            getattr(exc.diag, "constraint_name", None)
                            == "cayu_budget_reservations_pkey"
                        ):
                            raise BudgetReservationIdentityConflict(
                                "Budget ledger reused a reservation identity."
                            ) from exc
                        raise
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return _reservation_result(
            limit=limit,
            model_attempt_identity=model_attempt_identity,
            accepted=True,
            requested=requested,
            actual=projected,
            message=(f"Budget reserved: {requested} {limit.currency} for {provider_name}/{model}."),
            record=record,
        )

    async def mark_dispatched(
        self,
        *,
        reservation_ids: tuple[str, ...],
        dispatch_id: str,
        dispatched_at: datetime | None = None,
    ) -> tuple[BudgetReservationRecord, ...]:
        reservation_ids = _validate_reservation_id_batch(reservation_ids)
        dispatch_id = require_clean_nonblank(dispatch_id, "dispatch_id")
        marked_at = pg_support.to_utc(dispatched_at) if dispatched_at is not None else self._clock()
        await self._ensure_ready()
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    records_by_id = {
                        reservation_id: await self._load_record_for_update(
                            cur,
                            reservation_id,
                        )
                        for reservation_id in sorted(reservation_ids)
                    }
                    records = tuple(
                        records_by_id[reservation_id] for reservation_id in reservation_ids
                    )
                    for record in records:
                        if record.dispatch_id is not None and record.dispatch_id != dispatch_id:
                            raise ValueError(
                                "Budget reservation has a conflicting dispatch: "
                                f"{record.reservation_id}"
                            )
                        if record.dispatch_id is None and record.status != "active":
                            raise ValueError(
                                f"Budget reservation is not active: {record.reservation_id}"
                            )
                    dispatched_records = tuple(
                        (
                            record
                            if record.dispatch_id is not None
                            else record.model_copy(
                                update={
                                    "dispatch_id": dispatch_id,
                                    "dispatched_at": marked_at,
                                },
                                deep=True,
                            )
                        )
                        for record in records
                    )
                    for record in dispatched_records:
                        await self._update_record(cur, record)
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
        return dispatched_records

    async def heartbeat(self, *, reservation_id: str) -> bool:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    record = await self._load_record_for_update(cur, reservation_id)
                    now = self._clock()
                    if record.status != "active" or _reservation_is_expired(
                        record,
                        now=now,
                        ttl_seconds=self._reservation_ttl_seconds,
                    ):
                        await conn.commit()
                        return False
                    renewed = record.model_copy(update={"updated_at": now}, deep=True)
                    await self._update_record(cur, renewed)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return True

    async def reconcile(
        self,
        *,
        reservation_id: str,
        actual_amount: Decimal,
        settlement_kind: Literal["completed", "conservative"] = "completed",
        reason: str | None = None,
        occurred_at: datetime | None = None,
        billing_identity: BillingIdentity | None = None,
        pricing: BudgetReconciliationPricing | None = None,
    ) -> BudgetReconciliation:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        actual_amount = _validate_amount(actual_amount, "actual_amount")
        reconciled_at = pg_support.to_utc(occurred_at) if occurred_at is not None else self._clock()
        await self._ensure_ready()
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    record = await self._reconcilable_record_for_update(cur, reservation_id)
                    reconciled = _reconciled_record(
                        record,
                        actual_amount=actual_amount,
                        reason=reason,
                        updated_at=reconciled_at,
                        billing_identity=billing_identity,
                    )
                    reconciliation = _reconciliation_from_record(
                        reconciled,
                        settlement_kind=settlement_kind,
                        pricing=pricing,
                    )
                    await self._insert_or_validate_settlement(
                        cur,
                        _budget_settlement_record(record, reconciliation),
                    )
                    await self._update_record(cur, reconciled)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return reconciliation

    async def release(
        self,
        *,
        reservation_id: str,
        reason: str,
        occurred_at: datetime | None = None,
    ) -> BudgetReconciliation:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        reason = require_clean_nonblank(reason, "reason")
        released_at = pg_support.to_utc(occurred_at) if occurred_at is not None else self._clock()
        await self._ensure_ready()
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    record = await self._releasable_record_for_update(cur, reservation_id)
                    released = _released_record(
                        record,
                        reason=reason,
                        updated_at=released_at,
                    )
                    reconciliation = _reconciliation_from_record(
                        released,
                        settlement_kind="released",
                    )
                    await self._insert_or_validate_settlement(
                        cur,
                        _budget_settlement_record(record, reconciliation),
                    )
                    await self._update_record(cur, released)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return reconciliation

    async def load_settlement(self, settlement_id: str) -> BudgetSettlementRecord | None:
        settlement_id = require_clean_nonblank(settlement_id, "settlement_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT settlement_json, event_published
                FROM cayu_budget_settlements
                WHERE settlement_id = %s
                """,
                (settlement_id,),
            )
            row = await cur.fetchone()
            return None if row is None else self._settlement_from_row(row)

    async def list_pending_settlements(
        self,
        *,
        session_id: str | None = None,
        after: BudgetSettlementCursor | None = None,
        limit: int = 100,
    ) -> list[BudgetSettlementRecord]:
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        after = _copy_budget_settlement_cursor(after)
        limit = _validate_settlement_page_limit(limit)
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            filters = ["NOT event_published"]
            parameters: list[object] = []
            if session_id is not None:
                filters.append("session_id = %s")
                parameters.append(session_id)
            if after is not None:
                filters.append("(settled_at > %s OR (settled_at = %s AND settlement_id > %s))")
                parameters.extend(
                    [
                        pg_support.to_utc(after.settled_at),
                        pg_support.to_utc(after.settled_at),
                        after.settlement_id,
                    ]
                )
            parameters.append(limit)
            query = (
                """
                SELECT settlement_json, event_published
                FROM cayu_budget_settlements
                WHERE """
                + " AND ".join(filters)
                + """
                ORDER BY settled_at, settlement_id
                LIMIT %s
                """
            )
            await cur.execute(
                cast("LiteralString", query),
                parameters,
            )
            return [self._settlement_from_row(row) for row in await cur.fetchall()]

    async def mark_settlement_event_published(
        self,
        *,
        settlement_id: str,
        event_id: str,
    ) -> BudgetSettlementRecord:
        settlement_id = require_clean_nonblank(settlement_id, "settlement_id")
        event_id = require_clean_nonblank(event_id, "event_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT settlement_json, event_published
                        FROM cayu_budget_settlements
                        WHERE settlement_id = %s
                        FOR UPDATE
                        """,
                        (settlement_id,),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        raise KeyError(f"Budget settlement not found: {settlement_id}")
                    settlement = self._settlement_from_row(row)
                    if settlement.event.id != event_id:
                        raise ValueError(
                            "Budget settlement event acknowledgement has conflicting identity."
                        )
                    if not settlement.event_published:
                        await cur.execute(
                            """
                            UPDATE cayu_budget_settlements
                            SET event_published = TRUE
                            WHERE settlement_id = %s
                            """,
                            (settlement_id,),
                        )
                        settlement = settlement.model_copy(
                            update={"event_published": True},
                            deep=True,
                        )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return settlement

    async def _reap_expired(
        self,
        cur: Any,
        now: datetime,
        *,
        limit: _EffectiveBudgetLimit,
    ) -> None:
        if self._reservation_ttl_seconds is None:
            return
        cutoff = now - timedelta(seconds=self._reservation_ttl_seconds)
        await cur.execute(
            """
            SELECT reservation_id
            FROM cayu_budget_reservations
            WHERE status = 'active'
              AND dispatch_id IS NULL
              AND updated_at <= %s
              AND budget_limit_id = %s
            ORDER BY reservation_id
            FOR UPDATE
            """,
            (
                pg_support.to_utc(cutoff),
                limit.budget_limit_id,
            ),
        )
        for row in await cur.fetchall():
            record = await self._load_record_for_update(cur, row[0])
            released = _released_record(
                record,
                reason=(
                    record.settlement_fallback.expiration_reason
                    or _expired_reservation_reason(self._reservation_ttl_seconds)
                ),
                updated_at=record.settlement_fallback.settled_at,
            )
            reconciliation = _reconciliation_from_record(
                released,
                settlement_kind="released",
            )
            await self._insert_or_validate_settlement(
                cur,
                _budget_settlement_record(record, reconciliation),
            )
            await self._update_record(cur, released)

    async def _used_amount(
        self,
        cur: Any,
        limit: _EffectiveBudgetLimit,
        *,
        now: datetime,
    ) -> Decimal:
        since, until = limit.window.bounds(now=now)
        reconciled_bound_sql = ""
        params: list[object] = [
            limit.budget_limit_id,
        ]
        if since is not None:
            reconciled_bound_sql += " AND updated_at >= %s"
            params.append(pg_support.to_utc(since))
        if until is not None:
            reconciled_bound_sql += " AND updated_at < %s"
            params.append(pg_support.to_utc(until))
        legacy_params: list[object] = [
            limit.scope,
            limit.key,
            limit.window.storage_key,
            limit.currency.upper(),
        ]
        legacy_bound_sql = ""
        if since is not None:
            legacy_bound_sql += " AND updated_at >= %s"
            legacy_params.append(pg_support.to_utc(since))
        if until is not None:
            legacy_bound_sql += " AND updated_at < %s"
            legacy_params.append(pg_support.to_utc(until))
        await cur.execute(
            f"""
            SELECT 1
            FROM cayu_budget_reservations
            WHERE budget_limit_id IS NULL
              AND scope = %s
              AND budget_key IS NOT DISTINCT FROM %s
              AND budget_window = %s
              AND currency = %s
              AND status IN ('active', 'reconciled')
              AND (
                    status = 'active'
                    OR (status = 'reconciled' {legacy_bound_sql})
              )
            LIMIT 1
            """,
            legacy_params,
        )
        if await cur.fetchone() is not None:
            raise RuntimeError(
                "Budget ledger contains pre-identity reservations for this limit; "
                "exact capacity cannot be verified."
            )
        await cur.execute(
            f"""
            SELECT reserved_amount, actual_amount, status
            FROM cayu_budget_reservations
            WHERE budget_limit_id = %s
              AND status IN ('active', 'reconciled')
              AND (
                    status = 'active'
                    OR (status = 'reconciled' {reconciled_bound_sql})
              )
            """,
            params,
        )
        total = Decimal("0")
        for row in await cur.fetchall():
            if row[2] == "active":
                total += row[0]
            elif row[2] == "reconciled":
                total += Decimal("0") if row[1] is None else row[1]
        return total

    async def _insert_record(self, cur: Any, record: BudgetReservationRecord) -> None:
        await cur.execute(
            """
            INSERT INTO cayu_budget_reservations (
                reservation_id,
                budget_limit_id,
                model_step_id,
                model_attempt_id,
                scope,
                budget_key,
                budget_window,
                currency,
                session_id,
                agent_name,
                environment_name,
                provider_name,
                model,
                billing_identity,
                settlement_event_payload,
                settlement_fallback,
                dispatch_id,
                dispatched_at,
                reserved_amount,
                actual_amount,
                status,
                reason,
                created_at,
                updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
            """,
            (
                record.reservation_id,
                record.budget_limit_id,
                record.model_step_id,
                record.model_attempt_id,
                record.scope,
                record.key,
                record.window.storage_key,
                record.currency,
                record.session_id,
                record.agent_name,
                record.environment_name,
                record.provider_name,
                record.model,
                (
                    None
                    if record.billing_identity is None
                    else _dumps(record.billing_identity.model_dump(mode="json"))
                ),
                _dumps(record.settlement_event_payload),
                _dumps(record.settlement_fallback.model_dump(mode="json")),
                record.dispatch_id,
                pg_support.to_utc_optional(record.dispatched_at),
                record.reserved_amount,
                record.actual_amount,
                record.status,
                record.reason,
                pg_support.to_utc(record.created_at),
                pg_support.to_utc(record.updated_at),
            ),
        )

    async def _update_record(self, cur: Any, record: BudgetReservationRecord) -> None:
        await cur.execute(
            """
            UPDATE cayu_budget_reservations
            SET actual_amount = %s,
                billing_identity = %s,
                dispatch_id = %s,
                dispatched_at = %s,
                status = %s,
                reason = %s,
                updated_at = %s
            WHERE reservation_id = %s
            """,
            (
                record.actual_amount,
                (
                    None
                    if record.billing_identity is None
                    else _dumps(record.billing_identity.model_dump(mode="json"))
                ),
                record.dispatch_id,
                pg_support.to_utc_optional(record.dispatched_at),
                record.status,
                record.reason,
                pg_support.to_utc(record.updated_at),
                record.reservation_id,
            ),
        )
        if cur.rowcount != 1:
            raise KeyError(f"Budget reservation not found: {record.reservation_id}")

    async def _load_record_for_update(
        self,
        cur: Any,
        reservation_id: str,
    ) -> BudgetReservationRecord:
        await cur.execute(
            """
            SELECT reservation_id, budget_limit_id, model_step_id, model_attempt_id,
                   scope, budget_key, budget_window,
                   currency, session_id,
                   agent_name, environment_name, provider_name, model,
                   billing_identity, settlement_event_payload, settlement_fallback,
                   dispatch_id, dispatched_at,
                   reserved_amount, actual_amount,
                   status, reason, created_at, updated_at
            FROM cayu_budget_reservations
            WHERE reservation_id = %s
            FOR UPDATE
            """,
            (reservation_id,),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Budget reservation not found: {reservation_id}")
        if row[1] is None:
            raise RuntimeError(
                "Budget reservation predates durable budget-limit identity and "
                "cannot be reconciled safely."
            )
        if row[2] is None or row[3] is None:
            raise RuntimeError(
                "Budget reservation predates durable model-attempt identity and "
                "cannot be reconciled safely."
            )
        return BudgetReservationRecord(
            reservation_id=row[0],
            budget_limit_id=row[1],
            model_step_id=row[2],
            model_attempt_id=row[3],
            scope=row[4],
            key=row[5],
            window=row[6],
            currency=row[7],
            session_id=row[8],
            agent_name=row[9],
            environment_name=row[10],
            provider_name=row[11],
            model=row[12],
            billing_identity=(
                None if row[13] is None else BillingIdentity.model_validate(_json_obj(row[13]))
            ),
            settlement_event_payload=_json_obj(row[14]),
            settlement_fallback=BudgetSettlementFallback.model_validate(_json_obj(row[15])),
            dispatch_id=row[16],
            dispatched_at=pg_support.to_utc_optional(row[17]),
            reserved_amount=row[18],
            actual_amount=row[19],
            status=row[20],
            reason=row[21],
            created_at=pg_support.to_utc(row[22]),
            updated_at=pg_support.to_utc(row[23]),
        )

    async def _insert_or_validate_settlement(
        self,
        cur: Any,
        settlement: BudgetSettlementRecord,
    ) -> None:
        stored = settlement.model_copy(update={"event_published": False}, deep=True)
        await cur.execute(
            """
            INSERT INTO cayu_budget_settlements (
                settlement_id,
                reservation_id,
                session_id,
                settled_at,
                settlement_json,
                event_published
            )
            VALUES (%s, %s, %s, %s, %s, FALSE)
            ON CONFLICT (settlement_id) DO NOTHING
            RETURNING settlement_id
            """,
            (
                stored.settlement_id,
                stored.reservation_id,
                stored.session_id,
                pg_support.to_utc(stored.reconciliation.settled_at),
                _dumps(stored.model_dump(mode="json")),
            ),
        )
        if await cur.fetchone() is not None:
            return
        await cur.execute(
            """
            SELECT settlement_json, event_published
            FROM cayu_budget_settlements
            WHERE settlement_id = %s
            """,
            (stored.settlement_id,),
        )
        row = await cur.fetchone()
        if row is None:
            raise RuntimeError("Budget settlement disappeared during conflict.")
        existing = self._settlement_from_row(row).model_copy(
            update={"event_published": False},
            deep=True,
        )
        if existing != stored:
            raise ValueError(
                f"Budget reservation has a conflicting settlement: {stored.reservation_id}"
            )

    @staticmethod
    def _settlement_from_row(row: Any) -> BudgetSettlementRecord:
        settlement = BudgetSettlementRecord.model_validate(_json_obj(row[0]))
        return settlement.model_copy(
            update={"event_published": bool(row[1])},
            deep=True,
        )

    async def _active_record_for_update(
        self,
        cur: Any,
        reservation_id: str,
    ) -> BudgetReservationRecord:
        record = await self._load_record_for_update(cur, reservation_id)
        if record.status != "active":
            raise ValueError(f"Budget reservation is not active: {reservation_id}")
        return record

    async def _releasable_record_for_update(
        self,
        cur: Any,
        reservation_id: str,
    ) -> BudgetReservationRecord:
        record = await self._load_record_for_update(cur, reservation_id)
        if record.status == "active" and record.dispatch_id is not None:
            raise ValueError(f"Dispatched budget reservation cannot be released: {reservation_id}")
        if record.status in {"active", "released"}:
            return record
        raise ValueError(f"Budget reservation is not active: {reservation_id}")

    async def _reconcilable_record_for_update(
        self,
        cur: Any,
        reservation_id: str,
    ) -> BudgetReservationRecord:
        record = await self._load_record_for_update(cur, reservation_id)
        if record.status in {"active", "reconciled"}:
            return record
        raise ValueError(f"Budget reservation is not active: {reservation_id}")


class PostgresKnowledgeStore(_PostgresStoreBase, KnowledgeStore):
    """Postgres-backed durable knowledge store with full-text search."""

    async def put_entry(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        entry = copy_knowledge_entry(entry)
        await self._ensure_ready()
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    existing_entry = await self._load_entry(cur, entry.id)
                    existing_chunks = await self._load_chunks(cur, entry.id)
                    await self._upsert_entry(cur, entry)
                    if (
                        existing_entry is None
                        or not existing_chunks
                        or _knowledge_has_only_default_chunk(existing_entry, existing_chunks)
                    ):
                        await self._replace_chunks(cur, entry.id, [_default_chunk_for_entry(entry)])
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return copy_knowledge_entry(entry)

    async def get_entry(self, entry_id: str) -> KnowledgeEntry | None:
        entry_id = require_clean_nonblank(entry_id, "entry_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            return await self._load_entry(cur, entry_id)

    async def update_entry_status(
        self,
        entry_id: str,
        status: KnowledgeStatus,
    ) -> KnowledgeEntry:
        entry_id = require_clean_nonblank(entry_id, "entry_id")
        if not isinstance(status, KnowledgeStatus):
            raise ValueError("status must be a KnowledgeStatus.")
        await self._ensure_ready()
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    entry = await self._load_entry(cur, entry_id)
                    if entry is None:
                        raise KeyError(f"Knowledge entry {entry_id!r} does not exist.")
                    updated_at = max(datetime.now(UTC), entry.created_at, entry.updated_at)
                    await cur.execute(
                        """
                        UPDATE cayu_knowledge_entries
                        SET status = %s, updated_at = %s
                        WHERE id = %s
                        """,
                        (str(status), pg_support.to_utc(updated_at), entry_id),
                    )
                    loaded = await self._load_entry(cur, entry_id)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        if loaded is None:
            raise KeyError(f"Knowledge entry {entry_id!r} does not exist.")
        return loaded

    async def transition_entry_status(
        self,
        entry_id: str,
        *,
        from_status: KnowledgeStatus,
        to_status: KnowledgeStatus,
        expected_namespace: str | None = None,
        expected_labels: dict[str, str] | None = None,
    ) -> KnowledgeEntry:
        entry_id = require_clean_nonblank(entry_id, "entry_id")
        if not isinstance(from_status, KnowledgeStatus):
            raise ValueError("from_status must be a KnowledgeStatus.")
        if not isinstance(to_status, KnowledgeStatus):
            raise ValueError("to_status must be a KnowledgeStatus.")
        expected_namespace = (
            require_clean_nonblank(expected_namespace, "expected_namespace")
            if expected_namespace is not None
            else None
        )
        expected_labels = copy_label_map(expected_labels or {}, "expected_labels")
        await self._ensure_ready()
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    scope_clauses: list[str] = []
                    scope_params: list[object] = []
                    if expected_namespace is not None:
                        scope_clauses.append("e.namespace = %s")
                        scope_params.append(expected_namespace)
                    for key, value in expected_labels.items():
                        scope_clauses.append(
                            """
                            EXISTS (
                                SELECT 1
                                FROM cayu_knowledge_labels AS label
                                WHERE label.entry_id = e.id
                                  AND label.key = %s
                                  AND label.value = %s
                            )
                            """
                        )
                        scope_params.extend([key, value])
                    scope_sql = "".join(f" AND {clause}" for clause in scope_clauses)
                    update_sql = cast(
                        "LiteralString",
                        f"""
                        UPDATE cayu_knowledge_entries AS e
                        SET status = %s, updated_at = GREATEST(NOW(), created_at, updated_at)
                        WHERE e.id = %s AND e.status = %s
                        {scope_sql}
                        """,
                    )
                    await cur.execute(
                        update_sql,
                        (str(to_status), entry_id, str(from_status), *scope_params),
                    )
                    if cur.rowcount != 1:
                        entry = await self._load_entry(cur, entry_id)
                        if entry is None:
                            raise KeyError(f"Knowledge entry {entry_id!r} does not exist.")
                        if expected_namespace is not None and entry.namespace != expected_namespace:
                            raise ValueError(
                                f"Knowledge entry {entry_id!r} does not match expected namespace."
                            )
                        for key, value in expected_labels.items():
                            if entry.labels.get(key) != value:
                                raise ValueError(
                                    f"Knowledge entry {entry_id!r} does not match expected labels."
                                )
                        raise ValueError(
                            f"Knowledge entry {entry_id!r} is {entry.status.value!r}, "
                            f"not {from_status.value!r}."
                        )
                    loaded = await self._load_entry(cur, entry_id)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        if loaded is None:
            raise KeyError(f"Knowledge entry {entry_id!r} does not exist.")
        return loaded

    async def delete_entry(
        self,
        entry_id: str,
        *,
        hard: bool = False,
    ) -> KnowledgeEntry | None:
        entry_id = require_clean_nonblank(entry_id, "entry_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    entry = await self._load_entry(cur, entry_id)
                    if entry is None:
                        await conn.commit()
                        return None
                    if hard:
                        await cur.execute(
                            "DELETE FROM cayu_knowledge_entries WHERE id = %s",
                            (entry_id,),
                        )
                        await conn.commit()
                        return copy_knowledge_entry(entry)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return await self.update_entry_status(entry_id, KnowledgeStatus.DELETED)

    async def prune_expired(self, *, now: datetime | None = None) -> int:
        cutoff = datetime.now(UTC) if now is None else now
        await self._ensure_ready()
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    # The entries DELETE cascades (ON DELETE CASCADE) to chunks, labels, aspects, and
                    # — for the embedding subclass — cayu_knowledge_embeddings, so no override is needed.
                    await cur.execute(
                        "DELETE FROM cayu_knowledge_entries "
                        "WHERE expires_at IS NOT NULL AND expires_at <= %s",
                        (cutoff,),
                    )
                    pruned = cur.rowcount
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return pruned

    async def replace_chunks(
        self,
        entry_id: str,
        chunks: list[KnowledgeChunk],
    ) -> list[KnowledgeChunk]:
        entry_id = require_clean_nonblank(entry_id, "entry_id")
        copied_chunks = _copy_knowledge_entry_chunks(entry_id, chunks)
        await self._ensure_ready()
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    if await self._load_entry(cur, entry_id) is None:
                        raise KeyError(f"Knowledge entry {entry_id!r} does not exist.")
                    await self._replace_chunks(cur, entry_id, copied_chunks)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return [copy_knowledge_chunk(chunk) for chunk in copied_chunks]

    async def put_entry_with_chunks(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
    ) -> KnowledgeEntry:
        entry = copy_knowledge_entry(entry)
        copied_chunks = _copy_knowledge_entry_chunks(entry.id, chunks)
        await self._ensure_ready()
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await self._upsert_entry(cur, entry)
                    await self._replace_chunks(cur, entry.id, copied_chunks)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return copy_knowledge_entry(entry)

    async def read_chunks(
        self,
        entry_id: str,
        *,
        chunk_index: int | None = None,
        around: int = 0,
        max_chunks: int = DEFAULT_KNOWLEDGE_LIMIT,
        max_bytes: int = DEFAULT_KNOWLEDGE_MAX_BYTES,
    ) -> list[KnowledgeChunk]:
        entry_id = require_clean_nonblank(entry_id, "entry_id")
        if chunk_index is not None:
            _validate_knowledge_nonnegative_int(chunk_index, "chunk_index")
        _validate_knowledge_nonnegative_int(around, "around")
        if chunk_index is None and around != 0:
            raise ValueError("`around` requires `chunk_index`.")
        _validate_knowledge_positive_int(max_chunks, "max_chunks")
        _validate_knowledge_positive_int(max_bytes, "max_bytes")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            if await self._load_entry(cur, entry_id) is None:
                return []
            chunks = await self._load_chunks(cur, entry_id)
        if chunk_index is not None:
            chunks = _center_knowledge_chunk_window(
                chunks,
                chunk_index=chunk_index,
                max_chunks=max_chunks,
            )
        start_index = 0 if chunk_index is None else max(0, chunk_index - around)
        end_index = None if chunk_index is None else chunk_index + around
        return _bounded_knowledge_chunks(
            chunks,
            start_index=start_index,
            end_index=end_index,
            max_chunks=max_chunks,
            max_bytes=max_bytes,
        )

    async def search(self, query: KnowledgeQuery) -> KnowledgeSearchResult:
        query = copy_knowledge_query(query)
        if query.mode not in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.KEYWORD}:
            raise ValueError("PostgresKnowledgeStore supports only auto and keyword search modes.")
        ts_query, preview_terms = _postgres_knowledge_ts_query(query)
        search_filter_sql, search_filter_params = _postgres_knowledge_search_filter_sql(query)
        where_sql, params = _postgres_knowledge_filter_sql(query)
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            total_hits_known = await self._count_search_hits(
                cur,
                search_filter_sql,
                [*search_filter_params, *params],
                where_sql,
            )
            rows = await self._search_unique_rows(
                cur,
                ts_query=ts_query,
                search_filter_sql=search_filter_sql,
                where_sql=where_sql,
                params=[*search_filter_params, *params],
                limit=query.limit,
            )
            hits, byte_truncated = await self._hits_from_search_rows(
                cur,
                rows,
                query,
                preview_terms,
            )
        return KnowledgeSearchResult(
            query=query,
            hits=hits,
            truncated=byte_truncated or len(hits) < total_hits_known,
            limit=query.limit,
            max_bytes=query.max_bytes,
            total_hits_known=total_hits_known,
        )

    async def list_entries(self, query: KnowledgeListQuery) -> KnowledgeListResult:
        query = copy_knowledge_list_query(query)
        where_sql, params = _postgres_knowledge_list_filter_sql(query)
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            total_entries_known = await self._count_list_entries(cur, where_sql, params)
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    SELECT e.id
                    FROM cayu_knowledge_entries AS e
                    WHERE TRUE
                    {where_sql}
                    ORDER BY COALESCE(e.importance, 0.0) DESC,
                             e.updated_at DESC,
                             e.id ASC
                    LIMIT %s
                    """,
                ),
                [*params, query.limit],
            )
            rows = await cur.fetchall()
            entry_map = await self._load_entries(cur, [str(row[0]) for row in rows])
            entries = [entry for row in rows if (entry := entry_map.get(str(row[0]))) is not None]
            facets, facets_truncated = await self._list_facets(cur, query, where_sql, params)
            items, byte_truncated = await self._list_items(cur, entries, query)
        return KnowledgeListResult(
            query=query,
            entries=items,
            facets=facets,
            facets_truncated=facets_truncated,
            truncated=byte_truncated or len(items) < total_entries_known or facets_truncated,
            limit=query.limit,
            max_bytes=query.max_bytes,
            total_entries_known=total_entries_known,
        )

    async def _upsert_entry(self, cur: Any, entry: KnowledgeEntry) -> None:
        await cur.execute(
            """
            INSERT INTO cayu_knowledge_entries (
                id,
                namespace,
                text,
                kind,
                visibility,
                status,
                created_by_type,
                created_by,
                created_at,
                updated_at,
                source_type,
                source_uri,
                source_id,
                source_hash,
                importance,
                importance_source,
                confidence,
                last_used_at,
                expires_at,
                title,
                metadata
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (id) DO UPDATE SET
                namespace = excluded.namespace,
                text = excluded.text,
                kind = excluded.kind,
                visibility = excluded.visibility,
                status = excluded.status,
                created_by_type = excluded.created_by_type,
                created_by = excluded.created_by,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                source_type = excluded.source_type,
                source_uri = excluded.source_uri,
                source_id = excluded.source_id,
                source_hash = excluded.source_hash,
                importance = excluded.importance,
                importance_source = excluded.importance_source,
                confidence = excluded.confidence,
                last_used_at = excluded.last_used_at,
                expires_at = excluded.expires_at,
                title = excluded.title,
                metadata = excluded.metadata
            """,
            _knowledge_entry_row_values(entry),
        )
        await self._replace_entry_lists(cur, entry)

    async def _replace_entry_lists(self, cur: Any, entry: KnowledgeEntry) -> None:
        for table in (
            "cayu_knowledge_labels",
            "cayu_knowledge_aspects",
            "cayu_knowledge_impact_targets",
        ):
            await cur.execute(f"DELETE FROM {table} WHERE entry_id = %s", (entry.id,))
        if entry.labels:
            await cur.executemany(
                """
                INSERT INTO cayu_knowledge_labels (entry_id, key, value)
                VALUES (%s, %s, %s)
                """,
                [(entry.id, key, value) for key, value in sorted(entry.labels.items())],
            )
        if entry.aspects:
            await cur.executemany(
                """
                INSERT INTO cayu_knowledge_aspects (entry_id, aspect)
                VALUES (%s, %s)
                """,
                [(entry.id, aspect) for aspect in entry.aspects],
            )
        if entry.impact_targets:
            await cur.executemany(
                """
                INSERT INTO cayu_knowledge_impact_targets (entry_id, impact_target)
                VALUES (%s, %s)
                """,
                [(entry.id, target) for target in entry.impact_targets],
            )

    async def _replace_chunks(
        self,
        cur: Any,
        entry_id: str,
        chunks: list[KnowledgeChunk],
    ) -> None:
        await cur.execute("DELETE FROM cayu_knowledge_chunks WHERE entry_id = %s", (entry_id,))
        await cur.executemany(
            """
            INSERT INTO cayu_knowledge_chunks (
                id,
                entry_id,
                chunk_index,
                text,
                content_hash,
                source_uri,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            [_knowledge_chunk_row_values(chunk) for chunk in chunks],
        )

    async def _load_entry(self, cur: Any, entry_id: str) -> KnowledgeEntry | None:
        await cur.execute(
            """
            SELECT
                id,
                namespace,
                text,
                kind,
                visibility,
                status,
                created_by_type,
                created_by,
                created_at,
                updated_at,
                source_type,
                source_uri,
                source_id,
                source_hash,
                importance,
                importance_source,
                confidence,
                last_used_at,
                expires_at,
                title,
                metadata
            FROM cayu_knowledge_entries
            WHERE id = %s
            """,
            (entry_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return _knowledge_entry_from_row(
            row,
            labels=await self._load_labels(cur, entry_id),
            aspects=await self._load_aspects(cur, entry_id),
            impact_targets=await self._load_impact_targets(cur, entry_id),
        )

    async def _load_chunk(self, cur: Any, chunk_id: str) -> KnowledgeChunk | None:
        await cur.execute(
            """
            SELECT id, entry_id, chunk_index, text, content_hash, source_uri, metadata
            FROM cayu_knowledge_chunks
            WHERE id = %s
            """,
            (chunk_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return _knowledge_chunk_from_row(row)

    async def _load_chunks(self, cur: Any, entry_id: str) -> list[KnowledgeChunk]:
        await cur.execute(
            """
            SELECT id, entry_id, chunk_index, text, content_hash, source_uri, metadata
            FROM cayu_knowledge_chunks
            WHERE entry_id = %s
            ORDER BY chunk_index ASC
            """,
            (entry_id,),
        )
        return [_knowledge_chunk_from_row(row) for row in await cur.fetchall()]

    async def _load_labels(self, cur: Any, entry_id: str) -> dict[str, str]:
        await cur.execute(
            """
            SELECT key, value
            FROM cayu_knowledge_labels
            WHERE entry_id = %s
            ORDER BY key ASC
            """,
            (entry_id,),
        )
        return {row[0]: row[1] for row in await cur.fetchall()}

    async def _load_aspects(self, cur: Any, entry_id: str) -> list[str]:
        await cur.execute(
            """
            SELECT aspect
            FROM cayu_knowledge_aspects
            WHERE entry_id = %s
            ORDER BY aspect ASC
            """,
            (entry_id,),
        )
        return [row[0] for row in await cur.fetchall()]

    async def _load_impact_targets(self, cur: Any, entry_id: str) -> list[str]:
        await cur.execute(
            """
            SELECT impact_target
            FROM cayu_knowledge_impact_targets
            WHERE entry_id = %s
            ORDER BY impact_target ASC
            """,
            (entry_id,),
        )
        return [row[0] for row in await cur.fetchall()]

    async def _load_entries(
        self,
        cur: Any,
        entry_ids: list[str],
    ) -> dict[str, KnowledgeEntry]:
        unique_ids = list(dict.fromkeys(entry_ids))
        if not unique_ids:
            return {}
        await cur.execute(
            """
            SELECT
                id,
                namespace,
                text,
                kind,
                visibility,
                status,
                created_by_type,
                created_by,
                created_at,
                updated_at,
                source_type,
                source_uri,
                source_id,
                source_hash,
                importance,
                importance_source,
                confidence,
                last_used_at,
                expires_at,
                title,
                metadata
            FROM cayu_knowledge_entries
            WHERE id = ANY(%s)
            """,
            (unique_ids,),
        )
        rows = await cur.fetchall()
        labels = await self._load_labels_for_entries(cur, unique_ids)
        aspects = await self._load_aspects_for_entries(cur, unique_ids)
        impact_targets = await self._load_impact_targets_for_entries(cur, unique_ids)
        return {
            row[0]: _knowledge_entry_from_row(
                row,
                labels=labels.get(row[0], {}),
                aspects=aspects.get(row[0], []),
                impact_targets=impact_targets.get(row[0], []),
            )
            for row in rows
        }

    async def _load_chunks_by_ids(
        self,
        cur: Any,
        chunk_ids: list[str],
    ) -> dict[str, KnowledgeChunk]:
        unique_ids = list(dict.fromkeys(chunk_ids))
        if not unique_ids:
            return {}
        await cur.execute(
            """
            SELECT id, entry_id, chunk_index, text, content_hash, source_uri, metadata
            FROM cayu_knowledge_chunks
            WHERE id = ANY(%s)
            """,
            (unique_ids,),
        )
        return {row[0]: _knowledge_chunk_from_row(row) for row in await cur.fetchall()}

    async def _count_chunks_by_entry(
        self,
        cur: Any,
        entry_ids: list[str],
    ) -> dict[str, int]:
        unique_ids = list(dict.fromkeys(entry_ids))
        if not unique_ids:
            return {}
        await cur.execute(
            """
            SELECT entry_id, COUNT(*)
            FROM cayu_knowledge_chunks
            WHERE entry_id = ANY(%s)
            GROUP BY entry_id
            """,
            (unique_ids,),
        )
        return {row[0]: int(row[1]) for row in await cur.fetchall()}

    async def _load_labels_for_entries(
        self,
        cur: Any,
        entry_ids: list[str],
    ) -> dict[str, dict[str, str]]:
        if not entry_ids:
            return {}
        await cur.execute(
            """
            SELECT entry_id, key, value
            FROM cayu_knowledge_labels
            WHERE entry_id = ANY(%s)
            ORDER BY entry_id ASC, key ASC
            """,
            (entry_ids,),
        )
        result: dict[str, dict[str, str]] = {}
        for row in await cur.fetchall():
            result.setdefault(row[0], {})[row[1]] = row[2]
        return result

    async def _load_aspects_for_entries(
        self,
        cur: Any,
        entry_ids: list[str],
    ) -> dict[str, list[str]]:
        if not entry_ids:
            return {}
        await cur.execute(
            """
            SELECT entry_id, aspect
            FROM cayu_knowledge_aspects
            WHERE entry_id = ANY(%s)
            ORDER BY entry_id ASC, aspect ASC
            """,
            (entry_ids,),
        )
        result: dict[str, list[str]] = {}
        for row in await cur.fetchall():
            result.setdefault(row[0], []).append(row[1])
        return result

    async def _load_impact_targets_for_entries(
        self,
        cur: Any,
        entry_ids: list[str],
    ) -> dict[str, list[str]]:
        if not entry_ids:
            return {}
        await cur.execute(
            """
            SELECT entry_id, impact_target
            FROM cayu_knowledge_impact_targets
            WHERE entry_id = ANY(%s)
            ORDER BY entry_id ASC, impact_target ASC
            """,
            (entry_ids,),
        )
        result: dict[str, list[str]] = {}
        for row in await cur.fetchall():
            result.setdefault(row[0], []).append(row[1])
        return result

    async def _count_search_hits(
        self,
        cur: Any,
        search_filter_sql: str,
        params: list[object],
        where_sql: str,
    ) -> int:
        await cur.execute(
            f"""
            SELECT COUNT(DISTINCT e.id)
            FROM cayu_knowledge_chunks AS c
            JOIN cayu_knowledge_entries AS e ON e.id = c.entry_id
            WHERE {search_filter_sql}
            {where_sql}
            """,
            params,
        )
        row = await cur.fetchone()
        return 0 if row is None else int(row[0])

    async def _search_unique_rows(
        self,
        cur: Any,
        *,
        ts_query: str,
        search_filter_sql: str,
        where_sql: str,
        params: list[object],
        limit: int,
    ) -> list[tuple[Any, ...]]:
        unique_rows: list[tuple[Any, ...]] = []
        seen_entry_ids: set[str] = set()
        offset = 0
        while len(unique_rows) < limit:
            await cur.execute(
                f"""
                SELECT
                    e.id AS entry_id,
                    c.id AS chunk_id,
                    ts_rank_cd(
                        {_postgres_entry_search_vector_sql()},
                        to_tsquery('simple', %s)
                    ) AS score
                FROM cayu_knowledge_chunks AS c
                JOIN cayu_knowledge_entries AS e ON e.id = c.entry_id
                WHERE {search_filter_sql}
                {where_sql}
                ORDER BY score DESC,
                         COALESCE(e.importance, 0.0) DESC,
                         e.updated_at DESC,
                         e.id ASC,
                         c.chunk_index ASC
                LIMIT %s OFFSET %s
                """,
                [
                    ts_query,
                    *params,
                    _KNOWLEDGE_SEARCH_PAGE_SIZE,
                    offset,
                ],
            )
            rows = await cur.fetchall()
            if not rows:
                break
            for row in rows:
                entry_id = str(row[0])
                if entry_id in seen_entry_ids:
                    continue
                seen_entry_ids.add(entry_id)
                unique_rows.append(row)
                if len(unique_rows) >= limit:
                    break
            if len(rows) < _KNOWLEDGE_SEARCH_PAGE_SIZE:
                break
            offset += _KNOWLEDGE_SEARCH_PAGE_SIZE
        return unique_rows

    async def _hits_from_search_rows(
        self,
        cur: Any,
        rows: list[tuple[Any, ...]],
        query: KnowledgeQuery,
        terms: list[str],
    ) -> tuple[list[KnowledgeHit], bool]:
        entries = await self._load_entries(cur, [str(row[0]) for row in rows])
        chunks = await self._load_chunks_by_ids(cur, [str(row[1]) for row in rows])
        hits: list[KnowledgeHit] = []
        remaining = query.max_bytes
        truncated = False
        for row in rows:
            if remaining <= 0:
                truncated = True
                break
            entry = entries.get(str(row[0]))
            chunk = chunks.get(str(row[1]))
            if entry is None or chunk is None:
                continue
            reason, preview_text = _knowledge_preview_for_match(entry, chunk, terms)
            preview_bytes = len(preview_text.encode("utf-8"))
            preview = _truncate_knowledge_text_to_bytes(preview_text, remaining)
            if not preview:
                truncated = True
                break
            returned_bytes = len(preview.encode("utf-8"))
            if returned_bytes < preview_bytes:
                truncated = True
            remaining -= returned_bytes
            hits.append(
                KnowledgeHit(
                    entry=entry,
                    chunk=chunk,
                    score=float(row[2]),
                    score_kind="postgres_full_text",
                    rank=len(hits) + 1,
                    reason=reason,
                    text_preview=preview,
                )
            )
        return hits, truncated

    async def _count_list_entries(
        self,
        cur: Any,
        where_sql: str,
        params: list[object],
    ) -> int:
        await cur.execute(
            f"""
            SELECT COUNT(*)
            FROM cayu_knowledge_entries AS e
            WHERE TRUE
            {where_sql}
            """,
            params,
        )
        row = await cur.fetchone()
        return 0 if row is None else int(row[0])

    async def _list_items(
        self,
        cur: Any,
        entries: list[KnowledgeEntry],
        query: KnowledgeListQuery,
    ) -> tuple[list[KnowledgeListItem], bool]:
        chunk_counts = await self._count_chunks_by_entry(cur, [entry.id for entry in entries])
        items: list[KnowledgeListItem] = []
        remaining = query.max_bytes
        truncated = False
        for entry in entries:
            if remaining <= 0:
                truncated = True
                break
            preview_source = entry.title or entry.text
            preview_bytes = len(preview_source.encode("utf-8"))
            preview = _truncate_knowledge_text_to_bytes(preview_source, remaining)
            if not preview:
                truncated = True
                break
            returned_bytes = len(preview.encode("utf-8"))
            if returned_bytes < preview_bytes:
                truncated = True
            remaining -= returned_bytes
            items.append(
                KnowledgeListItem(
                    entry=entry,
                    chunk_count=chunk_counts.get(entry.id, 0),
                    text_preview=preview,
                )
            )
        return items, truncated

    async def _list_facets(
        self,
        cur: Any,
        query: KnowledgeListQuery,
        where_sql: str,
        params: list[object],
    ) -> tuple[list[KnowledgeFacet], bool]:
        if query.group_by is None:
            return [], False
        sql, facet_params = _postgres_list_facet_sql(
            query.group_by,
            where_sql,
            params,
            limit=query.limit + 1,
        )
        await cur.execute(sql, facet_params)
        rows = await cur.fetchall()
        return [
            KnowledgeFacet(
                field=query.group_by,
                key=str(row[0]) if row[0] is not None else None,
                value=str(row[1]),
                count=int(row[2]),
            )
            for row in rows[: query.limit]
        ], len(rows) > query.limit


def _warn_if_embedding_dims_exceed_hnsw(dimensions: int) -> None:
    """Warn (do not reject) when embedding dimensions exceed pgvector's HNSW cap.

    pgvector's HNSW index supports at most 2000 dimensions. Larger models (e.g. 3072-dim) are still
    allowed — the store just can't build the index, so semantic search falls back to an exact O(n)
    brute-force scan. Surface that loudly instead of failing silently.
    """
    if dimensions > _PGVECTOR_HNSW_VECTOR_MAX_DIMENSIONS:
        logger.warning(
            "Embedding dimensions (%d) exceed pgvector's HNSW limit (%d); the HNSW index will not be "
            "created and semantic search will fall back to an exact brute-force scan (O(n) per query).",
            dimensions,
            _PGVECTOR_HNSW_VECTOR_MAX_DIMENSIONS,
        )


class PostgresEmbeddingKnowledgeStore(PostgresKnowledgeStore):
    """Postgres knowledge store with pgvector-backed semantic chunk search."""

    def __init__(
        self,
        conninfo: str | None = None,
        *,
        pool: AsyncConnectionPool | None = None,
        min_size: int = 1,
        max_size: int = 8,
        schema_mode: schema.SchemaMode = schema.SchemaMode.VALIDATE,
        embedding_provider: TextEmbeddingProvider,
        embedding_model: str,
        embedding_dimensions: int,
        hybrid_keyword_weight: float = 0.35,
        semantic_min_score: float = 0.55,
    ) -> None:
        if not isinstance(embedding_provider, TextEmbeddingProvider):
            raise TypeError("embedding_provider must implement TextEmbeddingProvider.")
        _validate_positive_int(embedding_dimensions, "embedding_dimensions")
        _warn_if_embedding_dims_exceed_hnsw(embedding_dimensions)
        self.embedding_provider = embedding_provider
        self.embedding_model = require_clean_nonblank(embedding_model, "embedding_model")
        self.embedding_dimensions = embedding_dimensions
        self.hybrid_keyword_weight = _validate_nonnegative_float(
            hybrid_keyword_weight,
            "hybrid_keyword_weight",
        )
        self.semantic_min_score = _validate_unit_float(
            semantic_min_score,
            "semantic_min_score",
        )
        self._embedding_schema_ready = False
        super().__init__(
            conninfo,
            pool=pool,
            min_size=min_size,
            max_size=max_size,
            schema_mode=schema_mode,
        )

    def supported_search_modes(self) -> tuple[KnowledgeSearchMode, ...]:
        return (
            KnowledgeSearchMode.AUTO,
            KnowledgeSearchMode.KEYWORD,
            KnowledgeSearchMode.SEMANTIC,
            KnowledgeSearchMode.HYBRID,
        )

    async def _ensure_ready(self) -> None:
        await super()._ensure_ready()
        if self._embedding_schema_ready:
            return
        async with self._open_lock:
            if self._embedding_schema_ready:
                return
            await self._reconcile_embedding_schema()
            self._embedding_schema_ready = True

    async def put_entry(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        stored = await super().put_entry(entry)
        await self._embed_entry_chunks_best_effort(stored.id)
        return stored

    async def delete_entry(
        self,
        entry_id: str,
        *,
        hard: bool = False,
    ) -> KnowledgeEntry | None:
        deleted = await super().delete_entry(entry_id, hard=hard)
        if hard and deleted is not None:
            await self._delete_entry_embeddings(deleted.id)
        return deleted

    async def replace_chunks(
        self,
        entry_id: str,
        chunks: list[KnowledgeChunk],
    ) -> list[KnowledgeChunk]:
        stored_chunks = await super().replace_chunks(entry_id, chunks)
        await self._embed_entry_chunks_best_effort(entry_id, chunks=stored_chunks)
        return stored_chunks

    async def put_entry_with_chunks(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
    ) -> KnowledgeEntry:
        stored = await super().put_entry_with_chunks(entry, chunks)
        await self._embed_entry_chunks_best_effort(stored.id)
        return stored

    async def backfill_embeddings(
        self,
        query: KnowledgeListQuery | None = None,
        *,
        limit: int = 500,
        refresh_existing: bool = False,
    ) -> PostgresEmbeddingBackfillResult:
        """Embed a bounded batch of existing chunks matching knowledge filters.

        By default this only fills missing or stale embedding rows. Set
        ``refresh_existing=True`` to re-embed current rows for the configured
        model and dimensions.
        """

        _validate_positive_int(limit, "limit")
        if type(refresh_existing) is not bool:
            raise ValueError("`refresh_existing` must be a boolean.")
        query = copy_knowledge_list_query(query or KnowledgeListQuery())
        await self._ensure_ready()
        chunks = await self._backfill_candidate_chunks(
            query,
            limit,
            refresh_existing=refresh_existing,
        )
        embedded_chunks = await self._embed_chunks(
            chunks,
            refresh_existing=refresh_existing,
        )
        return PostgresEmbeddingBackfillResult(
            scanned_chunks=len(chunks),
            embedded_chunks=embedded_chunks,
            skipped_current_chunks=len(chunks) - embedded_chunks,
            limit=limit,
            refresh_existing=refresh_existing,
        )

    async def search(self, query: KnowledgeQuery) -> KnowledgeSearchResult:
        query = copy_knowledge_query(query)
        if query.mode is KnowledgeSearchMode.KEYWORD:
            return await super().search(query)
        if query.mode not in {
            KnowledgeSearchMode.AUTO,
            KnowledgeSearchMode.SEMANTIC,
            KnowledgeSearchMode.HYBRID,
        }:
            raise ValueError(
                "PostgresEmbeddingKnowledgeStore supports auto, keyword, semantic, "
                "and hybrid search modes."
            )
        await self._ensure_ready()
        await self._lazy_backfill_search_scope(query)
        semantic_query_text = _semantic_query_text(query)
        query_vector = await self._embed_query(query, semantic_query_text)
        rows, candidate_limit_reached = await self._semantic_search_rows(query, query_vector)
        async with self._pool.connection() as conn, conn.cursor() as cur:
            scored, byte_truncated = await self._scored_semantic_rows(
                cur,
                rows,
                query,
            )
        total_hits_known_floor = len(scored)
        if query.mode in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.HYBRID}:
            keyword_query = query.model_copy(update={"mode": KnowledgeSearchMode.KEYWORD})
            try:
                keyword_result = await super().search(keyword_query)
            except ValueError:
                keyword_result = None
            if keyword_result is not None:
                scored = self._merge_keyword_hits(scored, keyword_result)
                byte_truncated = byte_truncated or keyword_result.truncated
                keyword_total_hits_known = keyword_result.total_hits_known
                keyword_hits_floor = (
                    keyword_total_hits_known
                    if keyword_total_hits_known is not None
                    else len(keyword_result.hits)
                )
                total_hits_known_floor = max(
                    total_hits_known_floor,
                    keyword_hits_floor,
                )
        score_kind = (
            "postgres_semantic" if query.mode is KnowledgeSearchMode.SEMANTIC else "postgres_hybrid"
        )
        result = _search_result_from_scored_embeddings(
            scored,
            query,
            score_kind=score_kind,
        )
        return KnowledgeSearchResult(
            query=result.query,
            hits=result.hits,
            truncated=byte_truncated or result.truncated or candidate_limit_reached,
            limit=result.limit,
            max_bytes=result.max_bytes,
            total_hits_known=max(
                result.total_hits_known
                if result.total_hits_known is not None
                else len(result.hits),
                total_hits_known_floor,
            ),
        )

    async def _reconcile_embedding_schema(self) -> None:
        mode = self._schema_mode
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT pg_advisory_xact_lock(%s)", (_PGVECTOR_SCHEMA_ADVISORY_LOCK_KEY,)
                )
                if mode in {schema.SchemaMode.CREATE, schema.SchemaMode.MIGRATE}:
                    await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    await cur.execute(
                        cast(
                            "LiteralString",
                            f"""
                            CREATE TABLE IF NOT EXISTS cayu_knowledge_embeddings (
                                chunk_id TEXT PRIMARY KEY REFERENCES cayu_knowledge_chunks(id) ON DELETE CASCADE,
                                entry_id TEXT NOT NULL REFERENCES cayu_knowledge_entries(id) ON DELETE CASCADE,
                                content_hash TEXT NOT NULL,
                                model TEXT NOT NULL,
                                dimensions INTEGER NOT NULL,
                                embedding vector({self.embedding_dimensions}) NOT NULL,
                                embedding_space_version INTEGER NOT NULL DEFAULT 1,
                                created_at TIMESTAMPTZ NOT NULL,
                                updated_at TIMESTAMPTZ NOT NULL
                            )
                            """,
                        )
                    )
                    # Belt-and-suspenders for an existing embeddings table opened directly in CREATE
                    # mode (where `_apply_pending` won't re-run migrations on a non-fresh DB). The
                    # canonical path for existing DBs is revision 12 in `_MIGRATION_STEPS`, applied by
                    # `cayu storage migrate`; both are idempotent.
                    await cur.execute(
                        "ALTER TABLE cayu_knowledge_embeddings "
                        "ADD COLUMN IF NOT EXISTS embedding_space_version INTEGER NOT NULL DEFAULT 1"
                    )
                    await self._validate_embedding_schema(cur)
                    await cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_embeddings_entry
                        ON cayu_knowledge_embeddings(entry_id)
                        """
                    )
                    await cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_embeddings_model_dims
                        ON cayu_knowledge_embeddings(model, dimensions)
                        """
                    )
                    # HNSW tops out at 2000 dims; above the cap no index is built and semantic search
                    # falls back to an exact brute-force scan (the constructor warns — see
                    # _warn_if_embedding_dims_exceed_hnsw).
                    if self.embedding_dimensions <= _PGVECTOR_HNSW_VECTOR_MAX_DIMENSIONS:
                        await cur.execute(
                            """
                            CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_embeddings_embedding_hnsw
                            ON cayu_knowledge_embeddings USING hnsw (embedding vector_cosine_ops)
                            """
                        )
                elif mode is schema.SchemaMode.VALIDATE:
                    await cur.execute(
                        "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                    )
                    row = await cur.fetchone()
                    if row is None or not bool(row[0]):
                        raise RuntimeError(
                            "PostgresEmbeddingKnowledgeStore requires the pgvector extension. "
                            "Use schema_mode=CREATE/MIGRATE or create extension vector manually."
                        )
                    await cur.execute("SELECT to_regclass('cayu_knowledge_embeddings')")
                    row = await cur.fetchone()
                    if row is None or row[0] is None:
                        raise RuntimeError(
                            "Missing Postgres knowledge embedding schema. "
                            "Run with schema_mode=CREATE or MIGRATE first."
                        )
                await self._validate_embedding_schema(cur)
            await conn.commit()

    async def _validate_embedding_schema(self, cur: Any) -> None:
        await cur.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute AS a
            WHERE a.attrelid = 'cayu_knowledge_embeddings'::regclass
              AND a.attname = 'embedding'
              AND NOT a.attisdropped
            """
        )
        row = await cur.fetchone()
        expected = f"vector({self.embedding_dimensions})"
        actual = None if row is None else str(row[0])
        if actual != expected:
            raise RuntimeError(
                "Postgres knowledge embedding dimension mismatch: "
                f"expected {expected}, found {actual or 'missing embedding column'}."
            )
        await cur.execute(
            """
            SELECT 1
            FROM pg_attribute AS a
            WHERE a.attrelid = 'cayu_knowledge_embeddings'::regclass
              AND a.attname = 'embedding_space_version'
              AND NOT a.attisdropped
            """
        )
        if await cur.fetchone() is None:
            raise RuntimeError(
                "Postgres knowledge embedding schema is missing the embedding_space_version column. "
                "Run with schema_mode=CREATE or MIGRATE first."
            )

    async def _semantic_search_rows(
        self,
        query: KnowledgeQuery,
        query_vector: list[float],
    ) -> tuple[list[tuple[str, str, float]], bool]:
        where_sql, params = _postgres_knowledge_filter_sql(query)
        none_sql, none_params = _postgres_knowledge_none_filter_sql(query)
        vector_literal = _postgres_vector_literal(query_vector)
        candidate_limit = max(
            query.limit,
            query.limit * _PGVECTOR_SEMANTIC_CANDIDATE_MULTIPLIER,
        )
        semantic_min_score = self.semantic_min_score if query.min_score is None else query.min_score
        min_score_sql = (
            ""
            if query.mode in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.HYBRID}
            else "WHERE normalized_score >= %s"
        )
        min_score_params: list[object] = (
            []
            if query.mode in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.HYBRID}
            else [semantic_min_score]
        )
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    WITH nearest_chunks AS (
                        SELECT
                            e.id AS entry_id,
                            c.id AS chunk_id,
                            c.chunk_index AS chunk_index,
                            emb.embedding <=> %s::vector AS distance,
                            (1.0 + (1.0 - (emb.embedding <=> %s::vector))) / 2.0 AS normalized_score,
                            COALESCE(e.importance, 0.0) AS importance,
                            e.updated_at AS updated_at
                        FROM cayu_knowledge_embeddings AS emb
                        JOIN cayu_knowledge_chunks AS c ON c.id = emb.chunk_id
                        JOIN cayu_knowledge_entries AS e ON e.id = emb.entry_id
                        WHERE emb.model = %s
                          AND emb.dimensions = %s
                          AND emb.embedding_space_version = %s
                          AND (emb.content_hash = c.content_hash OR c.content_hash IS NULL)
                        {where_sql}
                        {none_sql}
                        ORDER BY emb.embedding <=> %s::vector
                        LIMIT %s
                    ),
                    best_entries AS (
                        SELECT DISTINCT ON (entry_id)
                            entry_id,
                            chunk_id,
                            normalized_score,
                            importance,
                            updated_at
                        FROM nearest_chunks
                        ORDER BY entry_id, distance ASC, chunk_index ASC
                    ),
                    filtered_entries AS (
                        SELECT *
                        FROM best_entries
                        {min_score_sql}
                    )
                    SELECT
                        entry_id,
                        chunk_id,
                        normalized_score,
                        (SELECT COUNT(*) FROM nearest_chunks) AS candidate_count
                    FROM filtered_entries
                    ORDER BY normalized_score DESC,
                             importance DESC,
                             updated_at DESC,
                             entry_id ASC
                    LIMIT %s
                    """,
                ),
                [
                    vector_literal,
                    vector_literal,
                    self.embedding_model,
                    self.embedding_dimensions,
                    _EMBEDDING_SPACE_VERSION,
                    *params,
                    *none_params,
                    vector_literal,
                    candidate_limit,
                    *min_score_params,
                    query.limit,
                ],
            )
            rows = await cur.fetchall()
        candidate_count = 0 if not rows else int(rows[0][3])
        candidate_limit_reached = candidate_count >= candidate_limit
        return [(str(row[0]), str(row[1]), float(row[2])) for row in rows], candidate_limit_reached

    async def _backfill_candidate_chunks(
        self,
        query: KnowledgeListQuery,
        limit: int,
        *,
        refresh_existing: bool,
    ) -> list[KnowledgeChunk]:
        where_sql, params = _postgres_knowledge_list_filter_sql(query)
        current_embedding_join_sql = ""
        missing_embedding_filter_sql = ""
        current_embedding_params: list[object] = []
        if not refresh_existing:
            current_embedding_join_sql = """
                    LEFT JOIN cayu_knowledge_embeddings AS emb
                      ON emb.chunk_id = c.id
                     AND emb.entry_id = c.entry_id
                     AND emb.model = %s
                     AND emb.dimensions = %s
                     AND emb.embedding_space_version = %s
                     AND (emb.content_hash = c.content_hash OR c.content_hash IS NULL)
                    """
            missing_embedding_filter_sql = "AND emb.chunk_id IS NULL"
            current_embedding_params = [
                self.embedding_model,
                self.embedding_dimensions,
                _EMBEDDING_SPACE_VERSION,
            ]
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    SELECT c.id, c.entry_id, c.chunk_index, c.text, c.content_hash, c.source_uri, c.metadata
                    FROM cayu_knowledge_chunks AS c
                    JOIN cayu_knowledge_entries AS e ON e.id = c.entry_id
                    {current_embedding_join_sql}
                    WHERE TRUE
                    {where_sql}
                    {missing_embedding_filter_sql}
                    ORDER BY COALESCE(e.importance, 0.0) DESC,
                             e.updated_at DESC,
                             e.id ASC,
                             c.chunk_index ASC
                    LIMIT %s
                    """,
                ),
                [*current_embedding_params, *params, limit],
            )
            return [_knowledge_chunk_from_row(row) for row in await cur.fetchall()]

    async def _scored_semantic_rows(
        self,
        cur: Any,
        rows: list[tuple[str, str, float]],
        query: KnowledgeQuery,
    ) -> tuple[
        list[tuple[float, KnowledgeEntry, KnowledgeChunk | None, str, str, float | None]], bool
    ]:
        scored: list[
            tuple[float, KnowledgeEntry, KnowledgeChunk | None, str, str, float | None]
        ] = []
        byte_truncated = False
        semantic_min_score = self.semantic_min_score if query.min_score is None else query.min_score
        for entry_id, chunk_id, normalized_score in rows:
            entry = await self._load_entry(cur, entry_id)
            chunk = await self._load_chunk(cur, chunk_id)
            if entry is None or chunk is None:
                continue
            semantic_matched = normalized_score >= semantic_min_score
            score = normalized_score if semantic_matched else 0.0
            reason = "semantic chunk match"
            preview_text = chunk.text
            score_normalized = normalized_score if semantic_matched else None
            if query.mode in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.HYBRID}:
                chunks = await self._load_chunks(cur, entry.id)
                keyword_score, keyword_chunk, keyword_reason, keyword_preview = _score_entry(
                    entry,
                    chunks,
                    query,
                )
                if keyword_score > 0:
                    keyword_boost = min(keyword_score, 10.0) / 10.0
                    score += self.hybrid_keyword_weight * keyword_boost
                    if keyword_chunk is not None:
                        chunk = keyword_chunk
                    reason = (
                        f"hybrid semantic chunk match; {keyword_reason}"
                        if semantic_matched
                        else f"hybrid keyword match; {keyword_reason}"
                    )
                    preview_text = keyword_preview
            elif not semantic_matched:
                continue
            if score <= 0:
                continue
            scored.append((score, entry, chunk, reason, preview_text, score_normalized))
        scored.sort(
            key=lambda item: (
                -item[0],
                -(item[1].importance or 0.0),
                -item[1].updated_at.timestamp(),
                item[1].id,
            )
        )
        return scored, byte_truncated

    def _merge_keyword_hits(
        self,
        scored: list[tuple[float, KnowledgeEntry, KnowledgeChunk | None, str, str, float | None]],
        keyword_result: KnowledgeSearchResult,
    ) -> list[tuple[float, KnowledgeEntry, KnowledgeChunk | None, str, str, float | None]]:
        merged = list(scored)
        seen_entry_ids = {entry.id for _, entry, _, _, _, _ in merged}
        for hit in keyword_result.hits:
            if hit.entry.id in seen_entry_ids:
                continue
            if hit.score is None:
                continue
            keyword_boost = min(float(hit.score), 10.0) / 10.0
            score = self.hybrid_keyword_weight * keyword_boost
            if score <= 0:
                continue
            text_preview = hit.text_preview
            if text_preview is None:
                continue
            seen_entry_ids.add(hit.entry.id)
            merged.append(
                (
                    score,
                    hit.entry,
                    hit.chunk,
                    f"hybrid keyword match; {hit.reason or 'keyword match'}",
                    hit.text_preview or hit.entry.title or hit.entry.id,
                    None,
                )
            )
        merged.sort(
            key=lambda item: (
                -item[0],
                -(item[1].importance or 0.0),
                -item[1].updated_at.timestamp(),
                item[1].id,
            )
        )
        return merged

    async def _embed_entry_chunks(
        self,
        entry_id: str,
        *,
        chunks: list[KnowledgeChunk] | None = None,
    ) -> None:
        await self._ensure_ready()
        if chunks is None:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                chunks = await self._load_chunks(cur, entry_id)
        await self._embed_chunks(chunks)
        await self._drop_stale_entry_embeddings(entry_id, chunks)

    async def _embed_entry_chunks_best_effort(
        self,
        entry_id: str,
        *,
        chunks: list[KnowledgeChunk] | None = None,
    ) -> None:
        """Embed an entry's chunks, flag-and-continuing on failure.

        The durable entry/chunk write has already committed by the time this runs,
        so an embedding-provider outage must not surface to the caller (which would
        make a successfully-stored entry look like a failed write). We swallow the
        error and leave the embedding rows absent; their absence is the flag that
        ``search`` reads to lazily backfill the embeddings on the next query.
        """
        try:
            await self._embed_entry_chunks(entry_id, chunks=chunks)
        except Exception:
            logger.warning(
                "Deferred embedding for knowledge entry %r after a durable write; "
                "embeddings will be backfilled lazily on the next search.",
                entry_id,
                exc_info=True,
            )

    async def _lazy_backfill_search_scope(self, query: KnowledgeQuery) -> None:
        """Backfill missing embeddings within the search's filter scope.

        Entries whose write-time embedding was deferred (provider outage) have no
        embedding rows and would be invisible to semantic search. Before running
        the semantic query we embed any such chunks that match this query's
        structural filters, bounded by ``_PGVECTOR_LAZY_BACKFILL_LIMIT``. In steady
        state the missing-embedding scan returns nothing, so this is a single cheap
        query. A provider failure here is itself flag-and-continued: the search
        proceeds against whatever embeddings already exist.
        """
        list_query = _knowledge_list_query_for_search(query)
        try:
            chunks = await self._backfill_candidate_chunks(
                list_query,
                _PGVECTOR_LAZY_BACKFILL_LIMIT,
                refresh_existing=False,
            )
            if chunks:
                await self._embed_chunks(chunks)
        except Exception:
            logger.warning(
                "Lazy embedding backfill during search failed; searching against "
                "already-embedded chunks only.",
                exc_info=True,
            )

    async def _embed_chunks(
        self,
        chunks: list[KnowledgeChunk],
        *,
        refresh_existing: bool = False,
    ) -> int:
        if not chunks:
            return 0
        missing = list(chunks) if refresh_existing else await self._missing_embedding_chunks(chunks)
        if not missing:
            return 0
        result = await self.embedding_provider.embed_texts(
            TextEmbeddingRequest(
                model=self.embedding_model,
                texts=[chunk.text for chunk in missing],
                dimensions=self.embedding_dimensions,
            )
        )
        if len(result.embeddings) != len(missing):
            raise ValueError("Embedding provider returned a different number of embeddings.")
        by_index = {embedding.index: embedding for embedding in result.embeddings}
        now = datetime.now(UTC)
        rows: list[tuple[object, ...]] = []
        for index, chunk in enumerate(missing):
            embedding = by_index.get(index)
            if embedding is None:
                raise ValueError("Embedding provider did not return every requested index.")
            self._validate_embedding_dimension(embedding.vector)
            rows.append(
                (
                    chunk.id,
                    chunk.entry_id,
                    _knowledge_chunk_content_hash(chunk),
                    self.embedding_model,
                    self.embedding_dimensions,
                    _EMBEDDING_SPACE_VERSION,
                    _postgres_vector_literal(embedding.vector),
                    now,
                    now,
                )
            )
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.executemany(
                """
                INSERT INTO cayu_knowledge_embeddings (
                    chunk_id,
                    entry_id,
                    content_hash,
                    model,
                    dimensions,
                    embedding_space_version,
                    embedding,
                    created_at,
                    updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    entry_id = excluded.entry_id,
                    content_hash = excluded.content_hash,
                    model = excluded.model,
                    dimensions = excluded.dimensions,
                    embedding_space_version = excluded.embedding_space_version,
                    embedding = excluded.embedding,
                    updated_at = excluded.updated_at
                """,
                rows,
            )
            await conn.commit()
        return len(rows)

    async def _missing_embedding_chunks(
        self,
        chunks: list[KnowledgeChunk],
    ) -> list[KnowledgeChunk]:
        chunk_ids = [chunk.id for chunk in chunks]
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT chunk_id, entry_id, content_hash, model, dimensions, embedding_space_version
                FROM cayu_knowledge_embeddings
                WHERE chunk_id = ANY(%s)
                """,
                (chunk_ids,),
            )
            existing = {str(row[0]): row for row in await cur.fetchall()}
        missing: list[KnowledgeChunk] = []
        for chunk in chunks:
            row = existing.get(chunk.id)
            if (
                row is None
                or str(row[1]) != chunk.entry_id
                or str(row[2]) != _knowledge_chunk_content_hash(chunk)
                or str(row[3]) != self.embedding_model
                or int(row[4]) != self.embedding_dimensions
                or int(row[5]) != _EMBEDDING_SPACE_VERSION
            ):
                missing.append(chunk)
        return missing

    async def _embed_query(self, query: KnowledgeQuery, text: str) -> list[float]:
        result = await self.embedding_provider.embed_texts(
            TextEmbeddingRequest(
                model=self.embedding_model,
                texts=[text],
                dimensions=self.embedding_dimensions,
            )
        )
        embedding = next((item for item in result.embeddings if item.index == 0), None)
        if embedding is None:
            raise ValueError("Embedding provider did not return query embedding index 0.")
        self._validate_embedding_dimension(embedding.vector)
        return list(embedding.vector)

    def _validate_embedding_dimension(self, vector: list[float]) -> None:
        if len(vector) != self.embedding_dimensions:
            raise ValueError("Embedding provider returned a vector with unexpected dimension.")

    async def _delete_entry_embeddings(self, entry_id: str) -> None:
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM cayu_knowledge_embeddings WHERE entry_id = %s",
                (entry_id,),
            )
            await conn.commit()

    async def _drop_stale_entry_embeddings(
        self,
        entry_id: str,
        chunks: list[KnowledgeChunk],
    ) -> None:
        current_ids = [chunk.id for chunk in chunks]
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                DELETE FROM cayu_knowledge_embeddings
                WHERE entry_id = %s
                  AND NOT (chunk_id = ANY(%s))
                """,
                (entry_id, current_ids),
            )
            await conn.commit()


class PostgresSessionStore(_PostgresStoreBase, SessionStore):
    """Postgres-backed session store for durable multi-tenant runtime state."""

    supports_usage_aggregates: ClassVar[bool] = True
    supports_mcp_manifest_history: ClassVar[bool] = True
    supports_session_topology: ClassVar[bool] = True
    _min_required_revision = _POSTGRES_SESSION_MIN_REQUIRED_REVISION
    _supports_read_only = True

    async def create(
        self,
        request: RunRequest,
        *,
        identity: SessionIdentity,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
    ) -> Session:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        request = copy_run_request(request)
        identity = copy_session_identity(identity)
        await self._ensure_ready()
        now = datetime.now(UTC)
        session_id = request.session_id if request.session_id is not None else _new_id()
        session = Session(
            id=session_id,
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
            metadata=copy_durable_json_object(request.metadata, "metadata"),
            labels=request.labels,
        )
        admission = _copy_optional_interaction_admission(
            session.id,
            interaction_started_event,
            interaction_source_messages,
            defer_transcript=True,
        )
        if admission is not None:
            session = session.model_copy(update={"status": SessionStatus.RUNNING, "run_epoch": 1})
        if session.parent_session_id == session.id:
            raise ValueError("Session cannot be its own parent.")
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"""
                        INSERT INTO cayu_sessions ({pg_support.SESSION_COLUMNS})
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        pg_support.session_insert_values(session),
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
                                _dumps(started_event.payload),
                                _dumps(started_event.model_dump(mode="json")),
                                lookup_key,
                                projection,
                                projection_bytes,
                            ),
                        )
                        await self._enqueue_persisted_event_side_effects(
                            cur,
                            session.id,
                            [started_event.id],
                        )
                        await cur.execute(
                            "INSERT INTO cayu_deferred_interaction_inputs "
                            "(session_id, interaction_id, source_messages) "
                            "VALUES (%s, %s, %s)",
                            (
                                session.id,
                                interaction_id,
                                _dumps(
                                    [message.model_dump(mode="json") for message in source_messages]
                                ),
                            ),
                        )
                        await self._upsert_checkpoint(
                            cur,
                            session.id,
                            _initial_transcript_pending_checkpoint(interaction_id),
                            session.updated_at,
                        )
                await conn.commit()
            except UniqueViolation as exc:
                await conn.rollback()
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

    async def create_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        expected_source_run_epoch: int,
    ) -> Session:
        return await self._create_fork(
            source_session_id=source_session_id,
            fork=fork,
            source_statuses=source_statuses,
            transcript_cursor=transcript_cursor,
            checkpoint_transform=checkpoint_transform,
            expected_source_run_epoch=expected_source_run_epoch,
            transcript_validator=None,
        )

    async def create_fork_with_transcript_validation(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        expected_source_run_epoch: int,
        transcript_validator: ForkTranscriptValidator,
    ) -> Session:
        return await self._create_fork(
            source_session_id=source_session_id,
            fork=fork,
            source_statuses=source_statuses,
            transcript_cursor=transcript_cursor,
            checkpoint_transform=checkpoint_transform,
            expected_source_run_epoch=expected_source_run_epoch,
            transcript_validator=transcript_validator,
        )

    async def _create_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        expected_source_run_epoch: int,
        transcript_validator: ForkTranscriptValidator | None,
    ) -> Session:
        source_session_id, fork, allowed_statuses, transcript_cursor = (
            _prepare_session_fork_request(
                source_session_id=source_session_id,
                fork=fork,
                source_statuses=source_statuses,
                transcript_cursor=transcript_cursor,
            )
        )

        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    source_session = _validate_session_fork_source(
                        source_session=await self._load_for_update(cur, source_session_id),
                        source_session_id=source_session_id,
                        fork=fork,
                        allowed_statuses=allowed_statuses,
                        expected_source_run_epoch=expected_source_run_epoch,
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
                        raise ValueError(
                            "Cannot fork a session while a model-completion stage is active: "
                            f"{source_session_id}"
                        )

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
                        SELECT message, interaction_id
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
                        Message(**_json_obj(row[0])) for row in selected_transcript_rows
                    ]
                    copied_interaction_ids = [row[1] for row in selected_transcript_rows]
                    selected_transcript_rows.clear()
                    if not fork_transcript_is_accepted(copied_messages, transcript_validator):
                        copied_messages.clear()
                        copied_messages = []
                        raise ValueError(FORK_TRANSCRIPT_VALIDATION_ERROR) from None

                    copied_checkpoint = None
                    if checkpoint_transform is not None:
                        checkpoint_input = await self._load_checkpoint(
                            cur,
                            source_session_id,
                        )
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

                    await cur.execute(
                        f"""
                        INSERT INTO cayu_sessions ({pg_support.SESSION_COLUMNS})
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        pg_support.session_insert_values(fork),
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
                                (session_id, interaction_id, message)
                            VALUES (%s, %s, %s)
                            """,
                            [
                                (
                                    fork.id,
                                    copied_interaction_ids[index],
                                    _dumps(message.model_dump(mode="json")),
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
                await conn.commit()
            except UniqueViolation as exc:
                await conn.rollback()
                raise ValueError(f"Session already exists: {fork.id}") from exc
            except Exception:
                await conn.rollback()
                raise

            async with conn.cursor() as cur:
                loaded = await self._load(cur, fork.id)
            await conn.commit()
            if loaded is None:
                raise KeyError(f"Session not found: {fork.id}")
            return loaded

    async def load(self, session_id: str) -> Session | None:
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

    async def inspect_identity(self, session_id: str) -> SessionInspectionIdentity:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, agent_name, provider_name, model, parent_session_id,
                       causal_budget_id, runtime_name, runtime_version, environment_name,
                       status, created_at, updated_at, last_activity_at, run_epoch
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

    async def update_model(self, session_id: str, model: str) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        model = require_clean_nonblank(model, "model")
        await self._ensure_ready()
        updated_at = datetime.now(UTC)
        expected_run_epoch = _current_session_run_epoch(session_id)
        async with self._connection() as conn:
            async with conn.cursor() as cur:
                if expected_run_epoch is None:
                    await cur.execute(
                        """
                        UPDATE cayu_sessions
                        SET model = %s, updated_at = %s, last_activity_at = %s
                        WHERE id = %s
                        """,
                        (model, updated_at, updated_at, session_id),
                    )
                else:
                    await cur.execute(
                        """
                        UPDATE cayu_sessions
                        SET model = %s, updated_at = %s, last_activity_at = %s
                        WHERE id = %s AND run_epoch = %s
                        """,
                        (model, updated_at, updated_at, session_id, expected_run_epoch),
                    )
                if cur.rowcount != 1:
                    if expected_run_epoch is not None:
                        await _raise_session_write_conflict(cur, session_id, expected_run_epoch)
                    raise KeyError(f"Session not found: {session_id}")
                loaded = await self._load(cur, session_id)
            await conn.commit()
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            return loaded

    async def delete_session(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    session = await self._load_for_update(cur, session_id)
                    if session is None:
                        await conn.rollback()
                        return
                    if session.status in DELETE_BLOCKED_SESSION_STATUSES:
                        raise ValueError(
                            f"Cannot delete a session while it is {session.status}; "
                            f"interrupt it first: {session_id}"
                        )
                    checkpoint = await self._load_checkpoint(cur, session_id)
                    deletion_now = datetime.now(UTC)
                    active_recovery_claim_id = _active_unexpired_incomplete_recovery_claim_id(
                        checkpoint,
                        now=deletion_now,
                    )
                    if active_recovery_claim_id is not None:
                        raise ValueError(
                            "Cannot delete a session while incomplete-session recovery claim "
                            f"{active_recovery_claim_id} is active: {session_id}"
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
                    await cur.execute(
                        "SELECT 1 FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s",
                        (
                            session_id,
                            MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                        ),
                    )
                    if await cur.fetchone() is not None:
                        raise ValueError(
                            "Cannot delete a session while a model-completion stage is active: "
                            f"{session_id}"
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
                    # ON DELETE CASCADE removes events/labels/checkpoint/transcript;
                    # the self-FK is ON DELETE SET NULL so children keep loading.
                    await cur.execute(
                        "DELETE FROM cayu_sessions WHERE id = %s",
                        (session_id,),
                    )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    async def update_labels(self, session_id: str, labels: dict[str, str]) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        new_labels = copy_label_map(labels, "labels", allow_reserved=False)
        await self._ensure_ready()
        updated_at = datetime.now(UTC)
        expected_run_epoch = _current_session_run_epoch(session_id)
        async with self._connection() as conn:
            async with conn.cursor() as cur:
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
                loaded = await self._load(cur, session_id)
            await conn.commit()
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            return loaded

    async def update_metadata(self, session_id: str, metadata: dict[str, Any]) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        user_metadata = copy_session_user_metadata(metadata)
        await self._ensure_ready()
        updated_at = datetime.now(UTC)
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT run_epoch, metadata FROM cayu_sessions WHERE id = %s FOR UPDATE",
                        (session_id,),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        raise KeyError(f"Session not found: {session_id}")
                    _assert_session_run_epoch_value(session_id, row[0])
                    new_metadata = replace_session_user_metadata(_json_obj(row[1]), user_metadata)
                    await cur.execute(
                        "UPDATE cayu_sessions SET metadata = %s, updated_at = %s WHERE id = %s",
                        (_dumps(new_metadata), updated_at, session_id),
                    )
                    loaded = await self._load(cur, session_id)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
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
        updated_at = datetime.now(UTC)
        expected_run_epoch = _current_session_run_epoch(session_id)
        async with self._connection() as conn:
            async with conn.cursor() as cur:
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
        checkpoint_transform: CheckpointTransform,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        continued_interaction_id: str | None = None,
        defer_interaction_source: bool = False,
    ) -> Session:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        admission = _copy_transition_interaction_admission(
            session_id,
            interaction_started_event,
            interaction_source_messages,
            continued_interaction_id=continued_interaction_id,
            defer_interaction_source=defer_interaction_source,
        )
        if admission is not None and to_status is not SessionStatus.RUNNING:
            raise ValueError("Interaction admission requires a transition to running.")
        await self._ensure_ready()
        updated_at = datetime.now(UTC)
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded = await self._load_for_update(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    _assert_session_run_epoch(session_id, loaded)
                    if loaded.status not in allowed_statuses:
                        raise SessionStatusConflict(
                            f"Session status transition not allowed: {loaded.status} -> {to_status}"
                        )
                    transformed_checkpoint = checkpoint_transform(
                        loaded,
                        await self._load_checkpoint(cur, session_id),
                    )
                    if transformed_checkpoint is not None:
                        transformed_checkpoint = copy_durable_json_object(
                            transformed_checkpoint, "checkpoint"
                        )

                    await cur.execute(
                        """
                        UPDATE cayu_sessions
                        SET status = %s, updated_at = %s, last_activity_at = %s,
                            run_epoch = run_epoch + %s,
                            event_seq = event_seq + %s
                        WHERE id = %s
                        RETURNING event_seq
                        """,
                        (
                            str(to_status),
                            updated_at,
                            updated_at,
                            1 if to_status == SessionStatus.RUNNING else 0,
                            1 if admission is not None and admission[0] is not None else 0,
                            session_id,
                        ),
                    )
                    order_row = await cur.fetchone()
                    if order_row is None:
                        raise KeyError(f"Session not found: {session_id}")
                    if transformed_checkpoint is not None:
                        await self._upsert_checkpoint(
                            cur, session_id, transformed_checkpoint, updated_at
                        )
                    if admission is not None:
                        started_event, interaction_id, source_messages, defer_source = admission
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
                        if started_event is not None:
                            lookup_key, projection, projection_bytes = (
                                pending_action_event_storage_values(started_event)
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
                                    started_event.id,
                                    interaction_id,
                                    str(started_event.type),
                                    pg_support.to_utc(started_event.timestamp),
                                    started_event.agent_name,
                                    started_event.environment_name,
                                    started_event.workflow_name,
                                    started_event.tool_name,
                                    _dumps(started_event.payload),
                                    _dumps(started_event.model_dump(mode="json")),
                                    lookup_key,
                                    projection,
                                    projection_bytes,
                                ),
                            )
                            await self._enqueue_persisted_event_side_effects(
                                cur,
                                session_id,
                                [started_event.id],
                            )
                        if defer_source:
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
                                    _dumps(
                                        [
                                            message.model_dump(mode="json")
                                            for message in source_messages
                                        ]
                                    ),
                                ),
                            )
                        else:
                            await cur.executemany(
                                "INSERT INTO cayu_transcript_messages "
                                "(session_id, interaction_id, message) "
                                "VALUES (%s, %s, %s)",
                                [
                                    (
                                        session_id,
                                        interaction_id,
                                        _dumps(message.model_dump(mode="json")),
                                    )
                                    for message in source_messages
                                ],
                            )
                await conn.commit()
            except UniqueViolation as exc:
                await conn.rollback()
                existing_event_id = (
                    None
                    if admission is None or admission[0] is None
                    else await self._first_existing_event_id(
                        session_id,
                        [admission[0].id],
                    )
                )
                if existing_event_id is not None:
                    raise ValueError(
                        f"Event already exists for session {session_id}: {existing_event_id}"
                    ) from exc
                raise
            except Exception:
                await conn.rollback()
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
        await self._ensure_ready()
        updated_at = datetime.now(UTC)
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded = await self._load_for_update(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    _assert_session_run_epoch(session_id, loaded)
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
                    await cur.execute(
                        "UPDATE cayu_sessions SET status = %s, updated_at = %s, "
                        "last_activity_at = %s, run_epoch = run_epoch + %s WHERE id = %s",
                        (
                            str(to_status),
                            updated_at,
                            updated_at,
                            1 if to_status == SessionStatus.RUNNING else 0,
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
    ) -> InteractionTransitionResult:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        (
            session_id,
            copied_event,
            allowed_statuses,
            target_status,
            conditional,
        ) = _prepare_interaction_transition(
            session_id,
            event=event,
            from_statuses=from_statuses,
            to_status=to_status,
            only_if_no_queued_messages=only_if_no_queued_messages,
        )
        receipt_storage_key = _interaction_transition_storage_key(copied_event.id)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded = await self._load_for_update(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
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
                    if receipt_row is not None:
                        receipt = _reconstruct_interaction_transition_receipt(
                            _json_obj(receipt_row[0]),
                            event=copied_event,
                            from_statuses=allowed_statuses,
                            to_status=target_status,
                            only_if_no_queued_messages=conditional,
                        )
                        if (
                            existing_row is not None
                            and Event(**_json_obj(existing_row[0])) != receipt.event
                        ):
                            raise RuntimeError(
                                "Interaction transition receipt conflicts with retained event history."
                            )
                        await conn.commit()
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
                    updated_at = datetime.now(UTC)
                    await cur.execute(
                        """
                        UPDATE cayu_sessions
                        SET status = CASE WHEN %s THEN status ELSE %s END,
                            updated_at = CASE WHEN %s THEN updated_at ELSE %s END,
                            last_activity_at = %s,
                            event_seq = event_seq + 1
                        WHERE id = %s
                        RETURNING event_seq
                        """,
                        (
                            queued,
                            str(target_status),
                            queued,
                            updated_at,
                            updated_at,
                            session_id,
                        ),
                    )
                    order_row = await cur.fetchone()
                    if order_row is None:
                        raise KeyError(f"Session not found: {session_id}")
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
                            _dumps(copied_event.payload),
                            _dumps(copied_event.model_dump(mode="json")),
                            lookup_key,
                            projection,
                            projection_bytes,
                        ),
                    )
                    await self._enqueue_persisted_event_side_effects(
                        cur,
                        session_id,
                        [copied_event.id],
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
                        status_changed=not queued,
                    )
                    await cur.execute(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record, updated_at) "
                        "VALUES (%s, %s, %s, %s)",
                        (
                            session_id,
                            receipt_storage_key,
                            _dumps(receipt_record),
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
            status_changed=not queued,
        )

    async def fence_stalled_run(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        inactive_before: datetime,
    ) -> Session | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(statuses, "statuses")
        if inactive_before.tzinfo is None or inactive_before.utcoffset() is None:
            raise ValueError("inactive_before must be timezone-aware.")
        await self._ensure_ready()
        now = datetime.now(UTC)
        async with self._connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE cayu_sessions
                    SET run_epoch = run_epoch + 1, last_activity_at = %s
                    WHERE id = %s AND status = ANY(%s) AND last_activity_at <= %s
                    RETURNING run_epoch
                    """,
                    (
                        now,
                        session_id,
                        [str(status) for status in allowed_statuses],
                        inactive_before,
                    ),
                )
                row = await cur.fetchone()
                if row is None:
                    loaded = await self._load(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    await conn.commit()
                    return None
                loaded = await self._load(cur, session_id)
            await conn.commit()
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            _activate_session_run_fence(loaded)
            return loaded

    async def fence_run_and_transform_checkpoint(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        checkpoint_transform: CheckpointTransform,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(statuses, "statuses")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        await self._ensure_ready()
        updated_at = datetime.now(UTC)
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded = await self._load_for_update(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    if loaded.status not in allowed_statuses:
                        raise SessionStatusConflict(
                            f"Session status cannot be fenced: {loaded.status}"
                        )
                    transformed = checkpoint_transform(
                        loaded,
                        await self._load_checkpoint(cur, session_id),
                    )
                    if transformed is None:
                        raise ValueError("Fenced checkpoint transform must return a checkpoint.")
                    transformed = copy_durable_json_object(transformed, "checkpoint")
                    await cur.execute(
                        "UPDATE cayu_sessions SET run_epoch = run_epoch + 1, "
                        "last_activity_at = %s WHERE id = %s",
                        (updated_at, session_id),
                    )
                    await self._upsert_checkpoint(cur, session_id, transformed, updated_at)
                    fenced = loaded.model_copy(
                        update={
                            "run_epoch": loaded.run_epoch + 1,
                            "last_activity_at": updated_at,
                        }
                    )
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

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        session_id, copied_events = _copy_session_event_batch(session_id, events)

        await self._ensure_ready()
        expected_run_epoch = _current_session_run_epoch(session_id)
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    # Reserve a contiguous block of per-session order values by
                    # advancing the session's event counter. UPDATE ... RETURNING
                    # row-locks the session row (serializing concurrent appends to
                    # the same session) and hands back the new counter in one round
                    # trip, replacing a SELECT ... FOR UPDATE + COALESCE(MAX())
                    # scan on this hot write path. A no-op (+0) update on an empty
                    # batch still returns the row, so a missing session is caught.
                    activity_at = datetime.now(UTC)
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
                            await _raise_session_write_conflict(cur, session_id, expected_run_epoch)
                        raise KeyError(f"Session not found: {session_id}")
                    if not copied_events:
                        await conn.commit()
                        return

                    await self._publish_budget_reservation_identities(cur, copied_events)
                    # RETURNING yields the post-increment counter, i.e. the order
                    # of the last event in this batch; walk back to the first.
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
                                _dumps(event.payload),
                                _dumps(event.model_dump(mode="json")),
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
                    await self._enqueue_persisted_event_side_effects(
                        cur,
                        session_id,
                        [event.id for event in copied_events],
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
        from cayu.runtime.pending_actions import pending_action_event_storage_values

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
                    activity_at = datetime.now(UTC)
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
                                _dumps(event.payload),
                                _dumps(event.model_dump(mode="json")),
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
                        [event.id for event in copied_events],
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
                                _dumps(baseline.model_dump(mode="json")),
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
        event_ids: list[str],
    ) -> None:
        if not event_ids:
            return
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
        claim_id = str(uuid4())
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    exact_filter = ""
                    params: list[Any] = []
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
            event=Event(**_json_obj(event_row[0])),
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
                            last_error = %s, updated_at = timing.now
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

    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
    ) -> EnqueueSessionMessageResult:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        request = copy_enqueue_session_message_request(request)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded = await self._load_for_update(cur, request.session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {request.session_id}")
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
                            event=Event(**_json_obj(event_row[0])),
                            replayed=True,
                        )
                    if loaded.status not in {SessionStatus.PENDING, SessionStatus.RUNNING}:
                        raise SessionStatusConflict(
                            "Session messages may be enqueued only while a session is pending or running."
                        )
                    transcript_cursor = await _transcript_cursor(cur, request.session_id)
                    accepted_at = datetime.now(UTC)
                    queue_id = str(uuid4())
                    accepted_event_id = str(uuid4())
                    await cur.execute(
                        """
                        INSERT INTO cayu_session_message_queue (
                            queue_id, session_id, idempotency_key, content,
                            delivery_mode, status, requested_by,
                            accepted_run_epoch, accepted_transcript_cursor,
                            accepted_event_id, accepted_at
                        )
                        VALUES (%s, %s, %s, %s, %s, 'queued', %s, %s, %s, %s, %s)
                        RETURNING ordering_key
                        """,
                        (
                            queue_id,
                            request.session_id,
                            request.idempotency_key,
                            request.content,
                            str(request.delivery_mode),
                            (
                                None
                                if request.requested_by is None
                                else _dumps(resolution_actor_payload(request.requested_by))
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
                    accepted_event = Event(
                        id=accepted_event_id,
                        type=EventType.SESSION_MESSAGE_QUEUED,
                        session_id=request.session_id,
                        agent_name=loaded.agent_name,
                        environment_name=loaded.environment_name,
                        timestamp=accepted_at,
                        payload=_queued_session_message_event_payload(
                            queue_id=queue_id,
                            delivery_mode=request.delivery_mode,
                            ordering_key=ordering_key,
                            actor=request.requested_by,
                            run_epoch=loaded.run_epoch,
                            transcript_cursor=transcript_cursor,
                        ),
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
                            _dumps(accepted_event.payload),
                            _dumps(accepted_event.model_dump(mode="json")),
                            lookup_key,
                            projection,
                            projection_bytes,
                        ),
                    )
                    await self._enqueue_persisted_event_side_effects(
                        cur,
                        request.session_id,
                        [accepted_event.id],
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
        delivery_id: str | None = None,
        eligible_through: int | None = None,
        limit: int = SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
        interaction_id: str | None = None,
        interaction_started_event: Event | None = None,
    ) -> SessionMessageDeliveryBatch:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

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
                               queue_ids, events
                        FROM cayu_session_message_deliveries
                        WHERE delivery_id = %s
                        """,
                        (delivery_id,),
                    )
                    delivery_row = await cur.fetchone()
                    if delivery_row is not None:
                        stored_started_event = (
                            None if delivery_row[7] is None else Event(**_json_obj(delivery_row[7]))
                        )
                        if (
                            delivery_row[0] != session_id
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
                            Event(**_json_obj(event)) for event in delivery_row[9]
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
                        await conn.commit()
                        return SessionMessageDeliveryBatch(
                            messages=tuple(queued_by_id[queue_id] for queue_id in queue_ids),
                            events=replayed_events,
                            delivery_id=delivery_id,
                            interaction_id=interaction_id,
                            eligible_through=delivery_row[4],
                            has_more=delivery_row[6],
                            replayed=True,
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
                                    else _dumps(interaction_started_event.model_dump(mode="json"))
                                ),
                                datetime.now(UTC),
                            ),
                        )
                        await conn.commit()
                        return SessionMessageDeliveryBatch(
                            delivery_id=delivery_id,
                            interaction_id=interaction_id,
                            eligible_through=boundary,
                            has_more=False,
                        )
                    transcript_cursor = await _transcript_cursor(cur, session_id)
                    delivered_at = datetime.now(UTC)
                    updated_messages: list[SessionQueuedMessage] = []
                    delivery_events: list[Event] = []
                    transcript_messages: list[Message] = []
                    for offset, row in enumerate(rows, start=1):
                        queued_message = _queued_session_message_from_row(row)
                        delivered_cursor = transcript_cursor + offset
                        delivery_event = Event(
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
                            },
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
                        transcript_messages.append(
                            Message.text(MessageRole.USER, queued_message.content)
                        )
                    await cur.executemany(
                        "INSERT INTO cayu_transcript_messages "
                        "(session_id, interaction_id, message) VALUES (%s, %s, %s)",
                        [
                            (
                                session_id,
                                interaction_id,
                                _dumps(message.model_dump(mode="json")),
                            )
                            for message in transcript_messages
                        ],
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
                    persisted_events = [
                        *(
                            [interaction_started_event]
                            if interaction_started_event is not None
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
                                _dumps(event.payload),
                                _dumps(event.model_dump(mode="json")),
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
                        [event.id for event in persisted_events],
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
                            remaining is not None,
                            (
                                None
                                if interaction_started_event is None
                                else _dumps(interaction_started_event.model_dump(mode="json"))
                            ),
                            _dumps([message.queue_id for message in updated_messages]),
                            _dumps([event.model_dump(mode="json") for event in persisted_events]),
                            delivered_at,
                        ),
                    )
                await conn.commit()
                return SessionMessageDeliveryBatch(
                    messages=tuple(updated_messages),
                    events=tuple(persisted_events),
                    delivery_id=delivery_id,
                    interaction_id=interaction_id,
                    eligible_through=boundary,
                    has_more=remaining is not None,
                )
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
            operation_commit_guard=None,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )

    async def load_session_operation(
        self,
        session_id: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        idempotency_key = _reject_reserved_runtime_publication_key(
            idempotency_key,
            "idempotency_key",
        )
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT record FROM cayu_session_operations "
                "WHERE session_id = %s AND idempotency_key = %s",
                (session_id, idempotency_key),
            )
            row = await cur.fetchone()
            if row is not None:
                return _json_obj(row[0])
            await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (session_id,))
            if await cur.fetchone() is None:
                raise KeyError(f"Session not found: {session_id}")
            return None

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
            transcript = [Message(**_json_obj(row[1])) for row in transcript_rows]
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
                    "WHERE session_id = %s AND event_id = ANY(%s) FOR SHARE",
                    (receipt.session_id, list(requested_event_ids)),
                )
                events_by_id = {row[0]: Event(**_json_obj(row[1])) for row in await cur.fetchall()}
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
                    await cur.execute(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record, updated_at) "
                        "VALUES (%s, %s, %s, %s)",
                        (
                            session_id,
                            prepared.preparation_storage_key,
                            _dumps(record),
                            prepared_at,
                        ),
                    )
                    active_record = _active_model_completion_stage_record(
                        stage,
                        activated_at=prepared_at,
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
                            _dumps(active_record),
                            prepared_at,
                        ),
                    )
                    await cur.execute(
                        "UPDATE cayu_sessions SET updated_at = %s, last_activity_at = %s "
                        "WHERE id = %s",
                        (prepared_at, prepared_at, session_id),
                    )
                await conn.commit()
                return ModelCompletionStageResult(
                    stage=stage,
                    replayed=False,
                    dispatch_authorized=True,
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
                    if stage.state == "completed":
                        _validate_model_completion_stage_terminal_replay(stage, prepared)
                        await conn.rollback()
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
                            _dumps(terminal_record),
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
                    await cur.execute(
                        "SELECT idempotency_key FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                        (
                            session_id,
                            [winner_storage_key, publication_storage_key],
                        ),
                    )
                    publication_keys = {row[0] for row in await cur.fetchall()}
                    _validate_model_completion_stage_for_abandonment(
                        session=loaded,
                        stage=stage,
                        active=active,
                        prepared=prepared,
                        abandonment_record=records.get(prepared.abandonment_storage_key),
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
                            _dumps(abandonment_record),
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
        from cayu.runtime.pending_actions import (
            pending_action_event_storage_values,
            pending_action_lookup_key,
        )

        session_id = prepared.session_id
        request = prepared.request
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
                            row[0]: Event(**_json_obj(row[1])) for row in await cur.fetchall()
                        }
                    _validate_runtime_publication_event_references(
                        request.referenced_events,
                        durable_references,
                        interaction_id=request.interaction_id,
                    )
                    current_checkpoint = await self._load_checkpoint(cur, session_id)
                    _validate_tool_round_checkpoint_mutation(
                        request,
                        current_checkpoint,
                    )
                    durable_tool_events: list[Event] = []
                    tool_round_identity = _tool_round_publication_identity(request)
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
                                RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS + 1,
                            ),
                        )
                        rows = await cur.fetchall()
                        if len(rows) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
                            raise ValueError(
                                "Tool-round lifecycle evidence exceeds the publication limit."
                            )
                        durable_tool_events = [Event(**_json_obj(row[1])) for row in rows]
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

                    checkpoint = _apply_runtime_publication_checkpoint_mutation(
                        request.mutation,
                        current_checkpoint,
                    )

                    transcript_rows = [
                        (session_id, request.interaction_id, _dumps(message_payload))
                        for message_payload in prepared.transcript_payloads
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
                                _dumps(event_payload["payload"]),
                                _dumps(event_payload),
                                lookup_key,
                                projection,
                                projection_bytes,
                            )
                        )

                    published_at = _next_runtime_publication_timestamp(loaded)
                    checkpoint_values = (
                        None
                        if checkpoint is None or not request.mutation.operations
                        else _checkpoint_row_values(
                            session_id,
                            checkpoint,
                            published_at,
                        )
                    )
                    receipt = _build_runtime_publication_receipt(
                        prepared,
                        source_session=loaded,
                        checkpoint=checkpoint,
                        transcript_start_cursor=transcript_start_cursor,
                        published_at=published_at,
                    )
                    receipt_record = _runtime_publication_receipt_record(receipt)
                    receipt_json = _dumps(receipt_record)

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
                                session_id, interaction_id, message
                            )
                            VALUES (%s, %s, %s)
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
                            [event.id for event in request.events],
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
                                _dumps(winner),
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
            operation_commit_guard=None,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
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
            operation_commit_guard=commit_guard,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )

    async def _publish_checkpoint_and_events(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform | None,
        operation_idempotency_key: str | None,
        operation_transform: SessionOperationTransform | None,
        operation_commit_guard: Callable[[], None] | None,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None,
        expected_run_epoch: int | None,
        expected_transcript_cursor: int | None,
    ) -> Session:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        session_id, copied_events = _copy_session_event_batch(session_id, events)
        if (checkpoint_transform is None) == (operation_transform is None):
            raise TypeError("Exactly one checkpoint publication transform is required.")
        if operation_transform is not None and operation_idempotency_key is None:
            raise TypeError("operation_idempotency_key is required.")
        allowed_statuses = (
            None
            if expected_statuses is None
            else _validate_status_set(expected_statuses, "expected_statuses")
        )
        updated_at = datetime.now(UTC)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    loaded = await self._load_for_update(cur, session_id)
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
                    current_cursor = await _transcript_cursor(cur, session_id)
                    if (
                        expected_transcript_cursor is not None
                        and current_cursor != expected_transcript_cursor
                    ):
                        raise ValueError(
                            "Session source transcript cursor is stale: expected "
                            f"{expected_transcript_cursor}, current {current_cursor}."
                        )
                    current_checkpoint = await self._load_checkpoint(cur, session_id)
                    operation_records: dict[str, dict[str, Any]] = {}
                    if operation_transform is not None:
                        await cur.execute(
                            "SELECT record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = %s",
                            (session_id, operation_idempotency_key),
                        )
                        operation_row = await cur.fetchone()
                        current_operation = (
                            None if operation_row is None else _json_obj(operation_row[0])
                        )
                        publication = operation_transform(
                            loaded,
                            current_checkpoint,
                            current_operation,
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
                        operation_records = copy_durable_json_object(
                            publication.operation_records,
                            "operation_records",
                        )
                        _validate_session_operation_record_keys(operation_records)
                    else:
                        assert checkpoint_transform is not None
                        transformed = checkpoint_transform(loaded, current_checkpoint)
                        if transformed is None:
                            raise ValueError("Checkpoint transform must return a checkpoint.")
                        transformed = copy_durable_json_object(transformed, "checkpoint")

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
                                (session_id, key, _dumps(record), updated_at)
                                for key, record in operation_records.items()
                            ],
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
                                _dumps(event.payload),
                                _dumps(event.model_dump(mode="json")),
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
                        await self._enqueue_persisted_event_side_effects(
                            cur,
                            session_id,
                            [event.id for event in copied_events],
                        )
                    if operation_commit_guard is not None:
                        operation_commit_guard()
                await conn.commit()
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
            return loaded.model_copy(
                update={"updated_at": updated_at, "last_activity_at": updated_at}
            )

    async def load_events(self, session_id: str) -> list[Event]:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
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
            return [Event(**_json_obj(row[0])) for row in rows]

    async def load_tool_round_lifecycle_events(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
    ) -> list[Event]:
        from cayu.runtime.pending_actions import pending_action_lookup_key

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
                    RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS + 1,
                ),
            )
            rows = await cur.fetchall()
            if len(rows) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
                raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
            return [Event(**_json_obj(row[0])) for row in rows]

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
                    RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS + 1,
                ),
            )
            rows = await cur.fetchall()
            if len(rows) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
                raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
            return [Event(**_json_obj(row[0])) for row in rows]

    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        query = copy_event_query(query)
        if len(query.session_ids) > _EVENT_QUERY_SESSION_IDS_BATCH_SIZE:
            return await self._query_events_by_session_id_batches(query)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            return await self._query_events(cur, query, safe_insert_xid=None)

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
            return [EventRecord(sequence=row[0], event=Event(**_json_obj(row[1]))) for row in rows]

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
                SELECT cayu_events.sequence, cayu_events.event
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
        return [EventRecord(sequence=row[0], event=Event(**_json_obj(row[1]))) for row in rows]

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
                        _event_query_with_session_ids(query, session_ids=batch),
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
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
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
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
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
                                          OR (cayu_sessions.created_at, cayu_sessions.id) >
                                             (
                                                 requested.cursor_created_at,
                                                 requested.cursor_id
                                             )
                                      )
                                    ORDER BY cayu_sessions.created_at ASC,
                                             cayu_sessions.id ASC
                                    LIMIT %s
                                ) AS child
                                ORDER BY requested.branch_order ASC,
                                         child.created_at ASC,
                                         child.id ASC
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
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

        return build_session_topology_result(
            focus=focus,
            ancestors=ancestors,
            expanded_parents=expanded_parents,
            branch_candidates=(candidates_by_parent[parent.id] for parent in expanded_parents),
            child_limit=query.child_limit,
        )

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

    async def query_pending_actions(
        self,
        query: PendingActionQuery | None = None,
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
            "cayu_checkpoints.pending_action_metrics_ready",
            "cayu_checkpoints.pending_action_flags <> 0",
        ]
        params: list[Any] = []
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
                    cayu_checkpoints.state -> 'pending_tool_round'
                )) AS pending_state
            FROM cayu_checkpoints
            WHERE cayu_checkpoints.session_id = ANY(%s)
        """
        checkpoint_preflight_sql = """
            SELECT
                cayu_checkpoints.session_id,
                cayu_checkpoints.pending_action_source_bytes AS pending_state_bytes,
                cayu_checkpoints.pending_action_tool_call_count
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
                    ('session.interrupted')
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
                                event=Event(**_json_obj(pending_event.get("event"))),
                            )
                        )
                    grouped[session_id] = (
                        copy_durable_json_object(_json_obj(row[1]), "checkpoint"),
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
        plan = session_store_sql.build_session_query_sql(query, dialect=_SQL_DIALECT)

        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
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
                    ORDER BY {plan.order_sql}, id ASC
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
                await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (session_id,))
                if await cur.fetchone() is None:
                    raise KeyError(f"Session not found: {session_id}")
                if copied_messages:
                    await _touch_session_activity(cur, session_id, datetime.now(UTC))
                    await cur.executemany(
                        """
                        INSERT INTO cayu_transcript_messages
                            (session_id, interaction_id, message)
                        VALUES (%s, %s, %s)
                        """,
                        [
                            (
                                session_id,
                                interaction_id,
                                _dumps(message.model_dump(mode="json")),
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
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        if interaction_id is None:
            raise ValueError("Initial transcript publication requires an interaction identity.")
        expected = copy_transcript_messages(expected_messages)
        replacement = copy_transcript_messages(replacement_messages)
        updated_at = datetime.now(UTC)
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
                    if row is None or row[0] != interaction_id:
                        raise RuntimeError(
                            "Deferred interaction input changed before finalization."
                        )
                    stored = [Message(**item) for item in _json_list(row[1])]
                    if stored != expected:
                        raise RuntimeError(
                            "Deferred interaction input changed before finalization."
                        )
                    await cur.execute(
                        "SELECT 1 FROM cayu_transcript_messages WHERE session_id = %s LIMIT 1",
                        (session_id,),
                    )
                    if await cur.fetchone() is not None:
                        raise RuntimeError("Initial transcript changed before finalization.")
                    if len(replacement) < len(expected) or (
                        expected and replacement[-len(expected) :] != expected
                    ):
                        raise RuntimeError(
                            "Initial transcript must preserve the admitted source suffix."
                        )
                    prefix_count = len(replacement) - len(expected)
                    checkpoint = _checkpoint_after_initial_transcript_publication(
                        await self._load_checkpoint(cur, session_id),
                        interaction_id=interaction_id,
                    )
                    await cur.executemany(
                        "INSERT INTO cayu_transcript_messages "
                        "(session_id, interaction_id, message) VALUES (%s, %s, %s)",
                        [
                            (
                                session_id,
                                None if index < prefix_count else interaction_id,
                                _dumps(message.model_dump(mode="json")),
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
                    messages = [Message(**item) for item in _json_list(row[1])]
                    await cur.executemany(
                        "INSERT INTO cayu_transcript_messages "
                        "(session_id, interaction_id, message) VALUES (%s, %s, %s)",
                        [
                            (
                                session_id,
                                interaction_id,
                                _dumps(message.model_dump(mode="json")),
                            )
                            for message in messages
                        ],
                    )
                    await cur.execute(
                        "DELETE FROM cayu_deferred_interaction_inputs WHERE session_id = %s",
                        (session_id,),
                    )
                    await _touch_session_activity(cur, session_id, datetime.now(UTC))
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
        return DeferredInteractionInput(
            interaction_id=row[0],
            source_messages=[Message(**item) for item in _json_list(row[1])],
        )

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
        updated_at = datetime.now(UTC)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    session = await self._load_for_update(cur, session_id)
                    if session is None:
                        raise KeyError(f"Session not found: {session_id}")
                    _assert_session_run_epoch(session_id, session)
                    transformed = checkpoint_transform(
                        session,
                        await self._load_checkpoint(cur, session_id),
                    )
                    if transformed is None:
                        raise ValueError("Checkpoint transform must return a checkpoint.")
                    transformed = copy_durable_json_object(transformed, "checkpoint")
                    await _touch_session_activity(cur, session_id, updated_at)
                    if copied_messages:
                        await cur.executemany(
                            """
                            INSERT INTO cayu_transcript_messages
                                (session_id, interaction_id, message)
                            VALUES (%s, %s, %s)
                            """,
                            [
                                (
                                    session_id,
                                    interaction_id,
                                    _dumps(message.model_dump(mode="json")),
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
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
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
            return [Message(**_json_obj(row[0])) for row in rows]

    async def query_transcript(self, query: TranscriptQuery) -> TranscriptPage:
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
                    message=Message(**_json_obj(row[2])),
                )
                for row in rows
            ]
            return TranscriptPage(
                records=filter_transcript_records(records, include_thinking=query.include_thinking),
                total_records=total_records,
            )

    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state must be a dictionary.")
        copied = copy_durable_json_object(state, "checkpoint")
        updated_at = datetime.now(UTC)
        await self._ensure_ready()
        async with self._connection() as conn:
            async with conn.cursor() as cur:
                if await self._load_for_update(cur, session_id) is None:
                    raise KeyError(f"Session not found: {session_id}")
                await _touch_session_activity(cur, session_id, updated_at)
                await self._upsert_checkpoint(cur, session_id, copied, updated_at)
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
        updated_at = datetime.now(UTC)
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    session = await self._load_for_update(cur, session_id)
                    if session is None:
                        raise KeyError(f"Session not found: {session_id}")
                    _assert_session_run_epoch(session_id, session)
                    transformed = checkpoint_transform(
                        session,
                        await self._load_checkpoint(cur, session_id),
                    )
                    if transformed is not None:
                        transformed = copy_durable_json_object(transformed, "checkpoint")
                        await _touch_session_activity(cur, session_id, updated_at)
                        await self._upsert_checkpoint(cur, session_id, transformed, updated_at)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            return await self._load_checkpoint(cur, session_id)

    async def load_interruption_cascade_marker(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                WITH marker AS (
                    SELECT state -> 'pending_interruption_cascade' AS value
                    FROM cayu_checkpoints
                    WHERE session_id = %s
                )
                SELECT
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
            field_types = {
                "attempt_id": row[1],
                "interrupt_payload": row[3],
                "generation": row[4],
                "failure_recorded": row[6],
                "claim_id": row[8],
                "claim_expires_at": row[10],
                "created_at": row[12],
            }
            field_values = {
                "attempt_id": row[2],
                "generation": row[5],
                "failure_recorded": row[7],
                "claim_id": row[9],
                "claim_expires_at": row[11],
                "created_at": row[13],
            }
            return _project_interruption_cascade_marker_fields(
                row[0],
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

    async def _load_checkpoint(self, cur: Any, session_id: str) -> dict[str, Any] | None:
        await cur.execute(
            "SELECT state FROM cayu_checkpoints WHERE session_id = %s",
            (session_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return copy_durable_json_object(_json_obj(row[0]), "checkpoint")

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


class PostgresTaskStore(_PostgresStoreBase, TaskStore):
    """Postgres-backed task store for durable multi-tenant work items."""

    supports_task_topology: ClassVar[bool] = True
    _min_required_revision = 27

    async def create_task(self, request: TaskCreate) -> Task:
        request = copy_task_create(request)
        await self._ensure_ready()
        task = _task_from_create(request)
        await self._insert_task(task)
        return task.model_copy(deep=True)

    async def create_running_task(self, request: TaskCreate) -> Task:
        request = copy_task_create(request)
        await self._ensure_ready()
        task = _running_task_from_create(request)
        await self._insert_task(task)
        return task.model_copy(deep=True)

    async def _insert_task(self, task: Task) -> None:
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"""
                        INSERT INTO cayu_tasks ({pg_support.TASK_COLUMNS})
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s
                        )
                        """,
                        pg_support.task_insert_values(task),
                    )
                await conn.commit()
            except UniqueViolation as exc:
                await conn.rollback()
                raise ValueError(f"Task already exists: {task.id}") from exc

    async def load_task(self, task_id: str) -> Task | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            return await self._load_task(cur, task_id)

    async def list_tasks(self, query: TaskQuery | None = None) -> list[Task]:
        query = copy_task_query(query)
        clauses: list[str] = []
        params: list[object] = []

        if query.q is not None:
            like = _ilike_contains_pattern(query.q)
            clauses.append(
                """
                (
                    id ILIKE %s ESCAPE '\\'
                    OR type ILIKE %s ESCAPE '\\'
                    OR title ILIKE %s ESCAPE '\\'
                    OR description ILIKE %s ESCAPE '\\'
                    OR status ILIKE %s ESCAPE '\\'
                    OR session_id ILIKE %s ESCAPE '\\'
                    OR parent_task_id ILIKE %s ESCAPE '\\'
                    OR assigned_agent_name ILIKE %s ESCAPE '\\'
                    OR worker_id ILIKE %s ESCAPE '\\'
                    OR status_reason ILIKE %s ESCAPE '\\'
                )
                """
            )
            params.extend([like] * 10)
        if query.status is not None:
            clauses.append("status = %s")
            params.append(str(query.status))
        if query.type is not None:
            clauses.append("type = %s")
            params.append(query.type)
        if query.session_id is not None:
            clauses.append("session_id = %s")
            params.append(query.session_id)
        if query.parent_task_id is not None:
            clauses.append("parent_task_id = %s")
            params.append(query.parent_task_id)
        if query.assigned_agent_name is not None:
            clauses.append("assigned_agent_name = %s")
            params.append(query.assigned_agent_name)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_sql = pg_support.task_order_sql(query.order_by)
        params.extend([query.limit, query.offset])

        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            # Interpolations are trusted: TASK_COLUMNS is a constant, order_sql is an
            # enum-derived literal, where_sql is hard-coded clauses; values bind via %s.
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    SELECT {pg_support.TASK_COLUMNS}
                    FROM cayu_tasks
                    {where_sql}
                    ORDER BY {order_sql}, id ASC
                    LIMIT %s OFFSET %s
                    """,
                ),
                params,
            )
            rows = await cur.fetchall()
            return [pg_support.task_from_row(row) for row in rows]

    async def query_task_topology(
        self,
        query: TaskTopologyQuery,
    ) -> TaskTopologyStoreResult:
        if type(query) is not TaskTopologyQuery:
            raise TypeError("Task topology queries must be TaskTopologyQuery instances.")
        query = TaskTopologyQuery.model_validate(query.model_dump(mode="python"))
        session_branch_limits, child_branch_limits = _allocate_task_topology_branch_limits(query)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                    await cur.execute("SELECT transaction_timestamp()")
                    observed_row = await cur.fetchone()
                    if observed_row is None:
                        raise RuntimeError("Postgres did not return a topology snapshot timestamp.")

                    expanded_parents: list[TaskTopologyNode] = []
                    if query.expanded_parent_ids:
                        await cur.execute(
                            cast(
                                "LiteralString",
                                f"""
                                SELECT {pg_support.TASK_TOPOLOGY_COLUMNS}
                                FROM cayu_tasks
                                WHERE id = ANY(%s)
                                """,
                            ),
                            (list(query.expanded_parent_ids),),
                        )
                        parents_by_id = {
                            row[0]: pg_support.task_topology_node_from_row(row)
                            for row in await cur.fetchall()
                        }
                        for parent_id in query.expanded_parent_ids:
                            parent = parents_by_id.get(parent_id)
                            if parent is None:
                                raise KeyError(f"Task not found: {parent_id}")
                            expanded_parents.append(parent)

                    async def read_branch_candidates(
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
                        cursor_created_ats: list[datetime | None] = []
                        cursor_ids: list[str | None] = []
                        for branch_id in branch_ids:
                            cursor = cursors.get(branch_id)
                            if cursor is None:
                                cursor_created_ats.append(None)
                                cursor_ids.append(None)
                                continue
                            cursor_created_at, cursor_id = decode_task_topology_cursor(
                                cursor,
                                scope_kind=scope_kind,
                                scope_id=branch_id,
                            )
                            cursor_created_ats.append(cursor_created_at)
                            cursor_ids.append(cursor_id)
                        branch_sql: LiteralString = f"""
                                WITH requested_branches AS (
                                    SELECT branch_id, cursor_created_at, cursor_id,
                                           candidate_limit, branch_order
                                    FROM unnest(
                                        %s::text[],
                                        %s::timestamptz[],
                                        %s::text[],
                                        %s::integer[]
                                    ) WITH ORDINALITY AS requested(
                                        branch_id,
                                        cursor_created_at,
                                        cursor_id,
                                        candidate_limit,
                                        branch_order
                                    )
                                )
                                SELECT requested.branch_order, candidate.*
                                FROM requested_branches AS requested
                                CROSS JOIN LATERAL (
                                    SELECT {pg_support.TASK_TOPOLOGY_COLUMNS}
                                    FROM cayu_tasks
                                    WHERE cayu_tasks.{scope_column} = requested.branch_id
                                      AND (
                                          requested.cursor_created_at IS NULL
                                          OR cayu_tasks.created_at >
                                             requested.cursor_created_at
                                          OR (
                                              cayu_tasks.created_at =
                                                  requested.cursor_created_at
                                              AND cayu_tasks.id COLLATE "C" >
                                                  requested.cursor_id COLLATE "C"
                                          )
                                      )
                                    ORDER BY cayu_tasks.created_at ASC,
                                             cayu_tasks.id COLLATE "C" ASC
                                    LIMIT requested.candidate_limit
                                ) AS candidate
                                ORDER BY requested.branch_order ASC,
                                         candidate.topology_created_at ASC,
                                         candidate.topology_id COLLATE "C" ASC
                                """
                        await cur.execute(
                            branch_sql,
                            (
                                list(branch_ids),
                                cursor_created_ats,
                                cursor_ids,
                                [limit + 1 for limit in branch_limits],
                            ),
                        )
                        for row in await cur.fetchall():
                            branch_index = int(row[0]) - 1
                            candidates[branch_index].append(
                                pg_support.task_topology_node_from_row(row[1:])
                            )
                        return candidates

                    session_candidates = await read_branch_candidates(
                        branch_ids=query.linked_session_ids,
                        cursors=query.session_cursors,
                        scope_kind="session",
                        scope_column="session_id",
                        branch_limits=session_branch_limits,
                    )
                    child_candidates = await read_branch_candidates(
                        branch_ids=query.expanded_parent_ids,
                        cursors=query.child_cursors,
                        scope_kind="parent_task",
                        scope_column="parent_task_id",
                        branch_limits=child_branch_limits,
                    )

                    async def load_parent_links(
                        task_ids: tuple[str, ...],
                    ) -> dict[str, str | None]:
                        await cur.execute(
                            cast(
                                "LiteralString",
                                f"""
                                SELECT
                                    id,
                                    CASE
                                        WHEN parent_task_id IS NULL
                                          OR octet_length(parent_task_id)
                                             <= {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
                                        THEN parent_task_id
                                    END AS topology_parent_task_id,
                                    parent_task_id IS NOT NULL
                                      AND octet_length(parent_task_id)
                                          > {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
                                        AS topology_parent_task_id_oversized
                                FROM cayu_tasks
                                WHERE id = ANY(%s)
                                """,
                            ),
                            (list(task_ids),),
                        )
                        links: dict[str, str | None] = {}
                        for task_id, parent_task_id, parent_id_oversized in await cur.fetchall():
                            if parent_id_oversized:
                                raise TaskTopologyInconsistent(
                                    "A task topology ancestor contains an oversized "
                                    "parent identifier."
                                )
                            links[task_id] = _bounded_optional_task_topology_parent_id(
                                parent_task_id
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
                        observed_at=observed_row[0],
                        linked_session_ids=query.linked_session_ids,
                        session_branch_candidates=session_candidates,
                        session_branch_limits=session_branch_limits,
                        expanded_parents=expanded_parents,
                        child_branch_candidates=child_candidates,
                        child_branch_limits=child_branch_limits,
                        session_task_limit=query.session_task_limit,
                        child_limit=query.child_limit,
                    )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return result

    async def aggregate_operational_snapshot(
        self,
        filters: TaskAggregateFilter | None = None,
    ) -> TaskOperationalSnapshot:
        filters = copy_task_aggregate_filter(filters)
        query = task_query_from_aggregate_filter(filters)
        clauses: list[str] = []
        params: list[object] = []
        for column, value in (
            ("type", query.type),
            ("session_id", query.session_id),
            ("parent_task_id", query.parent_task_id),
            ("assigned_agent_name", query.assigned_agent_name),
        ):
            if value is not None:
                clauses.append(f"{column} = %s")
                params.append(value)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        await self._ensure_ready()
        async with self._pool.connection() as conn:
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
                            FROM cayu_tasks
                            {where_sql}
                            GROUP BY status
                            """,
                        ),
                        params,
                    )
                    rows = await cur.fetchall()
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

        counts = {status: 0 for status in TaskStatus}
        for row in rows:
            counts[TaskStatus(row[0])] = row[1]
        return TaskOperationalSnapshot(
            as_of=as_of_row[0],
            total_count=sum(counts.values()),
            counts_by_status=TaskStatusCounts.model_validate(counts),
            accuracy=EXACT_AGGREGATE.model_copy(),
        )

    async def start_task(
        self,
        task_id: str,
        *,
        session_id: str | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        now = datetime.now(UTC)
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE cayu_tasks
                SET status = %s,
                    session_id = COALESCE(%s, session_id),
                    started_at = COALESCE(started_at, %s),
                    updated_at = %s
                WHERE id = %s AND status = %s
                """,
                (
                    str(TaskStatus.RUNNING),
                    session_id,
                    now,
                    now,
                    task_id,
                    str(TaskStatus.PENDING),
                ),
            )
            if cur.rowcount == 1:
                updated = await self._require_task(cur, task_id)
                await conn.commit()
                return updated.model_copy(deep=True)
            task = await self._require_task(cur, task_id)
            _ensure_can_transition(task, TaskStatus.RUNNING)
            raise ValueError(f"Task {task.id} cannot transition to running from {task.status}")

    async def attach_task(
        self,
        task_id: str,
        *,
        session_id: str,
        worker_id: str,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        session_id = require_clean_nonblank(session_id, "session_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        await self._ensure_ready()
        now = datetime.now(UTC)
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                f"""
                UPDATE cayu_tasks
                SET status = %s,
                    session_id = %s,
                    started_at = COALESCE(started_at, %s),
                    updated_at = %s
                WHERE id = %s
                  AND status = %s
                  AND worker_id = %s
                  AND session_id IS NULL
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at > %s
                RETURNING {pg_support.TASK_COLUMNS}
                """,
                (
                    str(TaskStatus.RUNNING),
                    session_id,
                    now,
                    now,
                    task_id,
                    str(TaskStatus.CLAIMED),
                    worker_id,
                    now,
                ),
            )
            row = await cur.fetchone()
            if row is None:
                await self._raise_task_claim_attach_error(cur, task_id, worker_id)
            assert row is not None
            updated = pg_support.task_from_row(row)
            await conn.commit()
            return updated.model_copy(deep=True)

    async def complete_task(
        self, task_id: str, result: dict[str, Any], *, worker_id: str | None = None
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        result = copy_durable_json_object(result, "result")
        return await self._finish_task(
            task_id, TaskStatus.COMPLETED, result=result, error=None, worker_id=worker_id
        )

    async def fail_task(
        self, task_id: str, error: dict[str, Any], *, worker_id: str | None = None
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        error = copy_durable_json_object(error, "error")
        return await self._finish_task(
            task_id, TaskStatus.FAILED, result=None, error=error, worker_id=worker_id
        )

    async def cancel_task(
        self,
        task_id: str,
        error: dict[str, Any] | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        copied_error = None if error is None else copy_durable_json_object(error, "error")
        return await self._finish_task(
            task_id, TaskStatus.CANCELLED, result=None, error=copied_error
        )

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
        await self._ensure_ready()
        now = datetime.now(UTC)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    UPDATE cayu_tasks
                    SET status = %s,
                        status_reason = NULL,
                        status_payload = NULL,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        updated_at = %s
                    WHERE id = %s
                      AND status IN (%s, %s, %s)
                    RETURNING {pg_support.TASK_COLUMNS}
                    """,
                    (
                        str(TaskStatus.PENDING),
                        now,
                        task_id,
                        str(TaskStatus.PAUSED),
                        str(TaskStatus.BLOCKED),
                        str(TaskStatus.NEEDS_ATTENTION),
                    ),
                )
                row = await cur.fetchone()
                if row is None:
                    task = await self._require_task(cur, task_id)
                    _ensure_can_resume_task(task)
                    raise ValueError(f"Task {task.id} cannot resume from {task.status}")
                updated = pg_support.task_from_row(row)
            await conn.commit()
            return updated.model_copy(deep=True)

    async def claim_task(
        self,
        worker_id: str,
        query: TaskQuery | None = None,
        *,
        lease_seconds: int = 300,
    ) -> Task | None:
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        lease_seconds = _validate_task_positive_int(lease_seconds, "lease_seconds")
        if query.status is not None and query.status is not TaskStatus.PENDING:
            return None
        clauses, params = self._task_filter_clauses(query)
        where_sql = " AND ".join(["status = %s", "session_id IS NULL", *clauses])
        # Claiming is always FIFO by creation time, independent of the query's
        # display ordering, so the oldest pending task is dispatched first.
        order_sql = pg_support.task_order_sql(TaskOrder.CREATED_AT_ASC)
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=lease_seconds)

        await self._ensure_ready()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    cast(
                        "LiteralString",
                        f"""
                        WITH candidate AS (
                            SELECT id
                            FROM cayu_tasks
                            WHERE {where_sql}
                            ORDER BY {order_sql}, id ASC
                            FOR UPDATE SKIP LOCKED
                            LIMIT 1
                        )
                        UPDATE cayu_tasks AS task
                        SET status = %s,
                            worker_id = %s,
                            lease_expires_at = %s,
                            updated_at = %s
                        FROM candidate
                        WHERE task.id = candidate.id
                        RETURNING {_TASK_RETURNING_COLUMNS}
                        """,
                    ),
                    [
                        str(TaskStatus.PENDING),
                        *params,
                        str(TaskStatus.CLAIMED),
                        worker_id,
                        lease_expires_at,
                        now,
                    ],
                )
                row = await cur.fetchone()
            await conn.commit()
        if row is None:
            return None
        return pg_support.task_from_row(row).model_copy(deep=True)

    async def heartbeat(
        self,
        task_id: str,
        worker_id: str,
        *,
        extend_seconds: int = 300,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        extend_seconds = _validate_task_positive_int(extend_seconds, "extend_seconds")
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=extend_seconds)

        await self._ensure_ready()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    UPDATE cayu_tasks
                    SET lease_expires_at = %s,
                        updated_at = %s
                    WHERE id = %s AND worker_id = %s AND status IN (%s, %s)
                      AND lease_expires_at IS NOT NULL AND lease_expires_at > %s
                    RETURNING {pg_support.TASK_COLUMNS}
                    """,
                    (
                        lease_expires_at,
                        now,
                        task_id,
                        worker_id,
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.RUNNING),
                        now,
                    ),
                )
                row = await cur.fetchone()
                if row is None:
                    await self._raise_task_active_lease_error(cur, task_id, worker_id)
                assert row is not None
                updated = pg_support.task_from_row(row)
            await conn.commit()
            return updated.model_copy(deep=True)

    async def release_task(self, task_id: str, worker_id: str) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        now = datetime.now(UTC)

        await self._ensure_ready()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    UPDATE cayu_tasks
                    SET status = %s,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        updated_at = %s
                    WHERE id = %s AND worker_id = %s AND status = %s
                      AND session_id IS NULL
                      AND lease_expires_at IS NOT NULL AND lease_expires_at > %s
                    RETURNING {pg_support.TASK_COLUMNS}
                    """,
                    (
                        str(TaskStatus.PENDING),
                        now,
                        task_id,
                        worker_id,
                        str(TaskStatus.CLAIMED),
                        now,
                    ),
                )
                row = await cur.fetchone()
                if row is None:
                    await self._raise_task_release_error(cur, task_id, worker_id)
                assert row is not None
                updated = pg_support.task_from_row(row)
            await conn.commit()
            return updated.model_copy(deep=True)

    async def release_attached_task_worker(self, task_id: str, worker_id: str) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        now = datetime.now(UTC)

        await self._ensure_ready()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    UPDATE cayu_tasks
                    SET worker_id = NULL,
                        lease_expires_at = NULL,
                        updated_at = %s
                    WHERE id = %s AND worker_id = %s AND status = %s
                      AND session_id IS NOT NULL
                      AND lease_expires_at IS NOT NULL AND lease_expires_at > %s
                    RETURNING {pg_support.TASK_COLUMNS}
                    """,
                    (
                        now,
                        task_id,
                        worker_id,
                        str(TaskStatus.RUNNING),
                        now,
                    ),
                )
                row = await cur.fetchone()
                if row is None:
                    await self._raise_attached_task_worker_release_error(
                        cur,
                        task_id,
                        worker_id,
                    )
                assert row is not None
                updated = pg_support.task_from_row(row)
            await conn.commit()
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
                "status = %s",
                "session_id IS NULL",
                "lease_expires_at IS NOT NULL",
                "lease_expires_at <= %s",
                *clauses,
            ]
        )
        now = datetime.now(UTC)

        await self._ensure_ready()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    cast(
                        "LiteralString",
                        f"""
                        WITH expired AS (
                            SELECT id
                            FROM cayu_tasks
                            WHERE {where_sql}
                            ORDER BY lease_expires_at ASC, id ASC
                            FOR UPDATE SKIP LOCKED
                            LIMIT %s
                        )
                        UPDATE cayu_tasks AS task
                        SET status = %s,
                            worker_id = NULL,
                            lease_expires_at = NULL,
                            updated_at = %s
                        FROM expired
                        WHERE task.id = expired.id
                        RETURNING {_TASK_RETURNING_COLUMNS}
                        """,
                    ),
                    [
                        str(TaskStatus.CLAIMED),
                        now,
                        *params,
                        max_reclaims,
                        str(TaskStatus.PENDING),
                        now,
                    ],
                )
                rows = await cur.fetchall()
            await conn.commit()
        return [pg_support.task_from_row(row).model_copy(deep=True) for row in rows]

    # -- internal helpers -------------------------------------------------

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
        await self._ensure_ready()
        now = datetime.now(UTC)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    UPDATE cayu_tasks
                    SET status = %s,
                        status_reason = %s,
                        status_payload = %s,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        updated_at = %s
                    WHERE id = %s
                      AND (
                        status = %s
                        OR status = %s
                        OR status = %s
                        OR status = %s
                        OR status = %s
                        OR (status = %s AND session_id IS NULL)
                      )
                    RETURNING {pg_support.TASK_COLUMNS}
                    """,
                    (
                        str(status),
                        reason,
                        None if payload is None else _dumps(payload),
                        now,
                        task_id,
                        str(TaskStatus.PENDING),
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.PAUSED),
                        str(TaskStatus.BLOCKED),
                        str(TaskStatus.NEEDS_ATTENTION),
                        str(TaskStatus.RUNNING),
                    ),
                )
                row = await cur.fetchone()
                if row is None:
                    task = await self._require_task(cur, task_id)
                    _ensure_can_hold_task(task, status)
                    raise ValueError(f"Task {task.id} cannot transition to {status}")
                updated = pg_support.task_from_row(row)
            await conn.commit()
            return updated.model_copy(deep=True)

    async def _finish_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
        worker_id: str | None = None,
    ) -> Task:
        await self._ensure_ready()
        now = datetime.now(UTC)
        # When a worker_id is given, only terminalize if that worker still owns an active
        # lease — a worker that lost its lease must not clobber a task another has reclaimed.
        owner_clause = ""
        owner_params: list[Any] = []
        if worker_id is not None:
            owner_clause = (
                "\n                      AND worker_id = %s"
                "\n                      AND lease_expires_at IS NOT NULL AND lease_expires_at > %s"
            )
            owner_params = [worker_id, now]
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    UPDATE cayu_tasks
                    SET status = %s,
                        status_reason = NULL,
                        status_payload = NULL,
                        result = %s,
                        error = %s,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        started_at = COALESCE(started_at, %s),
                        completed_at = %s,
                        updated_at = %s
                    WHERE id = %s
                      AND status NOT IN (%s, %s, %s){owner_clause}
                    """,
                    (
                        str(status),
                        None if result is None else _dumps(result),
                        None if error is None else _dumps(error),
                        now,
                        now,
                        now,
                        task_id,
                        str(TaskStatus.COMPLETED),
                        str(TaskStatus.FAILED),
                        str(TaskStatus.CANCELLED),
                        *owner_params,
                    ),
                )
                if cur.rowcount != 1:
                    if worker_id is not None:
                        await self._raise_task_active_lease_error(cur, task_id, worker_id)
                    task = await self._require_task(cur, task_id)
                    _ensure_can_transition(task, status)
                    raise ValueError(f"Task {task.id} cannot transition from {task.status}")
                updated = await self._require_task(cur, task_id)
            await conn.commit()
            return updated.model_copy(deep=True)

    async def _load_task(self, cur: Any, task_id: str) -> Task | None:
        await cur.execute(
            f"SELECT {pg_support.TASK_COLUMNS} FROM cayu_tasks WHERE id = %s",
            (task_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return pg_support.task_from_row(row)

    async def _require_task(self, cur: Any, task_id: str) -> Task:
        task = await self._load_task(cur, task_id)
        if task is None:
            raise KeyError(f"Task not found: {task_id}")
        return task

    def _task_filter_clauses(self, query: TaskQuery) -> tuple[list[str], list[object]]:
        clauses: list[str] = []
        params: list[object] = []
        if query.type is not None:
            clauses.append("type = %s")
            params.append(query.type)
        if query.session_id is not None:
            clauses.append("session_id = %s")
            params.append(query.session_id)
        if query.parent_task_id is not None:
            clauses.append("parent_task_id = %s")
            params.append(query.parent_task_id)
        if query.assigned_agent_name is not None:
            clauses.append("assigned_agent_name = %s")
            params.append(query.assigned_agent_name)
        return clauses, params

    async def _raise_task_active_lease_error(
        self,
        cur: Any,
        task_id: str,
        worker_id: str,
    ) -> None:
        task = await self._require_task(cur, task_id)
        _ensure_owned_active_task_lease(task, worker_id)
        raise RuntimeError(f"Task {task.id} active-lease mutation did not update a row.")

    async def _raise_task_release_error(
        self,
        cur: Any,
        task_id: str,
        worker_id: str,
    ) -> None:
        task = await self._require_task(cur, task_id)
        _ensure_owned_active_task_lease(task, worker_id)
        if task.session_id is not None:
            raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
        if task.status is not TaskStatus.CLAIMED:
            raise ValueError(f"Task {task.id} is not claimed.")
        raise RuntimeError(f"Task {task.id} active claim could not be released.")

    async def _raise_attached_task_worker_release_error(
        self,
        cur: Any,
        task_id: str,
        worker_id: str,
    ) -> None:
        task = await self._require_task(cur, task_id)
        _ensure_owned_active_task_lease(task, worker_id)
        if task.status is not TaskStatus.RUNNING:
            raise ValueError(f"Task {task.id} is not running.")
        if task.session_id is None:
            raise ValueError(f"Task {task.id} is not attached to a session.")
        raise RuntimeError(f"Task {task.id} active attached claim could not be released.")

    async def _raise_task_claim_attach_error(
        self,
        cur: Any,
        task_id: str,
        worker_id: str,
    ) -> None:
        task = await self._require_task(cur, task_id)
        _raise_task_claim_attach_error(task, worker_id)


def _new_id() -> str:
    from uuid import uuid4

    return str(uuid4())


def _dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _checkpoint_row_values(
    session_id: str,
    checkpoint: dict[str, Any],
    updated_at: datetime,
) -> tuple[object, ...]:
    from cayu.runtime.pending_actions import pending_action_checkpoint_metrics

    checkpoint = copy_durable_json_object(checkpoint, "checkpoint")
    source_bytes, tool_call_count, flags = pending_action_checkpoint_metrics(checkpoint)
    return (
        session_id,
        _dumps(checkpoint),
        pg_support.to_utc(updated_at),
        source_bytes,
        tool_call_count,
        flags,
        True,
    )


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if type(value) is not list:
        raise TypeError("Expected a JSON array.")
    return value


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


def _knowledge_entry_row_values(entry: KnowledgeEntry) -> tuple[object, ...]:
    return (
        entry.id,
        entry.namespace,
        entry.text,
        entry.kind,
        str(entry.visibility),
        str(entry.status),
        str(entry.created_by_type),
        entry.created_by,
        pg_support.to_utc(entry.created_at),
        pg_support.to_utc(entry.updated_at),
        entry.source_type,
        entry.source_uri,
        entry.source_id,
        entry.source_hash,
        entry.importance,
        entry.importance_source,
        entry.confidence,
        pg_support.to_utc_optional(entry.last_used_at),
        pg_support.to_utc_optional(entry.expires_at),
        entry.title,
        _dumps(entry.metadata),
    )


def _knowledge_entry_from_row(
    row: tuple[Any, ...],
    *,
    labels: dict[str, str],
    aspects: list[str],
    impact_targets: list[str],
) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=row[0],
        namespace=row[1],
        text=row[2],
        kind=row[3],
        visibility=KnowledgeVisibility(row[4]),
        status=KnowledgeStatus(row[5]),
        created_by_type=KnowledgeActorType(row[6]),
        created_by=row[7],
        created_at=pg_support.to_utc(row[8]),
        updated_at=pg_support.to_utc(row[9]),
        source_type=row[10],
        source_uri=row[11],
        source_id=row[12],
        source_hash=row[13],
        importance=row[14],
        importance_source=row[15],
        confidence=row[16],
        last_used_at=pg_support.to_utc_optional(row[17]),
        expires_at=pg_support.to_utc_optional(row[18]),
        title=row[19],
        labels=labels,
        aspects=aspects,
        impact_targets=impact_targets,
        metadata=_json_obj(row[20]),
    )


def _knowledge_chunk_row_values(chunk: KnowledgeChunk) -> tuple[object, ...]:
    return (
        chunk.id,
        chunk.entry_id,
        chunk.chunk_index,
        chunk.text,
        chunk.content_hash,
        chunk.source_uri,
        _dumps(chunk.metadata),
    )


def _knowledge_chunk_from_row(row: tuple[Any, ...]) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=row[0],
        entry_id=row[1],
        chunk_index=row[2],
        text=row[3],
        content_hash=row[4],
        source_uri=row[5],
        metadata=_json_obj(row[6]),
    )


def _copy_knowledge_entry_chunks(
    entry_id: str,
    chunks: list[KnowledgeChunk],
) -> list[KnowledgeChunk]:
    if type(chunks) is not list:
        raise ValueError("`chunks` must be a list.")
    if not chunks:
        raise ValueError("`chunks` cannot be empty.")
    copied_chunks = [copy_knowledge_chunk(chunk) for chunk in chunks]
    seen_ids: set[str] = set()
    seen_indexes: set[int] = set()
    for chunk in copied_chunks:
        if chunk.entry_id != entry_id:
            raise ValueError("Knowledge chunks must belong to the entry.")
        if chunk.id in seen_ids:
            raise ValueError("Knowledge chunk ids must be unique within an entry.")
        if chunk.chunk_index in seen_indexes:
            raise ValueError("Knowledge chunk indexes must be unique within an entry.")
        seen_ids.add(chunk.id)
        seen_indexes.add(chunk.chunk_index)
    return sorted(copied_chunks, key=lambda chunk: chunk.chunk_index)


def _postgres_knowledge_filter_sql(query: KnowledgeQuery) -> tuple[str, list[object]]:
    return _postgres_knowledge_metadata_filter_sql(
        namespace=query.namespace,
        labels=query.labels,
        kinds=query.kinds,
        statuses=query.statuses,
        visibilities=query.visibilities,
        aspects=query.aspects,
        impact_targets=query.impact_targets,
        source_type=query.source_type,
        source_id=query.source_id,
        include_expired=query.include_expired,
    )


def _knowledge_list_query_for_search(query: KnowledgeQuery) -> KnowledgeListQuery:
    """Project a search query's structural filters onto a list query.

    Used to bound the lazy embedding backfill to the entries a semantic search
    could actually return. Free-text terms are dropped (backfill is scope-based),
    but namespace/labels/kinds/statuses/visibility/aspect/impact/source/expiry
    carry over so the backfill never embeds chunks outside the query's reach.
    """
    return KnowledgeListQuery(
        namespace=query.namespace,
        labels=dict(query.labels),
        kinds=None if query.kinds is None else list(query.kinds),
        statuses=list(query.statuses),
        visibilities=None if query.visibilities is None else list(query.visibilities),
        aspects=list(query.aspects),
        impact_targets=list(query.impact_targets),
        source_type=query.source_type,
        source_id=query.source_id,
        include_expired=query.include_expired,
    )


def _postgres_knowledge_list_filter_sql(
    query: KnowledgeListQuery,
) -> tuple[str, list[object]]:
    return _postgres_knowledge_metadata_filter_sql(
        namespace=query.namespace,
        labels=query.labels,
        kinds=query.kinds,
        statuses=query.statuses,
        visibilities=query.visibilities,
        aspects=query.aspects,
        impact_targets=query.impact_targets,
        source_type=query.source_type,
        source_id=query.source_id,
        include_expired=query.include_expired,
    )


def _postgres_knowledge_metadata_filter_sql(
    *,
    namespace: str | None,
    labels: dict[str, str],
    kinds: list[str] | None,
    statuses: list[KnowledgeStatus],
    visibilities: list[KnowledgeVisibility] | None,
    aspects: list[str],
    impact_targets: list[str],
    source_type: str | None,
    source_id: str | None,
    include_expired: bool,
) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if namespace is not None:
        clauses.append("e.namespace = %s")
        params.append(namespace)
    for key, value in labels.items():
        clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_labels AS label
                WHERE label.entry_id = e.id
                  AND label.key = %s
                  AND label.value = %s
            )
            """
        )
        params.extend([key, value])
    if kinds is not None:
        if kinds:
            clauses.append("e.kind = ANY(%s)")
            params.append(kinds)
        else:
            clauses.append("FALSE")
    if statuses:
        clauses.append("e.status = ANY(%s)")
        params.append([str(status) for status in statuses])
    if visibilities is not None:
        clauses.append("e.visibility = ANY(%s)")
        params.append([str(visibility) for visibility in visibilities])
    if source_type is not None:
        clauses.append("e.source_type = %s")
        params.append(source_type)
    if source_id is not None:
        clauses.append("e.source_id = %s")
        params.append(source_id)
    if aspects:
        clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_aspects AS aspect
                WHERE aspect.entry_id = e.id
                  AND aspect.aspect = ANY(%s)
            )
            """
        )
        params.append(aspects)
    if impact_targets:
        clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_impact_targets AS target
                WHERE target.entry_id = e.id
                  AND target.impact_target = ANY(%s)
            )
            """
        )
        params.append(impact_targets)
    if not include_expired:
        clauses.append("(e.expires_at IS NULL OR e.expires_at > %s)")
        params.append(datetime.now(UTC))
    if not clauses:
        return "", params
    return " AND " + " AND ".join(clauses), params


def _postgres_knowledge_ts_query(query: KnowledgeQuery) -> tuple[str, list[str]]:
    any_terms = _dedupe_knowledge_search_tokens(
        [
            *_expand_knowledge_search_tokens(_tokenize_knowledge_search_text(query.text or "")),
            *(
                token
                for term in query.any_terms
                for group in _structured_knowledge_search_token_groups(term)
                for token in group
            ),
        ]
    )
    all_groups = _dedupe_knowledge_search_token_groups(
        [
            group
            for term in query.all_terms
            for group in _structured_knowledge_search_token_groups(term)
        ]
    )
    none_terms = _dedupe_knowledge_search_tokens(
        [
            token
            for term in query.none_terms
            for group in _structured_knowledge_search_token_groups(term)
            for token in group
        ]
    )
    phrase_queries = [_postgres_phrase_query(phrase) for phrase in query.phrases]
    phrase_terms = _dedupe_knowledge_search_tokens(
        [term for phrase in query.phrases for term in _tokenize_knowledge_search_text(phrase)]
    )
    positive_parts: list[str] = []
    if any_terms:
        positive_parts.append("(" + " | ".join(any_terms) + ")")
    if all_groups:
        positive_parts.append(" & ".join("(" + " | ".join(group) + ")" for group in all_groups))
    if phrase_queries:
        positive_parts.append("(" + " | ".join(phrase_queries) + ")")
    if not positive_parts:
        raise ValueError("Knowledge query requires positive search terms.")
    ts_query = " & ".join(positive_parts)
    for term in none_terms:
        ts_query += f" & !{term}"
    preview_terms = _dedupe_knowledge_search_tokens(
        [*any_terms, *(term for group in all_groups for term in group), *phrase_terms]
    )
    return ts_query, preview_terms


def _postgres_knowledge_search_filter_sql(query: KnowledgeQuery) -> tuple[str, list[object]]:
    any_terms = _dedupe_knowledge_search_tokens(
        [
            *_expand_knowledge_search_tokens(_tokenize_knowledge_search_text(query.text or "")),
            *(
                token
                for term in query.any_terms
                for group in _structured_knowledge_search_token_groups(term)
                for token in group
            ),
        ]
    )
    all_groups = _dedupe_knowledge_search_token_groups(
        [
            group
            for term in query.all_terms
            for group in _structured_knowledge_search_token_groups(term)
        ]
    )
    none_terms = _dedupe_knowledge_search_tokens(
        [
            token
            for term in query.none_terms
            for group in _structured_knowledge_search_token_groups(term)
            for token in group
        ]
    )
    phrase_queries = [_postgres_phrase_query(phrase) for phrase in query.phrases]
    clauses: list[str] = []
    params: list[object] = []
    if any_terms:
        clause, clause_params = _postgres_document_match_clause("(" + " | ".join(any_terms) + ")")
        clauses.append(clause)
        params.extend(clause_params)
    for group in all_groups:
        clause, clause_params = _postgres_document_match_clause("(" + " | ".join(group) + ")")
        clauses.append(clause)
        params.extend(clause_params)
    if phrase_queries:
        phrase_clauses: list[str] = []
        for phrase_query in phrase_queries:
            clause, clause_params = _postgres_document_match_clause(phrase_query)
            phrase_clauses.append(clause)
            params.extend(clause_params)
        clauses.append("(" + " OR ".join(phrase_clauses) + ")")
    for term in none_terms:
        clause, clause_params = _postgres_document_match_clause(term)
        clauses.append(f"NOT {clause}")
        params.extend(clause_params)
    if not any_terms and not all_groups and not phrase_queries:
        raise ValueError("Knowledge query requires positive search terms.")
    return cast("LiteralString", " AND ".join(clauses)), params


def _postgres_knowledge_none_filter_sql(query: KnowledgeQuery) -> tuple[str, list[object]]:
    none_terms = _dedupe_knowledge_search_tokens(
        [
            token
            for term in query.none_terms
            for group in _structured_knowledge_search_token_groups(term)
            for token in group
        ]
    )
    clauses: list[str] = []
    params: list[object] = []
    for term in none_terms:
        clause, clause_params = _postgres_document_match_clause(term)
        clauses.append(f"NOT {clause}")
        params.extend(clause_params)
    if not clauses:
        return "", params
    return cast("LiteralString", " AND " + " AND ".join(clauses)), params


def _postgres_document_match_clause(ts_query: str) -> tuple[LiteralString, list[object]]:
    return (
        cast(
            "LiteralString",
            """
            (
                to_tsvector('simple', COALESCE(e.title, '')) @@ to_tsquery('simple', %s)
                OR to_tsvector('simple', e.text) @@ to_tsquery('simple', %s)
                OR (
                    c.text <> e.text
                    AND to_tsvector('simple', c.text) @@ to_tsquery('simple', %s)
                )
            )
            """,
        ),
        [ts_query, ts_query, ts_query],
    )


def _postgres_vector_literal(vector: list[float]) -> str:
    values: list[str] = []
    for index, item in enumerate(vector):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError(f"embedding vector item {index} must be a number.")
        number = float(item)
        if number != number or number in {float("inf"), float("-inf")}:
            raise ValueError(f"embedding vector item {index} must be finite.")
        values.append(repr(number))
    if not values:
        raise ValueError("embedding vector cannot be empty.")
    return "[" + ",".join(values) + "]"


def _postgres_list_facet_sql(
    group_by: KnowledgeListGroup,
    where_sql: str,
    params: list[object],
    *,
    limit: int,
) -> tuple[LiteralString, list[object]]:
    limited_params = [*params, limit]
    if group_by is KnowledgeListGroup.KIND:
        return (
            cast(
                "LiteralString",
                f"""
                SELECT NULL AS key, e.kind AS value, COUNT(*) AS count
                FROM cayu_knowledge_entries AS e
                WHERE TRUE
                {where_sql}
                GROUP BY e.kind
                ORDER BY count DESC, value ASC
                LIMIT %s
                """,
            ),
            limited_params,
        )
    if group_by is KnowledgeListGroup.NAMESPACE:
        return (
            cast(
                "LiteralString",
                f"""
                SELECT NULL AS key, e.namespace AS value, COUNT(*) AS count
                FROM cayu_knowledge_entries AS e
                WHERE TRUE
                {where_sql}
                GROUP BY e.namespace
                ORDER BY count DESC, value ASC
                LIMIT %s
                """,
            ),
            limited_params,
        )
    if group_by is KnowledgeListGroup.LABEL:
        return (
            cast(
                "LiteralString",
                f"""
                SELECT label.key AS key, label.value AS value, COUNT(DISTINCT e.id) AS count
                FROM cayu_knowledge_entries AS e
                JOIN cayu_knowledge_labels AS label ON label.entry_id = e.id
                WHERE TRUE
                {where_sql}
                GROUP BY label.key, label.value
                ORDER BY count DESC, key ASC, value ASC
                LIMIT %s
                """,
            ),
            limited_params,
        )
    if group_by is KnowledgeListGroup.ASPECT:
        return (
            cast(
                "LiteralString",
                f"""
                SELECT NULL AS key, aspect.aspect AS value, COUNT(DISTINCT e.id) AS count
                FROM cayu_knowledge_entries AS e
                JOIN cayu_knowledge_aspects AS aspect ON aspect.entry_id = e.id
                WHERE TRUE
                {where_sql}
                GROUP BY aspect.aspect
                ORDER BY count DESC, value ASC
                LIMIT %s
                """,
            ),
            limited_params,
        )
    if group_by is KnowledgeListGroup.IMPACT_TARGET:
        return (
            cast(
                "LiteralString",
                f"""
                SELECT NULL AS key, target.impact_target AS value, COUNT(DISTINCT e.id) AS count
                FROM cayu_knowledge_entries AS e
                JOIN cayu_knowledge_impact_targets AS target ON target.entry_id = e.id
                WHERE TRUE
                {where_sql}
                GROUP BY target.impact_target
                ORDER BY count DESC, value ASC
                LIMIT %s
                """,
            ),
            limited_params,
        )
    if group_by is KnowledgeListGroup.VISIBILITY:
        return (
            cast(
                "LiteralString",
                f"""
                SELECT NULL AS key, e.visibility AS value, COUNT(*) AS count
                FROM cayu_knowledge_entries AS e
                WHERE TRUE
                {where_sql}
                GROUP BY e.visibility
                ORDER BY count DESC, value ASC
                LIMIT %s
                """,
            ),
            limited_params,
        )
    return (
        cast(
            "LiteralString",
            f"""
            SELECT NULL AS key, e.source_type AS value, COUNT(*) AS count
            FROM cayu_knowledge_entries AS e
            WHERE e.source_type IS NOT NULL
            {where_sql}
            GROUP BY e.source_type
            ORDER BY count DESC, value ASC
            LIMIT %s
            """,
        ),
        limited_params,
    )


def _structured_knowledge_search_token_groups(value: str) -> list[list[str]]:
    tokens = _tokenize_knowledge_search_text(value)
    if not tokens:
        raise ValueError("Structured knowledge search terms must contain at least one token.")
    return [_knowledge_search_token_variants(token) for token in tokens]


def _postgres_phrase_query(value: str) -> str:
    tokens = _tokenize_knowledge_search_text(value)
    if not tokens:
        raise ValueError("Structured knowledge search phrases must contain at least one token.")
    return " <-> ".join(tokens)


def _dedupe_knowledge_search_tokens(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _dedupe_knowledge_search_token_groups(groups: list[list[str]]) -> list[list[str]]:
    result: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for group in groups:
        key = tuple(group)
        if key not in seen:
            result.append(group)
            seen.add(key)
    return result


def _postgres_entry_search_vector_sql() -> LiteralString:
    return cast(
        "LiteralString",
        """
        setweight(to_tsvector('simple', COALESCE(e.title, '')), 'A')
        || setweight(to_tsvector('simple', e.text), 'B')
        || to_tsvector(
               'simple',
               CASE WHEN c.text = e.text THEN '' ELSE c.text END
           )
        """,
    )


def _center_knowledge_chunk_window(
    chunks: list[KnowledgeChunk],
    *,
    chunk_index: int,
    max_chunks: int,
) -> list[KnowledgeChunk]:
    if len(chunks) <= max_chunks:
        return chunks
    closest = sorted(
        chunks, key=lambda chunk: (abs(chunk.chunk_index - chunk_index), chunk.chunk_index)
    )
    return sorted(closest[:max_chunks], key=lambda chunk: chunk.chunk_index)


def _bounded_knowledge_chunks(
    chunks: list[KnowledgeChunk],
    *,
    start_index: int,
    end_index: int | None,
    max_chunks: int,
    max_bytes: int,
) -> list[KnowledgeChunk]:
    selected: list[KnowledgeChunk] = []
    remaining = max_bytes
    for chunk in chunks:
        if chunk.chunk_index < start_index:
            continue
        if end_index is not None and chunk.chunk_index > end_index:
            continue
        if len(selected) >= max_chunks or remaining <= 0:
            break
        copied = copy_knowledge_chunk(chunk)
        chunk_bytes = len(copied.text.encode("utf-8"))
        if chunk_bytes > remaining:
            truncated_text = _truncate_knowledge_text_to_bytes(copied.text, remaining)
            if not truncated_text:
                break
            selected.append(
                KnowledgeChunk(
                    id=copied.id,
                    entry_id=copied.entry_id,
                    text=truncated_text,
                    chunk_index=copied.chunk_index,
                    content_hash=None,
                    source_uri=copied.source_uri,
                    metadata=copied.metadata,
                )
            )
            break
        selected.append(copied)
        remaining -= chunk_bytes
    return selected


def _knowledge_preview_for_match(
    entry: KnowledgeEntry,
    chunk: KnowledgeChunk,
    terms: list[str],
) -> tuple[str, str]:
    if entry.title is not None:
        title_terms = set(_tokenize_knowledge_search_text(entry.title))
        if any(term in title_terms for term in terms):
            return "title match", entry.title
    entry_terms = set(_tokenize_knowledge_search_text(entry.text))
    if any(term in entry_terms for term in terms):
        return "entry text match", entry.text
    return "chunk text match", chunk.text


def _default_chunk_for_entry(entry: KnowledgeEntry) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=f"{entry.id}:0",
        entry_id=entry.id,
        text=entry.text,
        chunk_index=0,
        content_hash=sha256(entry.text.encode("utf-8")).hexdigest(),
        source_uri=entry.source_uri,
    )


def _knowledge_has_only_default_chunk(
    entry: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
) -> bool:
    if len(chunks) != 1:
        return False
    default_chunk = _default_chunk_for_entry(entry)
    chunk = chunks[0]
    return (
        chunk.id == default_chunk.id
        and chunk.entry_id == default_chunk.entry_id
        and chunk.text == default_chunk.text
        and chunk.chunk_index == default_chunk.chunk_index
        and chunk.content_hash == default_chunk.content_hash
        and chunk.source_uri == default_chunk.source_uri
        and chunk.metadata == default_chunk.metadata
    )


def _tokenize_knowledge_search_text(text: str) -> list[str]:
    return _KNOWLEDGE_SEARCH_TOKEN_RE.findall(text.casefold())


def _expand_knowledge_search_tokens(tokens: list[str]) -> list[str]:
    return [variant for token in tokens for variant in _knowledge_search_token_variants(token)]


def _knowledge_search_token_variants(token: str) -> list[str]:
    variants = [token]
    if len(token) < 3 or not token.isalpha():
        return variants
    if token.endswith("ies") and len(token) > 4:
        variants.append(token[:-3] + "y")
    elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
        variants.append(token[:-1])
    else:
        variants.append(_plural_knowledge_search_token(token))
    return _dedupe_knowledge_search_tokens(variants)


def _plural_knowledge_search_token(token: str) -> str:
    if token.endswith("y") and len(token) > 1 and token[-2] not in "aeiou":
        return token[:-1] + "ies"
    return token + "s"


def _truncate_knowledge_text_to_bytes(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _validate_knowledge_nonnegative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or type(value) is not int:
        raise ValueError(f"`{field_name}` must be an integer.")
    if value < 0:
        raise ValueError(f"`{field_name}` must be greater than or equal to 0.")


def _validate_knowledge_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or type(value) is not int:
        raise ValueError(f"`{field_name}` must be an integer.")
    if value < 1:
        raise ValueError(f"`{field_name}` must be greater than or equal to 1.")


def _validate_task_positive_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1.")
    return value


def _event_record_from_row(row: tuple[Any, Any] | None) -> EventRecord | None:
    """Build an EventRecord from a ``(sequence, event)`` row, or None for a missing row."""
    if row is None:
        return None
    return EventRecord(sequence=row[0], event=Event(**_json_obj(row[1])))


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


def _event_watcher_state_from_row(row: tuple[Any, ...]) -> EventWatcherState:
    return EventWatcherState(
        watcher_name=row[0],
        cursor_sequence=row[1],
        pending_event_id=row[2],
        pending_event_sequence=row[3],
        pending_attempt=row[4],
        pending_claim_id=row[5],
        delivery_status=None if row[6] is None else EventWatcherDeliveryStatus(row[6]),
        lease_expires_at=pg_support.to_utc_optional(row[7]),
        last_error=row[8],
        dead_lettered_count=row[9],
        updated_at=pg_support.to_utc(row[10]),
    )


def _event_watcher_dead_letter_from_row(row: tuple[Any, ...]) -> EventWatcherDeadLetter:
    return EventWatcherDeadLetter(
        watcher_name=row[0],
        event_id=row[1],
        event_sequence=row[2],
        attempts=row[3],
        error=row[4],
        dead_lettered_at=pg_support.to_utc(row[5]),
        resolved_at=pg_support.to_utc_optional(row[6]),
    )


def _validate_dead_letter_limit(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("limit must be an integer greater than or equal to 1.")
    return value


def _validate_event_sequence(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("event_sequence must be an integer greater than or equal to 1.")
    return value


def _event_watcher_delivery_from_claim(
    claim: EventWatcherClaim,
    *,
    status: EventWatcherDeliveryStatus,
    cursor_sequence: int,
    error: str | None = None,
) -> EventWatcherDelivery:
    return EventWatcherDelivery(
        watcher_name=claim.watcher_name,
        event_id=claim.event_id,
        event_sequence=claim.event_sequence,
        status=status,
        attempt=claim.attempt,
        cursor_sequence=cursor_sequence,
        error=error,
    )


def _validate_positive_float(value: float, field_name: str) -> float:
    if type(value) not in {int, float} or value <= 0:
        raise ValueError(f"{field_name} must be greater than 0.")
    return float(value)


def _clean_error(value: str) -> str:
    return require_durable_nonblank(value, "error")


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
    "PostgresEmbeddingBackfillResult",
    "PostgresEmbeddingKnowledgeStore",
    "PostgresEventWatcherStore",
    "PostgresKnowledgeStore",
    "PostgresSessionStore",
    "PostgresTaskStore",
]
