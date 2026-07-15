from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from typing import Any, LiteralString, cast

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
    JsonUtf8SizeCounter,
    copy_json_object,
    copy_json_value,
    copy_label_map,
    require_clean_nonblank,
    require_nonblank,
)
from cayu.core.events import Event, EventType
from cayu.core.messages import Message
from cayu.embeddings import TextEmbeddingProvider, TextEmbeddingRequest
from cayu.runtime.budgets import (
    DEFAULT_RESERVATION_TTL_SECONDS,
    BudgetLedger,
    BudgetLimit,
    BudgetReconciliation,
    BudgetReservationRecord,
    BudgetReservationResult,
    _budget_reservation_amount,
    _clock_or_utc_now,
    _expired_reservation_reason,
    _is_expired_reservation_reason,
    _reconciled_record,
    _reconciliation_from_record,
    _reservation_is_expired,
    _reservation_result,
    _validate_amount,
    _validate_reservation_ttl,
)
from cayu.runtime.event_watchers import (
    EventWatcherClaim,
    EventWatcherDeadLetter,
    EventWatcherDelivery,
    EventWatcherDeliveryStatus,
    EventWatcherState,
    EventWatcherStore,
)
from cayu.runtime.sessions import (
    DELETE_BLOCKED_SESSION_STATUSES,
    MAX_PENDING_ACTION_RESULT_BYTES,
    MAX_PENDING_ACTION_TOOL_CALLS,
    CheckpointTransform,
    EventQuery,
    EventRecord,
    EventSummary,
    PendingActionIssue,
    PendingActionKind,
    PendingActionListResult,
    PendingActionQuery,
    PendingActionSession,
    RunRequest,
    Session,
    SessionIdentity,
    SessionListResult,
    SessionOrder,
    SessionOutcome,
    SessionQuery,
    SessionRunFenced,
    SessionStateSnapshot,
    SessionStatus,
    SessionStatusConflict,
    SessionStore,
    TranscriptPage,
    TranscriptQuery,
    TranscriptRecord,
    _activate_session_run_fence,
    _assert_session_run_epoch,
    _copy_session_event_batch,
    _current_session_run_epoch,
    _deactivate_session_run_fence,
    _prepare_session_fork_request,
    _project_interruption_cascade_marker_fields,
    _validate_session_fork_source,
    _validate_status_set,
    copy_event_query,
    copy_run_request,
    copy_session_identity,
    copy_session_query,
    copy_transcript_messages,
    copy_transcript_query,
    decode_session_cursor,
    encode_session_cursor,
    enforce_pending_action_result_size,
    filter_transcript_records,
    session_next_cursor,
    session_outcome,
)
from cayu.runtime.tasks import (
    Task,
    TaskCreate,
    TaskOrder,
    TaskQuery,
    TaskStatus,
    TaskStore,
    _copy_optional_status_payload,
    _copy_optional_status_reason,
    _ensure_can_hold_task,
    _ensure_can_resume_task,
    _ensure_can_transition,
    _ensure_claim_query_supported,
    _ensure_not_terminal,
    _running_task_from_create,
    _task_from_create,
    copy_task_create,
    copy_task_query,
)
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
_POSTGRES_MIN_REQUIRED_REVISION = 17
_EVENT_QUERY_SESSION_IDS_BATCH_SIZE = 500
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
}

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
            WHEN jsonb_typeof(target.state #> '{{pending_tool_round,tool_calls}}') = 'array'
              AND jsonb_array_length(target.state #> '{{pending_tool_round,tool_calls}}')
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
        pending_action_tool_call_count = CASE
            WHEN jsonb_typeof(target.state #> '{{pending_tool_round,tool_calls}}') = 'array'
            THEN jsonb_array_length(target.state #> '{{pending_tool_round,tool_calls}}')
            ELSE 0
        END,
        pending_action_metrics_ready = TRUE
    FROM batch
    WHERE target.session_id = batch.session_id
    RETURNING target.session_id
"""

_REVISION_17_EVENT_BACKFILL_SMALL_EVENT_BYTES = 1024 * 1024


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
                            'approval', jsonb_strip_nulls(jsonb_build_object(
                                'approval_id', payload #> '{{approval,approval_id}}',
                                'reason', payload #> '{{approval,reason}}',
                                'tool_name', payload #> '{{approval,tool_name}}'
                            ))
                        ))
                    WHEN event_type = 'session.awaiting_user_input' THEN
                        jsonb_strip_nulls(jsonb_build_object(
                            'input_id', payload -> 'input_id',
                            'tool_call_id', payload -> 'tool_call_id',
                            'question', payload -> 'question',
                            'options', payload -> 'options'
                        ))
                    WHEN event_type = 'session.interrupted' THEN
                        jsonb_strip_nulls(jsonb_build_object(
                            'interruption_type', payload -> 'interruption_type',
                            'manual_recovery_required', payload -> 'manual_recovery_required',
                            'approval_id', payload -> 'approval_id',
                            'tool_call_id', payload -> 'tool_call_id',
                            'tool_round_id', payload -> 'tool_round_id',
                            'error', payload -> 'error',
                            'message', payload -> 'message',
                            'tool_name', payload -> 'tool_name',
                            'approval', jsonb_strip_nulls(jsonb_build_object(
                                'approval_id', payload #> '{{approval,approval_id}}',
                                'reason', payload #> '{{approval,reason}}',
                                'tool_name', payload #> '{{approval,tool_name}}'
                            )),
                            'user_input', jsonb_strip_nulls(jsonb_build_object(
                                'input_id', payload #> '{{user_input,input_id}}',
                                'tool_call_id', payload #> '{{user_input,tool_call_id}}',
                                'question', payload #> '{{user_input,question}}',
                                'options', payload #> '{{user_input,options}}'
                            ))
                        ))
                    WHEN event_type IN (
                        'tool.call.started',
                        'tool.call.completed',
                        'tool.call.failed',
                        'tool.call.blocked',
                        'tool.call.approval_denied'
                    ) THEN jsonb_strip_nulls(jsonb_build_object(
                        'tool_call_id', payload -> 'tool_call_id',
                        'tool_round_id', payload -> 'tool_round_id',
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
            ELSE NULL
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


async def _disable_prepared_statements(conn: Any) -> None:
    """Pool ``configure`` hook: disable psycopg3 server-side prepared statements.

    Required when the store's own pool runs behind a transaction-pooling pgbouncer
    (e.g. Fly Managed Postgres), where prepared statements raise
    "prepared statement ... already exists". Harmless on a direct connection.
    """
    conn.prepare_threshold = None


class _PostgresStoreBase:
    """Shared async connection-pool management for Postgres-backed stores.

    The pool is created eagerly (closed) and opened lazily on first use so that
    it is bound to the event loop that actually drives the store. This mirrors
    the way the SQLite store opens its connection in ``__init__`` while keeping
    psycopg's async pool happy about running inside a live loop.
    """

    def __init__(
        self,
        conninfo: str | None = None,
        *,
        pool: AsyncConnectionPool | None = None,
        min_size: int = 1,
        max_size: int = 8,
        schema_mode: schema.SchemaMode = schema.SchemaMode.VALIDATE,
    ) -> None:
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")
        self._schema_mode = schema_mode
        if pool is not None:
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
                configure=_disable_prepared_statements,
            )
            self._owns_pool = True
        self._open_lock = asyncio.Lock()
        self._opened = False
        self._schema_ready = False

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

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT pg_advisory_xact_lock(%s)", (_SCHEMA_ADVISORY_LOCK_KEY,))
                if self._schema_mode is not schema.SchemaMode.VALIDATE:
                    await cur.execute(pg_support.MIGRATIONS_TABLE_DDL)
                state = await self._read_schema_state(cur)
                if self._schema_mode is schema.SchemaMode.VALIDATE:
                    schema.validate(state)
                    await self._validate_postgres_schema(cur, state)
                elif self._schema_mode is schema.SchemaMode.CREATE:
                    if state.revision == schema.UNINITIALIZED:
                        await self._apply_pending(cur, state)
                    else:
                        schema.validate(state)
                        await self._validate_postgres_schema(cur, state)
            await conn.commit()

    async def _migrate_schema(self) -> None:
        while True:
            concurrent_revision: schema.Revision | None = None
            concurrent_indexes: tuple[_ConcurrentIndexMigration, ...] = ()
            recorded_indexes: tuple[_ConcurrentIndexMigration, ...] = ()
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT pg_advisory_xact_lock(%s)",
                        (_SCHEMA_ADVISORY_LOCK_KEY,),
                    )
                    await cur.execute(pg_support.MIGRATIONS_TABLE_DDL)
                    state = await self._read_schema_state(cur)
                    current = state.revision
                    if current == schema.UNINITIALIZED:
                        await self._apply_baseline(cur)
                        current = schema.BASELINE_REVISION
                    pending = schema.pending(current)
                    if not pending:
                        current_state = await self._read_schema_state(cur)
                        schema.validate(current_state)
                        self._validate_postgres_revision(current_state)
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
                await cur.execute(
                    "SELECT pg_advisory_xact_lock(%s)",
                    (_SCHEMA_ADVISORY_LOCK_KEY,),
                )
                state = await self._read_schema_state(cur)
                if state.revision < concurrent_revision.revision:
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
        if state.revision < _POSTGRES_MIN_REQUIRED_REVISION:
            raise schema.SchemaTooOld(
                f"Postgres schema is at revision {state.revision}; this build requires "
                f">= {_POSTGRES_MIN_REQUIRED_REVISION}. Run `cayu storage migrate` before "
                "starting."
            )

    async def _validate_postgres_schema(self, cur: Any, state: schema.SchemaState) -> None:
        self._validate_postgres_revision(state)
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

    async def _read_schema_state(self, cur: Any) -> schema.SchemaState:
        return await read_schema_state(cur)

    async def _apply_baseline(self, cur: Any) -> None:
        for statement in pg_support.SCHEMA_STATEMENTS:
            await cur.execute(statement)
        await self._record_revision(cur, schema.revision(schema.BASELINE_REVISION))

    async def _apply_pending(self, cur: Any, state: schema.SchemaState) -> None:
        current = state.revision
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
                    await asyncio.sleep(0.25)

            while True:
                async with conn.cursor() as cur:
                    existing = await self._concurrent_index_state(
                        cur,
                        index,
                    )
                    if existing == (True, False):
                        return
                    if existing is not None and existing[1]:
                        await asyncio.sleep(0.25)
                        continue
                    if existing is not None:
                        await cur.execute(index.drop_statement)
                    try:
                        await cur.execute(index.create_statement)
                    except (DeadlockDetected, DuplicateTable, UniqueViolation):
                        continue
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
                pg_get_expr(
                    index_definition.indpred,
                    index_definition.indrelid,
                    FALSE
                ),
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
        predicate = _normalize_postgres_index_expression(row[5])
        expected_predicate = _normalize_postgres_index_expression(index.predicate_definition)
        expected_definition = (
            bool(row[0])
            and bool(row[2])
            and bool(row[3])
            and key_definitions == expected_keys
            and predicate == expected_predicate
        )
        if not expected_definition:
            columns = ", ".join(index.key_definitions)
            raise RuntimeError(
                f"Postgres schema object {index.index_name!r} conflicts with the required "
                f"B-tree index on {index.table_name}({columns}). Remove or rename the "
                "conflicting object, then rerun `cayu storage migrate`."
            )
        return bool(row[1]), bool(row[6])

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
        if type(record) is not EventRecord:
            raise TypeError("record must be an EventRecord.")
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
        if type(claim) is not EventWatcherClaim:
            raise TypeError("claim must be an EventWatcherClaim.")
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
        if type(claim) is not EventWatcherClaim:
            raise TypeError("claim must be an EventWatcherClaim.")
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


def _budget_advisory_lock_key(limit: BudgetLimit) -> int:
    """Stable 63-bit advisory-lock key for one budget scope/key/window/currency."""
    material = "|".join(
        (
            "cayu_budget_reservations",
            limit.scope,
            limit.key or "",
            limit.window.storage_key,
            limit.currency.upper(),
        )
    )
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

    async def reserve(
        self,
        *,
        limit: BudgetLimit,
        session_id: str,
        agent_name: str,
        provider_name: str,
        model: str,
    ) -> BudgetReservationResult:
        if type(limit) is not BudgetLimit:
            raise TypeError("limit must be a BudgetLimit.")
        session_id = require_clean_nonblank(session_id, "session_id")
        agent_name = require_clean_nonblank(agent_name, "agent_name")
        provider_name = require_clean_nonblank(provider_name, "provider_name")
        model = require_clean_nonblank(model, "model")
        await self._ensure_ready()
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT pg_advisory_xact_lock(%s)",
                        (_budget_advisory_lock_key(limit),),
                    )
                    now = self._clock()
                    requested = _budget_reservation_amount(
                        limit=limit,
                        provider_name=provider_name,
                        model=model,
                        effective_at=now,
                    )
                    await self._reap_expired(cur, now, limit=limit)
                    current = await self._used_amount(cur, limit, now=now)
                    projected = current + requested
                    if projected > limit.max_estimated_cost:
                        await conn.rollback()
                        return _reservation_result(
                            limit=limit,
                            accepted=False,
                            requested=requested,
                            actual=projected,
                            message=(
                                "Budget reservation failed: "
                                f"{projected} > {limit.max_estimated_cost} {limit.currency}."
                            ),
                        )
                    record = BudgetReservationRecord(
                        scope=limit.scope,
                        key=limit.key,
                        window=limit.window,
                        currency=limit.currency,
                        session_id=session_id,
                        agent_name=agent_name,
                        provider_name=provider_name,
                        model=model,
                        reserved_amount=requested,
                        created_at=now,
                        updated_at=now,
                    )
                    await self._insert_record(cur, record)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return _reservation_result(
            limit=limit,
            accepted=True,
            requested=requested,
            actual=projected,
            message=(f"Budget reserved: {requested} {limit.currency} for {provider_name}/{model}."),
            record=record,
        )

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
        reason: str | None = None,
        occurred_at: datetime | None = None,
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
                    )
                    await self._update_record(cur, reconciled)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return _reconciliation_from_record(reconciled)

    async def release(
        self,
        *,
        reservation_id: str,
        reason: str,
    ) -> BudgetReconciliation:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        reason = require_clean_nonblank(reason, "reason")
        released_at = self._clock()
        await self._ensure_ready()
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    record = await self._releasable_record_for_update(cur, reservation_id)
                    if record.status == "released":
                        await conn.commit()
                        return _reconciliation_from_record(record)
                    released = record.model_copy(
                        update={
                            "status": "released",
                            "reason": reason,
                            "updated_at": released_at,
                        },
                        deep=True,
                    )
                    await self._update_record(cur, released)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return _reconciliation_from_record(released)

    async def _reap_expired(self, cur: Any, now: datetime, *, limit: BudgetLimit) -> None:
        if self._reservation_ttl_seconds is None:
            return
        cutoff = now - timedelta(seconds=self._reservation_ttl_seconds)
        # Keep the matching dimensions and inclusive expiry boundary aligned with
        # _reservation_matches_limit() and _reservation_is_expired(). ``IS NOT
        # DISTINCT FROM`` keeps the nullable budget key comparison null-safe.
        await cur.execute(
            """
            UPDATE cayu_budget_reservations
            SET status = 'released',
                reason = %s,
                updated_at = %s
            WHERE status = 'active'
              AND updated_at <= %s
              AND scope = %s
              AND budget_key IS NOT DISTINCT FROM %s
              AND budget_window = %s
              AND currency = %s
            """,
            (
                _expired_reservation_reason(self._reservation_ttl_seconds),
                pg_support.to_utc(now),
                pg_support.to_utc(cutoff),
                limit.scope,
                limit.key,
                limit.window.storage_key,
                limit.currency.upper(),
            ),
        )

    async def _used_amount(self, cur: Any, limit: BudgetLimit, *, now: datetime) -> Decimal:
        since, until = limit.window.bounds(now=now)
        reconciled_bound_sql = ""
        params: list[object] = [
            limit.scope,
            limit.key,
            limit.window.storage_key,
            limit.currency.upper(),
        ]
        if since is not None:
            reconciled_bound_sql += " AND updated_at >= %s"
            params.append(pg_support.to_utc(since))
        if until is not None:
            reconciled_bound_sql += " AND updated_at < %s"
            params.append(pg_support.to_utc(until))
        await cur.execute(
            f"""
            SELECT reserved_amount, actual_amount, status
            FROM cayu_budget_reservations
            WHERE scope = %s
              AND budget_key IS NOT DISTINCT FROM %s
              AND budget_window = %s
              AND currency = %s
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
                scope,
                budget_key,
                budget_window,
                currency,
                session_id,
                agent_name,
                provider_name,
                model,
                reserved_amount,
                actual_amount,
                status,
                reason,
                created_at,
                updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                record.reservation_id,
                record.scope,
                record.key,
                record.window.storage_key,
                record.currency,
                record.session_id,
                record.agent_name,
                record.provider_name,
                record.model,
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
                status = %s,
                reason = %s,
                updated_at = %s
            WHERE reservation_id = %s
            """,
            (
                record.actual_amount,
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
            SELECT reservation_id, scope, budget_key, budget_window, currency, session_id,
                   agent_name, provider_name, model, reserved_amount, actual_amount,
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
        return BudgetReservationRecord(
            reservation_id=row[0],
            scope=row[1],
            key=row[2],
            window=row[3],
            currency=row[4],
            session_id=row[5],
            agent_name=row[6],
            provider_name=row[7],
            model=row[8],
            reserved_amount=row[9],
            actual_amount=row[10],
            status=row[11],
            reason=row[12],
            created_at=pg_support.to_utc(row[13]),
            updated_at=pg_support.to_utc(row[14]),
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
        if record.status == "active":
            return record
        if record.status == "released" and _is_expired_reservation_reason(record.reason):
            return record
        raise ValueError(f"Budget reservation is not active: {reservation_id}")

    async def _reconcilable_record_for_update(
        self,
        cur: Any,
        reservation_id: str,
    ) -> BudgetReservationRecord:
        record = await self._load_record_for_update(cur, reservation_id)
        if record.status == "active":
            return record
        if record.status == "released" and _is_expired_reservation_reason(record.reason):
            # Reaped by the TTL while still in flight (a long step or a wall-clock jump).
            # Reconcile it anyway so the actual spend is recorded rather than crashing the
            # billed run and silently undercounting the shared budget window.
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

    async def create(
        self,
        request: RunRequest,
        *,
        identity: SessionIdentity,
    ) -> Session:
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
            metadata=copy_json_value(request.metadata, "metadata"),
            labels=request.labels,
        )
        if session.parent_session_id == session.id:
            raise ValueError("Session cannot be its own parent.")
        async with self._pool.connection() as conn:
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
        source_session_id, fork, allowed_statuses, transcript_cursor = (
            _prepare_session_fork_request(
                source_session_id=source_session_id,
                fork=fork,
                source_statuses=source_statuses,
                transcript_cursor=transcript_cursor,
            )
        )

        await self._ensure_ready()
        async with self._pool.connection() as conn:
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
                        """
                        SELECT message
                        FROM cayu_transcript_messages
                        WHERE session_id = %s
                        ORDER BY sequence ASC
                        """,
                        (source_session_id,),
                    )
                    transcript_rows = await cur.fetchall()
                    if transcript_cursor is None:
                        copied_messages = [Message(**_json_obj(row[0])) for row in transcript_rows]
                    else:
                        if transcript_cursor > len(transcript_rows):
                            raise ValueError(
                                "transcript_cursor is greater than source transcript length."
                            )
                        copied_messages = [
                            Message(**_json_obj(row[0]))
                            for row in transcript_rows[:transcript_cursor]
                        ]

                    copied_checkpoint = None
                    if checkpoint_transform is not None:
                        copied_checkpoint = checkpoint_transform(
                            source_session,
                            await self._load_checkpoint(cur, source_session_id),
                        )
                        if copied_checkpoint is not None:
                            copied_checkpoint = copy_json_value(copied_checkpoint, "checkpoint")

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
                            INSERT INTO cayu_transcript_messages (session_id, message)
                            VALUES (%s, %s)
                            """,
                            [
                                (fork.id, _dumps(message.model_dump(mode="json")))
                                for message in copied_messages
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
        async with self._pool.connection() as conn, conn.cursor() as cur:
            return await self._load(cur, session_id)

    async def load_state(self, session_id: str) -> SessionStateSnapshot | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
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
        async with self._pool.connection() as conn:
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
        blocked_statuses = [str(status) for status in DELETE_BLOCKED_SESSION_STATUSES]
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    DELETE FROM cayu_sessions
                    WHERE id = %s
                      AND status <> ALL(%s)
                    RETURNING id
                    """,
                    (session_id, blocked_statuses),
                )
                deleted = await cur.fetchone()
                if deleted is None:
                    await cur.execute(
                        "SELECT status FROM cayu_sessions WHERE id = %s",
                        (session_id,),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        return  # idempotent: deleting a missing session is a no-op
                    status = SessionStatus(row[0])
                    if status in DELETE_BLOCKED_SESSION_STATUSES:
                        raise ValueError(
                            f"Cannot delete a session while it is {status}; "
                            f"interrupt it first: {session_id}"
                        )
                    await cur.execute(
                        """
                        DELETE FROM cayu_sessions
                        WHERE id = %s
                          AND status <> ALL(%s)
                        RETURNING id
                        """,
                        (session_id, blocked_statuses),
                    )
                    deleted = await cur.fetchone()
                    if deleted is None:
                        raise ValueError(
                            f"Cannot delete a session while its status is changing; "
                            f"retry later: {session_id}"
                        )
                # ON DELETE CASCADE removes events/labels/checkpoint/transcript; the
                # self-FK is ON DELETE SET NULL so children keep loading with no parent.
            await conn.commit()

    async def update_labels(self, session_id: str, labels: dict[str, str]) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        new_labels = copy_label_map(labels, "labels", allow_reserved=False)
        await self._ensure_ready()
        updated_at = datetime.now(UTC)
        expected_run_epoch = _current_session_run_epoch(session_id)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                if expected_run_epoch is None:
                    await cur.execute(
                        "UPDATE cayu_sessions SET updated_at = %s, last_activity_at = %s "
                        "WHERE id = %s",
                        (updated_at, updated_at, session_id),
                    )
                else:
                    await cur.execute(
                        "UPDATE cayu_sessions SET updated_at = %s, last_activity_at = %s "
                        "WHERE id = %s AND run_epoch = %s",
                        (updated_at, updated_at, session_id, expected_run_epoch),
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
        new_metadata = copy_json_value(metadata, "metadata")
        if type(new_metadata) is not dict:
            raise TypeError("Session metadata must be an object.")
        await self._ensure_ready()
        updated_at = datetime.now(UTC)
        expected_run_epoch = _current_session_run_epoch(session_id)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                if expected_run_epoch is None:
                    await cur.execute(
                        "UPDATE cayu_sessions SET metadata = %s, updated_at = %s, "
                        "last_activity_at = %s WHERE id = %s",
                        (_dumps(new_metadata), updated_at, updated_at, session_id),
                    )
                else:
                    await cur.execute(
                        "UPDATE cayu_sessions SET metadata = %s, updated_at = %s, "
                        "last_activity_at = %s WHERE id = %s AND run_epoch = %s",
                        (
                            _dumps(new_metadata),
                            updated_at,
                            updated_at,
                            session_id,
                            expected_run_epoch,
                        ),
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
        async with self._pool.connection() as conn:
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
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        await self._ensure_ready()
        updated_at = datetime.now(UTC)
        async with self._pool.connection() as conn:
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
                        transformed_checkpoint = copy_json_value(
                            transformed_checkpoint, "checkpoint"
                        )

                    await cur.execute(
                        """
                        UPDATE cayu_sessions
                        SET status = %s, updated_at = %s, last_activity_at = %s,
                            run_epoch = run_epoch + %s
                        WHERE id = %s
                        """,
                        (
                            str(to_status),
                            updated_at,
                            updated_at,
                            1 if to_status == SessionStatus.RUNNING else 0,
                            session_id,
                        ),
                    )
                    if transformed_checkpoint is not None:
                        await self._upsert_checkpoint(
                            cur, session_id, transformed_checkpoint, updated_at
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
        async with self._pool.connection() as conn:
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

    async def release_run_fence(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        expected_run_epoch = _current_session_run_epoch(session_id)
        if expected_run_epoch is None:
            return
        await self._ensure_ready()
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "UPDATE cayu_sessions SET run_epoch = run_epoch + 1 "
                        "WHERE id = %s AND run_epoch = %s",
                        (session_id, expected_run_epoch),
                    )
                await conn.commit()
        finally:
            _deactivate_session_run_fence(session_id)

    async def append_event(self, session_id: str, event: Event) -> None:
        await self.append_events(session_id, [event])

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        session_id, copied_events = _copy_session_event_batch(session_id, events)

        await self._ensure_ready()
        expected_run_epoch = _current_session_run_epoch(session_id)
        async with self._pool.connection() as conn:
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
                            session_id, session_order, event_id, event_type, timestamp,
                            agent_name, environment_name, workflow_name, tool_name,
                            payload, event, pending_action_lookup_key,
                            pending_action_projection, pending_action_projection_bytes
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s
                        )
                        """,
                        rows,
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
                raise
            except Exception:
                await conn.rollback()
                raise

    async def load_events(self, session_id: str) -> list[Event]:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
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

    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        query = copy_event_query(query)
        if len(query.session_ids) > _EVENT_QUERY_SESSION_IDS_BATCH_SIZE:
            return await self._query_events_by_session_id_batches(query)
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            return await self._query_events(cur, query, safe_insert_xid=None)

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
        async with self._pool.connection() as conn, conn.cursor() as cur:
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
        async with self._pool.connection() as conn, conn.cursor() as cur:
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
        async with self._pool.connection() as conn, conn.cursor() as cur:
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
                SELECT id, encode(sha256(convert_to(
                    pending_state #>> '{{pending_tool_round,round_id}}',
                    'UTF8'
                )), 'hex')
                FROM candidates
                WHERE jsonb_typeof(
                    pending_state #> '{{pending_tool_round,round_id}}'
                ) = 'string'
                UNION
                SELECT candidates.id, encode(sha256(convert_to(
                    pending_call ->> 'tool_call_id',
                    'UTF8'
                )), 'hex')
                FROM candidates
                CROSS JOIN LATERAL jsonb_array_elements(
                    CASE
                        WHEN jsonb_typeof(
                            candidates.pending_state
                                #> '{{pending_tool_round,tool_calls}}'
                        ) = 'array'
                        THEN candidates.pending_state
                            #> '{{pending_tool_round,tool_calls}}'
                        ELSE '[]'::jsonb
                    END
                ) AS pending_call
                WHERE jsonb_typeof(pending_call -> 'tool_call_id') = 'string'
            ),
            pending_action_event_types(event_type) AS (
                VALUES
                    ('tool.call.approval_requested'),
                    ('session.awaiting_user_input'),
                    ('session.interrupted'),
                    ('tool.call.started'),
                    ('tool.call.completed'),
                    ('tool.call.failed'),
                    ('tool.call.blocked'),
                    ('tool.call.approval_denied')
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
            matched_event_sequences AS (
                SELECT candidate_session_id, sequence
                FROM matched_action_events
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
                        AND (
                            source_event.pending_action_projection IS NOT NULL
                            OR source_event.pending_action_projection_bytes
                                > {MAX_PENDING_ACTION_RESULT_BYTES}
                        )
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
        async with self._pool.connection() as conn, conn.cursor() as cur:
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
            if preflight_eligible_ids:
                await cur.execute(source_size_sql, (preflight_eligible_ids,))
                for row in await cur.fetchall():
                    sequence_values = copy_json_value(row[3], "matched event sequences")
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

            processable_ids: list[str] = []
            materializable_ids: list[str] = []
            materialized_source_bytes = 0
            stopped_for_bytes = preflight_stopped_for_bytes
            for session_id in preflight_processable_ids:
                session = candidate_sessions[session_id]
                if (
                    session_id in oversized_ids
                    or session_id in overcomplex_ids
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
                    pending_events = copy_json_value(row[2], "pending events")
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
                        copy_json_value(_json_obj(row[1]), "checkpoint"),
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
        async with self._pool.connection() as conn, conn.cursor() as cur:
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
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        copied_messages = copy_transcript_messages(messages)
        await self._ensure_ready()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (session_id,))
                if await cur.fetchone() is None:
                    raise KeyError(f"Session not found: {session_id}")
                if copied_messages:
                    await _touch_session_activity(cur, session_id, datetime.now(UTC))
                    await cur.executemany(
                        """
                        INSERT INTO cayu_transcript_messages (session_id, message)
                        VALUES (%s, %s)
                        """,
                        [
                            (session_id, _dumps(message.model_dump(mode="json")))
                            for message in copied_messages
                        ],
                    )
            await conn.commit()

    async def append_transcript_messages_and_transform_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint_transform: CheckpointTransform,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        copied_messages = copy_transcript_messages(messages)
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        updated_at = datetime.now(UTC)
        await self._ensure_ready()
        async with self._pool.connection() as conn:
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
                    transformed = copy_json_value(transformed, "checkpoint")
                    await _touch_session_activity(cur, session_id, updated_at)
                    if copied_messages:
                        await cur.executemany(
                            """
                            INSERT INTO cayu_transcript_messages (session_id, message)
                            VALUES (%s, %s)
                            """,
                            [
                                (session_id, _dumps(message.model_dump(mode="json")))
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
        async with self._pool.connection() as conn, conn.cursor() as cur:
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
        # The transcript role lives inside the stored ``message`` JSONB, so it is read
        # with ``message ->> 'role'`` rather than a dedicated column. This keeps the
        # existing transcript schema unchanged (no destructive NOT NULL migration) while
        # matching the SQLite store's role-filtering and stable, gap-free index semantics.
        role_clause = "WHERE role = %s" if query.role is not None else ""
        role_params: list[object] = [str(query.role)] if query.role is not None else []

        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (query.session_id,))
            if await cur.fetchone() is None:
                raise KeyError(f"Session not found: {query.session_id}")

            await cur.execute(
                f"""
                WITH ordered AS (
                    SELECT
                        message ->> 'role' AS role,
                        ROW_NUMBER() OVER (ORDER BY sequence ASC) - 1 AS transcript_index
                    FROM cayu_transcript_messages
                    WHERE session_id = %s
                )
                SELECT COUNT(*)
                FROM ordered
                {role_clause}
                """,
                [query.session_id, *role_params],
            )
            total_row = await cur.fetchone()
            total_records = int(total_row[0]) if total_row is not None else 0

            await cur.execute(
                f"""
                WITH ordered AS (
                    SELECT
                        message ->> 'role' AS role,
                        message,
                        ROW_NUMBER() OVER (ORDER BY sequence ASC) - 1 AS transcript_index
                    FROM cayu_transcript_messages
                    WHERE session_id = %s
                )
                SELECT transcript_index, message
                FROM ordered
                {role_clause}
                ORDER BY transcript_index ASC
                LIMIT %s OFFSET %s
                """,
                [query.session_id, *role_params, query.limit, query.offset],
            )
            rows = await cur.fetchall()
            records = [
                TranscriptRecord(index=row[0], message=Message(**_json_obj(row[1]))) for row in rows
            ]
            return TranscriptPage(
                records=filter_transcript_records(records, include_thinking=query.include_thinking),
                total_records=total_records,
            )

    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state must be a dictionary.")
        copied = copy_json_value(state, "checkpoint")
        updated_at = datetime.now(UTC)
        await self._ensure_ready()
        async with self._pool.connection() as conn:
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
        async with self._pool.connection() as conn:
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
                        transformed = copy_json_value(transformed, "checkpoint")
                        await _touch_session_activity(cur, session_id, updated_at)
                        await self._upsert_checkpoint(cur, session_id, transformed, updated_at)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            return await self._load_checkpoint(cur, session_id)

    async def load_interruption_cascade_marker(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
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
        return copy_json_value(_json_obj(row[0]), "checkpoint")

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
        async with self._pool.connection() as conn, conn.cursor() as cur:
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
        result = copy_json_object(result, "result")
        return await self._finish_task(
            task_id, TaskStatus.COMPLETED, result=result, error=None, worker_id=worker_id
        )

    async def fail_task(
        self, task_id: str, error: dict[str, Any], *, worker_id: str | None = None
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        error = copy_json_object(error, "error")
        return await self._finish_task(
            task_id, TaskStatus.FAILED, result=None, error=error, worker_id=worker_id
        )

    async def cancel_task(
        self,
        task_id: str,
        error: dict[str, Any] | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        copied_error = None if error is None else copy_json_object(error, "error")
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
        if task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
            raise ValueError(f"Task {task.id} is not claimed or running.")
        now = datetime.now(UTC)
        if task.lease_expires_at is None:
            raise ValueError(f"Task {task.id} has no active lease.")
        if task.lease_expires_at <= now:
            raise ValueError(f"Task {task.id} lease for worker {worker_id} has expired.")
        raise ValueError(f"Worker {worker_id} does not own task {task.id}.")

    async def _raise_task_release_error(
        self,
        cur: Any,
        task_id: str,
        worker_id: str,
    ) -> None:
        task = await self._require_task(cur, task_id)
        if task.session_id is not None:
            raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
        if task.status is not TaskStatus.CLAIMED:
            raise ValueError(f"Task {task.id} is not claimed.")
        await self._raise_task_active_lease_error(cur, task_id, worker_id)

    async def _raise_task_claim_attach_error(
        self,
        cur: Any,
        task_id: str,
        worker_id: str,
    ) -> None:
        task = await self._require_task(cur, task_id)
        if task.status is TaskStatus.RUNNING:
            if task.session_id is not None:
                raise ValueError(
                    f"Task {task.id} is already attached to session {task.session_id}."
                )
            raise ValueError(f"Task {task.id} is already running.")
        if task.status is not TaskStatus.CLAIMED:
            _ensure_not_terminal(task)
            raise ValueError(f"Task {task.id} is not claimed by worker {worker_id}.")
        if task.session_id is not None:
            raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
        if task.worker_id != worker_id:
            raise ValueError(f"Worker {worker_id} does not own task {task.id}.")
        await self._raise_task_active_lease_error(cur, task_id, worker_id)


def _new_id() -> str:
    from uuid import uuid4

    return str(uuid4())


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _checkpoint_row_values(
    session_id: str,
    checkpoint: dict[str, Any],
    updated_at: datetime,
) -> tuple[object, ...]:
    from cayu.runtime.pending_actions import pending_action_checkpoint_metrics

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
    if type(value) is not str or not value.strip():
        raise ValueError("error must be a non-empty string.")
    return value


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
