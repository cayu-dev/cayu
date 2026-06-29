from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, LiteralString, cast

from psycopg.errors import ForeignKeyViolation, UniqueViolation
from psycopg_pool import AsyncConnectionPool

from cayu._validation import (
    copy_json_object,
    copy_json_value,
    copy_label_map,
    require_clean_nonblank,
    require_nonblank,
)
from cayu.core.events import Event, EventType, copy_event
from cayu.core.messages import Message
from cayu.runtime.event_watchers import (
    EventWatcherClaim,
    EventWatcherDelivery,
    EventWatcherDeliveryStatus,
    EventWatcherState,
    EventWatcherStore,
)
from cayu.runtime.sessions import (
    DELETE_BLOCKED_SESSION_STATUSES,
    CheckpointTransform,
    EventQuery,
    EventRecord,
    EventSummary,
    LabelSelectorOperator,
    RunRequest,
    Session,
    SessionIdentity,
    SessionListResult,
    SessionOutcome,
    SessionQuery,
    SessionStatus,
    SessionStore,
    TranscriptPage,
    TranscriptQuery,
    TranscriptRecord,
    _validate_status_set,
    copy_event_query,
    copy_run_request,
    copy_session,
    copy_session_identity,
    copy_session_query,
    copy_transcript_messages,
    copy_transcript_query,
    decode_session_cursor,
    session_next_cursor,
    session_order_is_descending,
    session_outcome,
    session_sort_column,
)
from cayu.runtime.tasks import (
    Task,
    TaskCreate,
    TaskQuery,
    TaskStatus,
    TaskStore,
    _can_attach_claimed_task,
    _copy_optional_status_payload,
    _copy_optional_status_reason,
    _ensure_can_hold_task,
    _ensure_can_resume_task,
    _ensure_can_transition,
    _ensure_claim_query_supported,
    _raise_task_worker_start_error,
    _task_from_create,
    copy_task_create,
    copy_task_query,
)
from cayu.storage import _postgres_support as pg_support
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
_EVENT_QUERY_SESSION_IDS_BATCH_SIZE = 500
_KNOWLEDGE_SEARCH_PAGE_SIZE = 500
_KNOWLEDGE_SEARCH_TOKEN_RE = re.compile(r"\w+")
_TASK_RETURNING_COLUMNS = (
    "task.id, task.type, task.title, task.description, task.status, task.session_id, "
    "task.parent_task_id, task.assigned_agent_name, task.worker_id, task.lease_expires_at, "
    "task.status_reason, task.status_payload, task.input, task.result, task.error, task.metadata, "
    "task.created_at, task.updated_at, task.started_at, task.completed_at"
)


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
        causal_budget_id=query.causal_budget_id,
        event_type=query.event_type,
        agent_name=query.agent_name,
        environment_name=query.environment_name,
        workflow_name=query.workflow_name,
        tool_name=query.tool_name,
        since=query.since,
        until=query.until,
        after_sequence=query.after_sequence,
        limit=query.limit,
        order_by=query.order_by,
    )


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
}


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

        All work happens inside a transaction-scoped advisory lock so concurrent
        stores serialize (ADR 0001, Decision 4):

        - ``validate``: read the recorded revision and fail fast unless this binary
          can operate against it. Never runs DDL.
        - ``create``: initialize the baseline schema on an empty database; otherwise
          validate. The dev/test/local default.
        - ``migrate``: apply pending forward revisions under the lock, then validate.
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT pg_advisory_xact_lock(%s)", (_SCHEMA_ADVISORY_LOCK_KEY,))
                if self._schema_mode is not schema.SchemaMode.VALIDATE:
                    await cur.execute(pg_support.MIGRATIONS_TABLE_DDL)
                state = await self._read_schema_state(cur)
                if self._schema_mode is schema.SchemaMode.VALIDATE:
                    schema.validate(state)
                elif self._schema_mode is schema.SchemaMode.CREATE:
                    if state.revision == schema.UNINITIALIZED:
                        await self._apply_pending(cur, state)
                    else:
                        schema.validate(state)
                else:  # MIGRATE
                    await self._apply_pending(cur, state)
                    schema.validate(await self._read_schema_state(cur))
            await conn.commit()

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
            entries = [
                entry
                for row in rows
                if (entry := await self._load_entry(cur, str(row[0]))) is not None
            ]
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
        hits: list[KnowledgeHit] = []
        remaining = query.max_bytes
        truncated = False
        for row in rows:
            if remaining <= 0:
                truncated = True
                break
            entry = await self._load_entry(cur, row[0])
            chunk = await self._load_chunk(cur, row[1])
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
                    chunk_count=len(await self._load_chunks(cur, entry.id)),
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
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
    ) -> Session:
        source_session_id = require_clean_nonblank(source_session_id, "source_session_id")
        fork = copy_session(fork)
        allowed_statuses = _validate_status_set(source_statuses, "source_statuses")
        if fork.parent_session_id != source_session_id:
            raise ValueError("Fork parent_session_id must match source_session_id.")
        if transcript_cursor is not None and transcript_cursor < 0:
            raise ValueError("transcript_cursor must be greater than or equal to 0.")

        await self._ensure_ready()
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    source_session = await self._load_for_update(cur, source_session_id)
                    if source_session is None:
                        raise KeyError(f"Session not found: {source_session_id}")
                    if source_session.status not in allowed_statuses:
                        raise ValueError(
                            f"Source session status is not forkable: {source_session.status}"
                        )
                    if fork.status != source_session.status:
                        raise ValueError(
                            "Fork status must match source session status: "
                            f"{fork.status} != {source_session.status}"
                        )
                    if fork.provider_name != source_session.provider_name:
                        raise ValueError(
                            "Fork provider_name must match source session provider_name: "
                            f"{fork.provider_name} != {source_session.provider_name}"
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
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                            INSERT INTO cayu_checkpoints (session_id, state, updated_at)
                            VALUES (%s, %s, %s)
                            """,
                            (
                                fork.id,
                                _dumps(copied_checkpoint),
                                pg_support.to_utc(fork.updated_at),
                            ),
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

    async def update_status(self, session_id: str, status: SessionStatus) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(status, SessionStatus):
            raise ValueError("Session status must be a SessionStatus.")
        await self._ensure_ready()
        updated_at = datetime.now(UTC)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE cayu_sessions SET status = %s, updated_at = %s WHERE id = %s
                    """,
                    (str(status), updated_at, session_id),
                )
                if cur.rowcount != 1:
                    raise KeyError(f"Session not found: {session_id}")
                loaded = await self._load(cur, session_id)
            await conn.commit()
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            return loaded

    async def update_model(self, session_id: str, model: str) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        model = require_clean_nonblank(model, "model")
        await self._ensure_ready()
        updated_at = datetime.now(UTC)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE cayu_sessions SET model = %s, updated_at = %s WHERE id = %s
                    """,
                    (model, updated_at, session_id),
                )
                if cur.rowcount != 1:
                    raise KeyError(f"Session not found: {session_id}")
                loaded = await self._load(cur, session_id)
            await conn.commit()
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            return loaded

    async def delete_session(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
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
                # ON DELETE CASCADE removes events/labels/checkpoint/transcript; the
                # self-FK is ON DELETE SET NULL so children keep loading with no parent.
                await cur.execute("DELETE FROM cayu_sessions WHERE id = %s", (session_id,))
            await conn.commit()

    async def update_labels(self, session_id: str, labels: dict[str, str]) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        new_labels = copy_label_map(labels, "labels", allow_reserved=False)
        await self._ensure_ready()
        updated_at = datetime.now(UTC)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE cayu_sessions SET updated_at = %s WHERE id = %s",
                    (updated_at, session_id),
                )
                if cur.rowcount != 1:
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
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE cayu_sessions SET metadata = %s, updated_at = %s WHERE id = %s",
                    (_dumps(new_metadata), updated_at, session_id),
                )
                if cur.rowcount != 1:
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
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE cayu_sessions
                    SET status = %s, updated_at = %s
                    WHERE id = %s AND status = ANY(%s)
                    """,
                    (
                        str(to_status),
                        updated_at,
                        session_id,
                        [str(status) for status in allowed_statuses],
                    ),
                )
                if cur.rowcount != 1:
                    loaded = await self._load(cur, session_id)
                    if loaded is None:
                        raise KeyError(f"Session not found: {session_id}")
                    raise ValueError(
                        f"Session status transition not allowed: {loaded.status} -> {to_status}"
                    )
                loaded = await self._load(cur, session_id)
            await conn.commit()
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
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
                    if loaded.status not in allowed_statuses:
                        raise ValueError(
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
                        UPDATE cayu_sessions SET status = %s, updated_at = %s WHERE id = %s
                        """,
                        (str(to_status), updated_at, session_id),
                    )
                    if transformed_checkpoint is not None:
                        await self._upsert_checkpoint(
                            cur, session_id, transformed_checkpoint, updated_at
                        )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
            return loaded.model_copy(update={"status": to_status, "updated_at": updated_at})

    async def append_event(self, session_id: str, event: Event) -> None:
        await self.append_events(session_id, [event])

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if type(events) is not list:
            raise TypeError("Session events must be a list.")

        copied_events: list[Event] = []
        seen_event_ids: set[str] = set()
        for event in events:
            if type(event) is not Event:
                raise TypeError("Session events must be Event instances.")
            copied_event = copy_event(event)
            if copied_event.session_id != session_id:
                raise ValueError("Event session_id does not match target session.")
            if copied_event.id in seen_event_ids:
                raise ValueError(
                    f"Event already exists for session {session_id}: {copied_event.id}"
                )
            seen_event_ids.add(copied_event.id)
            copied_events.append(copied_event)

        await self._ensure_ready()
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    # Lock the session row so the per-session order assignment is
                    # serialized against concurrent appends to the same session.
                    await cur.execute(
                        "SELECT 1 FROM cayu_sessions WHERE id = %s FOR UPDATE",
                        (session_id,),
                    )
                    if await cur.fetchone() is None:
                        raise KeyError(f"Session not found: {session_id}")
                    if not copied_events:
                        await conn.commit()
                        return

                    await cur.execute(
                        """
                        SELECT COALESCE(MAX(session_order), 0)
                        FROM cayu_events
                        WHERE session_id = %s
                        """,
                        (session_id,),
                    )
                    order_row = await cur.fetchone()
                    # COALESCE(MAX(session_order), 0) always returns exactly one row.
                    next_order = order_row[0] if order_row is not None else 0
                    rows = []
                    for event in copied_events:
                        next_order += 1
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
                            )
                        )
                    await cur.executemany(
                        """
                        INSERT INTO cayu_events (
                            session_id, session_order, event_id, event_type, timestamp,
                            agent_name, environment_name, workflow_name, tool_name,
                            payload, event
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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

        clauses: list[str] = []
        params: list[object] = []

        if query.after_sequence is not None:
            clauses.append("cayu_events.sequence > %s")
            params.append(query.after_sequence)
        if query.session_id is not None:
            clauses.append("cayu_events.session_id = %s")
            params.append(query.session_id)
        if query.session_ids:
            placeholders = ", ".join("%s" for _ in query.session_ids)
            clauses.append(f"cayu_events.session_id IN ({placeholders})")
            params.extend(query.session_ids)
        if query.causal_budget_id is not None:
            clauses.append("cayu_sessions.causal_budget_id = %s")
            params.append(query.causal_budget_id)
        if query.since is not None:
            clauses.append("cayu_events.timestamp >= %s")
            params.append(pg_support.to_utc(query.since))
        if query.until is not None:
            clauses.append("cayu_events.timestamp < %s")
            params.append(pg_support.to_utc(query.until))
        if query.event_type is not None:
            clauses.append("cayu_events.event_type = %s")
            params.append(str(query.event_type))
        if query.agent_name is not None:
            clauses.append("cayu_events.agent_name = %s")
            params.append(query.agent_name)
        if query.environment_name is not None:
            clauses.append("cayu_events.environment_name = %s")
            params.append(query.environment_name)
        if query.workflow_name is not None:
            clauses.append("cayu_events.workflow_name = %s")
            params.append(query.workflow_name)
        if query.tool_name is not None:
            clauses.append("cayu_events.tool_name = %s")
            params.append(query.tool_name)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_direction = "DESC" if query.order_by.value == "sequence_desc" else "ASC"
        params.append(query.limit)

        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            # where_sql is built only from hard-coded clause literals; all values
            # are bound via %s params, so the assembled text carries no user input.
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    SELECT cayu_events.sequence, cayu_events.event
                    FROM cayu_events
                    JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
                    {where_sql}
                    ORDER BY cayu_events.sequence {order_direction}
                    LIMIT %s
                    """,
                ),
                params,
            )
            rows = await cur.fetchall()
            return [EventRecord(sequence=row[0], event=Event(**_json_obj(row[1]))) for row in rows]

    async def _query_events_by_session_id_batches(self, query: EventQuery) -> list[EventRecord]:
        records: list[EventRecord] = []
        for batch in _event_query_session_id_batches(query.session_ids):
            records.extend(
                await self.query_events(
                    _event_query_with_session_ids(query, session_ids=batch),
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
        query = copy_session_query(query)
        clauses: list[str] = []
        params: list[object] = []

        if query.status is not None:
            clauses.append("status = %s")
            params.append(str(query.status))
        if query.agent_name is not None:
            clauses.append("agent_name = %s")
            params.append(query.agent_name)
        if query.environment_name is not None:
            clauses.append("environment_name = %s")
            params.append(query.environment_name)
        if query.parent_session_id is not None:
            clauses.append("parent_session_id = %s")
            params.append(query.parent_session_id)
        if query.causal_budget_id is not None:
            clauses.append("causal_budget_id = %s")
            params.append(query.causal_budget_id)
        for key, value in query.labels.items():
            clauses.append(
                """
                EXISTS (
                    SELECT 1
                    FROM cayu_session_labels
                    WHERE cayu_session_labels.session_id = cayu_sessions.id
                      AND cayu_session_labels.key = %s
                      AND cayu_session_labels.value = %s
                )
                """
            )
            params.extend([key, value])
        for selector in query.label_selectors:
            if selector.operator == LabelSelectorOperator.EXISTS:
                clauses.append(
                    """
                    EXISTS (
                        SELECT 1
                        FROM cayu_session_labels
                        WHERE cayu_session_labels.session_id = cayu_sessions.id
                          AND cayu_session_labels.key = %s
                    )
                    """
                )
                params.append(selector.key)
            elif selector.operator == LabelSelectorOperator.NOT_EXISTS:
                clauses.append(
                    """
                    NOT EXISTS (
                        SELECT 1
                        FROM cayu_session_labels
                        WHERE cayu_session_labels.session_id = cayu_sessions.id
                          AND cayu_session_labels.key = %s
                    )
                    """
                )
                params.append(selector.key)
            else:
                placeholders = ", ".join("%s" for _ in selector.values)
                exists_sql = f"""
                    EXISTS (
                        SELECT 1
                        FROM cayu_session_labels
                        WHERE cayu_session_labels.session_id = cayu_sessions.id
                          AND cayu_session_labels.key = %s
                          AND cayu_session_labels.value IN ({placeholders})
                    )
                    """
                if selector.operator == LabelSelectorOperator.IN:
                    clauses.append(exists_sql)
                elif selector.operator == LabelSelectorOperator.NOT_IN:
                    clauses.append(f"NOT {exists_sql}")
                else:
                    raise ValueError(f"Unsupported label selector operator: {selector.operator}")
                params.extend([selector.key, *selector.values])

        where_filter = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_sql = pg_support.session_order_sql(query.order_by)

        # The paged query reuses the filters plus, when a cursor is given, a keyset
        # predicate; the COUNT uses only the filters so total_count is page-stable.
        page_clauses = list(clauses)
        page_params = list(params)
        sort_column = session_sort_column(query.order_by)
        if query.cursor is not None:
            cursor_dt, cursor_id = decode_session_cursor(query.cursor)
            comparison = "<" if session_order_is_descending(query.order_by) else ">"
            page_clauses.append(
                f"(({sort_column} {comparison} %s) OR ({sort_column} = %s AND id > %s))"
            )
            page_params.extend([cursor_dt, cursor_dt, cursor_id, query.limit + 1])
            pagination_sql = "LIMIT %s"
        else:
            page_params.extend([query.limit + 1, query.offset])
            pagination_sql = "LIMIT %s OFFSET %s"
        where_page = f"WHERE {' AND '.join(page_clauses)}" if page_clauses else ""

        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            # Interpolations are trusted: SESSION_COLUMNS is a constant, order_sql is
            # an enum-derived literal, the clauses are hard-coded; values bind via %s.
            total_count: int | None = None
            if query.include_total_count:
                await cur.execute(
                    cast("LiteralString", f"SELECT COUNT(*) FROM cayu_sessions {where_filter}"),
                    params,
                )
                count_row = await cur.fetchone()
                total_count = count_row[0] if count_row is not None else 0
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    SELECT {pg_support.SESSION_COLUMNS}
                    FROM cayu_sessions
                    {where_page}
                    ORDER BY {order_sql}, id ASC
                    {pagination_sql}
                    """,
                ),
                page_params,
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

    async def append_transcript_messages_and_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint: dict[str, Any],
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        copied_messages = copy_transcript_messages(messages)
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint state must be a dictionary.")
        copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
        updated_at = datetime.now(UTC)
        await self._ensure_ready()
        async with self._pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT 1 FROM cayu_sessions WHERE id = %s FOR UPDATE",
                        (session_id,),
                    )
                    if await cur.fetchone() is None:
                        raise KeyError(f"Session not found: {session_id}")
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
                    await self._upsert_checkpoint(cur, session_id, copied_checkpoint, updated_at)
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
            return TranscriptPage(
                records=[
                    TranscriptRecord(index=row[0], message=Message(**_json_obj(row[1])))
                    for row in rows
                ],
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
                await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (session_id,))
                if await cur.fetchone() is None:
                    raise KeyError(f"Session not found: {session_id}")
                await self._upsert_checkpoint(cur, session_id, copied, updated_at)
            await conn.commit()

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            return await self._load_checkpoint(cur, session_id)

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
            INSERT INTO cayu_checkpoints (session_id, state, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (session_id) DO UPDATE SET
                state = EXCLUDED.state,
                updated_at = EXCLUDED.updated_at
            """,
            (session_id, _dumps(checkpoint), updated_at),
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
        return task.model_copy(deep=True)

    async def load_task(self, task_id: str) -> Task | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            return await self._load_task(cur, task_id)

    async def list_tasks(self, query: TaskQuery | None = None) -> list[Task]:
        query = copy_task_query(query)
        clauses: list[str] = []
        params: list[object] = []

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
        worker_id: str | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        if worker_id is not None:
            worker_id = require_clean_nonblank(worker_id, "worker_id")
            if session_id is None:
                raise ValueError("Task worker handoff requires session_id.")
        await self._ensure_ready()
        now = datetime.now(UTC)
        async with self._pool.connection() as conn, conn.cursor() as cur:
            if worker_id is None:
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
            if _can_attach_claimed_task(task, now=now):
                if task.worker_id != worker_id:
                    raise ValueError(f"Worker {worker_id} does not own task {task.id}.")
                await cur.execute(
                    f"""
                    UPDATE cayu_tasks
                    SET session_id = COALESCE(%s, session_id),
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
                        session_id,
                        now,
                        task_id,
                        str(TaskStatus.RUNNING),
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
            if worker_id is not None:
                _raise_task_worker_start_error(task, worker_id, now=now)
            _ensure_can_transition(task, TaskStatus.RUNNING)
            raise ValueError(f"Task {task.id} cannot transition to running from {task.status}")

    async def complete_task(self, task_id: str, result: dict[str, Any]) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        result = copy_json_object(result, "result")
        return await self._finish_task(task_id, TaskStatus.COMPLETED, result=result, error=None)

    async def fail_task(self, task_id: str, error: dict[str, Any]) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        error = copy_json_object(error, "error")
        return await self._finish_task(task_id, TaskStatus.FAILED, result=None, error=error)

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
        order_sql = pg_support.task_order_sql(query.order_by)
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
                            started_at = COALESCE(task.started_at, %s),
                            updated_at = %s
                        FROM candidate
                        WHERE task.id = candidate.id
                        RETURNING {_TASK_RETURNING_COLUMNS}
                        """,
                    ),
                    [
                        str(TaskStatus.PENDING),
                        *params,
                        str(TaskStatus.RUNNING),
                        worker_id,
                        lease_expires_at,
                        now,
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
                    WHERE id = %s AND worker_id = %s AND status = %s
                      AND lease_expires_at IS NOT NULL AND lease_expires_at > %s
                    RETURNING {pg_support.TASK_COLUMNS}
                    """,
                    (
                        lease_expires_at,
                        now,
                        task_id,
                        worker_id,
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
                        str(TaskStatus.RUNNING),
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
        if query.status is not None and query.status is not TaskStatus.RUNNING:
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
                        str(TaskStatus.RUNNING),
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
    ) -> Task:
        await self._ensure_ready()
        now = datetime.now(UTC)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
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
                      AND status NOT IN (%s, %s, %s)
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
                    ),
                )
                if cur.rowcount != 1:
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
        if task.status is not TaskStatus.RUNNING:
            raise ValueError(f"Task {task.id} is not running.")
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
        if task.status is not TaskStatus.RUNNING:
            raise ValueError(f"Task {task.id} is not running.")
        if task.session_id is not None:
            raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
        await self._raise_task_active_lease_error(cur, task_id, worker_id)

    async def _raise_task_claim_attach_error(
        self,
        cur: Any,
        task_id: str,
        worker_id: str | None,
    ) -> None:
        task = await self._require_task(cur, task_id)
        if task.status is not TaskStatus.RUNNING:
            raise ValueError(f"Task {task.id} cannot transition to running from {task.status}")
        if task.session_id is not None:
            raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
        if task.worker_id != worker_id:
            raise ValueError(f"Worker {worker_id} does not own task {task.id}.")
        await self._raise_task_active_lease_error(cur, task_id, worker_id or "")


def _new_id() -> str:
    from uuid import uuid4

    return str(uuid4())


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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
        [group for term in query.all_terms for group in _structured_knowledge_search_token_groups(term)]
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
        positive_parts.append(
            " & ".join("(" + " | ".join(group) + ")" for group in all_groups)
        )
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
        [group for term in query.all_terms for group in _structured_knowledge_search_token_groups(term)]
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
        clause, clause_params = _postgres_document_match_clause(
            "(" + " | ".join(any_terms) + ")"
        )
        clauses.append(clause)
        params.extend(clause_params)
    for group in all_groups:
        clause, clause_params = _postgres_document_match_clause(
            "(" + " | ".join(group) + ")"
        )
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
    "PostgresEventWatcherStore",
    "PostgresKnowledgeStore",
    "PostgresSessionStore",
    "PostgresTaskStore",
]
