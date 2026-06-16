from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any, LiteralString, cast

from psycopg.errors import UniqueViolation
from psycopg_pool import AsyncConnectionPool

from cayu._validation import (
    copy_json_object,
    copy_json_value,
    require_clean_nonblank,
    require_nonblank,
)
from cayu.core.events import Event, EventType, copy_event
from cayu.core.messages import Message
from cayu.runtime.sessions import (
    CheckpointTransform,
    EventQuery,
    EventRecord,
    EventSummary,
    RunRequest,
    Session,
    SessionIdentity,
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
    session_outcome,
)
from cayu.runtime.tasks import (
    Task,
    TaskCreate,
    TaskQuery,
    TaskStatus,
    TaskStore,
    _ensure_can_transition,
    _task_from_create,
    copy_task_create,
    copy_task_query,
)
from cayu.storage import _postgres_support as pg_support
from cayu.storage import migrations as schema

# A fixed 63-bit advisory-lock key. Every Cayu store sharing a database takes this
# lock before touching schema, so concurrent creators/migrators (the production
# PostgresSessionStore + PostgresTaskStore on one pool) serialize: one runs the
# DDL, the rest wait and then validate (ADR 0001, Decision 4). The value is the
# ASCII bytes of "cayuschm" masked to stay positive (signed bigint); its only
# requirement is being a stable constant unlikely to collide with app locks.
_SCHEMA_ADVISORY_LOCK_KEY = 0x6361_7975_7363_686D & 0x7FFF_FFFF_FFFF_FFFF

# Per-revision forward-migration DDL, keyed by revision number. The baseline
# (revision 1) is applied from pg_support.SCHEMA_STATEMENTS, so it is not listed
# here; future additive/breaking revisions append their ALTER/CREATE statements.
_MIGRATION_STEPS: dict[int, tuple[str, ...]] = {}


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
                        await self._apply_baseline(cur)
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
            causal_budget_id=request.causal_budget_id or request.task_id or session_id,
            runtime_name=identity.runtime_name,
            runtime_version=identity.runtime_version,
            environment_name=request.environment_name,
            status=SessionStatus.PENDING,
            created_at=now,
            updated_at=now,
            metadata=copy_json_value(request.metadata, "metadata"),
        )
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
                await conn.commit()
            except UniqueViolation as exc:
                await conn.rollback()
                raise ValueError(f"Session already exists: {session.id}") from exc
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
        clauses: list[str] = []
        params: list[object] = []

        if query.after_sequence is not None:
            clauses.append("cayu_events.sequence > %s")
            params.append(query.after_sequence)
        if query.session_id is not None:
            clauses.append("cayu_events.session_id = %s")
            params.append(query.session_id)
        if query.causal_budget_id is not None:
            clauses.append("cayu_sessions.causal_budget_id = %s")
            params.append(query.causal_budget_id)
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
                    ORDER BY cayu_events.sequence ASC
                    LIMIT %s
                    """,
                ),
                params,
            )
            rows = await cur.fetchall()
            return [EventRecord(sequence=row[0], event=Event(**_json_obj(row[1]))) for row in rows]

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

    async def list_sessions(self, query: SessionQuery | None = None) -> list[Session]:
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

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_sql = pg_support.session_order_sql(query.order_by)
        params.extend([query.limit, query.offset])

        await self._ensure_ready()
        async with self._pool.connection() as conn, conn.cursor() as cur:
            # Interpolations are trusted: SESSION_COLUMNS is a constant, order_sql is
            # an enum-derived literal, where_sql is hard-coded clauses; values bind via %s.
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    SELECT {pg_support.SESSION_COLUMNS}
                    FROM cayu_sessions
                    {where_sql}
                    ORDER BY {order_sql}, id ASC
                    LIMIT %s OFFSET %s
                    """,
                ),
                params,
            )
            rows = await cur.fetchall()
            return [pg_support.session_from_row(row) for row in rows]

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
        return pg_support.session_from_row(row)

    async def _load_for_update(self, cur: Any, session_id: str) -> Session | None:
        await cur.execute(
            f"SELECT {pg_support.SESSION_COLUMNS} FROM cayu_sessions WHERE id = %s FOR UPDATE",
            (session_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return pg_support.session_from_row(row)

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
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        now = datetime.now(UTC)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
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
                if cur.rowcount != 1:
                    task = await self._require_task(cur, task_id)
                    _ensure_can_transition(task, TaskStatus.RUNNING)
                    raise ValueError(
                        f"Task {task.id} cannot transition to running from {task.status}"
                    )
                updated = await self._require_task(cur, task_id)
            await conn.commit()
            return updated.model_copy(deep=True)

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

    # -- internal helpers -------------------------------------------------

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
                        result = %s,
                        error = %s,
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


def _new_id() -> str:
    from uuid import uuid4

    return str(uuid4())


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _event_record_from_row(row: tuple[Any, Any] | None) -> EventRecord | None:
    """Build an EventRecord from a ``(sequence, event)`` row, or None for a missing row."""
    if row is None:
        return None
    return EventRecord(sequence=row[0], event=Event(**_json_obj(row[1])))


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
    "PostgresSessionStore",
    "PostgresTaskStore",
]
