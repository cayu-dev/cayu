from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from cayu._validation import (
    copy_json_object,
    copy_json_value,
    copy_label_map,
    require_clean_nonblank,
    require_nonblank,
)
from cayu.core.events import Event
from cayu.core.messages import Message
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
    _BoundedJsonSize,
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
    _task_from_create,
    copy_task_create,
    copy_task_query,
)
from cayu.storage import _session_store_sql as session_store_sql
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema

_EVENT_QUERY_SESSION_IDS_BATCH_SIZE = 500
_SQLITE_SESSION_MIN_REQUIRED_REVISION = 17
_SQL_DIALECT = session_store_sql.SessionStoreSqlDialect(
    placeholder="?",
    contains_style="sqlite_nocase_like",
    datetime_param=sqlite_support.format_datetime,
)

_T = TypeVar("_T")


def _like_contains_pattern(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _session_exists(connection: sqlite3.Connection, session_id: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM cayu_sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    return row is not None


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
        SELECT id, agent_name, provider_name, model, parent_session_id,
               causal_budget_id, runtime_name, runtime_version, environment_name,
               status, created_at, updated_at, last_activity_at, run_epoch,
               metadata_json
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
    return copy_json_value(json.loads(row["state_json"]), "checkpoint")


def _load_interruption_cascade_marker(
    connection: sqlite3.Connection,
    session_id: str,
) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT
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


# Columns needed to reconstruct an Event, in a stable order. The formerly-stored
# event_json blob duplicated exactly these (plus payload_json), so the store now
# rebuilds Events from the individual columns instead of parsing a redundant copy.
_EVENT_COLUMN_NAMES: tuple[str, ...] = (
    "session_id",
    "event_id",
    "event_type",
    "timestamp",
    "agent_name",
    "environment_name",
    "workflow_name",
    "tool_name",
    "payload_json",
)


def _event_from_row(row: sqlite3.Row) -> Event:
    """Reconstruct an :class:`Event` from its individual cayu_events columns."""
    return Event(
        type=row["event_type"],
        session_id=row["session_id"],
        id=row["event_id"],
        timestamp=row["timestamp"],
        agent_name=row["agent_name"],
        environment_name=row["environment_name"],
        workflow_name=row["workflow_name"],
        tool_name=row["tool_name"],
        payload=json.loads(row["payload_json"]),
    )


def _event_record_from_row(row: sqlite3.Row | None) -> EventRecord | None:
    if row is None:
        return None
    return EventRecord(
        sequence=row["sequence"],
        event=_event_from_row(row),
    )


class SQLiteSessionStore(SessionStore):
    """SQLite-backed session store for durable local runtime state."""

    def __init__(
        self,
        path: str | Path,
        *,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
    ) -> None:
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str:
            db_path = Path(require_nonblank(path, "path"))
        else:
            raise TypeError("SQLiteSessionStore path must be a string or Path.")
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")

        self.path = db_path
        self._schema_mode = schema_mode
        self._lock = asyncio.Lock()
        self._connection = self._connect(db_path)
        self._initialize_schema()
        # Hot-path queries run on a dedicated read-only connection in worker
        # threads so the event loop never blocks on SQLite I/O and reads never
        # queue behind the writer connection's transactions. In-memory
        # databases are private to their connection, so they fall back to the
        # writer connection (and its lock).
        if str(db_path) == ":memory:":
            self._read_connection = self._connection
            self._read_lock = self._lock
        else:
            self._read_connection = self._connect_read_only(db_path)
            self._read_lock = asyncio.Lock()

    async def _run_read(self, query: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run a read-only query off the event loop on the read connection."""
        async with self._read_lock:
            return await asyncio.to_thread(query, self._read_connection)

    async def _run_write(self, statement: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run a write statement off the event loop on the writer connection."""
        async with self._lock:
            return await asyncio.to_thread(statement, self._connection)

    async def create(
        self,
        request: RunRequest,
        *,
        identity: SessionIdentity,
    ) -> Session:
        request = copy_run_request(request)
        identity = copy_session_identity(identity)
        async with self._lock:
            session = sqlite_support.session_from_request(request, identity=identity)
            if session.parent_session_id == session.id:
                raise ValueError("Session cannot be its own parent.")
            try:
                with self._connection:
                    self._connection.execute(
                        """
                        INSERT INTO cayu_sessions (
                            id,
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
                            metadata_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session.id,
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
                            sqlite_support.json_dumps(session.metadata),
                        ),
                    )
                    if session.labels:
                        self._connection.executemany(
                            """
                            INSERT INTO cayu_session_labels (session_id, key, value)
                            VALUES (?, ?, ?)
                            """,
                            sqlite_support.session_label_row_values(session),
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

        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                source_session = _validate_session_fork_source(
                    source_session=self._load_unlocked(source_session_id),
                    source_session_id=source_session_id,
                    fork=fork,
                    allowed_statuses=allowed_statuses,
                    expected_source_run_epoch=expected_source_run_epoch,
                )
                transcript_rows = self._connection.execute(
                    """
                    SELECT message_json
                    FROM cayu_transcript_messages
                    WHERE session_id = ?
                    ORDER BY sequence ASC
                    """,
                    (source_session_id,),
                ).fetchall()
                if transcript_cursor is None:
                    copied_messages = [
                        Message(**json.loads(row["message_json"])) for row in transcript_rows
                    ]
                else:
                    if transcript_cursor > len(transcript_rows):
                        raise ValueError(
                            "transcript_cursor is greater than source transcript length."
                        )
                    copied_messages = [
                        Message(**json.loads(row["message_json"]))
                        for row in transcript_rows[:transcript_cursor]
                    ]
                copied_checkpoint = None
                if checkpoint_transform is not None:
                    copied_checkpoint = checkpoint_transform(
                        source_session,
                        self._load_checkpoint_unlocked(source_session_id),
                    )
                    if copied_checkpoint is not None:
                        copied_checkpoint = copy_json_value(
                            copied_checkpoint,
                            "checkpoint",
                        )

                self._connection.execute(
                    """
                    INSERT INTO cayu_sessions (
                        id,
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
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    sqlite_support.session_to_row_values(fork),
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
                            message_json
                        )
                        VALUES (?, ?, ?)
                        """,
                        [
                            (
                                fork.id,
                                str(message.role),
                                sqlite_support.json_dumps(message.model_dump(mode="json")),
                            )
                            for message in copied_messages
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

    async def update_model(self, session_id: str, model: str) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        model = require_clean_nonblank(model, "model")
        updated_at = datetime.now(UTC)
        expected_run_epoch = _current_session_run_epoch(session_id)
        async with self._lock:
            with self._connection:
                epoch_clause = "" if expected_run_epoch is None else " AND run_epoch = ?"
                params: list[object] = [
                    model,
                    sqlite_support.format_datetime(updated_at),
                    sqlite_support.format_datetime(updated_at),
                    session_id,
                ]
                if expected_run_epoch is not None:
                    params.append(expected_run_epoch)
                cursor = self._connection.execute(
                    f"""
                    UPDATE cayu_sessions
                    SET model = ?, updated_at = ?, last_activity_at = ?
                    WHERE id = ?{epoch_clause}
                    """,
                    params,
                )
            if cursor.rowcount != 1:
                if expected_run_epoch is not None:
                    _raise_session_write_conflict(self._connection, session_id, expected_run_epoch)
                raise KeyError(f"Session not found: {session_id}")

            loaded = self._load_unlocked(session_id)
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            return loaded

    async def delete_session(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        blocked = [str(status) for status in DELETE_BLOCKED_SESSION_STATUSES]
        placeholders = ", ".join("?" for _ in blocked)
        async with self._lock:
            with self._connection:
                # Guard the status check inside the statement so a concurrent
                # transition into a delete-blocked status cannot slip between a
                # separate SELECT and the DELETE. ON DELETE CASCADE removes
                # events/labels/checkpoint/transcript; the self-FK is ON DELETE
                # SET NULL so children keep loading with no parent.
                cursor = self._connection.execute(
                    f"""
                    DELETE FROM cayu_sessions
                    WHERE id = ? AND status NOT IN ({placeholders})
                    """,
                    (session_id, *blocked),
                )
            if cursor.rowcount == 0:
                # Nothing was deleted: either the session is already gone
                # (idempotent no-op) or it is in a delete-blocked status.
                row = self._connection.execute(
                    "SELECT status FROM cayu_sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                if row is None:
                    return  # idempotent: deleting a missing session is a no-op
                status = SessionStatus(row["status"])
                raise ValueError(
                    f"Cannot delete a session while it is {status}; "
                    f"interrupt it first: {session_id}"
                )

    async def update_labels(self, session_id: str, labels: dict[str, str]) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        new_labels = copy_label_map(labels, "labels", allow_reserved=False)
        updated_at = datetime.now(UTC)
        expected_run_epoch = _current_session_run_epoch(session_id)
        async with self._lock:
            with self._connection:
                epoch_clause = "" if expected_run_epoch is None else " AND run_epoch = ?"
                params: list[object] = [
                    sqlite_support.format_datetime(updated_at),
                    sqlite_support.format_datetime(updated_at),
                    session_id,
                ]
                if expected_run_epoch is not None:
                    params.append(expected_run_epoch)
                cursor = self._connection.execute(
                    "UPDATE cayu_sessions SET updated_at = ?, last_activity_at = ? "
                    f"WHERE id = ?{epoch_clause}",
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
            return loaded

    async def update_metadata(self, session_id: str, metadata: dict[str, Any]) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        new_metadata = copy_json_value(metadata, "metadata")
        if type(new_metadata) is not dict:
            raise TypeError("Session metadata must be an object.")
        updated_at = datetime.now(UTC)
        expected_run_epoch = _current_session_run_epoch(session_id)
        async with self._lock:
            with self._connection:
                epoch_clause = "" if expected_run_epoch is None else " AND run_epoch = ?"
                params: list[object] = [
                    sqlite_support.json_dumps(new_metadata),
                    sqlite_support.format_datetime(updated_at),
                    sqlite_support.format_datetime(updated_at),
                    session_id,
                ]
                if expected_run_epoch is not None:
                    params.append(expected_run_epoch)
                cursor = self._connection.execute(
                    "UPDATE cayu_sessions SET metadata_json = ?, updated_at = ?, "
                    f"last_activity_at = ? WHERE id = ?{epoch_clause}",
                    params,
                )
                if cursor.rowcount != 1:
                    if expected_run_epoch is not None:
                        _raise_session_write_conflict(
                            self._connection, session_id, expected_run_epoch
                        )
                    raise KeyError(f"Session not found: {session_id}")
            loaded = self._load_unlocked(session_id)
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

        updated_at = datetime.now(UTC)
        async with self._lock:
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
            with self._connection:
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

            loaded = self._load_unlocked(session_id)
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

        updated_at = datetime.now(UTC)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, loaded)
                if loaded.status not in allowed_statuses:
                    raise SessionStatusConflict(
                        f"Session status transition not allowed: {loaded.status} -> {to_status}"
                    )
                transformed_checkpoint = checkpoint_transform(
                    loaded,
                    self._load_checkpoint_unlocked(session_id),
                )
                if transformed_checkpoint is not None:
                    transformed_checkpoint = copy_json_value(
                        transformed_checkpoint,
                        "checkpoint",
                    )

                placeholders = ", ".join("?" for _ in allowed_statuses)
                cursor = self._connection.execute(
                    f"""
                    UPDATE cayu_sessions
                    SET status = ?, updated_at = ?, last_activity_at = ?,
                        run_epoch = run_epoch + ?
                    WHERE id = ? AND status IN ({placeholders})
                    """,
                    (
                        str(to_status),
                        sqlite_support.format_datetime(updated_at),
                        sqlite_support.format_datetime(updated_at),
                        1 if to_status == SessionStatus.RUNNING else 0,
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
                self._connection.commit()
                transitioned = loaded.model_copy(
                    update={
                        "status": to_status,
                        "updated_at": updated_at,
                        "last_activity_at": updated_at,
                        "run_epoch": loaded.run_epoch + (to_status == SessionStatus.RUNNING),
                    }
                )
            except Exception:
                self._connection.rollback()
                raise

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
        now = datetime.now(UTC)
        placeholders = ", ".join("?" for _ in allowed_statuses)
        async with self._lock:
            with self._connection:
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
            if cursor.rowcount != 1:
                if not self._session_exists_unlocked(session_id):
                    raise KeyError(f"Session not found: {session_id}")
                return None
            loaded = self._load_unlocked(session_id)
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            _activate_session_run_fence(loaded)
            return loaded

    async def release_run_fence(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        expected_run_epoch = _current_session_run_epoch(session_id)
        if expected_run_epoch is None:
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

    async def append_event(self, session_id: str, event: Event) -> None:
        await self.append_events(session_id, [event])

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        session_id, copied_events = _copy_session_event_batch(session_id, events)

        def statement(connection: sqlite3.Connection) -> None:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            if not copied_events:
                return

            try:
                with connection:
                    _touch_session_activity(connection, session_id, datetime.now(UTC))
                    rows = []
                    for event in copied_events:
                        lookup_key, projection, projection_bytes = (
                            pending_action_event_storage_values(event)
                        )
                        rows.append(
                            (
                                session_id,
                                event.id,
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
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        rows,
                    )
            except sqlite3.IntegrityError as exc:
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

        await self._run_write(statement)

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
        async with self._lock:
            if not self._session_exists_unlocked(session_id):
                raise KeyError(f"Session not found: {session_id}")

            total_row = self._connection.execute(
                """
                SELECT COUNT(*) AS total_events
                FROM cayu_events
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            count_rows = self._connection.execute(
                """
                SELECT event_type, COUNT(*) AS count
                FROM cayu_events
                WHERE session_id = ?
                GROUP BY event_type
                ORDER BY event_type ASC
                """,
                (session_id,),
            ).fetchall()
            latest_row = self._connection.execute(
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

    async def summarize_outcome(self, session_id: str) -> SessionOutcome:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._load_unlocked(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")

            terminal_row = self._connection.execute(
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
            retry_row = self._connection.execute(
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
            with connection:
                if session_id is None:
                    cursor = connection.execute(
                        "DELETE FROM cayu_events WHERE timestamp < ?",
                        (cutoff,),
                    )
                else:
                    cursor = connection.execute(
                        "DELETE FROM cayu_events WHERE session_id = ? AND timestamp < ?",
                        (session_id, cutoff),
                    )
            return cursor.rowcount

        return await self._run_write(statement)

    async def compact_transcript(self, session_id: str, *, keep_last: int) -> int:
        """Compact a session's transcript, keeping only its most recent messages.

        Retains the ``keep_last`` newest transcript messages (by insertion order)
        for ``session_id`` and deletes the rest, bounding transcript growth for
        long-lived sessions. Returns the number of messages deleted.
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
            return cursor.rowcount

        return await self._run_write(statement)

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
                cayu_sessions.updated_at
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
        checkpoint_preflight_sql = """
            SELECT
                cayu_checkpoints.session_id,
                cayu_checkpoints.pending_action_source_bytes AS pending_state_bytes,
                cayu_checkpoints.pending_action_tool_call_count AS pending_tool_call_count
            FROM cayu_checkpoints
            WHERE cayu_checkpoints.session_id IN (
                SELECT CAST(value AS TEXT) FROM json_each(?)
            )
        """
        projected_event_sql = "json(source_event.pending_action_projection_json)"
        pending_action_ctes = f"""
            WITH candidates AS ({selected_candidate_sql}),
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
                SELECT id,
                    cayu_pending_action_lookup_key(
                        json_extract(pending_state_json, '$.pending_tool_round.round_id')
                    )
                FROM candidates
                WHERE json_type(
                    pending_state_json,
                    '$.pending_tool_round.round_id'
                ) = 'text'
                UNION
                SELECT candidates.id,
                    cayu_pending_action_lookup_key(
                        json_extract(pending_call.value, '$.tool_call_id')
                    )
                FROM candidates
                JOIN json_each(
                    CASE
                        WHEN json_type(
                            candidates.pending_state_json,
                            '$.pending_tool_round.tool_calls'
                        ) = 'array'
                        THEN json_extract(
                            candidates.pending_state_json,
                            '$.pending_tool_round.tool_calls'
                        )
                        ELSE json('[]')
                    END
                ) AS pending_call
                WHERE json_type(pending_call.value, '$.tool_call_id') = 'text'
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
            matched_event_sequences AS (
                SELECT
                    matched_action.candidate_session_id,
                    matched_action.sequence
                FROM matched_action_sequences AS matched_action
                WHERE matched_action.sequence IS NOT NULL
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
                        AND (
                            source_event.pending_action_projection_json IS NOT NULL
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
                    session_size = _BoundedJsonSize(query.max_result_bytes)
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
                            copy_json_value(
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
        plan = session_store_sql.build_session_query_sql(query, dialect=_SQL_DIALECT)

        async with self._lock:
            total_count: int | None = None
            if query.include_total_count:
                total_count = self._connection.execute(
                    f"SELECT COUNT(*) FROM {session_source_sql} {plan.filter_where_sql}",
                    plan.filter_params,
                ).fetchone()[0]
            rows = self._connection.execute(
                f"""
                SELECT id, agent_name, provider_name, model, parent_session_id,
                       causal_budget_id, runtime_name, runtime_version, environment_name,
                       status, created_at, updated_at, last_activity_at, run_epoch,
                       metadata_json
                FROM {session_source_sql}
                {plan.page_where_sql}
                ORDER BY {plan.order_sql}, id ASC
                {plan.pagination_sql}
                """,
                plan.page_params,
            ).fetchall()
            has_more = len(rows) > query.limit
            rows = rows[: query.limit]
            labels_by_session_id = self._load_labels_for_sessions_unlocked(
                [row["id"] for row in rows]
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
            sessions=sessions, next_cursor=next_cursor, total_count=total_count
        )

    async def append_transcript_messages(
        self,
        session_id: str,
        messages: list[Message],
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        copied_messages = copy_transcript_messages(messages)

        def statement(connection: sqlite3.Connection) -> None:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            if not copied_messages:
                return
            with connection:
                _touch_session_activity(connection, session_id, datetime.now(UTC))
                connection.executemany(
                    """
                    INSERT INTO cayu_transcript_messages (
                        session_id,
                        role,
                        message_json
                    )
                    VALUES (?, ?, ?)
                    """,
                    [
                        (
                            session_id,
                            str(message.role),
                            sqlite_support.json_dumps(message.model_dump(mode="json")),
                        )
                        for message in copied_messages
                    ],
                )

        await self._run_write(statement)

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

        def statement(connection: sqlite3.Connection) -> None:
            try:
                connection.execute("BEGIN IMMEDIATE")
                session = self._load_unlocked(session_id)
                if session is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, session)
                transformed = checkpoint_transform(
                    session,
                    self._load_checkpoint_unlocked(session_id),
                )
                if transformed is None:
                    raise ValueError("Checkpoint transform must return a checkpoint.")
                transformed = copy_json_value(transformed, "checkpoint")
                _touch_session_activity(connection, session_id, updated_at)
                if copied_messages:
                    connection.executemany(
                        """
                        INSERT INTO cayu_transcript_messages (
                            session_id,
                            role,
                            message_json
                        )
                        VALUES (?, ?, ?)
                        """,
                        [
                            (
                                session_id,
                                str(message.role),
                                sqlite_support.json_dumps(message.model_dump(mode="json")),
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

    async def query_transcript(self, query: TranscriptQuery) -> TranscriptPage:
        query = copy_transcript_query(query)
        role_clause = "WHERE role = ?" if query.role is not None else ""
        role_params: list[object] = [str(query.role)] if query.role is not None else []

        async with self._lock:
            if not self._session_exists_unlocked(query.session_id):
                raise KeyError(f"Session not found: {query.session_id}")

            count_params: list[object] = [query.session_id, *role_params]
            total_row = self._connection.execute(
                f"""
                WITH ordered AS (
                    SELECT
                        role,
                        ROW_NUMBER() OVER (ORDER BY sequence ASC) - 1 AS transcript_index
                    FROM cayu_transcript_messages
                    WHERE session_id = ?
                )
                SELECT COUNT(*) AS total_records
                FROM ordered
                {role_clause}
                """,
                count_params,
            ).fetchone()
            total_records = int(total_row["total_records"])

            page_params: list[object] = [
                query.session_id,
                *role_params,
                query.limit,
                query.offset,
            ]
            rows = self._connection.execute(
                f"""
                WITH ordered AS (
                    SELECT
                        role,
                        message_json,
                        ROW_NUMBER() OVER (ORDER BY sequence ASC) - 1 AS transcript_index
                    FROM cayu_transcript_messages
                    WHERE session_id = ?
                )
                SELECT transcript_index, message_json
                FROM ordered
                {role_clause}
                ORDER BY transcript_index ASC
                LIMIT ? OFFSET ?
                """,
                page_params,
            ).fetchall()
            records = [
                TranscriptRecord(
                    index=row["transcript_index"],
                    message=Message(**json.loads(row["message_json"])),
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
        checkpoint = copy_json_value(state, "checkpoint")
        updated_at = datetime.now(UTC)

        def statement(connection: sqlite3.Connection) -> None:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            with connection:
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
                    sqlite_support.checkpoint_row_values(session_id, checkpoint, updated_at),
                )

        await self._run_write(statement)

    async def transform_checkpoint(
        self,
        session_id: str,
        checkpoint_transform: CheckpointTransform,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        updated_at = datetime.now(UTC)

        def statement(connection: sqlite3.Connection) -> None:
            try:
                connection.execute("BEGIN IMMEDIATE")
                session = self._load_unlocked(session_id)
                if session is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, session)
                transformed = checkpoint_transform(
                    session,
                    self._load_checkpoint_unlocked(session_id),
                )
                if transformed is not None:
                    transformed = copy_json_value(transformed, "checkpoint")
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
            except Exception:
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
    ) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        return await self._run_read(
            lambda connection: _load_interruption_cascade_marker(connection, session_id)
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
        return sqlite_support.connect(path, read_only=True)

    def _initialize_schema(self) -> None:
        sqlite_support.reconcile_schema(self._connection, self._schema_mode)
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

    def __init__(
        self,
        path: str | Path,
        *,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
    ) -> None:
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str:
            db_path = Path(require_nonblank(path, "path"))
        else:
            raise TypeError("SQLiteTaskStore path must be a string or Path.")
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")

        self.path = db_path
        self._schema_mode = schema_mode
        self._lock = asyncio.Lock()
        self._connection = self._connect(db_path)
        self._initialize_schema()

    async def create_task(self, request: TaskCreate) -> Task:
        request = copy_task_create(request)
        async with self._lock:
            task = _task_from_create(request)
            try:
                with self._connection:
                    self._connection.execute(
                        """
                        INSERT INTO cayu_tasks (
                            id,
                            type,
                            title,
                            description,
                            status,
                            session_id,
                            parent_task_id,
                            assigned_agent_name,
                            worker_id,
                            lease_expires_at,
                            status_reason,
                            status_payload_json,
                            input_json,
                            result_json,
                            error_json,
                            metadata_json,
                            created_at,
                            updated_at,
                            started_at,
                            completed_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        sqlite_support.task_to_row_values(task),
                    )
            except sqlite3.IntegrityError as exc:
                if self._task_exists_unlocked(task.id):
                    raise ValueError(f"Task already exists: {task.id}") from exc
                raise
            return task.model_copy(deep=True)

    async def load_task(self, task_id: str) -> Task | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        async with self._lock:
            return self._load_task_unlocked(task_id)

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

    async def start_task(
        self,
        task_id: str,
        *,
        session_id: str | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            now = datetime.now(UTC)
            with self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?,
                        session_id = COALESCE(?, session_id),
                        started_at = COALESCE(started_at, ?),
                        updated_at = ?
                    WHERE id = ? AND status = ?
                    """,
                    (
                        str(TaskStatus.RUNNING),
                        session_id,
                        sqlite_support.format_datetime(now),
                        sqlite_support.format_datetime(now),
                        task_id,
                        str(TaskStatus.PENDING),
                    ),
                )
            if cursor.rowcount == 1:
                updated = self._require_task_unlocked(task_id)
                return updated.model_copy(deep=True)
            task = self._require_task_unlocked(task_id)
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
        now = datetime.now(UTC)
        async with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?,
                        session_id = ?,
                        started_at = COALESCE(started_at, ?),
                        updated_at = ?
                    WHERE id = ?
                      AND status = ?
                      AND worker_id = ?
                      AND session_id IS NULL
                      AND lease_expires_at IS NOT NULL
                      AND lease_expires_at > ?
                    """,
                    (
                        str(TaskStatus.RUNNING),
                        session_id,
                        sqlite_support.format_datetime(now),
                        sqlite_support.format_datetime(now),
                        task_id,
                        str(TaskStatus.CLAIMED),
                        worker_id,
                        sqlite_support.format_datetime(now),
                    ),
                )
            if cursor.rowcount != 1:
                self._raise_task_claim_attach_error(task_id, worker_id)
            updated = self._require_task_unlocked(task_id)
            return updated.model_copy(deep=True)

    async def complete_task(
        self, task_id: str, result: dict[str, Any], *, worker_id: str | None = None
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        result = copy_json_object(result, "result")
        async with self._lock:
            return self._finish_task_unlocked(
                task_id,
                TaskStatus.COMPLETED,
                result=result,
                error=None,
                worker_id=worker_id,
            )

    async def fail_task(
        self, task_id: str, error: dict[str, Any], *, worker_id: str | None = None
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        error = copy_json_object(error, "error")
        async with self._lock:
            return self._finish_task_unlocked(
                task_id,
                TaskStatus.FAILED,
                result=None,
                error=error,
                worker_id=worker_id,
            )

    async def cancel_task(
        self,
        task_id: str,
        error: dict[str, Any] | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        copied_error = None if error is None else copy_json_object(error, "error")
        async with self._lock:
            return self._finish_task_unlocked(
                task_id,
                TaskStatus.CANCELLED,
                result=None,
                error=copied_error,
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
        now = datetime.now(UTC)
        async with self._lock:
            with self._connection:
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
                    """,
                    (
                        str(TaskStatus.PENDING),
                        sqlite_support.format_datetime(now),
                        task_id,
                        str(TaskStatus.PAUSED),
                        str(TaskStatus.BLOCKED),
                        str(TaskStatus.NEEDS_ATTENTION),
                    ),
                )
            if cursor.rowcount != 1:
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
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        lease_seconds = _validate_task_positive_int(lease_seconds, "lease_seconds")
        if query.status is not None and query.status is not TaskStatus.PENDING:
            return None
        clauses, params = self._task_filter_clauses(query)
        where_sql = " AND ".join(["status = ?", "session_id IS NULL", *clauses])
        # Claiming is always FIFO by creation time, independent of the query's
        # display ordering, so the oldest pending task is dispatched first.
        order_sql = sqlite_support.task_order_sql(TaskOrder.CREATED_AT_ASC)
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=lease_seconds)

        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    f"""
                    SELECT id
                    FROM cayu_tasks
                    WHERE {where_sql}
                    ORDER BY {order_sql}, id ASC
                    LIMIT 1
                    """,
                    [str(TaskStatus.PENDING), *params],
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
        extend_seconds: int = 300,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        extend_seconds = _validate_task_positive_int(extend_seconds, "extend_seconds")
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=extend_seconds)
        async with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET lease_expires_at = ?,
                        updated_at = ?
                    WHERE id = ? AND worker_id = ? AND status IN (?, ?)
                      AND lease_expires_at IS NOT NULL AND lease_expires_at > ?
                    """,
                    (
                        sqlite_support.format_datetime(lease_expires_at),
                        sqlite_support.format_datetime(now),
                        task_id,
                        worker_id,
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.RUNNING),
                        sqlite_support.format_datetime(now),
                    ),
                )
            if cursor.rowcount != 1:
                self._raise_task_active_lease_error(task_id, worker_id)
            updated = self._require_task_unlocked(task_id)
            return updated.model_copy(deep=True)

    async def release_task(self, task_id: str, worker_id: str) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        now = datetime.now(UTC)
        async with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?
                    WHERE id = ? AND worker_id = ? AND status = ?
                      AND session_id IS NULL
                      AND lease_expires_at IS NOT NULL AND lease_expires_at > ?
                    """,
                    (
                        str(TaskStatus.PENDING),
                        sqlite_support.format_datetime(now),
                        task_id,
                        worker_id,
                        str(TaskStatus.CLAIMED),
                        sqlite_support.format_datetime(now),
                    ),
                )
            if cursor.rowcount != 1:
                self._raise_task_release_error(task_id, worker_id)
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
                *clauses,
            ]
        )
        now = datetime.now(UTC)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
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
                        *params,
                        max_reclaims,
                    ],
                ).fetchall()
                task_ids = [row["id"] for row in rows]
                if task_ids:
                    self._connection.executemany(
                        """
                        UPDATE cayu_tasks
                        SET status = ?,
                            worker_id = NULL,
                            lease_expires_at = NULL,
                            updated_at = ?
                        WHERE id = ? AND status = ? AND session_id IS NULL
                        """,
                        [
                            (
                                str(TaskStatus.PENDING),
                                sqlite_support.format_datetime(now),
                                task_id,
                                str(TaskStatus.CLAIMED),
                            )
                            for task_id in task_ids
                        ],
                    )
                reclaimed = [self._require_task_unlocked(task_id) for task_id in task_ids]
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
        sqlite_support.reconcile_schema(self._connection, self._schema_mode)

    def _load_task_unlocked(self, task_id: str) -> Task | None:
        row = self._connection.execute(
            "SELECT * FROM cayu_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return sqlite_support.task_from_row(row)

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

    def _finish_task_unlocked(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
        worker_id: str | None = None,
    ) -> Task:
        now = datetime.now(UTC)
        # When a worker_id is given, only terminalize if that worker still owns an active
        # lease — a worker that lost its lease must not clobber a task another has reclaimed.
        owner_clause = ""
        owner_params: list[str] = []
        if worker_id is not None:
            owner_clause = (
                "\n                  AND worker_id = ?"
                "\n                  AND lease_expires_at IS NOT NULL AND lease_expires_at > ?"
            )
            owner_params = [worker_id, sqlite_support.format_datetime(now)]
        with self._connection:
            cursor = self._connection.execute(
                f"""
                UPDATE cayu_tasks
                SET status = ?,
                    status_reason = NULL,
                    status_payload_json = NULL,
                    result_json = ?,
                    error_json = ?,
                    worker_id = NULL,
                    lease_expires_at = NULL,
                    started_at = COALESCE(started_at, ?),
                    completed_at = ?,
                    updated_at = ?
                WHERE id = ?
                  AND status NOT IN (?, ?, ?){owner_clause}
                """,
                (
                    str(status),
                    None if result is None else sqlite_support.json_dumps(result),
                    None if error is None else sqlite_support.json_dumps(error),
                    sqlite_support.format_datetime(now),
                    sqlite_support.format_datetime(now),
                    sqlite_support.format_datetime(now),
                    task_id,
                    str(TaskStatus.COMPLETED),
                    str(TaskStatus.FAILED),
                    str(TaskStatus.CANCELLED),
                    *owner_params,
                ),
            )
        if cursor.rowcount != 1:
            if worker_id is not None:
                self._raise_task_active_lease_error(task_id, worker_id)
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
        now = datetime.now(UTC)
        async with self._lock:
            with self._connection:
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
                    ),
                )
            if cursor.rowcount != 1:
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

    def _raise_task_active_lease_error(self, task_id: str, worker_id: str) -> None:
        task = self._require_task_unlocked(task_id)
        if task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
            raise ValueError(f"Task {task.id} is not claimed or running.")
        now = datetime.now(UTC)
        if task.lease_expires_at is None:
            raise ValueError(f"Task {task.id} has no active lease.")
        if task.lease_expires_at <= now:
            raise ValueError(f"Task {task.id} lease for worker {worker_id} has expired.")
        raise ValueError(f"Worker {worker_id} does not own task {task.id}.")

    def _raise_task_release_error(self, task_id: str, worker_id: str) -> None:
        task = self._require_task_unlocked(task_id)
        if task.session_id is not None:
            raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
        if task.status is not TaskStatus.CLAIMED:
            raise ValueError(f"Task {task.id} is not claimed.")
        self._raise_task_active_lease_error(task_id, worker_id)

    def _raise_task_claim_attach_error(self, task_id: str, worker_id: str) -> None:
        task = self._require_task_unlocked(task_id)
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
        self._raise_task_active_lease_error(task_id, worker_id)


def _validate_task_positive_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1.")
    return value
