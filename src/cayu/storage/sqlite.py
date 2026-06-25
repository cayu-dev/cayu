from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cayu._validation import (
    copy_json_object,
    copy_json_value,
    copy_label_map,
    require_clean_nonblank,
    require_nonblank,
)
from cayu.core.events import Event, copy_event
from cayu.core.messages import Message
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
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema

_EVENT_QUERY_SESSION_IDS_BATCH_SIZE = 500


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
    )


def _event_record_from_row(row: sqlite3.Row | None) -> EventRecord | None:
    if row is None:
        return None
    return EventRecord(
        sequence=row["sequence"],
        event=Event(**json.loads(row["event_json"])),
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
                            metadata_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    ) -> Session:
        source_session_id = require_clean_nonblank(source_session_id, "source_session_id")
        fork = copy_session(fork)
        allowed_statuses = _validate_status_set(source_statuses, "source_statuses")
        if fork.parent_session_id != source_session_id:
            raise ValueError("Fork parent_session_id must match source_session_id.")
        if transcript_cursor is not None and transcript_cursor < 0:
            raise ValueError("transcript_cursor must be greater than or equal to 0.")

        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                source_session = self._load_unlocked(source_session_id)
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
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        INSERT INTO cayu_checkpoints (session_id, state_json, updated_at)
                        VALUES (?, ?, ?)
                        """,
                        (
                            fork.id,
                            sqlite_support.json_dumps(copied_checkpoint),
                            sqlite_support.format_datetime(fork.updated_at),
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
        async with self._lock:
            row = self._connection.execute(
                """
                SELECT id, agent_name, provider_name, model, parent_session_id,
                       causal_budget_id, runtime_name, runtime_version, environment_name,
                       status, created_at,
                       updated_at, metadata_json
                FROM cayu_sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return sqlite_support.session_from_row(
                row,
                labels=self._load_labels_unlocked(session_id),
            )

    async def update_status(self, session_id: str, status: SessionStatus) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(status, SessionStatus):
            raise ValueError("Session status must be a SessionStatus.")

        updated_at = datetime.now(UTC)
        async with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_sessions
                    SET status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (str(status), sqlite_support.format_datetime(updated_at), session_id),
                )
            if cursor.rowcount != 1:
                raise KeyError(f"Session not found: {session_id}")

            loaded = self._load_unlocked(session_id)
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            return loaded

    async def update_model(self, session_id: str, model: str) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        model = require_clean_nonblank(model, "model")
        updated_at = datetime.now(UTC)
        async with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_sessions
                    SET model = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (model, sqlite_support.format_datetime(updated_at), session_id),
                )
            if cursor.rowcount != 1:
                raise KeyError(f"Session not found: {session_id}")

            loaded = self._load_unlocked(session_id)
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            return loaded

    async def delete_session(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            row = self._connection.execute(
                "SELECT status FROM cayu_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return  # idempotent: deleting a missing session is a no-op
            status = SessionStatus(row["status"])
            if status in DELETE_BLOCKED_SESSION_STATUSES:
                raise ValueError(
                    f"Cannot delete a session while it is {status}; "
                    f"interrupt it first: {session_id}"
                )
            with self._connection:
                # ON DELETE CASCADE removes events/labels/checkpoint/transcript; the
                # self-FK is ON DELETE SET NULL so children keep loading with no parent.
                self._connection.execute(
                    "DELETE FROM cayu_sessions WHERE id = ?",
                    (session_id,),
                )

    async def update_labels(self, session_id: str, labels: dict[str, str]) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        new_labels = copy_label_map(labels, "labels", allow_reserved=False)
        updated_at = datetime.now(UTC)
        async with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    "UPDATE cayu_sessions SET updated_at = ? WHERE id = ?",
                    (sqlite_support.format_datetime(updated_at), session_id),
                )
                if cursor.rowcount != 1:
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
        async with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    "UPDATE cayu_sessions SET metadata_json = ?, updated_at = ? WHERE id = ?",
                    (
                        sqlite_support.json_dumps(new_metadata),
                        sqlite_support.format_datetime(updated_at),
                        session_id,
                    ),
                )
                if cursor.rowcount != 1:
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
            placeholders = ", ".join("?" for _ in allowed_statuses)
            params: list[object] = [
                str(to_status),
                sqlite_support.format_datetime(updated_at),
                session_id,
                *[str(status) for status in allowed_statuses],
            ]
            with self._connection:
                cursor = self._connection.execute(
                    f"""
                    UPDATE cayu_sessions
                    SET status = ?, updated_at = ?
                    WHERE id = ? AND status IN ({placeholders})
                    """,
                    params,
                )
            if cursor.rowcount != 1:
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                raise ValueError(
                    f"Session status transition not allowed: {loaded.status} -> {to_status}"
                )

            loaded = self._load_unlocked(session_id)
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

        updated_at = datetime.now(UTC)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                if loaded.status not in allowed_statuses:
                    raise ValueError(
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
                    SET status = ?, updated_at = ?
                    WHERE id = ? AND status IN ({placeholders})
                    """,
                    (
                        str(to_status),
                        sqlite_support.format_datetime(updated_at),
                        session_id,
                        *(str(status) for status in allowed_statuses),
                    ),
                )
                if cursor.rowcount != 1:
                    current = self._load_unlocked(session_id)
                    if current is None:
                        raise KeyError(f"Session not found: {session_id}")
                    raise ValueError(
                        f"Session status transition not allowed: {current.status} -> {to_status}"
                    )
                if transformed_checkpoint is not None:
                    self._connection.execute(
                        """
                        INSERT INTO cayu_checkpoints (session_id, state_json, updated_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT(session_id) DO UPDATE SET
                            state_json = excluded.state_json,
                            updated_at = excluded.updated_at
                        """,
                        (
                            session_id,
                            sqlite_support.json_dumps(transformed_checkpoint),
                            sqlite_support.format_datetime(updated_at),
                        ),
                    )
                self._connection.commit()
                transitioned = loaded.model_copy(
                    update={
                        "status": to_status,
                        "updated_at": updated_at,
                    }
                )
            except Exception:
                self._connection.rollback()
                raise

            return transitioned

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

        async with self._lock:
            if not self._session_exists_unlocked(session_id):
                raise KeyError(f"Session not found: {session_id}")
            if not copied_events:
                return

            try:
                with self._connection:
                    self._connection.executemany(
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
                            event_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
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
                                sqlite_support.json_dumps(event.model_dump(mode="json")),
                            )
                            for event in copied_events
                        ],
                    )
            except sqlite3.IntegrityError as exc:
                existing_event_id = self._first_existing_event_id_unlocked(
                    session_id,
                    [event.id for event in copied_events],
                )
                if existing_event_id is not None:
                    raise ValueError(
                        f"Event already exists for session {session_id}: {existing_event_id}"
                    ) from exc
                raise

    async def load_events(self, session_id: str) -> list[Event]:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if not self._session_exists_unlocked(session_id):
                raise KeyError(f"Session not found: {session_id}")
            rows = self._connection.execute(
                """
                SELECT event_json
                FROM cayu_events
                WHERE session_id = ?
                ORDER BY sequence ASC
                """,
                (session_id,),
            ).fetchall()
            return [Event(**json.loads(row["event_json"])) for row in rows]

    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        query = copy_event_query(query)
        if len(query.session_ids) > _EVENT_QUERY_SESSION_IDS_BATCH_SIZE:
            return await self._query_events_by_session_id_batches(query)

        clauses: list[str] = []
        params: list[object] = []

        if query.after_sequence is not None:
            clauses.append("cayu_events.sequence > ?")
            params.append(query.after_sequence)
        if query.session_id is not None:
            clauses.append("cayu_events.session_id = ?")
            params.append(query.session_id)
        if query.session_ids:
            placeholders = ", ".join("?" for _ in query.session_ids)
            clauses.append(f"cayu_events.session_id IN ({placeholders})")
            params.extend(query.session_ids)
        if query.causal_budget_id is not None:
            clauses.append("cayu_sessions.causal_budget_id = ?")
            params.append(query.causal_budget_id)
        if query.since is not None:
            clauses.append("cayu_events.timestamp >= ?")
            params.append(sqlite_support.format_datetime(query.since))
        if query.until is not None:
            clauses.append("cayu_events.timestamp < ?")
            params.append(sqlite_support.format_datetime(query.until))
        if query.event_type is not None:
            clauses.append("cayu_events.event_type = ?")
            params.append(str(query.event_type))
        if query.agent_name is not None:
            clauses.append("cayu_events.agent_name = ?")
            params.append(query.agent_name)
        if query.environment_name is not None:
            clauses.append("cayu_events.environment_name = ?")
            params.append(query.environment_name)
        if query.workflow_name is not None:
            clauses.append("cayu_events.workflow_name = ?")
            params.append(query.workflow_name)
        if query.tool_name is not None:
            clauses.append("cayu_events.tool_name = ?")
            params.append(query.tool_name)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(query.limit)

        async with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT cayu_events.sequence, cayu_events.event_json
                FROM cayu_events
                JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
                {where_sql}
                ORDER BY cayu_events.sequence ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [
                EventRecord(
                    sequence=row["sequence"],
                    event=Event(**json.loads(row["event_json"])),
                )
                for row in rows
            ]

    async def _query_events_by_session_id_batches(self, query: EventQuery) -> list[EventRecord]:
        records: list[EventRecord] = []
        for batch in _event_query_session_id_batches(query.session_ids):
            records.extend(
                await self.query_events(
                    _event_query_with_session_ids(query, session_ids=batch),
                )
            )
        records.sort(key=lambda record: record.sequence)
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
                """
                SELECT sequence, event_json
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
                latest_event=(
                    None
                    if latest_row is None
                    else EventRecord(
                        sequence=latest_row["sequence"],
                        event=Event(**json.loads(latest_row["event_json"])),
                    )
                ),
            )

    async def summarize_outcome(self, session_id: str) -> SessionOutcome:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._load_unlocked(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")

            terminal_row = self._connection.execute(
                """
                SELECT sequence, event_json
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
                """
                SELECT sequence, event_json
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

    async def list_sessions(self, query: SessionQuery | None = None) -> SessionListResult:
        query = copy_session_query(query)
        clauses: list[str] = []
        params: list[object] = []

        if query.status is not None:
            clauses.append("status = ?")
            params.append(str(query.status))
        if query.agent_name is not None:
            clauses.append("agent_name = ?")
            params.append(query.agent_name)
        if query.environment_name is not None:
            clauses.append("environment_name = ?")
            params.append(query.environment_name)
        if query.parent_session_id is not None:
            clauses.append("parent_session_id = ?")
            params.append(query.parent_session_id)
        if query.causal_budget_id is not None:
            clauses.append("causal_budget_id = ?")
            params.append(query.causal_budget_id)
        for key, value in query.labels.items():
            clauses.append(
                """
                EXISTS (
                    SELECT 1
                    FROM cayu_session_labels
                    WHERE cayu_session_labels.session_id = cayu_sessions.id
                      AND cayu_session_labels.key = ?
                      AND cayu_session_labels.value = ?
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
                          AND cayu_session_labels.key = ?
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
                          AND cayu_session_labels.key = ?
                    )
                    """
                )
                params.append(selector.key)
            else:
                placeholders = ", ".join("?" for _ in selector.values)
                exists_sql = f"""
                    EXISTS (
                        SELECT 1
                        FROM cayu_session_labels
                        WHERE cayu_session_labels.session_id = cayu_sessions.id
                          AND cayu_session_labels.key = ?
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
        order_sql = sqlite_support.session_order_sql(query.order_by)

        # The paged query reuses the filters plus, when a cursor is given, a keyset
        # predicate; the COUNT uses only the filters so total_count is page-stable.
        page_clauses = list(clauses)
        page_params = list(params)
        sort_column = session_sort_column(query.order_by)
        if query.cursor is not None:
            cursor_dt, cursor_id = decode_session_cursor(query.cursor)
            # SQLite compares timestamps as TEXT; format the cursor datetime with the
            # same encoder used to store the column so the comparison is byte-exact.
            cursor_value = sqlite_support.format_datetime(cursor_dt)
            comparison = "<" if session_order_is_descending(query.order_by) else ">"
            page_clauses.append(
                f"(({sort_column} {comparison} ?) OR ({sort_column} = ? AND id > ?))"
            )
            page_params.extend([cursor_value, cursor_value, cursor_id])
            page_params.append(query.limit + 1)
            pagination_sql = "LIMIT ?"
        else:
            page_params.extend([query.limit + 1, query.offset])
            pagination_sql = "LIMIT ? OFFSET ?"
        where_page = f"WHERE {' AND '.join(page_clauses)}" if page_clauses else ""

        async with self._lock:
            total_count: int | None = None
            if query.include_total_count:
                total_count = self._connection.execute(
                    f"SELECT COUNT(*) FROM cayu_sessions {where_filter}",
                    params,
                ).fetchone()[0]
            rows = self._connection.execute(
                f"""
                SELECT id, agent_name, provider_name, model, parent_session_id,
                       causal_budget_id, runtime_name, runtime_version, environment_name,
                       status, created_at,
                       updated_at, metadata_json
                FROM cayu_sessions
                {where_page}
                ORDER BY {order_sql}, id ASC
                {pagination_sql}
                """,
                page_params,
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

        async with self._lock:
            if not self._session_exists_unlocked(session_id):
                raise KeyError(f"Session not found: {session_id}")
            if not copied_messages:
                return
            with self._connection:
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
                            session_id,
                            str(message.role),
                            sqlite_support.json_dumps(message.model_dump(mode="json")),
                        )
                        for message in copied_messages
                    ],
                )

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

        async with self._lock:
            if not self._session_exists_unlocked(session_id):
                raise KeyError(f"Session not found: {session_id}")
            with self._connection:
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
                                session_id,
                                str(message.role),
                                sqlite_support.json_dumps(message.model_dump(mode="json")),
                            )
                            for message in copied_messages
                        ],
                    )
                self._connection.execute(
                    """
                    INSERT INTO cayu_checkpoints (session_id, state_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        state_json = excluded.state_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        session_id,
                        sqlite_support.json_dumps(copied_checkpoint),
                        sqlite_support.format_datetime(updated_at),
                    ),
                )

    async def load_transcript(self, session_id: str) -> list[Message]:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if not self._session_exists_unlocked(session_id):
                raise KeyError(f"Session not found: {session_id}")
            rows = self._connection.execute(
                """
                SELECT message_json
                FROM cayu_transcript_messages
                WHERE session_id = ?
                ORDER BY sequence ASC
                """,
                (session_id,),
            ).fetchall()
            return [Message(**json.loads(row["message_json"])) for row in rows]

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
            return TranscriptPage(
                records=[
                    TranscriptRecord(
                        index=row["transcript_index"],
                        message=Message(**json.loads(row["message_json"])),
                    )
                    for row in rows
                ],
                total_records=total_records,
            )

    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state must be a dictionary.")
        checkpoint = copy_json_value(state, "checkpoint")
        updated_at = datetime.now(UTC)

        async with self._lock:
            if not self._session_exists_unlocked(session_id):
                raise KeyError(f"Session not found: {session_id}")
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO cayu_checkpoints (session_id, state_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        state_json = excluded.state_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        session_id,
                        sqlite_support.json_dumps(checkpoint),
                        sqlite_support.format_datetime(updated_at),
                    ),
                )

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            return self._load_checkpoint_unlocked(session_id)

    def _load_checkpoint_unlocked(self, session_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
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

    async def close(self) -> None:
        async with self._lock:
            self._connection.close()

    def _connect(self, path: Path) -> sqlite3.Connection:
        return sqlite_support.connect(path)

    def _initialize_schema(self) -> None:
        sqlite_support.reconcile_schema(self._connection, self._schema_mode)

    def _load_unlocked(self, session_id: str) -> Session | None:
        row = self._connection.execute(
            """
            SELECT id, agent_name, provider_name, model, parent_session_id,
                   causal_budget_id, runtime_name, runtime_version, environment_name, status, created_at,
                   updated_at, metadata_json
            FROM cayu_sessions
            WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return sqlite_support.session_from_row(
            row,
            labels=self._load_labels_unlocked(session_id),
        )

    def _load_labels_unlocked(self, session_id: str) -> dict[str, str]:
        rows = self._connection.execute(
            """
            SELECT key, value
            FROM cayu_session_labels
            WHERE session_id = ?
            ORDER BY key ASC
            """,
            (session_id,),
        ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    def _load_labels_for_sessions_unlocked(
        self,
        session_ids: list[str],
    ) -> dict[str, dict[str, str]]:
        if not session_ids:
            return {}
        placeholders = ", ".join("?" for _ in session_ids)
        rows = self._connection.execute(
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
        row = self._connection.execute(
            "SELECT 1 FROM cayu_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        return row is not None

    def _first_existing_event_id_unlocked(
        self,
        session_id: str,
        event_ids: list[str],
    ) -> str | None:
        for event_id in event_ids:
            row = self._connection.execute(
                "SELECT 1 FROM cayu_events WHERE session_id = ? AND event_id = ?",
                (session_id, event_id),
            ).fetchone()
            if row is not None:
                return event_id
        return None


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
        worker_id: str | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        if worker_id is not None:
            worker_id = require_clean_nonblank(worker_id, "worker_id")
            if session_id is None:
                raise ValueError("Task worker handoff requires session_id.")
        async with self._lock:
            now = datetime.now(UTC)
            if worker_id is None:
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
            if _can_attach_claimed_task(task, now=now):
                if task.worker_id != worker_id:
                    raise ValueError(f"Worker {worker_id} does not own task {task.id}.")
                with self._connection:
                    cursor = self._connection.execute(
                        """
                        UPDATE cayu_tasks
                        SET session_id = COALESCE(?, session_id),
                            updated_at = ?
                        WHERE id = ?
                          AND status = ?
                          AND worker_id = ?
                          AND session_id IS NULL
                          AND lease_expires_at IS NOT NULL
                          AND lease_expires_at > ?
                        """,
                        (
                            session_id,
                            sqlite_support.format_datetime(now),
                            task_id,
                            str(TaskStatus.RUNNING),
                            worker_id,
                            sqlite_support.format_datetime(now),
                        ),
                    )
                if cursor.rowcount != 1:
                    self._raise_task_claim_attach_error(task_id, worker_id)
                updated = self._require_task_unlocked(task_id)
                return updated.model_copy(deep=True)
            if worker_id is not None:
                _raise_task_worker_start_error(task, worker_id, now=now)
            _ensure_can_transition(task, TaskStatus.RUNNING)
            raise ValueError(f"Task {task.id} cannot transition to running from {task.status}")

    async def complete_task(self, task_id: str, result: dict[str, Any]) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        result = copy_json_object(result, "result")
        async with self._lock:
            return self._finish_task_unlocked(
                task_id,
                TaskStatus.COMPLETED,
                result=result,
                error=None,
            )

    async def fail_task(self, task_id: str, error: dict[str, Any]) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        error = copy_json_object(error, "error")
        async with self._lock:
            return self._finish_task_unlocked(
                task_id,
                TaskStatus.FAILED,
                result=None,
                error=error,
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
        order_sql = sqlite_support.task_order_sql(query.order_by)
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
                        started_at = COALESCE(started_at, ?),
                        updated_at = ?
                    WHERE id = ? AND status = ?
                    """,
                    (
                        str(TaskStatus.RUNNING),
                        worker_id,
                        sqlite_support.format_datetime(lease_expires_at),
                        sqlite_support.format_datetime(now),
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
                    WHERE id = ? AND worker_id = ? AND status = ?
                      AND lease_expires_at IS NOT NULL AND lease_expires_at > ?
                    """,
                    (
                        sqlite_support.format_datetime(lease_expires_at),
                        sqlite_support.format_datetime(now),
                        task_id,
                        worker_id,
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
                        str(TaskStatus.RUNNING),
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
        if query.status is not None and query.status is not TaskStatus.RUNNING:
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
                        str(TaskStatus.RUNNING),
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
                                str(TaskStatus.RUNNING),
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
    ) -> Task:
        now = datetime.now(UTC)
        with self._connection:
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
                    started_at = COALESCE(started_at, ?),
                    completed_at = ?,
                    updated_at = ?
                WHERE id = ?
                  AND status NOT IN (?, ?, ?)
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
                ),
            )
        if cursor.rowcount != 1:
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
        if task.status is not TaskStatus.RUNNING:
            raise ValueError(f"Task {task.id} is not running.")
        now = datetime.now(UTC)
        if task.lease_expires_at is None:
            raise ValueError(f"Task {task.id} has no active lease.")
        if task.lease_expires_at <= now:
            raise ValueError(f"Task {task.id} lease for worker {worker_id} has expired.")
        raise ValueError(f"Worker {worker_id} does not own task {task.id}.")

    def _raise_task_release_error(self, task_id: str, worker_id: str) -> None:
        task = self._require_task_unlocked(task_id)
        if task.status is not TaskStatus.RUNNING:
            raise ValueError(f"Task {task.id} is not running.")
        if task.session_id is not None:
            raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
        self._raise_task_active_lease_error(task_id, worker_id)

    def _raise_task_claim_attach_error(self, task_id: str, worker_id: str | None) -> None:
        task = self._require_task_unlocked(task_id)
        if task.status is not TaskStatus.RUNNING:
            raise ValueError(f"Task {task.id} cannot transition to running from {task.status}")
        if task.session_id is not None:
            raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
        if task.worker_id != worker_id:
            raise ValueError(f"Worker {worker_id} does not own task {task.id}.")
        self._raise_task_active_lease_error(task_id, worker_id or "")


def _validate_task_positive_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1.")
    return value
